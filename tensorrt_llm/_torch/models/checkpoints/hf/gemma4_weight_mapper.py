# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

_LANG_PREFIX = "model.language_model."
_MODEL_PREFIX = "model."


@register_mapper("HF", "Gemma4ForCausalLM")
@register_mapper("HF", "Gemma4ForConditionalGeneration")
class Gemma4HfWeightMapper(HfWeightMapper):

    @property
    def _is_vlm(self) -> bool:
        """Check if the model is a VLM (has vision/audio towers) or text-only."""
        return hasattr(self.model, "vision_tower")

    def preprocess_weights(self, weights: dict) -> dict:
        """Rename HF checkpoint keys to TRT-LLM module names and handle
        buffers / special tensors that the generic loader cannot handle.

        For the text-only model (Gemma4ForCausalLM):
        1. Strip ``model.language_model.`` -> ``model.`` prefix.

        For the VLM (Gemma4ForConditionalGeneration):
        1. Strip ``model.`` prefix from all keys so that the VLM's
           ``load_weights`` can use ``filter_weights("language_model", ...)``
           to split by component: ``language_model.*``, ``vision_tower.*``,
           ``embed_vision.*``, etc.

        This method is idempotent: if the weights have already been processed
        (no key starts with the expected raw-checkpoint prefix), it returns
        them unchanged.  This is important because ``Gemma4ForCausalLM.load_weights``
        calls ``preprocess_weights`` again when invoked as a sub-model of the VLM.

        Common transformations:
        2. Load ``layer_scalar`` buffers directly into the model.
        3. For ``attention_k_eq_v`` layers, duplicate ``k_proj`` into ``v_proj``.
        """
        # Detect if any key still has the raw checkpoint prefix.
        # Raw Gemma4 checkpoint keys always start with "model.language_model.",
        # "model.vision_tower.", or "model.embed_vision.".  Keys like
        # "model.layers.*" (from the LLM after filter_weights) are already
        # processed and should not be re-transformed.
        _RAW_PREFIXES = (
            _LANG_PREFIX, "model.vision_tower.", "model.embed_vision.",
            "model.embed_audio.", "model.audio_tower.",
        )
        sample_keys = list(weights.keys())[:20]
        has_raw_prefix = any(
            any(k.startswith(rp) for rp in _RAW_PREFIXES)
            for k in sample_keys
        )
        if not has_raw_prefix:
            # Already preprocessed or text-only LLM sub-model call from VLM.
            # Still need to handle layer_scalar and k_eq_v.
            return self._handle_buffers_and_kvdup(weights)

        new_weights: dict = {}
        if self._is_vlm:
            # VLM: strip top-level "model." prefix only.
            for key in list(weights.keys()):
                new_key = key
                if new_key.startswith(_MODEL_PREFIX):
                    new_key = new_key[len(_MODEL_PREFIX):]
                new_weights[new_key] = weights[key]
        else:
            # Text-only: strip "model.language_model." -> "model."
            for key in list(weights.keys()):
                new_key = key
                if new_key.startswith(_LANG_PREFIX):
                    new_key = "model." + new_key[len(_LANG_PREFIX):]
                new_weights[new_key] = weights[key]

        return self._handle_buffers_and_kvdup(new_weights)

    def _handle_buffers_and_kvdup(self, weights: dict) -> dict:
        """Load layer_scalar buffers and duplicate k_proj for k_eq_v layers."""
        # Determine the layer scalar key pattern and accessor based on
        # whether any key starts with "language_model." (VLM sub-model
        # weights after filter_weights) or "model." (text-only).
        # Navigate to decoder layers regardless of model structure
        # (multimodal wrapper has .llm.model.layers, text-only has .model.layers)
        _root = self.model
        if hasattr(_root, 'llm'):  # Gemma4ForConditionalGeneration
            _layers = _root.llm.model.layers
        elif hasattr(_root, 'model') and hasattr(_root.model, 'layers'):
            _layers = _root.model.layers
        else:
            _layers = None

        sample = next(iter(weights), "")
        if sample.startswith("language_model."):
            scalar_pattern = r"language_model\.model\.layers\.(\d+)\.layer_scalar"
            get_layer = lambda idx: _layers[idx] if _layers else None
            key_tmpl = "language_model.model.layers.{}.self_attn.{}_proj.weight"
        else:
            scalar_pattern = r"model\.layers\.(\d+)\.layer_scalar"
            get_layer = lambda idx: _layers[idx] if _layers else None
            key_tmpl = "model.layers.{}.self_attn.{}_proj.weight"

        layer_scalar_keys = [
            k for k in weights if k.endswith(".layer_scalar")
        ]
        for key in layer_scalar_keys:
            m = re.match(scalar_pattern, key)
            if m:
                layer_idx = int(m.group(1))
                try:
                    layer = get_layer(layer_idx)
                    layer.layer_scalar.copy_(weights[key])
                except (AttributeError, IndexError):
                    pass
            del weights[key]

        config = self.model.config
        if getattr(config, "attention_k_eq_v", False):
            layer_types = getattr(config, "layer_types", [])
            for layer_idx, lt in enumerate(layer_types):
                if lt == "full_attention":
                    k_key = key_tmpl.format(layer_idx, "k")
                    v_key = key_tmpl.format(layer_idx, "v")
                    if k_key in weights and v_key not in weights:
                        weights[v_key] = weights[k_key]

        return weights

    def should_skip_module(self, module_name: str) -> bool:
        config = self.model.config
        if getattr(config, 'tie_word_embeddings', False) and module_name.startswith("lm_head"):
            return True

        # Determine the "inner" model (for VLM: self.model.llm, else: self.model)
        inner = self.model.llm if hasattr(self.model, 'llm') else self.model

        if (hasattr(inner, "model") and hasattr(inner.model, 'has_custom_embed_tokens')
                and inner.model.has_custom_embed_tokens
                and module_name == "model.embed_tokens"):
            return True
        if (hasattr(inner, 'has_custom_lm_head')
                and inner.has_custom_lm_head
                and module_name == "lm_head"):
            return True

        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def handle_manual_copy(self,
                           module_name: str,
                           module_weights: dict,
                           n: str,
                           p: nn.Parameter,
                           allow_partial_loading: bool = False) -> None:
        # Gemma uses weight+1 convention for RMSNorm weights.
        if 'norm' in module_name:
            if not allow_partial_loading:
                assert n in module_weights
            if n in module_weights:
                p.data.copy_(module_weights[n][:] + 1)
        else:
            super().handle_manual_copy(
                module_name,
                module_weights,
                n,
                p,
                allow_partial_loading=allow_partial_loading)
