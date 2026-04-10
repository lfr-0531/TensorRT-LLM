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

import torch
from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper

_LANG_PREFIX = "model.language_model."


@register_mapper("HF", "Gemma4ForCausalLM")
@register_mapper("HF", "Gemma4ForConditionalGeneration")
class Gemma4HfWeightMapper(HfWeightMapper):

    def preprocess_weights(self, weights: dict) -> dict:
        """Rename HF checkpoint keys to TRT-LLM module names and handle
        buffers / special tensors that the generic loader cannot handle.

        Key transformations:
        1. Strip ``model.language_model.`` → ``model.`` prefix.
        2. Load ``layer_scalar`` buffers directly into the model (they are
           registered with ``register_buffer`` and therefore invisible to the
           parameter-based weight loader).
        3. For ``attention_k_eq_v`` layers, duplicate ``k_proj`` into ``v_proj``
           so the downstream ``qkv_proj`` fusing finds all three projections.
        """
        new_weights: dict = {}
        for key in list(weights.keys()):
            new_key = key
            # 1. Strip VLM language-model prefix
            if new_key.startswith(_LANG_PREFIX):
                new_key = "model." + new_key[len(_LANG_PREFIX):]
            new_weights[new_key] = weights[key]

        # 2. Load layer_scalar buffers directly into model
        layer_scalar_keys = [
            k for k in new_weights if k.endswith(".layer_scalar")
        ]
        for key in layer_scalar_keys:
            # key looks like "model.layers.N.layer_scalar"
            m = re.match(r"model\.layers\.(\d+)\.layer_scalar", key)
            if m:
                layer_idx = int(m.group(1))
                layer = self.model.model.layers[layer_idx]
                layer.layer_scalar.copy_(new_weights[key])
            del new_weights[key]

        # 3. Duplicate k_proj -> v_proj for k_eq_v layers
        config = self.model.config
        if getattr(config, "attention_k_eq_v", False):
            layer_types = getattr(config, "layer_types", [])
            for layer_idx, lt in enumerate(layer_types):
                if lt == "full_attention":
                    k_key = f"model.layers.{layer_idx}.self_attn.k_proj.weight"
                    v_key = f"model.layers.{layer_idx}.self_attn.v_proj.weight"
                    if k_key in new_weights and v_key not in new_weights:
                        new_weights[v_key] = new_weights[k_key]

        return new_weights

    def should_skip_module(self, module_name: str) -> bool:
        if self.model.config.tie_word_embeddings and module_name.startswith(
                "lm_head"):
            return True

        # Skip loading weights for embedding and lm_head if LoRA is enabled and has custom values
        if hasattr(self.model, "model") and hasattr(
                self.model.model, 'has_custom_embed_tokens'
        ) and self.model.model.has_custom_embed_tokens and module_name == "model.embed_tokens":
            return True
        if hasattr(
                self.model, 'has_custom_lm_head'
        ) and self.model.has_custom_lm_head and module_name == "lm_head":
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
