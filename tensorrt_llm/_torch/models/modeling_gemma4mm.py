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
"""Gemma4 multimodal model: multimodal embedder, input processor,
and the Gemma4ForConditionalGeneration wrapper.

Vision and audio towers use native transformers models via
AutoModel.from_config() (requires transformers>=5.5.0).
"""

import copy
import dataclasses
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, Gemma4Config, PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (
    BaseMultimodalDummyInputsBuilder,
    BaseMultimodalInputProcessor,
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..modules.linear import Linear
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


class _RMSNormNoScale(nn.Module):
    """RMSNorm without learnable scale (for multimodal embedder pre-projection)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return normed.type_as(x)


# ---------------------------------------------------------------------------
# Multimodal embedder
# ---------------------------------------------------------------------------


class Gemma4MultimodalEmbedder(nn.Module):
    """Projects tower outputs into LM embedding space.

    Architecture (matching HF Gemma4MultimodalEmbedder):
        embedding_pre_projection_norm (RMSNorm, no learnable scale)
        -> embedding_projection (Linear, no bias)

    Only ``embedding_projection.weight`` exists in the checkpoint
    because the norm has no learnable parameters (with_scale=False).
    """

    def __init__(
        self,
        mm_hidden_size: int,
        text_hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        mapping=None,
    ):
        super().__init__()
        self.embedding_pre_projection_norm = _RMSNormNoScale(mm_hidden_size, eps=eps)
        self.embedding_projection = Linear(
            in_features=mm_hidden_size,
            out_features=text_hidden_size,
            bias=False,
            dtype=dtype,
            mapping=mapping,
        )

    def load_weights(self, weights: Dict):
        proj_weight = weights.get("embedding_projection.weight")
        if proj_weight is not None:
            self.embedding_projection.weight.data.copy_(proj_weight)

    @torch.inference_mode()
    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        normed = self.embedding_pre_projection_norm(inputs_embeds)
        return self.embedding_projection(normed)


# ---------------------------------------------------------------------------
# Input processor
# ---------------------------------------------------------------------------


class Gemma4InputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Preprocesses image inputs for Gemma4.

    Tries to use the HF ``Gemma4Processor`` if available in the installed
    transformers; otherwise falls back to manual tokenization + image
    processing using the image processor saved in the model directory.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._config = config
        self._tokenizer = tokenizer
        self._model_path = model_path
        self._dtype = getattr(config, "torch_dtype", torch.bfloat16)

        self._processor = None
        try:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                use_fast=self.use_fast,
            )
        except Exception:
            logger.warning(
                "Could not load AutoProcessor for Gemma4. "
                "Image preprocessing will use manual fallback."
            )

        self._image_processor = None
        if self._processor is not None and hasattr(self._processor, "image_processor"):
            self._image_processor = self._processor.image_processor
        elif self._processor is None:
            try:
                from transformers import AutoImageProcessor

                self._image_processor = AutoImageProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code
                )
            except Exception:
                logger.warning("Could not load image processor for Gemma4.")

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self):
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @nvtx_range("[Vision] preprocess")
    def _preprocess(self, inputs):
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data", {})
        if mm_data and "image" not in mm_data:
            raise KeyError("Expected image data in multimodal data for Gemma4.")

        images = mm_data.get("image")
        pixel_values = None
        image_position_ids = None

        if self._processor is not None and images is not None:
            do_rescale = self._image_processor.do_rescale
            if isinstance(images[0], torch.Tensor):
                do_rescale = False
            proc_out = self._processor(
                text=text_prompt,
                images=images,
                do_rescale=do_rescale,
                return_tensors="pt",
            ).to(dtype=self.dtype)
            input_ids = proc_out["input_ids"]
            pixel_values = proc_out.get("pixel_values")
            image_position_ids = proc_out.get("image_position_ids")
        else:
            input_ids = self._tokenizer(
                text_prompt,
                return_tensors="pt",
            )["input_ids"]
            if images is not None and self._image_processor is not None:
                img_out = self._image_processor(images=images, return_tensors="pt")
                pixel_values = img_out.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=self.dtype)
                image_position_ids = img_out.get("image_position_ids")

        return input_ids, pixel_values, image_position_ids

    @torch.inference_mode()
    def __call__(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        input_ids, pixel_values, image_position_ids = self._preprocess(inputs)
        multimodal_data = None
        if pixel_values is not None:
            image_data: Dict = {"pixel_values": pixel_values}
            if image_position_ids is not None:
                image_data["image_position_ids"] = image_position_ids
            multimodal_data = {
                "multimodal_data": {"image": image_data},
            }
        return input_ids[0].to(torch.int32).tolist(), multimodal_data


# ---------------------------------------------------------------------------
# Gemma4ForConditionalGeneration (multimodal wrapper)
# ---------------------------------------------------------------------------


@register_auto_model("Gemma4ForConditionalGeneration")
@register_input_processor(
    Gemma4InputProcessor,
    model_type="gemma4",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={"image": "<|image|>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.OPENAI,
    ),
)
class Gemma4ForConditionalGeneration(PreTrainedModel):
    """Gemma4 multimodal model: LLM + vision tower + multimodal embedder.

    Follows the Gemma3VLM pattern but adapted for Gemma4's architecture:
    - Custom vision tower with 2D RoPE and spatial pooling
    - Multimodal embedder with pre-projection RMSNorm (no scale)
    - Support for image_position_ids (2D patch coordinates)
    - mm_token_type_ids-based bidirectional masking
    """

    def __init__(self, model_config: ModelConfig[Gemma4Config]):
        if _is_disagg():
            raise NotImplementedError(
                "Gemma4ForConditionalGeneration does not support "
                "disaggregated inference yet. Please unset the "
                f"{_MULTIMODAL_ENV_NAME} environment variable, "
                "or set it to '0'."
            )

        config = model_config.pretrained_config
        super().__init__(config)

        self._device = "cuda"
        self.model_dtype = getattr(config, "torch_dtype", torch.bfloat16)
        self._top_config = config  # Preserve before post_config replaces it

        self.image_token_ids = torch.tensor(
            [config.image_token_id], dtype=torch.int32, device=self._device
        )
        self.audio_token_ids = (
            torch.tensor([config.audio_token_id], dtype=torch.int32, device=self._device)
            if getattr(config, "audio_token_id", None) is not None
            else None
        )

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        # --- Language model ---
        llm_model_config = self.get_sub_model_config(model_config_cp, "text_config")
        self.llm = Gemma4ForCausalLM(llm_model_config)

        # --- Vision tower (native transformers, eager mode) ---
        if config.vision_config is not None:
            self.vision_tower = AutoModel.from_config(config.vision_config).eval().to(self._device)
            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            vision_eps = config.vision_config.rms_norm_eps
            self.embed_vision = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=vision_hidden,
                    text_hidden_size=text_hidden,
                    eps=vision_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # --- Audio tower (native transformers, eager mode) ---
        if config.audio_config is not None:
            self.audio_tower = AutoModel.from_config(config.audio_config).eval().to(self._device)
            audio_hidden = getattr(
                config.audio_config, "output_proj_dims", config.audio_config.hidden_size
            )
            text_hidden = config.text_config.hidden_size
            audio_eps = config.audio_config.rms_norm_eps
            self.embed_audio = (
                Gemma4MultimodalEmbedder(
                    mm_hidden_size=audio_hidden,
                    text_hidden_size=text_hidden,
                    eps=audio_eps,
                    dtype=self.model_dtype,
                    mapping=model_config.mapping,
                )
                .eval()
                .to(self._device)
            )
        else:
            self.audio_tower = None
            self.embed_audio = None

        self.post_config()
        self.is_loaded = True

    @staticmethod
    def get_sub_model_config(
        model_config: ModelConfig[Gemma4Config],
        name: str,
    ) -> ModelConfig:
        assert name in ["text_config", "vision_config", "audio_config"], (
            f"Expected subconfig name to be 'text_config', 'vision_config', "
            f"or 'audio_config'. Got {name} instead."
        )
        pretrained_config = getattr(model_config.pretrained_config, name)
        quant_config = model_config.quant_config if name == "text_config" else None
        preferred_backend = "FLASHINFER" if name == "text_config" else "TRTLLM"
        sub_config: ModelConfig = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            attn_backend=preferred_backend,
            quant_config=quant_config,
        )
        if (
            hasattr(sub_config.pretrained_config, "torch_dtype")
            and sub_config.pretrained_config.torch_dtype is None
        ):
            sub_config.pretrained_config.torch_dtype = model_config.pretrained_config.torch_dtype
        return sub_config

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        # Gemma4 checkpoint keys: model.language_model.X -> need model.X for LLM
        # Remap: "model.language_model.layers.0..." -> "model.layers.0..."
        _LANG = "model.language_model."
        llm_weights = {}
        for k, v in weights.items():
            if k.startswith(_LANG):
                llm_weights["model." + k[len(_LANG) :]] = v
        self.llm.load_weights(llm_weights, weight_mapper)

        # Strip outer "model." for non-LLM components
        stripped = {
            (k[len("model.") :] if k.startswith("model.") else k): v for k, v in weights.items()
        }

        if self.vision_tower is not None:
            vit_weights = filter_weights("vision_tower", stripped)
            # Native transformers models use load_state_dict, not load_weights
            self.vision_tower.load_state_dict(vit_weights, strict=False)

        if self.embed_vision is not None:
            embed_v_weights = filter_weights("embed_vision", stripped)
            self.embed_vision.load_weights(embed_v_weights)

        if self.audio_tower is not None:
            audio_weights = filter_weights("audio_tower", stripped)
            self.audio_tower.load_state_dict(audio_weights, strict=False)

        if self.embed_audio is not None:
            embed_a_weights = filter_weights("embed_audio", stripped)
            self.embed_audio.load_weights(embed_a_weights)

    def post_config(self):
        self.config = self.llm.config
        self.model_config.pretrained_config = self.llm.config

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    @nvtx_range("[Vision] process")
    def _get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooling_k2 = self._top_config.vision_config.pooling_kernel_size**2
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        per_image_features = []
        for i in range(pixel_values.shape[0]):
            pv = pixel_values[i].unsqueeze(0)
            pp = image_position_ids[i].unsqueeze(0) if image_position_ids is not None else None

            max_patches = pv.shape[1]

            if pp is None:
                side = int(math.sqrt(max_patches))
                pp = torch.stack(
                    torch.meshgrid(
                        torch.arange(side, device=pv.device),
                        torch.arange(side, device=pv.device),
                        indexing="ij",
                    ),
                    dim=-1,
                ).reshape(1, -1, 2)

            output_length = max_patches // pooling_k2

            with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                output = self.vision_tower(pv, pp, output_length=output_length)
                hidden = output.last_hidden_state
                projected = self.embed_vision(hidden.unsqueeze(0).to(target_dtype)).squeeze(0)
            per_image_features.append(projected)

        return torch.cat(per_image_features, dim=0).contiguous()

    @nvtx_range("[Audio] process")
    def _get_audio_features(
        self,
        audio_features: torch.Tensor,
    ) -> torch.Tensor:
        """Process audio features through audio tower and embedder."""
        target_dtype = self.embed_audio.embedding_projection.weight.dtype
        with torch.autocast(device_type="cuda", dtype=self.model_dtype):
            output = self.audio_tower(audio_features)
            hidden = output.last_hidden_state
            projected = self.embed_audio(hidden.to(target_dtype))
        return projected.contiguous()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        multimodal_params = kwargs.get("multimodal_params", [])

        # --- Extract image data ---
        pixel_values_list = []
        image_position_ids_list = []
        # --- Extract audio data ---
        audio_features_list = []
        for mp in multimodal_params:
            img_data = mp.multimodal_data.get("image", {})
            pv = img_data.get("pixel_values")
            if pv is not None:
                pixel_values_list.append(pv)
                pid = img_data.get("image_position_ids")
                if pid is not None:
                    image_position_ids_list.append(pid)

            aud_data = mp.multimodal_data.get("audio", {})
            af = aud_data.get("audio_features")
            if af is not None:
                audio_features_list.append(af)

        mm_embeds = []
        all_mm_token_ids = []
        mm_token_type_ids = None

        # --- Process image features ---
        if len(pixel_values_list) > 0:
            pixel_values = torch.cat(pixel_values_list)
            image_position_ids = (
                torch.cat(image_position_ids_list)
                if len(image_position_ids_list) == len(pixel_values_list)
                else None
            )
            image_features = self._get_image_features(
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
            )
            mm_embeds.append(image_features)
            all_mm_token_ids.append(self.image_token_ids)

        # --- Process audio features ---
        if len(audio_features_list) > 0 and self.audio_tower is not None:
            audio_input = torch.cat(audio_features_list)
            audio_embeds = self._get_audio_features(audio_input)
            mm_embeds.append(audio_embeds)
            if self.audio_token_ids is not None:
                all_mm_token_ids.append(self.audio_token_ids)

        # Build integer mm_token_type_ids: 0=text, 1=image, 2=audio
        # _get_token_type_mask expects integer type IDs, not a boolean mask.
        if len(mm_embeds) > 0:
            mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
            mm_token_type_ids[torch.isin(input_ids, self.image_token_ids)] = 1
            if self.audio_token_ids is not None:
                mm_token_type_ids[torch.isin(input_ids, self.audio_token_ids)] = 2

        fuse_token_ids = torch.cat(all_mm_token_ids) if all_mm_token_ids else self.image_token_ids

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=fuse_token_ids,
            **kwargs,
        )
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            mm_token_type_ids=mm_token_type_ids,
            lora_params=kwargs.get("lora_params", None),
        )
        return logits

    @property
    def mm_token_ids(self):
        ids = [self.image_token_ids]
        if self.audio_token_ids is not None:
            ids.append(self.audio_token_ids)
        return torch.cat(ids) if len(ids) > 1 else self.image_token_ids
