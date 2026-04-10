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
"""Gemma4 multimodal model: vision tower, multimodal embedder, input processor,
and the Gemma4ForConditionalGeneration wrapper.

The vision tower is implemented as a self-contained eager-mode module that
matches the HuggingFace checkpoint weight layout, because the installed
transformers (4.57.3) does not include Gemma4 model classes.

Audio tower support is stubbed: the 26B-A4B-it checkpoint has no audio_config,
so the audio path is not exercised. It will be implemented when an
audio-capable checkpoint becomes available.
"""
import copy
import dataclasses
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.configs.gemma4 import (Gemma4AudioConfig,
                                                  Gemma4Config,
                                                  Gemma4VisionConfig)
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper

from ..._utils import nvtx_range
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ContentFormat,
                       ExtraProcessedInputs, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.linear import Linear
from ..modules.rms_norm import RMSNorm
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_multimodal_utils import fuse_input_embeds
from .modeling_utils import ModelConfig, filter_weights, register_auto_model

_MULTIMODAL_ENV_NAME = "TLLM_MULTIMODAL_DISAGGREGATED"


def _is_disagg() -> bool:
    return os.getenv(_MULTIMODAL_ENV_NAME, "0") == "1"


# ---------------------------------------------------------------------------
# Vision tower components (eager-mode, matching HF checkpoint layout)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """RMSNorm matching HF Gemma4RMSNorm with optional learnable scale."""

    def __init__(self, dim: int, eps: float = 1e-6,
                 with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float() * torch.pow(
            x.float().pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class _ClippableLinear(nn.Module):
    """Matches HF Gemma4ClippableLinear: ``linear`` sub-module + optional
    input/output clamping buffers."""

    def __init__(self, in_features: int, out_features: int,
                 use_clipped: bool = False):
        super().__init__()
        self.use_clipped_linears = use_clipped
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if self.use_clipped_linears:
            self.register_buffer("input_min",
                                 torch.tensor(-float("inf")))
            self.register_buffer("input_max",
                                 torch.tensor(float("inf")))
            self.register_buffer("output_min",
                                 torch.tensor(-float("inf")))
            self.register_buffer("output_max",
                                 torch.tensor(float("inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_clipped_linears:
            x = torch.clamp(x, self.input_min, self.input_max)
        x = self.linear(x)
        if self.use_clipped_linears:
            x = torch.clamp(x, self.output_min, self.output_max)
        return x


# --- Patch embedder ---

class _VisionPatchEmbedder(nn.Module):

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.position_embedding_size = config.position_embedding_size

        self.input_proj = nn.Linear(
            3 * self.patch_size ** 2, self.hidden_size, bias=False)
        self.position_embedding_table = nn.Parameter(
            torch.ones(2, self.position_embedding_size, self.hidden_size))

    def _position_embeddings(
        self, pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        clamped = pixel_position_ids.clamp(min=0)
        one_hot = F.one_hot(
            clamped, num_classes=self.position_embedding_size)
        one_hot = one_hot.permute(0, 2, 1, 3).to(
            self.position_embedding_table)
        pos_emb = one_hot @ self.position_embedding_table
        pos_emb = pos_emb.sum(dim=1)
        pos_emb = torch.where(padding_positions.unsqueeze(-1), 0.0, pos_emb)
        return pos_emb

    def forward(
        self, pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = 2 * (pixel_values - 0.5)
        hidden = self.input_proj(
            pixel_values.to(self.input_proj.weight.dtype))
        pos_emb = self._position_embeddings(
            pixel_position_ids, padding_positions)
        return hidden + pos_emb


# --- Vision RoPE ---

class _VisionRotaryEmbedding(nn.Module):

    def __init__(self, config: Gemma4VisionConfig, device=None):
        super().__init__()
        rope_params = config.rope_parameters
        base = rope_params["rope_theta"]
        dim = config.head_dim
        spatial_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, spatial_dim, 2,
                                  dtype=torch.float, device=device)
                     / spatial_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1).to(x.device)
        all_cos, all_sin = [], []
        for i in range(2):
            dim_pos = position_ids[:, :, i]
            dim_pos_exp = dim_pos[:, None, :].float()
            freqs = (inv_freq_expanded.float()
                     @ dim_pos_exp.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            all_cos.append(emb.cos() * self.attention_scaling)
            all_sin.append(emb.sin() * self.attention_scaling)
        cos = torch.cat(all_cos, dim=-1).to(dtype=x.dtype)
        sin = torch.cat(all_sin, dim=-1).to(dtype=x.dtype)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary(x, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _apply_multidimensional_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    position_ids: torch.Tensor, unsqueeze_dim: int = 2,
) -> torch.Tensor:
    ndim = position_ids.shape[-1]
    num_channels = x.shape[-1]
    channels_per_dim = 2 * (num_channels // (2 * ndim))
    split_sizes = [channels_per_dim] * ndim
    x_parts = torch.split(x, split_sizes, dim=-1)
    cos_parts = torch.split(cos, split_sizes, dim=-1)
    sin_parts = torch.split(sin, split_sizes, dim=-1)
    y_parts = [
        _apply_rotary(x_parts[k], cos_parts[k], sin_parts[k],
                       unsqueeze_dim=unsqueeze_dim)
        for k in range(ndim)
    ]
    return torch.cat(y_parts, dim=-1)


# --- Vision attention ---

class _VisionAttention(nn.Module):

    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0
        use_clip = getattr(config, "use_clipped_linears", False)

        self.q_proj = _ClippableLinear(
            config.hidden_size, self.num_heads * self.head_dim, use_clip)
        self.k_proj = _ClippableLinear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_clip)
        self.v_proj = _ClippableLinear(
            config.hidden_size, self.num_kv_heads * self.head_dim, use_clip)
        self.o_proj = _ClippableLinear(
            self.num_heads * self.head_dim, config.hidden_size, use_clip)

        self.q_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = _RMSNorm(self.head_dim, eps=config.rms_norm_eps,
                                with_scale=False)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = _apply_multidimensional_rope(q, cos, sin, position_ids)
        q = q.transpose(1, 2)

        k = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        k = _apply_multidimensional_rope(k, cos, sin, position_ids)
        k = k.transpose(1, 2)

        v = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        # Repeat KV heads for GQA
        if self.num_key_value_groups > 1:
            k = k[:, :, None, :, :].expand(
                -1, -1, self.num_key_value_groups, -1, -1).reshape(
                    bsz, self.num_heads, seq_len, self.head_dim)
            v = v[:, :, None, :, :].expand(
                -1, -1, self.num_key_value_groups, -1, -1).reshape(
                    bsz, self.num_heads, seq_len, self.head_dim)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(
            bsz, seq_len, -1)
        return self.o_proj(attn_out)


# --- Vision MLP ---

class _VisionMLP(nn.Module):

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        use_clip = getattr(config, "use_clipped_linears", False)
        self.gate_proj = _ClippableLinear(
            config.hidden_size, config.intermediate_size, use_clip)
        self.up_proj = _ClippableLinear(
            config.hidden_size, config.intermediate_size, use_clip)
        self.down_proj = _ClippableLinear(
            config.intermediate_size, config.hidden_size, use_clip)
        from transformers.activations import ACT2FN
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --- Vision encoder layer ---

class _VisionEncoderLayer(nn.Module):

    def __init__(self, config: Gemma4VisionConfig, layer_idx: int):
        super().__init__()
        self.self_attn = _VisionAttention(config, layer_idx)
        self.mlp = _VisionMLP(config)
        self.input_layernorm = _RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = _RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = _RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = _RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, position_ids=position_ids)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# --- Vision encoder ---

class _VisionEncoder(nn.Module):

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.rotary_emb = _VisionRotaryEmbedding(config)
        self.layers = nn.ModuleList([
            _VisionEncoderLayer(config, i)
            for i in range(config.num_hidden_layers)
        ])

    def _make_bidirectional_mask(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = inputs_embeds.shape
        mask = attention_mask[:, None, None, :].expand(
            bsz, 1, seq_len, seq_len).to(inputs_embeds.dtype)
        mask = (1.0 - mask) * torch.finfo(inputs_embeds.dtype).min
        return mask

    def forward(
        self, inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = self._make_bidirectional_mask(inputs_embeds, attention_mask)
        position_embeddings = self.rotary_emb(inputs_embeds, pixel_position_ids)
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(
                hidden, position_embeddings=position_embeddings,
                attention_mask=attn_mask, position_ids=pixel_position_ids)
        return hidden


# --- Vision pooler ---

class _VisionPooler(nn.Module):

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size ** 0.5

    def _avg_pool_by_positions(
        self, hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor, length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = hidden_states.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_sq = k ** 2
        clamped = pixel_position_ids.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        weights = F.one_hot(kernel_idxs.long(), length).float() / k_sq
        output = weights.transpose(1, 2) @ hidden_states.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output.to(hidden_states.dtype), mask

    def forward(
        self, hidden_states: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.masked_fill(
            padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] != output_length:
            hidden_states, padding_positions = self._avg_pool_by_positions(
                hidden_states, pixel_position_ids, output_length)
        hidden_states = hidden_states * self.root_hidden_size
        return hidden_states, padding_positions


# --- Full vision model ---

class Gemma4VisionModel(nn.Module):
    """Eager-mode Gemma4 vision tower matching HF checkpoint layout.

    Weight prefix in checkpoint: ``model.vision_tower.``
    """

    def __init__(self, config: Gemma4VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embedder = _VisionPatchEmbedder(config)
        self.encoder = _VisionEncoder(config)
        self.pooler = _VisionPooler(config)

        if getattr(config, "standardize", False):
            self.register_buffer(
                "std_bias", torch.zeros(config.hidden_size))
            self.register_buffer(
                "std_scale", torch.ones(config.hidden_size))
            self._standardize = True
        else:
            self._standardize = False

    @torch.inference_mode()
    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        output_length: Optional[int] = None,
    ) -> torch.Tensor:
        """Run vision encoder and return flat hidden states with padding stripped.

        Args:
            pixel_values: (B, max_patches, patch_pixels)
            pixel_position_ids: (B, max_patches, 2), padding = (-1, -1)
            output_length: number of pooled output tokens per image

        Returns:
            Flat tensor of valid (non-padding) hidden states, shape
            (total_valid_tokens, hidden_size).
        """
        pooling_k2 = self.config.pooling_kernel_size ** 2
        if output_length is None:
            output_length = pixel_values.shape[1] // pooling_k2

        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        inputs_embeds = self.patch_embedder(
            pixel_values, pixel_position_ids, padding_positions)

        hidden = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~padding_positions,
            pixel_position_ids=pixel_position_ids,
        )

        hidden, pooler_mask = self.pooler(
            hidden_states=hidden,
            pixel_position_ids=pixel_position_ids,
            padding_positions=padding_positions,
            output_length=output_length,
        )

        # Strip padding: pooler_mask True = valid
        hidden = hidden[pooler_mask]

        if self._standardize:
            hidden = (hidden - self.std_bias) * self.std_scale

        return hidden

    def load_weights(self, weights: Dict):
        self.load_state_dict(weights, strict=True)


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
        self.embedding_pre_projection_norm = _RMSNorm(
            mm_hidden_size, eps=eps, with_scale=False)
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


class Gemma4InputProcessor(BaseMultimodalInputProcessor,
                           BaseMultimodalDummyInputsBuilder):
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
        if self._processor is not None and hasattr(self._processor,
                                                     'image_processor'):
            self._image_processor = self._processor.image_processor
        elif self._processor is None:
            try:
                from transformers import AutoImageProcessor
                self._image_processor = AutoImageProcessor.from_pretrained(
                    model_path, trust_remote_code=trust_remote_code)
            except Exception:
                logger.warning(
                    "Could not load image processor for Gemma4.")

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
            raise KeyError(
                "Expected image data in multimodal data for Gemma4.")

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
                text_prompt, return_tensors="pt",
            )["input_ids"]
            if images is not None and self._image_processor is not None:
                img_out = self._image_processor(
                    images=images, return_tensors="pt")
                pixel_values = img_out.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(dtype=self.dtype)
                image_position_ids = img_out.get("image_position_ids")

        return input_ids, pixel_values, image_position_ids

    @torch.inference_mode()
    def __call__(
        self, inputs: TextPrompt, sampling_params: SamplingParams,
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
        placeholder_map={"image": "<start_of_image>"},
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ))
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
            [config.image_token_id],
            dtype=torch.int32, device=self._device)

        model_config_cp = copy.deepcopy(model_config)
        self.model_config = model_config_cp

        # --- Language model ---
        llm_model_config = self.get_sub_model_config(
            model_config_cp, "text_config")
        self.llm = Gemma4ForCausalLM(llm_model_config)

        # --- Vision tower (eager) ---
        if config.vision_config is not None:
            self.vision_tower = Gemma4VisionModel(
                config.vision_config).eval().to(self._device)

            vision_hidden = config.vision_config.hidden_size
            text_hidden = config.text_config.hidden_size
            vision_eps = config.vision_config.rms_norm_eps
            self.embed_vision = Gemma4MultimodalEmbedder(
                mm_hidden_size=vision_hidden,
                text_hidden_size=text_hidden,
                eps=vision_eps,
                dtype=self.model_dtype,
                mapping=model_config.mapping,
            ).eval().to(self._device)
        else:
            self.vision_tower = None
            self.embed_vision = None

        # --- Audio tower (stub for future audio-capable checkpoints) ---
        if config.audio_config is not None:
            audio_hidden = getattr(
                config.audio_config, "output_proj_dims",
                config.audio_config.hidden_size)
            text_hidden = config.text_config.hidden_size
            audio_eps = config.audio_config.rms_norm_eps
            self.embed_audio = Gemma4MultimodalEmbedder(
                mm_hidden_size=audio_hidden,
                text_hidden_size=text_hidden,
                eps=audio_eps,
                dtype=self.model_dtype,
                mapping=model_config.mapping,
            ).eval().to(self._device)
            self.audio_tower = None
            logger.warning(
                "Gemma4 audio tower initialization is not yet implemented. "
                "Audio inputs will not be processed.")
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
        assert name in [
            "text_config", "vision_config", "audio_config"
        ], (f"Expected subconfig name to be 'text_config', 'vision_config', "
            f"or 'audio_config'. Got {name} instead.")
        pretrained_config = getattr(model_config.pretrained_config, name)
        quant_config = (
            model_config.quant_config if name == "text_config" else None)
        preferred_backend = (
            "FLASHINFER" if name == "text_config" else "TRTLLM")
        sub_config: ModelConfig = dataclasses.replace(
            model_config,
            pretrained_config=pretrained_config,
            attn_backend=preferred_backend,
            quant_config=quant_config,
        )
        if (hasattr(sub_config.pretrained_config, "torch_dtype")
                and sub_config.pretrained_config.torch_dtype is None):
            sub_config.pretrained_config.torch_dtype = (
                model_config.pretrained_config.torch_dtype)
        return sub_config

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        # Gemma4 checkpoint keys: model.language_model.X -> need model.X for LLM
        # Remap: "model.language_model.layers.0..." -> "model.layers.0..."
        _LANG = "model.language_model."
        llm_weights = {}
        for k, v in weights.items():
            if k.startswith(_LANG):
                llm_weights["model." + k[len(_LANG):]] = v
        self.llm.load_weights(llm_weights, weight_mapper)

        # Strip outer "model." for non-LLM components
        stripped = {(k[len("model."):] if k.startswith("model.") else k): v
                    for k, v in weights.items()}

        if self.vision_tower is not None:
            vit_weights = filter_weights("vision_tower", stripped)
            self.vision_tower.load_weights(vit_weights)

        if self.embed_vision is not None:
            embed_v_weights = filter_weights("embed_vision", stripped)
            self.embed_vision.load_weights(embed_v_weights)

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
        self, pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pooling_k2 = self._top_config.vision_config.pooling_kernel_size ** 2
        target_dtype = self.embed_vision.embedding_projection.weight.dtype

        per_image_features = []
        for i in range(pixel_values.shape[0]):
            pv = pixel_values[i].unsqueeze(0)
            pp = (image_position_ids[i].unsqueeze(0)
                  if image_position_ids is not None else None)

            if pp is None:
                max_patches = pv.shape[1]
                side = int(math.sqrt(max_patches))
                pp = torch.stack(
                    torch.meshgrid(
                        torch.arange(side, device=pv.device),
                        torch.arange(side, device=pv.device),
                        indexing="ij"),
                    dim=-1,
                ).reshape(1, -1, 2)

            max_patches = pv.shape[1]
            output_length = max_patches // pooling_k2

            with torch.autocast(device_type="cuda", dtype=self.model_dtype):
                hidden = self.vision_tower(
                    pv, pp, output_length=output_length)
                projected = self.embed_vision(
                    hidden.unsqueeze(0).to(target_dtype)).squeeze(0)
            per_image_features.append(projected)

        return torch.cat(per_image_features, dim=0).contiguous()

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
        num_ctx, num_gen = (
            attn_metadata.num_contexts, attn_metadata.num_generations)
        logger.debug(
            f"[Gemma4ForConditionalGeneration::forward] "
            f"{num_ctx=}, {num_gen=}")

        multimodal_params = kwargs.get("multimodal_params", [])

        pixel_values_list = []
        image_position_ids_list = []
        for mp in multimodal_params:
            img_data = mp.multimodal_data.get("image", {})
            pv = img_data.get("pixel_values")
            if pv is not None:
                pixel_values_list.append(pv)
                pid = img_data.get("image_position_ids")
                if pid is not None:
                    image_position_ids_list.append(pid)

        mm_embeds = []
        mm_token_mask = None
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
            mm_embeds = [image_features]
            mm_token_mask = torch.isin(input_ids, self.image_token_ids)

        input_ids, inputs_embeds = fuse_input_embeds(
            embedding_layer=self.llm.model.embed_tokens,
            input_ids=input_ids,
            mm_embeds=mm_embeds,
            mm_token_ids=self.image_token_ids,
            **kwargs,
        )
        logits = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
            image_token_mask=mm_token_mask,
            lora_params=kwargs.get("lora_params", None),
        )
        return logits

    @property
    def mm_token_ids(self):
        return self.image_token_ids
