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
"""Local Gemma4 config shims for transformers compatibility.

The installed transformers==4.57.3 does not support Gemma4 (model_type="gemma4").
These config classes are ported from the transformers repo to enable loading
Gemma4 models until upstream support is available.
"""
import logging

from transformers.configuration_utils import PretrainedConfig

logger = logging.getLogger(__name__)


class Gemma4AudioConfig(PretrainedConfig):

    model_type = "gemma4_audio"

    def __init__(
        self,
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=8,
        hidden_act="silu",
        subsampling_conv_channels=None,
        conv_kernel_size=5,
        residual_weight=0.5,
        attention_chunk_size=12,
        attention_context_left=13,
        attention_context_right=0,
        attention_logit_cap=50.0,
        attention_invalid_logits_value=-1.0e9,
        use_clipped_linears=True,
        rms_norm_eps=1e-6,
        gradient_clipping=1e10,
        output_proj_dims=1536,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.subsampling_conv_channels = subsampling_conv_channels or [128, 32]
        self.conv_kernel_size = conv_kernel_size
        self.residual_weight = residual_weight
        self.attention_chunk_size = attention_chunk_size
        self.attention_context_left = attention_context_left
        self.attention_context_right = attention_context_right
        self.attention_logit_cap = attention_logit_cap
        self.attention_invalid_logits_value = attention_invalid_logits_value
        self.use_clipped_linears = use_clipped_linears
        self.rms_norm_eps = rms_norm_eps
        self.gradient_clipping = gradient_clipping
        self.output_proj_dims = output_proj_dims
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class Gemma4TextConfig(PretrainedConfig):

    model_type = "gemma4_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=30,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=256,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=2,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=512,
        layer_types=None,
        final_logit_softcapping=None,
        use_bidirectional_attention=None,
        vocab_size_per_layer_input=262144,
        hidden_size_per_layer_input=256,
        num_global_key_value_heads=None,
        global_head_dim=512,
        attention_k_eq_v=False,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        num_experts=None,
        top_k_experts=None,
        moe_intermediate_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.final_logit_softcapping = final_logit_softcapping
        self.use_bidirectional_attention = use_bidirectional_attention
        self.vocab_size_per_layer_input = vocab_size_per_layer_input
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.num_global_key_value_heads = num_global_key_value_heads
        self.global_head_dim = global_head_dim
        self.attention_k_eq_v = attention_k_eq_v
        self.num_kv_shared_layers = num_kv_shared_layers
        self.enable_moe_block = enable_moe_block
        self.use_double_wide_mlp = use_double_wide_mlp
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size

        # Generate default layer_types if not provided
        if self.layer_types is None:
            sliding_window_pattern = 6  # default 5:1 pattern
            self.layer_types = [
                "sliding_attention"
                if bool((i + 1) % sliding_window_pattern) else
                "full_attention" for i in range(self.num_hidden_layers)
            ]

        # Ensure last layer is full_attention
        if self.layer_types and self.layer_types[-1] != "full_attention":
            logger.warning(
                "Last layer must use full_attention. Forcing last layer to full_attention."
            )
            self.layer_types[-1] = "full_attention"

        # Default rope_parameters
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10_000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1_000_000.0,
                },
            }

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class Gemma4VisionConfig(PretrainedConfig):

    model_type = "gemma4_vision"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=16,
        num_attention_heads=12,
        num_key_value_heads=12,
        head_dim=64,
        hidden_activation="gelu_pytorch_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        attention_bias=False,
        attention_dropout=0.0,
        rope_parameters=None,
        pooling_kernel_size=3,
        patch_size=16,
        position_embedding_size=10240,
        use_clipped_linears=False,
        standardize=False,
        initializer_range=0.02,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters
        self.pooling_kernel_size = pooling_kernel_size
        self.patch_size = patch_size
        self.position_embedding_size = position_embedding_size
        self.use_clipped_linears = use_clipped_linears
        self.standardize = standardize
        self.initializer_range = initializer_range

        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_type": "default",
                "rope_theta": 100.0,
            }

        super().__init__(**kwargs)


class Gemma4Config(PretrainedConfig):

    model_type = "gemma4"

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        audio_config=None,
        boi_token_id=255999,
        eoi_token_id=258882,
        image_token_id=258880,
        video_token_id=258884,
        boa_token_id=256000,
        eoa_token_index=258883,
        audio_token_id=258881,
        initializer_range=0.02,
        tie_word_embeddings=True,
        **kwargs,
    ):
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.boa_token_id = boa_token_id
        self.eoa_token_index = eoa_token_index
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range

        if text_config is None:
            self.text_config = Gemma4TextConfig()
            logger.info(
                "text_config is None. Using default Gemma4TextConfig.")
        elif isinstance(text_config, dict):
            self.text_config = Gemma4TextConfig(**text_config)
        else:
            self.text_config = text_config

        if vision_config is None:
            self.vision_config = None
            logger.info("vision_config is None. Vision tower disabled.")
        elif isinstance(vision_config, dict):
            self.vision_config = Gemma4VisionConfig(**vision_config)
        else:
            self.vision_config = vision_config

        if audio_config is None:
            self.audio_config = None
            logger.info("audio_config is None. Audio tower disabled.")
        elif isinstance(audio_config, dict):
            self.audio_config = Gemma4AudioConfig(**audio_config)
        else:
            self.audio_config = audio_config

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
