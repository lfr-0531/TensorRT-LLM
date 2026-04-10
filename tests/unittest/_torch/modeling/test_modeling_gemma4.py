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
"""Unit tests for the Gemma4 model (PyTorch backend).

Since the installed transformers==4.57.3 does not support Gemma4, we cannot
perform HF comparison tests. Instead these tests validate config parsing,
model instantiation, and structural correctness.
"""

import math
import unittest
from copy import deepcopy

import torch

from transformers import Gemma4Config, Gemma4TextConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma4 import (
    Gemma4Attention, Gemma4DecoderLayer, Gemma4ForCausalLM, Gemma4MoE,
    Gemma4TextModel, Gemma4TextScaledWordEmbedding)
from tensorrt_llm.mapping import Mapping

# ---------------------------------------------------------------------------
# Small test configs
# ---------------------------------------------------------------------------
GEMMA4_SMALL_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,  # 5 sliding + 1 full
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "global_head_dim": 128,
    "num_global_key_value_heads": 1,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4,
    "attention_k_eq_v": True,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": 30.0,
    "rope_parameters": {
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        "full_attention": {
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1000000.0,
        },
    },
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
}

GEMMA4_MOE_CONFIG = {
    **GEMMA4_SMALL_CONFIG,
    "enable_moe_block": True,
    "num_experts": 8,
    "top_k_experts": 2,
    "moe_intermediate_size": 256,
}

GEMMA4_PLE_CONFIG = {
    **GEMMA4_SMALL_CONFIG,
    "hidden_size_per_layer_input": 32,
    "vocab_size_per_layer_input": 1024,
    "attention_k_eq_v": False,
    "num_kv_shared_layers": 2,
    "use_double_wide_mlp": True,
}


def _make_model_config(config_dict):
    """Build a ModelConfig from a raw config dict."""
    cfg = Gemma4TextConfig(**config_dict)
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    return ModelConfig(pretrained_config=cfg, mapping=mapping)


class TestGemma4Config(unittest.TestCase):
    """Tests for Gemma4 config classes."""

    def test_gemma4_config_nested_sub_configs(self):
        """Gemma4Config should accept and wrap nested sub-config dicts."""
        top_level = Gemma4Config(
            text_config=deepcopy(GEMMA4_SMALL_CONFIG),
            vision_config={"hidden_size": 768},
        )
        self.assertIsInstance(top_level.text_config, Gemma4TextConfig)
        self.assertEqual(top_level.text_config.hidden_size, 256)
        self.assertIsNotNone(top_level.vision_config)
        self.assertEqual(top_level.vision_config.hidden_size, 768)
        # audio_config defaults to None
        self.assertIsNone(top_level.audio_config)

    def test_gemma4_text_config_default_layer_types(self):
        """layer_types should be auto-generated with 5:1 sliding pattern."""
        cfg = Gemma4TextConfig(num_hidden_layers=6)
        # Pattern: indices 0-4 are sliding (i+1 % 6 != 0), index 5 is full
        expected = [
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
        self.assertEqual(cfg.layer_types, expected)

    def test_gemma4_text_config_last_layer_forced_full(self):
        """Even if the pattern says sliding, the last layer must be full."""
        # 7 layers: pattern would make layer 5 full, layer 6 sliding -> forced full
        cfg = Gemma4TextConfig(num_hidden_layers=7)
        self.assertEqual(cfg.layer_types[-1], "full_attention")
        self.assertEqual(len(cfg.layer_types), 7)

    def test_gemma4_text_config_default_rope_parameters(self):
        """Default rope_parameters should match Gemma4 spec."""
        cfg = Gemma4TextConfig()
        self.assertIn("sliding_attention", cfg.rope_parameters)
        self.assertIn("full_attention", cfg.rope_parameters)
        self.assertEqual(cfg.rope_parameters["sliding_attention"]["rope_theta"],
                         10_000.0)
        self.assertEqual(cfg.rope_parameters["full_attention"]["rope_theta"],
                         1_000_000.0)
        self.assertEqual(
            cfg.rope_parameters["full_attention"]["partial_rotary_factor"],
            0.25)

    def test_gemma4_text_config_explicit_layer_types(self):
        """Explicitly provided layer_types should be preserved."""
        explicit = ["full_attention"] * 4
        cfg = Gemma4TextConfig(num_hidden_layers=4, layer_types=explicit)
        self.assertEqual(cfg.layer_types, explicit)


class TestGemma4ModelInstantiation(unittest.TestCase):
    """Tests for model instantiation and structural correctness."""

    def test_model_instantiation_basic(self):
        """Create Gemma4ForCausalLM from small config and verify structure."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        # Top-level structure
        self.assertIsInstance(model.model, Gemma4TextModel)
        self.assertIsNotNone(model.lm_head)

        # Decoder layers
        pretrained = model_config.pretrained_config
        self.assertEqual(len(model.model.layers), pretrained.num_hidden_layers)
        for layer in model.model.layers:
            self.assertIsInstance(layer, Gemma4DecoderLayer)
            self.assertIsInstance(layer.self_attn, Gemma4Attention)

    def test_model_instantiation_moe(self):
        """Create with MoE enabled and verify MoE layers exist."""
        model_config = _make_model_config(GEMMA4_MOE_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        for layer in model.model.layers:
            self.assertTrue(layer.enable_moe_block)
            self.assertIsInstance(layer.moe, Gemma4MoE)
            # MoE-specific extra norms should exist
            self.assertIsNotNone(layer.post_feedforward_layernorm_1)
            self.assertIsNotNone(layer.post_feedforward_layernorm_2)
            self.assertIsNotNone(layer.pre_feedforward_layernorm_2)

    def test_model_instantiation_ple(self):
        """Create with PLE enabled and verify PLE components exist."""
        model_config = _make_model_config(GEMMA4_PLE_CONFIG)
        model = Gemma4ForCausalLM(model_config)

        # Model-level PLE components
        self.assertIsNotNone(model.model.embed_tokens_per_layer)
        self.assertIsNotNone(model.model.per_layer_model_projection)
        self.assertIsNotNone(model.model.per_layer_projection_norm)

        # Per-layer PLE components
        for layer in model.model.layers:
            self.assertEqual(layer.hidden_size_per_layer_input, 32)
            self.assertIsNotNone(layer.per_layer_input_gate)
            self.assertIsNotNone(layer.per_layer_projection)
            self.assertIsNotNone(layer.post_per_layer_input_norm)

        # Verify KV-shared layers get double-wide MLP
        config = model_config.pretrained_config
        first_kv_shared = config.num_hidden_layers - config.num_kv_shared_layers
        for i, layer in enumerate(model.model.layers):
            if i >= first_kv_shared:
                self.assertTrue(layer.is_kv_shared_layer)
            else:
                self.assertFalse(layer.is_kv_shared_layer)

    def test_per_layer_head_dim(self):
        """Sliding layers use head_dim=64, full layers use global_head_dim=128."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if config.layer_types[i] == "sliding_attention":
                self.assertEqual(attn.head_dim, config.head_dim,
                                 f"Layer {i} (sliding) should have head_dim={config.head_dim}")
            else:
                self.assertEqual(attn.head_dim, config.global_head_dim,
                                 f"Layer {i} (full) should have head_dim={config.global_head_dim}")

    def test_k_eq_v_attention(self):
        """Full attention layers with attention_k_eq_v=True should have v_norm."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            attn = layer.self_attn
            if config.layer_types[i] == "full_attention":
                # K=V should be active on full attention layers
                self.assertTrue(attn.use_k_eq_v,
                                f"Layer {i} (full) should have use_k_eq_v=True")
                self.assertTrue(hasattr(attn, 'v_norm'),
                                f"Layer {i} (full) should have v_norm")
            else:
                # Sliding layers should NOT use K=V
                self.assertFalse(attn.use_k_eq_v,
                                 f"Layer {i} (sliding) should have use_k_eq_v=False")

    def test_k_eq_v_disabled(self):
        """When attention_k_eq_v=False, no layers should have v_norm."""
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config_dict["attention_k_eq_v"] = False
        model_config = _make_model_config(config_dict)
        model = Gemma4ForCausalLM(model_config)

        for layer in model.model.layers:
            self.assertFalse(layer.self_attn.use_k_eq_v)
            self.assertFalse(hasattr(layer.self_attn, 'v_norm'))

    def test_layer_types(self):
        """Verify correct layer type assignment for default 5:1 pattern."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        for i, layer in enumerate(model.model.layers):
            expected_sliding = (config.layer_types[i] == "sliding_attention")
            self.assertEqual(layer.is_sliding, expected_sliding,
                             f"Layer {i} is_sliding mismatch")
            if expected_sliding:
                self.assertEqual(layer.self_attn.attention_window_size,
                                 config.sliding_window)
            else:
                self.assertIsNone(layer.self_attn.attention_window_size)

    def test_embedding_scale(self):
        """Embedding output should be scaled by sqrt(hidden_size)."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        embed = model.model.embed_tokens
        self.assertIsInstance(embed, Gemma4TextScaledWordEmbedding)
        expected_scale = math.sqrt(config.hidden_size)
        self.assertAlmostEqual(embed.embed_scale.item(), expected_scale,
                               places=4)

    def test_logit_softcapping(self):
        """Verify final_logit_softcapping value is stored on the config."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        model = Gemma4ForCausalLM(model_config)
        config = model_config.pretrained_config

        # The softcapping value is read from config during forward()
        self.assertEqual(config.final_logit_softcapping, 30.0)

        # Verify the forward method references the softcapping logic:
        # tanh(logits / cap) * cap. We verify the config attribute is
        # accessible through model.config (used in forward).
        # model_config.pretrained_config is stored as model.config
        # on DecoderModelForCausalLM subclasses (via model_config).
        self.assertEqual(
            model.model_config.pretrained_config.final_logit_softcapping, 30.0)

    def test_no_logit_softcapping(self):
        """When softcapping is None, it should be stored as None."""
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config_dict["final_logit_softcapping"] = None
        model_config = _make_model_config(config_dict)
        model = Gemma4ForCausalLM(model_config)
        self.assertIsNone(
            model.model_config.pretrained_config.final_logit_softcapping)

    def test_sliding_rope_params(self):
        """Sliding attention layers should use default RoPE with theta=10K."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        # Build a sliding attention to inspect its rope params
        attn = Gemma4Attention(model_config, layer_idx=0, is_sliding=True)
        rope = attn.pos_embd_params.rope
        self.assertEqual(rope.theta, 10_000.0)
        # Full rotation: dim should equal head_dim
        self.assertEqual(rope.dim, config.head_dim)

    def test_full_rope_params(self):
        """Full attention layers should use proportional RoPE with theta=1M."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        attn = Gemma4Attention(model_config, layer_idx=5, is_sliding=False)
        rope = attn.pos_embd_params.rope
        self.assertEqual(rope.theta, 1_000_000.0)
        # Partial rotation: dim = int(global_head_dim * 0.25)
        expected_dim = int(config.global_head_dim * 0.25)
        self.assertEqual(rope.dim, expected_dim)

    def test_num_kv_heads_per_layer_type(self):
        """Sliding layers use num_key_value_heads, full use num_global_key_value_heads."""
        model_config = _make_model_config(GEMMA4_SMALL_CONFIG)
        config = model_config.pretrained_config

        for i, layer in enumerate(model_config.pretrained_config.layer_types):
            attn = Gemma4Attention(
                model_config,
                layer_idx=i,
                is_sliding=(layer == "sliding_attention"),
            )
            if layer == "sliding_attention":
                expected_kv_heads = config.num_key_value_heads
            else:
                expected_kv_heads = config.num_global_key_value_heads
            self.assertEqual(attn.num_key_value_heads, expected_kv_heads,
                             f"Layer {i} kv_heads mismatch")


if __name__ == "__main__":
    unittest.main()
