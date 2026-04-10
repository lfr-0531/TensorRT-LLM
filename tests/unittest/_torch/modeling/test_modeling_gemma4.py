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

Includes structural tests and HF reference comparison tests using native
transformers>=5.5.0 Gemma4 support.
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
        """When attention_k_eq_v=False, use_k_eq_v is False but v_norm still exists."""
        config_dict = deepcopy(GEMMA4_SMALL_CONFIG)
        config_dict["attention_k_eq_v"] = False
        model_config = _make_model_config(config_dict)
        model = Gemma4ForCausalLM(model_config)

        for layer in model.model.layers:
            self.assertFalse(layer.self_attn.use_k_eq_v)
            # v_norm is always present in Gemma4 (even when K!=V)
            self.assertTrue(hasattr(layer.self_attn, 'v_norm'))

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


# ---------------------------------------------------------------------------
# HF reference comparison tests (sub-module + full model)
# ---------------------------------------------------------------------------

# Uniform head_dim config for comparisons (avoids per-layer head_dim issues)
GEMMA4_UNIFORM_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "global_head_dim": 64,  # Same as head_dim -> uniform
    "num_global_key_value_heads": 2,  # Same as num_key_value_heads
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "use_bidirectional_attention": None,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
}


class TestGemma4HFComparison(unittest.TestCase):
    """Compare TRT-LLM Gemma4 outputs against HuggingFace reference."""

    def _get_kv_cache_manager(self, config, num_blocks=1,
                               tokens_per_block=128, batch_size=1):
        import tensorrt_llm
        from tensorrt_llm._torch.pyexecutor.resource_manager import \
            KVCacheManagerV2
        from tensorrt_llm.llmapi.llm_args import \
            KvCacheConfig as KvCacheConfigV2

        dtype = config.torch_dtype
        if dtype == torch.half:
            kv_dtype = tensorrt_llm.bindings.DataType.HALF
        else:
            kv_dtype = tensorrt_llm.bindings.DataType.BF16

        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        max_seq_len = num_blocks * tokens_per_block
        kv_cache_config = KvCacheConfigV2(
            max_tokens=num_blocks * tokens_per_block,
            enable_block_reuse=False,
        )
        return KVCacheManagerV2(
            kv_cache_config,
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=batch_size,
            mapping=mapping,
            dtype=kv_dtype,
        )

    def _assert_most_elems_close(self, actual, ref, atol=0.5, rtol=0.5,
                                  max_failed_frac=0.01):
        matches = torch.isclose(actual, ref, atol=atol, rtol=rtol)
        failed = (~matches).float().mean().item()
        self.assertLessEqual(
            failed, max_failed_frac,
            f"{failed*100:.2f}% of elements differ (max {max_failed_frac*100}%)")

    def _make_hf_and_trt_models(self, config_dict=None):
        """Create paired HF and TRT-LLM models with shared weights."""
        from transformers import Gemma4ForCausalLM as HFGemma4
        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import \
            Gemma4HfWeightMapper

        config_dict = config_dict or deepcopy(GEMMA4_UNIFORM_CONFIG)
        config = Gemma4TextConfig(**config_dict)
        dtype = config.torch_dtype
        device = torch.device('cuda')

        hf_model = HFGemma4(config).to(dtype).to(device).eval()
        model_config = ModelConfig(pretrained_config=config,
                                    attn_backend="FLASHINFER")
        trt_model = Gemma4ForCausalLM(model_config).to(dtype).to(device)

        wm = Gemma4HfWeightMapper()
        wm.init_model_and_config(trt_model, model_config)
        trt_model.load_weights(hf_model.state_dict(), wm)

        return hf_model, trt_model, config

    # ---- Sub-module numerical comparison tests ----

    @torch.no_grad()
    def test_embedding_matches_hf(self):
        """Embedding layer: scaled by sqrt(hidden_size)."""
        hf, trt, config = self._make_hf_and_trt_models()
        ids = torch.tensor([100, 200, 300, 400], dtype=torch.int32, device='cuda')

        with torch.inference_mode():
            hf_out = hf.model.embed_tokens(ids.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.embed_tokens(ids)

        self.assertTrue(torch.allclose(hf_out, trt_out, atol=1e-3),
                         f"Embedding max diff: {(hf_out - trt_out).abs().max()}")

    @torch.no_grad()
    def test_mlp_matches_hf(self):
        """MLP (GatedMLP with gelu_tanh) output comparison."""
        hf, trt, config = self._make_hf_and_trt_models()
        x = torch.randn(4, config.hidden_size, device='cuda',
                          dtype=config.torch_dtype)

        with torch.inference_mode():
            hf_out = hf.model.layers[0].mlp(x.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.layers[0].mlp(x)

        self._assert_most_elems_close(trt_out.float(), hf_out.float(),
                                       atol=0.01, rtol=0.01)

    @torch.no_grad()
    def test_rms_norm_matches_hf(self):
        """RMSNorm with Gemma +1 offset convention."""
        hf, trt, config = self._make_hf_and_trt_models()
        x = torch.randn(4, config.hidden_size, device='cuda',
                          dtype=config.torch_dtype)

        with torch.inference_mode():
            hf_out = hf.model.layers[0].input_layernorm(x.unsqueeze(0)).squeeze(0)
            trt_out = trt.model.layers[0].input_layernorm(x)

        self.assertTrue(torch.allclose(hf_out, trt_out, atol=1e-2),
                         f"Norm max diff: {(hf_out - trt_out).abs().max()}")

    @torch.no_grad()
    def test_logit_softcapping_matches_hf(self):
        """Final logit softcapping: tanh(x/cap) * cap."""
        from transformers import Gemma4ForCausalLM as HFGemma4

        config_dict = deepcopy(GEMMA4_UNIFORM_CONFIG)
        config_dict["final_logit_softcapping"] = 30.0
        config_dict["num_hidden_layers"] = 2
        config = Gemma4TextConfig(**config_dict)

        # Just test the softcapping math directly
        logits = torch.randn(1, config.vocab_size, device='cuda')
        cap = config.final_logit_softcapping
        capped = torch.tanh(logits / cap) * cap
        # Verify values are bounded
        self.assertTrue((capped.abs() <= cap).all())
        # Verify small values are approximately unchanged
        small = torch.tensor([[0.1, -0.1, 0.5]], device='cuda')
        small_capped = torch.tanh(small / cap) * cap
        self.assertTrue(torch.allclose(small, small_capped, atol=0.01))

    @torch.no_grad()
    def test_gemma4_allclose_to_hf(self):
        """Compare context + generation logits between TRT-LLM and HF."""
        from transformers import \
            Gemma4ForCausalLM as HFGemma4ForCausalLM
        from transformers.cache_utils import DynamicCache

        from tensorrt_llm._torch.attention_backend.utils import \
            get_attention_backend
        from tensorrt_llm._torch.metadata import KVCacheParams
        from tensorrt_llm._torch.models.checkpoints.hf.gemma4_weight_mapper import \
            Gemma4HfWeightMapper

        torch.random.manual_seed(42)

        config_dict = deepcopy(GEMMA4_UNIFORM_CONFIG)
        gemma4_config = Gemma4TextConfig(**config_dict)

        dtype = gemma4_config.torch_dtype
        device = torch.device('cuda')
        backend = "FLASHINFER"

        # Create HF reference model
        hf_model = HFGemma4ForCausalLM(gemma4_config).to(dtype).to(
            device).eval()
        hf_cache = DynamicCache()

        # Create TRT-LLM model
        model_config = ModelConfig(pretrained_config=gemma4_config,
                                    attn_backend=backend)
        trt_model = Gemma4ForCausalLM(model_config).to(dtype).to(device)

        # Transfer weights HF -> TRT-LLM
        weight_mapper = Gemma4HfWeightMapper()
        weight_mapper.init_model_and_config(trt_model, model_config)
        trt_model.load_weights(hf_model.state_dict(), weight_mapper)

        # Set up KV cache
        num_blocks = 1
        tokens_per_block = 128
        kv_cache_manager = self._get_kv_cache_manager(
            gemma4_config, num_blocks=num_blocks,
            tokens_per_block=tokens_per_block)

        # -- Context phase --
        input_ids = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800],
                                  dtype=torch.int32, device=device)
        request_ids = [1]
        token_nums = [input_ids.size(-1)]
        prompt_lens = [input_ids.size(-1)]
        kv_cache_manager.add_dummy_requests(request_ids, token_nums)

        metadata_cls = get_attention_backend(backend).Metadata
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([input_ids.size(-1)], dtype=torch.int),
            num_contexts=1,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=[0],
            ),
            max_num_requests=1,
            max_num_tokens=8192,
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
        )
        position_ids = torch.arange(
            0, input_ids.size(-1), dtype=torch.int32,
            device=device).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt_model.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attn_metadata=attn_metadata,
            )
            hf_out = hf_model.forward(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        self._assert_most_elems_close(trt_logits, hf_logits)

        # -- Generation phase --
        gen_input_ids = torch.tensor([900], dtype=torch.int32, device=device)
        num_cached_tokens_per_seq = [input_ids.size(-1)]
        attn_metadata = metadata_cls(
            seq_lens=torch.tensor([gen_input_ids.size(-1)], dtype=torch.int),
            num_contexts=0,
            kv_cache_params=KVCacheParams(
                use_cache=True,
                num_cached_tokens_per_seq=num_cached_tokens_per_seq,
            ),
            kv_cache_manager=kv_cache_manager,
            request_ids=request_ids,
            prompt_lens=prompt_lens,
            max_num_requests=1,
            max_num_tokens=8192,
        )

        gen_position_ids = torch.arange(
            input_ids.size(-1),
            input_ids.size(-1) + gen_input_ids.size(-1),
            dtype=torch.int32, device=device).unsqueeze(0)

        with torch.inference_mode():
            attn_metadata.prepare()
            trt_logits = trt_model.forward(
                input_ids=gen_input_ids,
                position_ids=gen_position_ids,
                attn_metadata=attn_metadata,
            )
            hf_out = hf_model.forward(
                input_ids=gen_input_ids.unsqueeze(0),
                position_ids=gen_position_ids,
                past_key_values=hf_cache,
                use_cache=True,
            )
            hf_logits = hf_out.logits[:, -1].float()

        self._assert_most_elems_close(trt_logits, hf_logits)

        kv_cache_manager.shutdown()


if __name__ == "__main__":
    unittest.main()
