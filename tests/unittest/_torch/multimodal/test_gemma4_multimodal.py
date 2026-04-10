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
"""Unit tests for the Gemma4 multimodal model components.

Tests Gemma4MultimodalEmbedder, Gemma4ForConditionalGeneration,
and Gemma4InputProcessor with HF reference comparison.
"""

import unittest

import torch
from transformers import AutoModel, Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_gemma4mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from tensorrt_llm.mapping import Mapping

# Small vision config for testing
SMALL_VISION_CONFIG = {
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 16,
    "hidden_activation": "gelu_pytorch_tanh",
    "rms_norm_eps": 1e-6,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "position_embedding_size": 1024,
    "model_type": "gemma4_vision",
}

SMALL_TEXT_CONFIG = {
    "model_type": "gemma4_text",
    "vocab_size": 1024,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "global_head_dim": 32,
    "num_global_key_value_heads": 2,
    "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "sliding_window": 128,
    "attention_k_eq_v": False,
    "enable_moe_block": False,
    "num_kv_shared_layers": 0,
    "hidden_size_per_layer_input": 0,
    "use_double_wide_mlp": False,
    "final_logit_softcapping": None,
    "torch_dtype": "bfloat16",
    "tie_word_embeddings": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "rope_parameters": {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
}


class TestGemma4MultimodalEmbedder(unittest.TestCase):
    """Test the multimodal embedder projection layer."""

    def test_embedder_instantiation(self):
        """Embedder creates norm + projection with correct dimensions."""
        embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=64,
            text_hidden_size=128,
            eps=1e-6,
            dtype=torch.bfloat16,
        )
        self.assertEqual(embedder.embedding_projection.weight.shape, torch.Size([128, 64]))

    @torch.no_grad()
    def test_embedder_forward(self):
        """Embedder forward: norm → linear, output matches expected shape."""
        embedder = (
            Gemma4MultimodalEmbedder(
                mm_hidden_size=64,
                text_hidden_size=128,
                dtype=torch.bfloat16,
            )
            .to("cuda")
            .to(torch.bfloat16)
        )

        x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            out = embedder(x)
        self.assertEqual(out.shape, torch.Size([4, 128]))
        self.assertFalse(out.isnan().any())

    @torch.no_grad()
    def test_embedder_matches_hf(self):
        """Compare embedder output with HF Gemma4MultimodalEmbedder."""
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder as HFEmbedder,
        )

        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        dtype = torch.bfloat16
        device = "cuda"

        hf_embedder = HFEmbedder(vision_cfg, text_cfg).to(dtype).to(device).eval()

        trt_embedder = Gemma4MultimodalEmbedder(
            mm_hidden_size=vision_cfg.hidden_size,
            text_hidden_size=text_cfg.hidden_size,
            eps=vision_cfg.rms_norm_eps,
            dtype=dtype,
        ).to(device)

        # Copy HF weights to TRT
        trt_embedder.embedding_projection.weight.data.copy_(
            hf_embedder.embedding_projection.weight.data
        )

        x = torch.randn(4, vision_cfg.hidden_size, device=device, dtype=dtype)
        with torch.inference_mode():
            hf_out = hf_embedder(x)
            trt_out = trt_embedder(x)

        self.assertTrue(
            torch.allclose(hf_out, trt_out, atol=1e-2),
            f"Embedder max diff: {(hf_out - trt_out).abs().max()}",
        )


class TestGemma4VisionTower(unittest.TestCase):
    """Test the vision tower (native transformers AutoModel)."""

    def test_vision_tower_creation(self):
        """Vision tower can be created from config."""
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        tower = AutoModel.from_config(vision_cfg)
        self.assertIsNotNone(tower)
        params = sum(p.numel() for p in tower.parameters())
        self.assertGreater(params, 0)


class TestGemma4ForConditionalGeneration(unittest.TestCase):
    """Test the multimodal VLM wrapper."""

    def test_instantiation_with_vision(self):
        """VLM wrapper creates LLM + vision tower + embedder."""
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        vision_cfg = Gemma4VisionConfig(**SMALL_VISION_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=vision_cfg,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.embed_vision)
        self.assertIsNone(model.audio_tower)
        self.assertIsNone(model.embed_audio)

    def test_instantiation_without_vision(self):
        """VLM wrapper works text-only when vision_config is None."""
        text_cfg = Gemma4TextConfig(**SMALL_TEXT_CONFIG)
        config = Gemma4Config(
            text_config=text_cfg,
            vision_config=None,
            audio_config=None,
        )

        mc = ModelConfig(
            pretrained_config=config,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            attn_backend="FLASHINFER",
        )
        model = Gemma4ForConditionalGeneration(mc)

        self.assertIsNotNone(model.llm)
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.embed_vision)


if __name__ == "__main__":
    unittest.main()
