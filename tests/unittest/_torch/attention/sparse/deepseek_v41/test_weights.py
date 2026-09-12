# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF attention tensor shape validation and TP checkpoint slicing."""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.weights import (
    load_attention_weights,
)


def _parameters():
    module = nn.Module()
    module.wq_b = nn.Linear(32, 64, bias=False)
    module.o_a_proj = nn.Parameter(torch.zeros(1, 128, 128))
    module.o_b_proj = nn.Linear(128, 16, bias=False)
    module.attn_sink = nn.Parameter(torch.zeros(4))
    return module


def _weights():
    return {
        "wq_b.weight": torch.arange(128 * 32).reshape(128, 32).float(),
        "wo_a.weight": torch.arange(256 * 128).reshape(256, 128).float(),
        "wo_b.weight": torch.arange(16 * 256).reshape(16, 256).float(),
        "attn_sink": torch.arange(8).float(),
    }


@pytest.mark.parametrize("rank", [0, 1])
def test_weight_sharding_preserves_output_groups(rank):
    module, weights = _parameters(), _weights()
    load_attention_weights(module, weights, "", 2, 128, 2, rank)
    torch.testing.assert_close(
        module.wq_b.weight, weights["wq_b.weight"][rank * 64 : (rank + 1) * 64]
    )
    torch.testing.assert_close(
        module.o_a_proj[0], weights["wo_a.weight"][rank * 128 : (rank + 1) * 128]
    )
    torch.testing.assert_close(
        module.o_b_proj.weight, weights["wo_b.weight"][:, rank * 128 : (rank + 1) * 128]
    )
    torch.testing.assert_close(module.attn_sink, weights["attn_sink"][rank * 4 : (rank + 1) * 4])


@pytest.mark.parametrize("name,dim", [("wq_b.weight", 0), ("attn_sink", 0), ("wo_b.weight", 1)])
def test_oversized_checkpoint_rejected_before_copy(name, dim):
    module, weights = _parameters(), _weights()
    original = {k: p.clone() for k, p in module.named_parameters()}
    weights[name] = torch.cat((weights[name], weights[name]), dim=dim)
    with pytest.raises(ValueError, match="shape mismatch"):
        load_attention_weights(module, weights, "", 2, 128, 2, 0)
    for k, p in module.named_parameters():
        torch.testing.assert_close(p, original[k], atol=0, rtol=0)


@pytest.mark.parametrize("block", [32, 128])
def test_wo_a_fp8_scales(block):
    module, weights = _parameters(), _weights()
    weights["wo_a.weight"] = torch.ones(256, 128).to(torch.float8_e4m3fn)
    scales = (
        torch.arange(256 // block * 128 // block).reshape(256 // block, 128 // block).float() + 1
    )
    weights["wo_a.weight_scale_inv"] = scales
    load_attention_weights(module, weights, "", 2, 128, 2, 1)
    expanded = scales.repeat_interleave(block, 0).repeat_interleave(block, 1)
    torch.testing.assert_close(module.o_a_proj[0], expanded[128:])


def test_invalid_fp8_scale_shape_rejected():
    module, weights = _parameters(), _weights()
    weights["wo_a.weight"] = torch.ones(256, 128).to(torch.float8_e4m3fn)
    weights["wo_a.weight_scale_inv"] = torch.ones(3, 2)
    with pytest.raises(ValueError, match="scale geometry"):
        load_attention_weights(module, weights, "", 2, 128, 2, 0)
