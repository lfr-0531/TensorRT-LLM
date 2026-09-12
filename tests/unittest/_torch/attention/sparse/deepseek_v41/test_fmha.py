# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 gathered-cache attention against an independent dense softmax."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.fmha import run_flash_mla

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("heads", [8, 64])
@torch.inference_mode()
def test_flash_mla_combined_pool_and_sink(heads):
    torch.manual_seed(345)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(3, 17, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(3, 17, device="cuda", dtype=torch.bool)
    valid[0, 9:] = False
    valid[1] = False
    sink = torch.linspace(-1, 2, heads, device="cuda")
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    weights = torch.cat((scores, sink[None, :, None].expand(3, -1, -1)), -1).softmax(-1)
    expected = torch.einsum("qhk,qkd->qhd", weights[..., :-1], kv.float()).bfloat16()
    actual = run_flash_mla(q, kv, valid, sink, 512**-0.5)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)


@torch.inference_mode()
def test_flash_mla_cuda_graph_changes_selection():
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 128, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")
    for _ in range(3):
        run_flash_mla(q, kv, valid, sink, 512**-0.5)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run_flash_mla(q, kv, valid, sink, 512**-0.5)
    for end in (128, 3, 64):
        valid.zero_()
        valid[:, :end] = True
        kv.mul_(-1)
        graph.replay()
        expected = run_flash_mla(q, kv, valid, sink, 512**-0.5)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
