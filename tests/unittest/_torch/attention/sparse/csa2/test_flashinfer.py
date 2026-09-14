# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer selected-row BF16 path and persistent custom masks."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.flashinfer import (
    FlashInferCSA2,
    pack_query_masks,
)


def test_segmented_mask_packing():
    mask = torch.tensor(
        [
            [True, False, True, False, False, False, False, True, True],
            [False, True, False, False, False, False, False, False, False],
        ]
    )
    assert pack_query_masks(mask).tolist() == [133, 1, 2, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads,width", [(8, 17), (64, 640)])
@torch.inference_mode()
def test_flashinfer_bf16_and_graph(heads, width):
    torch.manual_seed(544)
    attn = FlashInferCSA2()
    q = torch.randn(2, heads, 512, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(2, width, 512, dtype=torch.bfloat16, device="cuda")
    valid = torch.ones(2, width, dtype=torch.bool, device="cuda")
    sink = torch.linspace(-2, 2, heads, device="cuda")

    def reference():
        scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * 512**-0.5
        scores.masked_fill_(~valid[:, None, :], -torch.inf)
        probs = torch.cat((scores, sink[None, :, None].expand(2, -1, -1)), -1).softmax(-1)[..., :-1]
        return torch.einsum("qhk,qkd->qhd", probs, kv.float()).bfloat16()

    for _ in range(3):
        output = attn(q, kv, valid, sink, 512**-0.5)
    torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attn(q, kv, valid, sink, 512**-0.5)
    for end in (0, 9, width):
        valid.zero_()
        valid[:, :end] = True
        valid[1, ::2] = False
        kv.mul_(-1)
        graph.replay()
        torch.testing.assert_close(output, reference(), atol=0.03, rtol=0.03)
