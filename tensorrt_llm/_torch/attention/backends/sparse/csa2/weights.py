# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 checkpoint validation, FP8 dequantization and grouped TP sharding."""

import torch
from torch import nn


def load_attention_weights(
    module: nn.Module,
    weights: dict[str, torch.Tensor],
    prefix: str,
    num_groups: int,
    o_lora_rank: int,
    tp_size: int,
    tp_rank: int,
) -> None:
    """Validate every global checkpoint tensor before modifying any parameter."""
    if tp_size <= 0 or not 0 <= tp_rank < tp_size or num_groups % tp_size:
        raise ValueError("Invalid CSA2 output-group tensor parallel mapping")
    aliases = {
        "q_norm_weight": "q_norm.weight",
        "kv_norm_weight": "kv_norm.weight",
        "o_a_proj": "wo_a.weight",
        "o_b_proj.weight": "wo_b.weight",
        "compressor.norm_weight": "compressor.norm.weight",
        "index_wq_b.weight": "indexer.wq_b.weight",
        "index_weights_proj.weight": "indexer.weights_proj.weight",
        "index_wk.weight": "indexer.wk.weight",
        "index_k_norm_weight": "indexer.k_norm.weight",
    }
    prepared = {}
    for name, parameter in module.named_parameters():
        key = prefix + aliases.get(name, name)
        tensor = weights[key]
        expected_shape = tuple(parameter.shape)
        if name == "o_a_proj":
            expected_shape = (num_groups * o_lora_rank, parameter.shape[2])
        elif name in ("wq_b.weight", "attn_sink"):
            expected_shape = (parameter.shape[0] * tp_size, *parameter.shape[1:])
        elif name == "o_b_proj.weight":
            expected_shape = (parameter.shape[0], parameter.shape[1] * tp_size)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"CSA2 checkpoint shape mismatch for {key}: {tensor.shape} != {expected_shape}"
            )
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = key.removesuffix("weight") + "weight_scale_inv"
            scales = weights[scale_key]
            supported_blocks = (32, 128) if name == "o_a_proj" else (32,)
            block_size = next(
                (
                    block
                    for block in supported_blocks
                    if tuple(scales.shape)
                    == tuple((dim + block - 1) // block for dim in tensor.shape)
                ),
                None,
            )
            if block_size is None:
                raise ValueError(f"CSA2 checkpoint has unsupported FP8 scale geometry for {key}")
            if scales.dtype == torch.uint8:
                scales = torch.exp2(scales.float() - 127)
            else:
                scales = scales.float()
            tensor = (
                tensor.float()
                * scales.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[
                    : tensor.shape[0], : tensor.shape[1]
                ]
            )
        if name == "o_a_proj":
            tensor = tensor.reshape(num_groups, *parameter.shape[1:])
            tensor = tensor.narrow(0, tp_rank * (num_groups // tp_size), (num_groups // tp_size))
        elif name in ("wq_b.weight", "attn_sink"):
            tensor = tensor.narrow(0, tp_rank * parameter.shape[0], parameter.shape[0])
        elif name == "o_b_proj.weight":
            tensor = tensor.narrow(1, tp_rank * parameter.shape[1], parameter.shape[1])
        if tensor.shape != parameter.shape:
            raise ValueError(
                f"CSA2 checkpoint shape mismatch for {key}: {tensor.shape} != {parameter.shape}"
            )
        prepared[name] = tensor.to(device=parameter.device, dtype=parameter.dtype)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(prepared[name])
