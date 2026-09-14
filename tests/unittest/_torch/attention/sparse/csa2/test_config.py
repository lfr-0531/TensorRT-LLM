# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 checkpoint geometry and standard factory configuration."""

from types import SimpleNamespace

from pydantic import TypeAdapter

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.registry import get_sparse_attn_kv_cache_manager
from tensorrt_llm.llmapi.llm_args import CSA2SparseAttentionConfig, SparseAttentionConfig


def test_checkpoint_owned_config_and_cache_factory():
    text = dict(
        compress_ratios=[0, 2, 2, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 2, 3],
        candidate_source_layer_id=3,
        candidate_topk_blocks=4,
        candidate_block_size=8,
        index_topk=16,
        sliding_window=128,
    )
    config = TypeAdapter(SparseAttentionConfig).validate_python({"algorithm": "csa2"})
    assert isinstance(config, CSA2SparseAttentionConfig)
    assert config.model_dump() == {"algorithm": "csa2"}
    lowered = config.to_sparse_params(pretrained_config={"text_config": text})
    metadata = config.to_sparse_metadata_params(pretrained_config=SimpleNamespace(**text))
    assert lowered.layout == metadata.layout
    assert metadata.layout.compress_ratios == (0, 2, 2, 1)
    assert get_sparse_attn_kv_cache_manager(config) is CSA2CacheManager
    assert config.supports_backend("pytorch")
    assert not config.supports_backend("autodeploy")
