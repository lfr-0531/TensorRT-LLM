from .rocket import RocketKVCacheManager, RocketVanillaAttention


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_backend(sparse_attn_config,
                                    layer_idx,
                                    num_heads,
                                    head_dim,
                                    num_kv_heads=None,
                                    quant_config=None,
                                    q_scaling=None):
    if sparse_attn_config is None:
        return None
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention(layer_idx, num_heads, head_dim,
                                      sparse_attn_config, num_kv_heads,
                                      quant_config, q_scaling)
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_backend(sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config is None:
        return None
    raise ValueError(
        f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
    )


def get_flashinfer_sparse_attn_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config is None:
        return None
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )


def get_vanilla_sparse_attn_metadata(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config is None:
        return None
    if sparse_attn_config.algorithm == "rocket":
        return None
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention metadata: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_metadata(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config is None:
        return None
    raise ValueError(
        f"Unsupported sparse attention algorithm in trtllm attention metadata: {sparse_attn_config.algorithm}"
    )


def get_flashinfer_sparse_attn_metadata(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config is None:
        return None
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention metadata: {sparse_attn_config.algorithm}"
    )
