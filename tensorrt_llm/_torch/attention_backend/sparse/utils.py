from .rocket import RocketKVCacheManager


def get_sparse_attn_kv_cache_manager(sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    else:
        raise ValueError(f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}")

