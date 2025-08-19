#pragma once

namespace tensorrt_llm
{
namespace kernels
{

struct SparseAttentionParams
{
    int32_t* sparse_kv_indices;   // [num_sparse_kv_indices]
    int32_t* sparse_attn_indices; // [num_sparse_attn_indices]
    int32_t* sparse_kv_offsets;   // [num_contexts + 1]
    int32_t* sparse_attn_offsets; // [num_generations + 1]
};

} // namespace kernels
} // namespace tensorrt_llm
