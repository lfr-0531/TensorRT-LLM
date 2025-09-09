#include "tensorrt_llm/kernels/sparseAttentionKernels.h"

namespace tensorrt_llm
{
namespace kernels
{

#define BLOCK_SIZE 64

template <int const kWarpSize = 32>
__device__ int warp_reduce_sum(int val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask > 0; mask >>= 1)
    {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <int const NUM_THREADS = 64, int const WARP_SIZE = 32>
__device__ int block_reduce_sum(int val)
{
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int const warp = threadIdx.x / WARP_SIZE;
    int const lane = threadIdx.x % WARP_SIZE;
    __shared__ int smem[NUM_WARPS];

    val = warp_reduce_sum<WARP_SIZE>(val);
    if (lane == 0)
    {
        smem[warp] = val;
    }
    __syncthreads();
    val = (lane < NUM_WARPS) ? smem[lane] : 0;
    val = warp_reduce_sum<NUM_WARPS>(val);
    return val;
}

__global__ void gatherKvPageOffsetsKernel(
    KVCacheIndex::UnderlyingType* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int* output_seq_lengths,                              // [num_head_kv, batch_size]
    KVCacheIndex::UnderlyingType const* kv_page_offsets,  // [batch_size, 2, max_num_pages_per_seq]
    int const* seq_lengths,                               // [batch_size]
    SparseAttentionParams const& sparse_params)
{
    // Each CUDA block processes one sequence from the batch for one head.
    int const head_idx = blockIdx.x;
    int const batch_idx = blockIdx.y;
    int const tid = threadIdx.x;

    // Get the range of sparse indices.
    int const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int const num_sparse_pages = end_offset - start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    int const max_num_pages_per_seq = sparse_params.max_num_pages_per_seq;
    int const src_base_offset = batch_idx * 2 * max_num_pages_per_seq;
    int const dst_base_offset = head_idx * sparse_params.batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    extern __shared__ int shared_kv_cache[];

    // Count valid pages and accumulate sequence length as we gather.
    int const tokens_per_page = sparse_params.tokens_per_page;
    int const original_seq_len = seq_lengths[batch_idx];
    int const total_pages = (original_seq_len + tokens_per_page - 1) / tokens_per_page;

    // Precompute constants for index calculation
    int const head_stride = sparse_params.num_head_kv;
    int const sparse_base_idx = start_offset * head_stride + head_idx;

    int const seq_len_offset = head_idx * sparse_params.batch_size + batch_idx;
    // Local accumulator for tokens
    int local_tokens = 0;

    for (int i = tid; i < num_sparse_pages; i += blockDim.x)
    {
        int const sparse_idx_global = sparse_base_idx + i * head_stride;
        int const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global];

        bool const is_valid = (src_idx != -1);

        if (is_valid)
        {
            int tokens_to_add = tokens_per_page;
            bool const is_last_page = (src_idx == total_pages - 1);
            if (is_last_page)
            {
                // For the last page, only add the remaining tokens
                int const remaining_tokens = original_seq_len - (total_pages - 1) * tokens_per_page;
                tokens_to_add = remaining_tokens;
            }

            local_tokens += tokens_to_add;

            // Load sparse data and store in shared memory cache
            int const src_offset_dim0 = src_base_offset + src_idx;
            int const src_offset_dim1 = src_base_offset + max_num_pages_per_seq + src_idx;

            shared_kv_cache[i] = kv_page_offsets[src_offset_dim0];
            shared_kv_cache[num_sparse_pages + i] = kv_page_offsets[src_offset_dim1];
        }
        else
        {
            shared_kv_cache[i] = -1;
            shared_kv_cache[num_sparse_pages + i] = -1;
        }
    }

    __syncthreads();

    for (int i = tid; i < num_sparse_pages; i += blockDim.x)
    {
        int const dst_offset_dim0 = dst_base_offset + i;
        output_kv_page_offsets[dst_offset_dim0] = shared_kv_cache[i];
    }

    for (int i = tid; i < num_sparse_pages; i += blockDim.x)
    {
        int const dst_offset_dim1 = dst_base_offset + max_num_pages_per_seq + i;
        output_kv_page_offsets[dst_offset_dim1] = shared_kv_cache[num_sparse_pages + i];
    }

    output_seq_lengths[seq_len_offset] = block_reduce_sum<BLOCK_SIZE, 32>(local_tokens);
}

// Host-side launcher function
void invokeGatherKvPageOffsets(KVCacheIndex::UnderlyingType* output_kv_page_offsets, int* output_seq_lengths,
    KVCacheIndex::UnderlyingType const* kv_page_offsets, int const* seq_lengths,
    SparseAttentionParams const& sparse_params, cudaStream_t stream)
{
    dim3 grid(sparse_params.num_head_kv, sparse_params.batch_size, 1);

    dim3 block(BLOCK_SIZE, 1, 1);

    int smem_size = sparse_params.max_num_pages_per_seq * 2 * sizeof(int);

    // Launch the kernel with optimized configuration
    gatherKvPageOffsetsKernel<<<grid, block, smem_size, stream>>>(
        output_kv_page_offsets, output_seq_lengths, kv_page_offsets, seq_lengths, sparse_params);
}

#undef BLOCK_SIZE

} // namespace kernels
} // namespace tensorrt_llm
