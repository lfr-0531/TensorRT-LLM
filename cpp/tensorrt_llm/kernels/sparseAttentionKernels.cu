#include "tensorrt_llm/kernels/sparseAttentionKernels.h"
#include <cub/cub.cuh>

namespace tensorrt_llm
{
namespace kernels
{
template <int THREADS_PER_BLOCK, int MAX_NUM_PAGES>
__global__ void gatherKvPageOffsetsKernel(
    int32_t* output_kv_page_offsets, // [num_head_kv, batch_size, 2, max_num_pages_per_seq]
    int32_t* output_seq_lengths,     // [num_head_kv, batch_size]
    int32_t const* kv_page_offsets,  // [batch_size, 2, max_num_pages_per_seq]
    int32_t const* seq_lengths,      // [batch_size]
    SparseAttentionParams const sparse_params, int32_t const batch_size, int32_t const tokens_per_page,
    int32_t const max_num_pages_per_seq)
{
    // Each CUDA block processes one sequence from the batch for one head.
    int32_t const head_idx = blockIdx.x;
    int32_t const batch_idx = blockIdx.y;
    int32_t const indices_block_size = sparse_params.sparse_attn_indices_block_size.value_or(1);
    if (batch_idx >= batch_size)
    {
        return;
    }

    // Shared memory for reduction.
    using BlockScan = cub::BlockScan<int32_t, THREADS_PER_BLOCK>;
    using BlockReduce = cub::BlockReduce<Pair, THREADS_PER_BLOCK>;
    __shared__ typename BlockScan::TempStorage temp_storage_scan;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ int32_t s_page_mask[MAX_NUM_PAGES];
    __shared__ int32_t s_cu_page_mask[MAX_NUM_PAGES];

    // Get the range of sparse indices and the sequence length.
    int32_t const start_offset = sparse_params.sparse_attn_offsets[batch_idx];
    int32_t const end_offset = sparse_params.sparse_attn_offsets[batch_idx + 1];
    int32_t const total_pages = sparse_params.sparse_attn_offsets[batch_size];
    int32_t const num_sparse_pages = end_offset - start_offset;
    int32_t const original_seq_len = seq_lengths[batch_idx];
    int32_t const ori_valid_pages = (original_seq_len + tokens_per_page - 1) / tokens_per_page;
    int32_t const page_loops = (ori_valid_pages + MAX_NUM_PAGES - 1) / MAX_NUM_PAGES;

    // Get global sparse index.
    int32_t const sparse_idx_global = head_idx * total_pages + start_offset;

    // Get the base memory offset. shape: [batch_size, 2, max_num_pages_per_seq]
    size_t const src_base_offset = (size_t) batch_idx * 2 * max_num_pages_per_seq;
    size_t const dst_base_offset = (size_t) head_idx * batch_size * 2 * max_num_pages_per_seq + src_base_offset;

    // Initialize the local max page index and number of valid pages.
    int32_t local_max_page_index = -1;
    int32_t local_num_valid_pages = 0;

    // Calculate the page mask.
    int32_t src_page_idx_offset = 0;
    int32_t dst_page_idx_offset = 0;
    for (int32_t loop_idx = 0; loop_idx < page_loops; loop_idx++)
    {
        src_page_idx_offset = loop_idx * MAX_NUM_PAGES;
        int32_t loop_num_valid_pages = min(MAX_NUM_PAGES, ori_valid_pages - src_page_idx_offset);
        for (int32_t i = threadIdx.x; i < loop_num_valid_pages; i += blockDim.x)
        {
            s_page_mask[i] = 0;
        }
        __syncthreads();

        for (int32_t i = threadIdx.x; i < num_sparse_pages; i += blockDim.x)
        {
            int32_t const src_idx = sparse_params.sparse_attn_indices[sparse_idx_global + i];
            if (src_idx < 0)
            {
                continue;
            }
            int32_t const src_page_idx = src_idx * indices_block_size / tokens_per_page;
            if (src_page_idx >= src_page_idx_offset && src_page_idx < src_page_idx_offset + loop_num_valid_pages)
            {
                s_page_mask[src_page_idx - src_page_idx_offset] = 1;
            }
        }
        __syncthreads();

        BlockScan(temp_storage_scan).ExclusiveSum(s_page_mask, s_cu_page_mask);

        // Perform the gather operation.
        for (int32_t i = threadIdx.x; i < loop_num_valid_pages; i += blockDim.x)
        {
            if (s_page_mask[i] == 0)
            {
                continue;
            }
            // Get the source and destination offsets.
            int32_t const src_idx = src_page_idx_offset + i;
            int32_t const dst_idx = dst_page_idx_offset + s_cu_page_mask[i];

            // Update the local max page index.
            local_max_page_index = max(local_max_page_index, src_idx);
            local_num_valid_pages++;

            // Get the source and destination offsets.
            size_t const src_offset_dim0 = src_base_offset + 0 * max_num_pages_per_seq + src_idx;
            size_t const src_offset_dim1 = src_base_offset + 1 * max_num_pages_per_seq + src_idx;
            size_t const dst_offset_dim0 = dst_base_offset + 0 * max_num_pages_per_seq + dst_idx;
            size_t const dst_offset_dim1 = dst_base_offset + 1 * max_num_pages_per_seq + dst_idx;

            // Perform the gather operation: read from the sparse location and write to the dense location.
            output_kv_page_offsets[dst_offset_dim0] = kv_page_offsets[src_offset_dim0];
            output_kv_page_offsets[dst_offset_dim1] = kv_page_offsets[src_offset_dim1];
        }
        dst_page_idx_offset += s_cu_page_mask[loop_num_valid_pages - 1];
    }

    // Reduce the local max page indices and number of valid pages.
    Pair local_pair = {local_max_page_index, local_num_valid_pages};
    Pair result = BlockReduce(temp_storage).Reduce(local_pair, PairReduceOp());

    // Update sequence length for this head and batch.
    if (threadIdx.x == 0)
    {
        int32_t const max_page_index = result.max_val;
        int32_t const num_valid_pages = result.sum_val;
        size_t const seq_len_offset = (size_t) head_idx * batch_size + batch_idx;
        if (num_valid_pages > 0)
        {
            int32_t seq_len = original_seq_len - (ori_valid_pages - num_valid_pages) * tokens_per_page;
            int32_t seq_len_remain = original_seq_len % tokens_per_page;
            if (max_page_index != ori_valid_pages - 1 && seq_len_remain != 0)
            {
                seq_len += tokens_per_page - seq_len_remain;
            }
            output_seq_lengths[seq_len_offset] = seq_len;
        }
        else
        {
            output_seq_lengths[seq_len_offset] = 0;
        }
    }
}

// Host-side launcher function
void invokeGatherKvPageOffsets(int32_t* output_kv_page_offsets, int32_t* output_seq_lengths,
    int32_t const* kv_page_offsets, int32_t const* seq_lengths, SparseAttentionParams const sparse_params,
    int32_t const batch_size, int32_t const num_head_kv, int32_t const tokens_per_page,
    int32_t const max_num_pages_per_seq, cudaStream_t stream)
{
    // The grid.
    dim3 grid(num_head_kv, batch_size, 1);
    // The block.
    dim3 block(256, 1, 1);
    // Shared memory size.
    size_t smem_size = sizeof(Pair) * 256 + sizeof(int32_t) * (512 * 2 + 256);

    // Launch the kernel.
    gatherKvPageOffsetsKernel<256, 512><<<grid, block, smem_size, stream>>>(output_kv_page_offsets, output_seq_lengths,
        kv_page_offsets, seq_lengths, sparse_params, batch_size, tokens_per_page, max_num_pages_per_seq);
}
} // namespace kernels
} // namespace tensorrt_llm
