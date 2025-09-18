import math

import torch
import triton
import triton.language as tl

########################################################
# Index gather kernel
########################################################


@triton.jit
def _index_gather_kernel(output_ptr, input_ptr, index_ptr, in_row_stride,
                         in_token_stride, in_head_stride, idx_row_stride,
                         idx_token_stride, idx_head_stride, dim_size,
                         BLOCK_SIZE: tl.constexpr):
    # get program id and block offset
    row_pid = tl.program_id(0)
    token_pid = tl.program_id(1)
    head_pid = tl.program_id(2)
    token_block_num = tl.num_programs(1)
    head_num = tl.num_programs(2)

    # get index
    indices_idx = row_pid * idx_row_stride + token_pid * idx_token_stride + head_pid * idx_head_stride
    token_idx = tl.load(index_ptr + indices_idx)

    # get input and output base address
    input_base = (row_pid * in_row_stride + token_idx * in_token_stride +
                  head_pid * in_head_stride)
    output_base = (row_pid * token_block_num * head_num * dim_size +
                   token_pid * head_num * dim_size + head_pid * dim_size)

    # process elements in blocks
    for dim_offset in tl.range(0, dim_size, BLOCK_SIZE):
        # get offsets
        offsets = tl.arange(0, BLOCK_SIZE)
        dim_indices = dim_offset + offsets
        mask = dim_indices < dim_size

        # load input and store output
        input_val = tl.load(input_ptr + input_base + dim_indices,
                            mask=mask,
                            other=0.0)
        tl.store(output_ptr + output_base + dim_indices, input_val, mask=mask)


def triton_index_gather(input, indices):
    assert input.ndim == 4, "Input must be a 4D tensor, [row, token, head, dim]"
    assert indices.ndim == 3, "Indices must be a 3D tensor, [row, token, head]"

    # shape of input and indices
    row_size = input.shape[0]
    head_num = input.shape[2]
    dim_size = input.shape[3]
    num_tokens = indices.shape[1]

    # create output tensor
    output = torch.empty((row_size, num_tokens, head_num, dim_size),
                         device='cuda',
                         dtype=input.dtype)

    # launch kernel
    grid = (row_size, num_tokens, head_num)
    _index_gather_kernel[grid](output,
                               input,
                               indices,
                               input.stride(0),
                               input.stride(1),
                               input.stride(2),
                               indices.stride(0),
                               indices.stride(1),
                               indices.stride(2),
                               dim_size,
                               BLOCK_SIZE=1024)
    return output


########################################################
# QK split kernel
########################################################


@triton.jit
def _extract_general_kernel(input_ptr, output_ptr, context_cumsum_ptr,
                            valid_seq_indices_ptr, extract_start_offsets_ptr,
                            extract_lengths_ptr, output_offsets_ptr, head_dim,
                            total_tokens, total_output_tokens, num_heads,
                            valid_batch_size, BLOCK_SIZE: tl.constexpr,
                            BLOCK_SIZE_M: tl.constexpr):
    """General kernel for extracting custom token ranges from sequences - optimized with token parallelism

    Args:
        input_ptr: Input tensor [num_heads, total_tokens, head_dim]
        output_ptr: Output tensor [num_heads, total_output_tokens, head_dim]
        context_cumsum_ptr: Cumulative sum of context lengths [batch_size + 1]
        valid_seq_indices_ptr: Valid sequence indices [valid_batch_size]
        extract_start_offsets_ptr: Start offset within each sequence [valid_batch_size]
        extract_lengths_ptr: Number of tokens to extract per sequence [valid_batch_size]
        output_offsets_ptr: Output offset for each sequence [valid_batch_size]
    """
    valid_seq_idx = tl.program_id(0)  # Which valid sequence
    head_idx = tl.program_id(1)  # Which head
    dim_offset = tl.program_id(2) * BLOCK_SIZE  # Which dimension block

    if valid_seq_idx >= valid_batch_size or head_idx >= num_heads:
        return

    # Get original sequence index
    orig_seq_idx = tl.load(valid_seq_indices_ptr + valid_seq_idx)

    # Get sequence start offset in original tensor
    seq_start_offset = tl.load(context_cumsum_ptr + orig_seq_idx)

    # Get extraction parameters for this sequence
    extract_start_offset = tl.load(extract_start_offsets_ptr + valid_seq_idx)
    extract_length = tl.load(extract_lengths_ptr + valid_seq_idx)
    output_offset = tl.load(output_offsets_ptr + valid_seq_idx)

    # Calculate dimension indices and mask
    dim_indices = dim_offset + tl.arange(0, BLOCK_SIZE)
    dim_mask = dim_indices < head_dim

    # Use tl.range for dynamic loop over token blocks
    for token_block_start in tl.range(0, extract_length, BLOCK_SIZE_M):
        # Generate token indices for current block
        token_indices = token_block_start + tl.arange(0, BLOCK_SIZE_M)
        token_mask = token_indices < extract_length

        # Calculate source and destination positions for current token block
        src_token_pos = seq_start_offset + extract_start_offset + token_indices  # [BLOCK_SIZE_M]
        dst_token_pos = output_offset + token_indices  # [BLOCK_SIZE_M]

        # Create 2D index arrays for parallel access
        # src_indices: [BLOCK_SIZE_M, BLOCK_SIZE]
        src_indices = (head_idx * total_tokens +
                       src_token_pos[:, None]) * head_dim + dim_indices[None, :]
        dst_indices = (head_idx * total_output_tokens +
                       dst_token_pos[:, None]) * head_dim + dim_indices[None, :]

        # Create 2D mask combining token and dimension masks
        full_mask = token_mask[:, None] & dim_mask[
            None, :]  # [BLOCK_SIZE_M, BLOCK_SIZE]

        # Parallel load and store for current token block
        data = tl.load(input_ptr + src_indices, mask=full_mask, other=0.0)
        tl.store(output_ptr + dst_indices, data, mask=full_mask)


def qk_split(q: torch.Tensor, k: torch.Tensor, window_size: int,
             prompt_budget: int, metadata):
    """
    Split Q and K tensors for sparse attention computation.

    Args:
        q: Query tensor [num_heads, total_tokens, head_dim]
        k: Key tensor [num_kv_heads, total_tokens, head_dim]
        context_lens: Context lengths for each sequence [batch_size]
        window_size: Size of attention window
        prompt_budget: Minimum sequence length to be considered valid

    Returns:
        q_window: Window queries [num_heads, window_size * valid_batch_size, head_dim]
        k_context: Context keys [num_kv_heads, sum(valid_context_lens), head_dim]
        k_context_lens: Context lengths for valid sequences [valid_batch_size]
        valid_seq_map: Mapping from valid sequence index to original sequence index [valid_batch_size]
    """
    num_heads, total_tokens, head_dim = q.shape
    num_kv_heads = k.shape[0]

    context_cumsum = metadata.context_cumsum_cuda[:metadata.num_contexts + 1]
    valid_seq_indices = metadata.valid_seq_indices_cuda[:metadata.num_contexts]
    q_extract_start_offsets = metadata.q_extract_start_offsets_cuda[:metadata.
                                                                    num_contexts]
    q_extract_lengths = metadata.q_extract_lengths_cuda[:metadata.num_contexts]
    q_output_offsets = metadata.q_output_offsets_cuda[:metadata.num_contexts]
    k_extract_start_offsets = metadata.k_extract_start_offsets_cuda[:metadata.
                                                                    num_contexts]
    k_extract_lengths = metadata.k_extract_lengths_cuda[:metadata.num_contexts]
    k_output_offsets = metadata.k_output_offsets_cuda[:metadata.num_contexts]
    valid_batch_size = len(valid_seq_indices)

    # Calculate total K context tokens
    total_k_context_tokens = k_extract_lengths.sum().item()

    # Create output tensors
    q_window = torch.empty(
        (num_heads, window_size * valid_batch_size, head_dim),
        device=q.device,
        dtype=q.dtype)
    k_context = torch.empty((num_kv_heads, total_k_context_tokens, head_dim),
                            device=k.device,
                            dtype=k.dtype)

    # Launch general extraction kernel - parallel execution
    BLOCK_SIZE = 128  # Dimension block size
    BLOCK_SIZE_M = 128  # Token block size for parallel processing

    # Create separate CUDA streams for parallel execution
    stream_q = torch.cuda.Stream()
    stream_k = torch.cuda.Stream()

    # Asynchronously launch both kernels in parallel
    with torch.cuda.stream(stream_q):
        # First call: Extract Q window (last window_size tokens)
        grid_q = (valid_batch_size, num_heads,
                  triton.cdiv(head_dim, BLOCK_SIZE))
        _extract_general_kernel[grid_q](q,
                                        q_window,
                                        context_cumsum,
                                        valid_seq_indices,
                                        q_extract_start_offsets,
                                        q_extract_lengths,
                                        q_output_offsets,
                                        head_dim,
                                        total_tokens,
                                        window_size * valid_batch_size,
                                        num_heads,
                                        valid_batch_size,
                                        BLOCK_SIZE=BLOCK_SIZE,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M)

    with torch.cuda.stream(stream_k):
        # Second call: Extract K context (first context_len - window_size tokens)
        grid_k = (valid_batch_size, num_kv_heads,
                  triton.cdiv(head_dim, BLOCK_SIZE))
        _extract_general_kernel[grid_k](k,
                                        k_context,
                                        context_cumsum,
                                        valid_seq_indices,
                                        k_extract_start_offsets,
                                        k_extract_lengths,
                                        k_output_offsets,
                                        head_dim,
                                        total_tokens,
                                        total_k_context_tokens,
                                        num_kv_heads,
                                        valid_batch_size,
                                        BLOCK_SIZE=BLOCK_SIZE,
                                        BLOCK_SIZE_M=BLOCK_SIZE_M)

    # Synchronize both streams to ensure completion before returning
    stream_q.synchronize()
    stream_k.synchronize()

    return q_window, k_context


########################################################
# BMM softmax kernel
########################################################


@triton.jit
def bmm_softmax_kernel(
    q_ptr,
    k_ptr,
    scores_ptr,
    m_i_stored_ptr,
    q_cu_seqlens_ptr,
    k_cu_seqlens_ptr,
    total_q_tokens,
    total_k_tokens,
    head_dim,
    batch_size,
    num_q_heads,
    num_k_heads,
    q_len_per_seq,
    sm_scale,
    causal,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized bmm softmax kernel for computing softmax(QK^T) with online softmax..

    Args:
        q_ptr: Query tensor [num_q_heads, total_q_tokens, head_dim]
        k_ptr: Key tensor [num_kv_heads, total_k_tokens, head_dim]
        scores_ptr: Output tensor [num_q_heads, q_len_per_seq, total_k_tokens]
                   where q_len_per_seq = total_q_tokens // batch_size (uniform seq assumption)
        m_i_stored_ptr: Tensor to store m_i_new values [num_q_heads, q_len_per_seq, total_k_tokens]
                       for correct final normalization while maintaining numerical stability
        BLOCK_M: Query block size (compile-time constant)
        BLOCK_N: Key block size (compile-time constant)
        BLOCK_K: Head dimension block size for tiled matmul (compile-time constant)
    """

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_q_heads:
        return

    # Continuous mapping of query heads to key heads
    k_head_idx = head_idx // (num_q_heads // num_k_heads)

    q_seq_start = tl.load(q_cu_seqlens_ptr + batch_idx)
    q_seq_end = tl.load(q_cu_seqlens_ptr + batch_idx + 1)
    k_seq_start = tl.load(k_cu_seqlens_ptr + batch_idx)
    k_seq_end = tl.load(k_cu_seqlens_ptr + batch_idx + 1)

    q_seqlen = q_seq_end - q_seq_start
    k_seqlen = k_seq_end - k_seq_start

    if q_seqlen <= 0 or k_seqlen <= 0:
        return

    # Process queries in this batch with BLOCK_M parallelization
    for q_block_start in tl.range(0, q_seqlen, BLOCK_M):
        q_offsets = q_block_start + tl.arange(0, BLOCK_M)
        q_mask = q_offsets < q_seqlen
        q_global_offsets = q_seq_start + q_offsets

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float(
            "inf")  # Running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # Running sum

        for k_block_start in tl.range(0, k_seqlen, BLOCK_N):
            k_offsets = k_block_start + tl.arange(0, BLOCK_N)
            k_mask = k_offsets < k_seqlen
            k_global_offsets = k_seq_start + k_offsets

            # Initialize QK^T accumulator for this (M, N) block
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k_dim_start in tl.range(0, head_dim, BLOCK_K):
                k_dim_offsets = k_dim_start + tl.arange(0, BLOCK_K)
                k_dim_mask = k_dim_offsets < head_dim

                # Load query chunk [BLOCK_M, BLOCK_K]
                q_indices = head_idx * total_q_tokens * head_dim + q_global_offsets[:, None] * head_dim + k_dim_offsets[
                    None, :]
                q_chunk = tl.load(q_ptr + q_indices,
                                  mask=q_mask[:, None] & k_dim_mask[None, :],
                                  other=0.0)

                # Load key chunk [BLOCK_N, BLOCK_K]
                k_indices = k_head_idx * total_k_tokens * head_dim + k_global_offsets[:, None] * head_dim + k_dim_offsets[
                    None, :]
                k_chunk = tl.load(k_ptr + k_indices,
                                  mask=k_mask[:, None] & k_dim_mask[None, :],
                                  other=0.0)

                qk += tl.dot(q_chunk, tl.trans(k_chunk))

            # Scale the accumulated QK^T
            qk = qk * sm_scale

            valid_mask = q_mask[:, None] & k_mask[None, :]
            if causal:
                # Create causal mask based on positions within this batch's sequence
                q_pos_in_seq = q_offsets[:, None]  # [BLOCK_M, 1]
                k_pos_in_seq = k_offsets[None, :]  # [1, BLOCK_N]
                causal_mask = q_pos_in_seq >= k_pos_in_seq
                qk = tl.where(causal_mask & valid_mask, qk, float("-inf"))
            else:
                qk = tl.where(valid_mask, qk, float("-inf"))

            # Online softmax update
            m_ij = tl.max(qk, 1)  # Max across keys [BLOCK_M]
            m_i_new = tl.maximum(m_i, m_ij)

            # Rescale previous sum
            alpha = tl.exp(m_i - m_i_new)
            l_i = l_i * alpha

            # Add contribution from current block
            p = tl.exp(
                qk -
                m_i_new[:, None])  # [BLOCK_M, BLOCK_N] - numerically stable

            l_ij = tl.sum(p, 1)  # Sum across keys [BLOCK_M]
            l_i = l_i + l_ij

            # Update running max
            m_i = m_i_new

            # Vectorized output index calculation
            output_indices = (head_idx * q_len_per_seq * total_k_tokens +
                              q_offsets[:, None] * total_k_tokens +
                              k_global_offsets[None, :])  # [BLOCK_M, BLOCK_N]

            # Store exp(qk - m_i_new) for numerical stability
            tl.store(scores_ptr + output_indices, p, mask=valid_mask)

            # Store corresponding m_i_new values for each position, this is needed for correct final normalization
            tl.store(m_i_stored_ptr + output_indices,
                     m_i_new[:, None],
                     mask=valid_mask)

        # Perform normalization for this q_block only after all k_blocks are processed
        for k_block_start in tl.range(0, k_seqlen, BLOCK_N):
            k_offsets = k_block_start + tl.arange(0, BLOCK_N)
            k_mask = k_offsets < k_seqlen
            k_global_offsets = k_seq_start + k_offsets

            valid_mask = q_mask[:, None] & k_mask[None, :]

            output_indices = (head_idx * q_len_per_seq * total_k_tokens +
                              q_offsets[:, None] * total_k_tokens +
                              k_global_offsets[None, :])

            # Load current scores exp(qk - m_i_new_block)
            stored_scores = tl.load(scores_ptr + output_indices,
                                    mask=valid_mask,
                                    other=0.0)

            # Load the stored m_i_new values for each position
            stored_m_i_new = tl.load(m_i_stored_ptr + output_indices,
                                     mask=valid_mask,
                                     other=float("-inf"))

            # Apply correct normalization:
            correction_factor = tl.exp(stored_m_i_new - m_i[:, None])

            normalized_scores = tl.where(
                valid_mask, stored_scores * correction_factor / l_i[:, None],
                tl.zeros_like(stored_scores))

            # Store normalized scores
            tl.store(scores_ptr + output_indices,
                     normalized_scores,
                     mask=valid_mask)


def bmm_softmax(q: torch.Tensor,
                k: torch.Tensor,
                q_cu_seqlens: torch.Tensor,
                k_cu_seqlens: torch.Tensor,
                sm_scale: float = None,
                causal: bool = False) -> torch.Tensor:
    """
    Compute softmax(QK^T) using optimized bmm softmax algorithm with tiled matrix multiplication.

    Args:
        q: Query tensor [num_q_heads, total_q_tokens, head_dim]
        k: Key tensor [num_kv_heads, total_k_tokens, head_dim]
        q_cu_seqlens: Query cumulative sequence lengths [batch_size + 1]
        k_cu_seqlens: Key cumulative sequence lengths [batch_size + 1]
        sm_scale: Scaling factor (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking

    Returns:
        scores: Attention scores [num_q_heads, q_len_per_seq, total_k_tokens]
                where q_len_per_seq = total_q_tokens // batch_size
                Each batch's results are concatenated along the last dimension
    """
    num_q_heads, total_q_tokens, head_dim = q.shape
    num_k_heads, total_k_tokens, _ = k.shape
    batch_size = len(q_cu_seqlens) - 1

    assert total_q_tokens % batch_size == 0, "total_q_tokens must be divisible by batch_size"
    q_len_per_seq = total_q_tokens // batch_size

    if total_k_tokens == 0:
        return torch.zeros((num_q_heads, q_len_per_seq, total_k_tokens),
                           dtype=torch.float32,
                           device=q.device)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    # Create output tensor with correct shape: [num_heads, q_len_per_seq, total_k_tokens]
    scores = torch.zeros((num_q_heads, q_len_per_seq, total_k_tokens),
                         dtype=torch.float32,
                         device=q.device)

    # Create tensor to store m_i_new values for each position
    m_i_stored = torch.zeros((num_q_heads, q_len_per_seq, total_k_tokens),
                             dtype=torch.float32,
                             device=q.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64

    grid = lambda meta: (batch_size, num_q_heads)

    bmm_softmax_kernel[grid](
        q,
        k,
        scores,
        m_i_stored,
        q_cu_seqlens,
        k_cu_seqlens,
        total_q_tokens,
        total_k_tokens,
        head_dim,
        batch_size,
        num_q_heads,
        num_k_heads,
        q_len_per_seq,
        sm_scale,
        causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return scores


########################################################
# Reshape flatten to batched kernel
########################################################


@triton.jit
def reshape_flatten_to_batched_kernel(
    input_ptr,
    output_ptr,
    context_lens,
    cu_context_lens,
    num_heads,
    total_tokens,
    padding_size,
    padding_value,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for reshaping with padding.

    Grid: (batch_size, num_heads)
    Each program handles one (batch, head) combination.
    """

    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    context_len = tl.load(context_lens + batch_idx)
    k_offset = tl.load(cu_context_lens + batch_idx)

    # Process in blocks
    for block_start in tl.range(0, padding_size, BLOCK_SIZE):
        pos_offsets = block_start + tl.arange(0, BLOCK_SIZE)

        pos_mask = pos_offsets < padding_size
        valid_mask = pos_offsets < context_len
        combined_mask = pos_mask & valid_mask

        input_indices = head_idx * total_tokens + k_offset + pos_offsets

        output_indices = batch_idx * num_heads * padding_size + head_idx * padding_size + pos_offsets

        values = tl.where(
            combined_mask,
            tl.load(input_ptr + input_indices,
                    mask=combined_mask,
                    other=padding_value), padding_value)

        tl.store(output_ptr + output_indices, values, mask=pos_mask)


def reshape_flatten_to_batched(
    input_tensor: torch.Tensor,
    context_lens: torch.Tensor,
    cu_context_lens: torch.Tensor,
    padding_size: int,
    padding_value=float('-inf')) -> torch.Tensor:
    """
    Reshape input_tensor from [num_heads, total_tokens] to [batch_size, num_heads, padding_size]

    Args:
        input_tensor: Input tensor tensor [num_heads, total_tokens]
        context_lens: List of context lengths for each batch
        cu_context_lens: Cumulative sum of context lengths [batch_size + 1]
        padding_size: Target padding size

    Returns:
        batched_tensor: Output tensor [batch_size, num_heads, padding_size]
    """
    batch_size = len(context_lens)
    num_heads, total_tokens = input_tensor.shape

    # Create output tensor filled with -inf
    batched_tensor = torch.full((batch_size, num_heads, padding_size),
                                padding_value,
                                device=input_tensor.device,
                                dtype=input_tensor.dtype)

    # Launch kernel
    grid = lambda meta: (batch_size, num_heads)
    reshape_flatten_to_batched_kernel[grid](input_tensor,
                                            batched_tensor,
                                            context_lens,
                                            cu_context_lens,
                                            num_heads,
                                            total_tokens,
                                            padding_size,
                                            padding_value,
                                            BLOCK_SIZE=1024)

    return batched_tensor


########################################################
# Sparse indices flattening kernel
########################################################


@triton.jit
def flatten_sparse_indices_kernel(
    selected_indices_ptr,
    context_lens_ptr,
    valid_seq_indices_ptr,
    sparse_indices_ptr,
    sparse_offsets_ptr,
    batch_size,
    valid_batch_size,
    num_kv_heads,
    prompt_budget,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Flatten sparse indices from selected valid sequences and generate indices for invalid sequences.

    Args:
        selected_indices_ptr: Selected indices [valid_batch_size, num_kv_heads, prompt_budget]
        context_lens_ptr: Context lengths for all sequences [batch_size]
        valid_seq_indices_ptr: Valid sequence indices [valid_batch_size]
        sparse_indices_ptr: Output flattened sparse indices [num_kv_heads, total_sparse_tokens]
        sparse_offsets_ptr: Output offset for each batch [batch_size + 1]
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    # Get context length for this batch
    context_len = tl.load(context_lens_ptr + batch_idx)

    # Check if this batch is valid (appears in valid_seq_indices)
    is_valid = 0
    valid_idx_in_selected = -1

    # Search for current batch_idx in valid_seq_indices
    for valid_idx in tl.range(0, valid_batch_size):
        orig_idx = tl.load(valid_seq_indices_ptr + valid_idx)
        if orig_idx == batch_idx:
            is_valid = 1
            valid_idx_in_selected = valid_idx

    # Calculate output offset for this batch
    output_offset = tl.load(sparse_offsets_ptr + batch_idx)
    total_sparse_tokens = tl.load(sparse_offsets_ptr + batch_size)

    if is_valid:
        # Valid batch: copy from selected_indices
        num_tokens = prompt_budget
        for token_block_start in tl.range(0, num_tokens, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < num_tokens

            # Load from selected_indices
            selected_indices = valid_idx_in_selected * num_kv_heads * prompt_budget + head_idx * prompt_budget + token_offsets
            selected_values = tl.load(selected_indices_ptr + selected_indices,
                                      mask=token_mask,
                                      other=0)

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(sparse_indices_ptr + output_indices,
                     selected_values,
                     mask=token_mask)
    else:
        # Invalid batch: generate [0, 1, ..., context_len-1]
        num_tokens = context_len
        for token_block_start in tl.range(0, num_tokens, BLOCK_SIZE):
            token_offsets = token_block_start + tl.arange(0, BLOCK_SIZE)
            token_mask = token_offsets < num_tokens

            # Generate sequential indices
            sequential_indices = token_offsets

            # Store to output
            output_indices = head_idx * total_sparse_tokens + output_offset + token_offsets
            tl.store(sparse_indices_ptr + output_indices,
                     sequential_indices,
                     mask=token_mask)


def flatten_sparse_indices(
        selected_indices: torch.Tensor, context_lens: torch.Tensor,
        valid_seq_indices: torch.Tensor, sparse_offsets: torch.Tensor,
        prompt_budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten sparse indices considering both valid and invalid batches.

    Args:
        selected_indices: Selected indices [valid_batch_size, num_kv_heads, prompt_budget]
        context_lens: Context lengths for all sequences [batch_size]
        valid_seq_indices: Valid sequence indices [valid_batch_size]
        sparse_offsets: Offset for each batch [batch_size + 1]
        prompt_budget: Number of tokens for valid sequences

    Returns:
        sparse_indices: Flattened sparse indices [num_kv_heads, total_sparse_tokens]
        sparse_offsets: Offset for each batch [batch_size + 1]
    """
    batch_size = len(context_lens)
    valid_batch_size, num_kv_heads, _ = selected_indices.shape

    total_sparse_tokens = sparse_offsets[-1].item()

    # Create output tensor
    sparse_indices = torch.zeros((num_kv_heads, total_sparse_tokens),
                                 dtype=selected_indices.dtype,
                                 device=selected_indices.device)

    # Launch kernel
    BLOCK_SIZE = 256
    grid = (batch_size, num_kv_heads)

    flatten_sparse_indices_kernel[grid](selected_indices,
                                        context_lens,
                                        valid_seq_indices,
                                        sparse_indices,
                                        sparse_offsets,
                                        batch_size,
                                        valid_batch_size,
                                        num_kv_heads,
                                        prompt_budget,
                                        BLOCK_SIZE=BLOCK_SIZE)

    return sparse_indices


########################################################
# Sparse KT cache update kernel
########################################################


@triton.jit
def sparse_update_kt_cache_kernel(k_tensor_ptr, kt_cache_tensor_ptr,
                                  sparse_kv_indices_ptr, sparse_kv_offsets_ptr,
                                  kt_cache_slots_ptr, num_kv_heads, head_dim,
                                  page_size, num_pages_per_block,
                                  num_total_tokens, num_total_sparse_tokens,
                                  batch_size, BLOCK_SIZE: tl.constexpr):
    """
    Batched sparse KT cache update kernel for RocketKV algorithm.

    Args:
        k_tensor_ptr: Input K tensor [num_kv_heads, num_total_tokens, head_dim]
        kt_cache_tensor_ptr: KT cache tensor [max_batch_size, num_kv_heads, 2*head_dim, num_pages_per_block]
        sparse_kv_indices_ptr: Sparse indices [num_kv_heads, num_total_sparse_tokens]
        sparse_kv_offsets_ptr: Sparse offsets [batch_size + 1]
        kt_cache_slots_ptr: Cache slot indices for each batch [batch_size]
        num_kv_heads: Number of KV heads
        head_dim: Head dimension
        page_size: Page size for grouping tokens
        num_pages_per_block: Maximum pages per block in cache
        batch_size: Number of batches to process
    """
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    page_idx = tl.program_id(2)

    if batch_idx >= batch_size or head_idx >= num_kv_heads:
        return

    # Get cache slot for current batch
    cache_slot = tl.load(kt_cache_slots_ptr + batch_idx)

    # Get sparse token range for current batch
    sparse_start = tl.load(sparse_kv_offsets_ptr + batch_idx)
    sparse_end = tl.load(sparse_kv_offsets_ptr + batch_idx + 1)
    num_sparse_tokens = sparse_end - sparse_start

    if num_sparse_tokens <= 0:
        return

    # Calculate page boundaries
    page_token_start = page_idx * page_size
    page_token_end = tl.minimum(page_token_start + page_size, num_sparse_tokens)

    if page_token_start >= num_sparse_tokens or page_idx >= num_pages_per_block:
        return

    # Load existing cache values using cache_slot
    cache_base = cache_slot * num_kv_heads * 2 * head_dim * num_pages_per_block + head_idx * 2 * head_dim * num_pages_per_block

    # Process head dimensions in blocks using BLOCK_SIZE
    for dim_block_start in tl.range(0, head_dim, BLOCK_SIZE):
        # Calculate dimension offsets for current block
        dim_offsets = dim_block_start + tl.arange(0, BLOCK_SIZE)
        dim_mask = dim_offsets < head_dim

        # Initialize min/max values for this dimension block
        k_min = tl.full([BLOCK_SIZE], float('inf'), dtype=tl.float32)
        k_max = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

        # Process tokens in current page for this dimension block
        for token_offset in tl.range(page_token_start, page_token_end):
            # Get global sparse index for current head and token
            sparse_global_idx = sparse_start + token_offset
            sparse_indices_base = head_idx * num_total_sparse_tokens + sparse_global_idx
            global_token_idx = tl.load(sparse_kv_indices_ptr +
                                       sparse_indices_base)

            # Load K values for current token and dimension block
            # K tensor shape: [num_kv_heads, num_total_tokens, head_dim]
            k_base_idx = head_idx * num_total_tokens * head_dim + global_token_idx * head_dim
            k_indices = k_base_idx + dim_offsets

            # Load values for current dimension block
            k_values = tl.load(k_tensor_ptr + k_indices,
                               mask=dim_mask,
                               other=0.0)

            # Update min/max for this page and dimension block
            k_min = tl.where(dim_mask, tl.minimum(k_min, k_values), k_min)
            k_max = tl.where(dim_mask, tl.maximum(k_max, k_values), k_max)

        # Calculate indices for min and max cache locations for current dimension block
        cache_min_indices = cache_base + dim_offsets * num_pages_per_block + page_idx
        cache_max_indices = cache_base + (
            head_dim + dim_offsets) * num_pages_per_block + page_idx

        # Store updated values back to cache for current dimension block
        tl.store(kt_cache_tensor_ptr + cache_min_indices, k_min, mask=dim_mask)
        tl.store(kt_cache_tensor_ptr + cache_max_indices, k_max, mask=dim_mask)


def batched_update_kt_cache(k_tensor: torch.Tensor,
                            kt_cache_tensor: torch.Tensor,
                            kt_cache_slots: torch.Tensor,
                            sparse_kv_indices: torch.Tensor,
                            sparse_kv_offsets: torch.Tensor, max_num_pages: int,
                            page_size: int) -> None:
    """
    Batched update KT cache with sparse K tensor using Triton kernel.

    Args:
        k_tensor: Input K tensor [num_kv_heads, num_total_tokens, head_dim]
        kt_cache_tensor: KT cache tensor [max_batch_size, num_kv_heads, 2*head_dim, num_pages_per_block]
        kt_cache_slots: Cache slot indices for each batch [batch_size] (tensor or list)
        sparse_kv_indices: Sparse indices [num_kv_heads, num_total_sparse_tokens]
        sparse_kv_offsets: Sparse offsets [batch_size + 1]
        max_num_pages: Maximum number of pages across all batches
        page_size: Page size for grouping tokens
    """
    num_kv_heads, num_total_tokens, head_dim = k_tensor.shape
    max_batch_size, _, _, num_pages_per_block = kt_cache_tensor.shape
    num_total_sparse_tokens = sparse_kv_indices.shape[1]
    batch_size = len(sparse_kv_offsets) - 1

    grid = (batch_size, num_kv_heads, max_num_pages)
    BLOCK_SIZE = 128

    sparse_update_kt_cache_kernel[grid](k_tensor,
                                        kt_cache_tensor,
                                        sparse_kv_indices,
                                        sparse_kv_offsets,
                                        kt_cache_slots,
                                        num_kv_heads,
                                        head_dim,
                                        page_size,
                                        num_pages_per_block,
                                        num_total_tokens,
                                        num_total_sparse_tokens,
                                        batch_size,
                                        BLOCK_SIZE=BLOCK_SIZE)
