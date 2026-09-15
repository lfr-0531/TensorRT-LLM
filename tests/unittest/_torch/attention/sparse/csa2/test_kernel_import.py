# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional CuTe kernels must not affect ordinary CSA2 cache-kernel imports."""

import subprocess
import sys
from pathlib import Path


def test_kernel_import_without_optional_cute_or_cuda_context():
    kernel = (
        Path(__file__).resolve().parents[6]
        / "tensorrt_llm/_torch/attention/backends/sparse/csa2/kernel.py"
    )
    program = """
import importlib.abc
import importlib.util
import sys
import torch
import triton

class RejectOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'cutlass', 'cuda'}:
            raise AssertionError('Unexpected optional import: ' + fullname)

sys.meta_path.insert(0, RejectOptional())
assert not torch.cuda.is_initialized()
spec = importlib.util.spec_from_file_location('csa2_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not torch.cuda.is_initialized()
assert module._indexer_projection_runner_type.cache_info().currsize == 0
assert module._packed_attention_kernel_type.cache_info().currsize == 0
assert hasattr(torch.ops.trtllm, 'csa2_indexer_q_gemm_rope_fp4')
"""
    subprocess.run([sys.executable, "-c", program, str(kernel)], check=True, timeout=60)
