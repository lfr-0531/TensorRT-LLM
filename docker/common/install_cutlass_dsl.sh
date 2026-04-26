#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Install nvidia-cutlass-dsl with the CUDA-version-specific extra.
#
# cutlass-dsl 4.4.x splits the package into a wrapper plus separate library
# bundles: `nvidia-cutlass-dsl-libs-base` (CUDA 12 ABI) and
# `nvidia-cutlass-dsl-libs-cu13` (CUDA 13 ABI). On CUDA 13 hosts a plain
# `pip install nvidia-cutlass-dsl==<ver>` only pulls the wrapper plus
# libs-base, leaving the `cutlass` python namespace unimportable.
# Detect torch's CUDA major version and add the `[cu13]` extra when needed.
# Mirrors flashinfer's setup_test_env.sh (see flashinfer PR #2833): only
# CUDA 13+ needs the extra; on CUDA 12 the plain install pulls libs-cu12 by
# default.

set -euo pipefail

CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.4.2}"

CUDA_MAJOR=$(python3 -c "import torch; print(torch.version.cuda.split('.')[0])" 2>/dev/null || echo "12")
if [ "$CUDA_MAJOR" = "13" ]; then
    PKG="nvidia-cutlass-dsl[cu13]==${CUTLASS_DSL_VERSION}"
else
    PKG="nvidia-cutlass-dsl==${CUTLASS_DSL_VERSION}"
fi

echo "Installing ${PKG} (detected CUDA ${CUDA_MAJOR})"

# Clean any stale cutlass-dsl install before reinstalling so the wrapper and
# libs match. `[cu12,cu13]` together breaks the `cutlass` namespace.
pip3 uninstall -y \
    nvidia-cutlass-dsl \
    nvidia-cutlass-dsl-libs-base \
    nvidia-cutlass-dsl-libs-cu12 \
    nvidia-cutlass-dsl-libs-cu13 \
    2>/dev/null || true

pip3 install --no-cache-dir "${PKG}"

python3 -c "import cutlass.cute"
echo "cutlass.cute import verified"
