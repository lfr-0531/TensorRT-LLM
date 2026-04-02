# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generic sparse attention method protocol for MLA integration.

Sparse attention algorithms (DSA, RocketKV, etc.) implement this protocol
to plug into MLAAttention without the attention module needing algorithm-specific
code. The MLA instance is passed as the first argument so implementations can
access projection layers, absorption methods, and attention backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from ...modules.attention import MLA

from ..interface import AttentionMetadata


@runtime_checkable
class SparseAttentionMethod(Protocol):
    """Protocol for sparse attention methods that plug into MLA.

    Implementations receive the MLA instance (``mla``) to access shared
    building blocks (projections, absorption paths, attention backends)
    without owning them.

    The two-phase interface (``forward_graph_capturable`` +
    ``forward_non_capturable``) supports CUDA graph capture where Op 1
    must be straight-line code with no batch-dependent branching, while
    Op 2 runs outside graph capture and may access batch metadata.
    Algorithms that don't need two-phase splitting can implement only
    ``forward`` and leave the other two as no-ops.
    """

    def forward(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Main forward pass. Writes result into ``output`` in-place."""
        ...

    def forward_graph_capturable(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> List[torch.Tensor]:
        """CUDA-graph-capturable projection phase (Op 1).

        Must NOT access batch-specific metadata or slice by num_tokens.
        Returns intermediate tensors consumed by ``forward_non_capturable``.
        """
        ...

    def forward_non_capturable(
        self,
        mla: MLA,
        proj_outputs: List[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Non-graph-capturable attention dispatch (Op 2).

        Receives outputs from ``forward_graph_capturable``.
        Batch-dependent slicing and sparse routing happen here.
        """
        ...
