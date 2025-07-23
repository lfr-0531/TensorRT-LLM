from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

from torch import Tensor, nn

from tensorrt_llm.models.modeling_utils import QuantConfig


@dataclass
class SparseAttentionMetadata:
    """Base metadata class for sparse attention algorithms."""

    def __post_init__(self):
        pass

    def prepare(self):
        """
        Hook to be called before the forward step of the model.
        """


class VanillaSparseAttention(nn.Module):
    """
    Abstract base class for sparse attention algorithms.

    This class provides a framework for implementing different sparse attention
    algorithms by requiring subclasses to implement specific prediction and
    calculation methods.
    """

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 sparse_attention_config: "SparseAttentionConfig",
                 num_kv_heads: Optional[int] = None,
                 quant_config: Optional[QuantConfig] = None,
                 q_scaling: Optional[float] = None,
                 **kwargs):
        self.sparse_attn_cfg = sparse_attention_config
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.quant_config = quant_config
        self.q_scaling = q_scaling

    @abstractmethod
    def single_request_sparse_attn_predict(self, q: Tensor, k: Optional[Tensor],
                                           v: Optional[Tensor],
                                           metadata: "VanillaAttentionMetadata",
                                           past_seen_token: int, cache_idx: int,
                                           **kwargs) -> Optional[Tensor]:
        pass

    @abstractmethod
    def single_request_sparse_kv_predict(self, q: Optional[Tensor],
                                         k: Optional[Tensor],
                                         v: Optional[Tensor],
                                         metadata: "VanillaAttentionMetadata",
                                         past_seen_token: int, cache_idx: int,
                                         **kwargs) -> Optional[Tensor]:
        pass
