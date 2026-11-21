import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 创建参数
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, device=device, dtype=dtype)
        )
        # 初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # 创建参数
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # 初始化
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)
        
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]