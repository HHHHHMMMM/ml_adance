
from typing import Optional, Union, List, Dict, Tuple

import torch
import torch.nn as nn


class DeepFeatureExtractor(nn.Module):
    """深度学习特征提取器"""
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float = 0.3):
        super().__init__()
        self.encoder = self._build_encoder(input_dim, hidden_dims, dropout_rate)
        self.decoder = self._build_decoder(hidden_dims)

    def _build_encoder(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        return nn.Sequential(*layers)

    def _build_decoder(self, hidden_dims: List[int]):
        layers = []
        hidden_dims_reversed = hidden_dims[::-1]
        current_dim = hidden_dims[-1]
        for hidden_dim in hidden_dims_reversed[1:]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, hidden_dims[0]))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded