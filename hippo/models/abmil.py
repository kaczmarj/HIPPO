"""Implementations of attention-based multiple instance learning models."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F


# Adapted from
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/bf1ee90af01eaec6e65fd8c685f996ac868934cf/model.py#L27-L31
def AttentionLayer(L: int, D: int, K: int):
    """Attention layer (without gating)."""
    return nn.Sequential(nn.Linear(L, D), nn.Tanh(), nn.Linear(D, K))  # NxK


# Adapted from
# https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/bf1ee90af01eaec6e65fd8c685f996ac868934cf/model.py#L72C33-L128
class GatedAttentionLayer(nn.Module):
    """Gated attention layer."""

    def __init__(self, L: int, D: int, K: int, *, dropout: float = 0.25):
        super().__init__()
        self.L = L
        self.D = D
        self.K = K
        self.dropout = dropout

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Dropout(self.dropout),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid(),
            nn.Dropout(self.dropout),
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U)  # element wise multiplication. NxK
        return A


class AttentionMILModelOutput(NamedTuple):
    """A container for the outputs of an attention MIL model."""

    logits: torch.Tensor
    attention: torch.Tensor


class AttentionMILModel(nn.Module):
    """Attention multiple-instance learning model."""

    def __init__(
        self,
        *,
        in_features: int,
        L: int,
        D: int,
        K: int = 1,
        num_classes: int,
        dropout: float = 0.25,
        gated_attention: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.L = L
        self.D = D
        self.K = K
        self.num_classes = num_classes
        self.dropout = dropout
        self.gated_attention = gated_attention

        self.encoder = nn.Sequential(nn.Linear(in_features, L), nn.ReLU(), nn.Dropout(dropout))
        if gated_attention:
            self.attention_weights = GatedAttentionLayer(L=L, D=D, K=K, dropout=dropout)
        else:
            self.attention_weights = AttentionLayer(L=L, D=D, K=K)
        self.head = nn.Linear(L, self.num_classes)

    def forward(self, H: torch.Tensor) -> AttentionMILModelOutput:
        # H.shape is N x in_features
        if H.ndim != 2:
            raise ValueError(f"Expected H to have 2 dimensions but got {H.ndim}")

        H = self.encoder(H)  # NxL
        A = self.attention_weights(H)  # NxK
        A_raw = A
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N
        M = torch.mm(A, H)  # KxL
        logits = self.head(M)  # 1xK
        return AttentionMILModelOutput(logits=logits, attention=A_raw)
