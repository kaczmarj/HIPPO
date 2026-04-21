"""Dual-stream MIL model.

Adapted from
https://github.com/binli123/dsmil-wsi/blob/dbb5cab415fb4079f89d8c977c34efd533ee87fa/dsmil.py
"""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DSMILModelOutput(NamedTuple):
    bag_logits: torch.Tensor  # 1 x C
    instance_logits: torch.Tensor  # N x C
    A: torch.Tensor
    B: torch.Tensor


class DSMILModel(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        num_classes: int,
        dropout_v: float = 0.0,
        nonlinear: bool = False,
        passing_v: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)

        if nonlinear:
            self.q = nn.Sequential(
                nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh()
            )
        else:
            self.q = nn.Linear(in_features, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v), nn.Linear(in_features, in_features), nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(num_classes, num_classes, kernel_size=in_features)

    def forward(self, feats: torch.Tensor):
        device = feats.device

        assert feats.ndim == 2
        assert feats.shape[1] == self.in_features
        c = self.fc(feats)  # N x C

        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        # sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        # compute queries of critical instances, q_max in shape C x Q
        q_max = self.q(m_feats)
        # compute inner product of Q to each entry of q_max, A in shape N x C, each
        # column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        # normalize attention scores, A in shape N x C,
        A = F.softmax(
            A
            / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
            0,
        )
        # compute bag representation, B in shape C x V
        B = torch.mm(A.transpose(0, 1), V)

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C: torch.Tensor = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)

        return DSMILModelOutput(bag_logits=C, instance_logits=c, A=A, B=B)
