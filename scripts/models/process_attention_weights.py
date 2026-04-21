from typing import Literal

import numpy as np
import torch
from torch import Tensor


@torch.no_grad()
def _torch_compute_joint_attention(attn: Tensor, add_residual: bool = True) -> Tensor:
    """
    Torch version of attention rollout.

    Args:
        attn: (B, L, T, T)  row-stochastic per layer (or raw if add_residual=True)
        add_residual: whether to add identity before normalization

    Returns:
        joint: (B, L, T, T) cumulative A[i] @ A[i-1] @ ... @ A[0]
    """
    assert attn.dim() == 4 and attn.size(-1) == attn.size(
        -2
    ), f"attn shape must be (B, L, T, T); got {tuple(attn.shape)}"
    B, L, T, _ = attn.shape
    device = attn.device
    dtype = attn.dtype

    if add_residual:
        eye = (
            torch.eye(T, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        )  # (1,1,T,T)
        aug = attn + eye
        aug = aug / (aug.sum(dim=-1, keepdim=True).clamp_min(1e-12))  # row-normalize
    else:
        aug = attn

    joint = torch.zeros_like(aug)
    joint[:, 0] = aug[:, 0]
    for i in range(1, L):
        # (B,T,T) = (B,T,T) @ (B,T,T)
        joint[:, i] = aug[:, i] @ joint[:, i - 1]
    return joint


@torch.no_grad()
def _torch_attentions_to_explanation(
    attn_layers: list[Tensor],
    mode: str = "rollout",
    include_first_token: bool = False,
) -> Tensor:
    """
    Convert per-head attention weights to a CLS-centric explanation.

    Args:
        attn_layers: list of length L; each item has shape (B, H, T, T)
        mode:
            - "rollout": attention rollout across layers (recommended)
            - "last":    last layer (after residual + normalization)
            - "layer_k": specific layer index, e.g., "layer_3"
        include_first_token: if False, drop CLS→CLS and return only CLS→non-CLS

    Returns:
        explanation: (B, T-1) if include_first_token=False else (B, T)
                     (CLS row over tokens)
    """
    assert len(attn_layers) > 0, "No attention weights provided."
    # Stack to (B, L, H, T, T)
    attn = torch.stack(attn_layers, dim=1)
    assert attn.ndim == 5 and attn.shape[-1] == attn.shape[-2], (
        "attentions must be (B, L, H, T, T); got " f"{attn.shape}"
    )
    B, L, H, T, _ = attn.shape

    # Mean over heads → (B, L, T, T)
    attn_mean = attn.mean(dim=2)

    if mode == "last":
        M = attn_mean[:, -1]  # (B, T, T)
    elif mode.startswith("layer_"):
        k = int(mode.split("_")[-1])
        if not (0 <= k < L):
            raise ValueError(f"Requested layer {k}, but there are only {L} layers.")
        M = attn_mean[:, k]  # (B, T, T)
    elif mode == "rollout":
        # Add residual & row-normalize
        eye = (
            torch.eye(T, device=attn_mean.device, dtype=attn_mean.dtype)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # (1,1,T,T)
        attn_res = attn_mean + eye
        attn_res = attn_res / (attn_res.sum(dim=-1, keepdim=True).clamp_min(1e-12))

        joint = _torch_compute_joint_attention(
            attn_res, add_residual=False
        )  # already residualized+normalized
        M = joint[:, -1]  # (B, T, T)
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'. Use 'rollout', 'last', or 'layer_k'."
        )

    cls_row = M[:, 0, :]  # (B, T)
    return cls_row if include_first_token else cls_row[:, 1:]
