"""
from https://github.com/KatherLab/STAMP/blob/main/src/stamp/modeling/vision_transformer.py
In parts from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
"""

from collections.abc import Iterable
from typing import cast

import torch
from einops import repeat
from torch import Tensor, nn

from .process_attention_weights import _torch_attentions_to_explanation


def feed_forward(
    dim: int,
    hidden_dim: int,
    dropout: float = 0.5,
) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout),
    )


class SelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.heads = num_heads
        self.norm = nn.LayerNorm(dim)

        self.mhsa = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        x,
        attn_mask=None,
        return_attn_weights=False,
    ):
        """
        Args:
            attn_mask:
                Which of the features to ignore during self-attention.
                `attn_mask[b,q,k] == False` means that
                query `q` of batch `b` can attend to key `k`.
                If `attn_mask` is `None`, all tokens can attend to all others.
        """
        x = self.norm(x)
        match self.mhsa:
            case nn.MultiheadAttention():
                if return_attn_weights:
                    attn_output, attn_output_weights = self.mhsa(
                        x,
                        x,
                        x,
                        need_weights=True,
                        average_attn_weights=False,
                        attn_mask=(
                            attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                            if attn_mask is not None
                            else None
                        ),
                    )
                else:
                    attn_output, attn_output_weights = self.mhsa(
                        x,
                        x,
                        x,
                        need_weights=False,
                        attn_mask=(
                            attn_mask.repeat(self.mhsa.num_heads, 1, 1)
                            if attn_mask is not None
                            else None
                        ),
                    )
            case _ as unreachable:
                raise RuntimeError(f"unreachable: {unreachable}")

        return attn_output, attn_output_weights


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        SelfAttention(
                            dim=dim,
                            num_heads=heads,
                            dropout=dropout,
                        ),
                        feed_forward(
                            dim,
                            mlp_dim,
                        ),
                    ]
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x,
        attn_mask=None,
        return_attn_weights=False,
    ):
        attn_weights_per_layer: list[Tensor] = [] if return_attn_weights else None
        for attn, ff in cast(Iterable[tuple[nn.Module, nn.Module]], self.layers):
            x_attn, x_attn_weights = attn(
                x, attn_mask=attn_mask, return_attn_weights=return_attn_weights
            )
            if return_attn_weights:
                attn_weights_per_layer.append(x_attn_weights)
            x = x_attn + x
            x = ff(x) + x

        x = self.norm(x)
        return x, attn_weights_per_layer


class VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        in_features: int,
        dim_model: int,
        n_layers: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(dim_model))

        self.project_features = nn.Sequential(
            nn.Linear(in_features, dim_model, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.transformer = Transformer(
            dim=dim_model,
            depth=n_layers,
            heads=n_heads,
            mlp_dim=dim_feedforward,
            dropout=dropout,
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim_model, num_classes))

    def forward(
        self,
        H,
        mask=None,
        *,
        return_attn: bool = False,
        attn_mode: str = "last",
        include_first_token: bool = False,
    ):
        """
        Args:
            H: (N_tokens, in_features).
            mask: (optional) padding mask you may support later.
            return_attn: If True, return attention-derived explanation or raw weights.
            attn_mode:
                - "rollout": CLS-centric attention rollout across layers.
                - "last":    last layer (after residual + normalization).
                - "layer_k": specific layer index, e.g., "layer_3".
                - "raw":     return list of raw per-layer per-head weights
                             (list of length L; each (B, H, T, T)).
            include_first_token: When returning processed attention, include CLS→CLS
                                 if True; otherwise drop it.

        Returns:
            logits: (1, num_classes)
            attention_score:
                - If attn_mode == "raw": list[Tensor] of length L, each (1, H, T, T)
                - Else (processed): Tensor of shape (1, T-1) or (1, T) if include_first_token
                - If return_attn is False: None
        """
        H = H.unsqueeze(0)
        batch_size, _n_tiles, _n_features = H.shape

        # Map input sequence to latent space of TransMIL
        H = self.project_features(H)

        # Prepend a class token to every bag,
        # include it in the mask.
        # TODO should the tiles be able to refer to the class token? Test!
        cls_tokens = repeat(self.class_token, "d -> b 1 d", b=batch_size)
        H = torch.cat([cls_tokens, H], dim=1)

        match mask:
            case None:
                H, attn_layers = self.transformer(
                    H, attn_mask=None, return_attn_weights=return_attn
                )

            case _:
                raise RuntimeError(
                    "`mask` argument was provided but it is not required as long as padding is not used."
                )
                mask_with_class_token = torch.cat(
                    [torch.zeros(mask.shape[0], 1).type_as(mask), mask], dim=1
                )
                square_attn_mask = torch.einsum(
                    "bq,bk->bqk", mask_with_class_token, mask_with_class_token
                )
                # Don't allow other tiles to reference the class token
                square_attn_mask[:, 1:, 0] = True

                H, attn_weights = self.transformer(
                    H,
                    attn_mask=square_attn_mask,
                    return_attn_weights=return_attn,
                )

        attention_score = None
        if return_attn:
            if attn_mode == "raw":
                attention_score = (
                    attn_layers  # list length L, each (B, H, T, T) with T = 1+N
                )
            else:
                # attn_layers: list length L, each (B, H, T, T) with T = 1+N
                attention_score = _torch_attentions_to_explanation(
                    attn_layers, mode=attn_mode, include_first_token=include_first_token
                )
                # (B, N)

        # Only take class token
        H = H[:, 0]

        logits = self.mlp_head(H)

        return logits, attention_score
