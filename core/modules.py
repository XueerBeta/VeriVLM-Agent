# -*- coding: utf-8 -*-
"""
Core neural modules for VeriVLM-Agent (Beta-BLIP).
Includes Variance-induced Perceptual Attention Redistribution (ViPER)
and the Beta-Semantic Entailment Adapter (Beta-SEA).
"""

from typing import Tuple, List
import torch
import torch.nn as nn


class ViPER(nn.Module):
    """
    Module A: Variance-induced Perceptual Attention Redistribution (ViPER).
    Implements Eq. (1)-(3) from the paper:
      1. Channel-wise variance standardization: \bar{\sigma}_i^2 = (\sigma_i^2 - E[\sigma^2]) / sqrt(Var(\sigma^2) + \eps)
      2. Bidirectional hyperbolic tangent routing: W_perc = tanh(\bar{\sigma}^2)
      3. Constrained residual fusion: \tilde{V} = V * (1 + \sigma(\alpha) * W_perc),
         where \sigma(\alpha) = \lambda_max / (1 + exp(-\alpha)), \lambda_max = 0.5.
    """

    def __init__(
        self,
        scale: float = 0.083,
        lambda_max: float = 0.5,
        trainable: bool = False,
        use_variance_routing: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.lambda_max = lambda_max
        self.trainable = trainable
        self.use_variance_routing = use_variance_routing
        self.eps = eps

        # Warm-start initialization via inverse Sigmoid projection (Section 4.2 of paper)
        clamped_scale = max(1e-4, min(scale, lambda_max - 1e-4))
        init_raw_alpha = torch.logit(torch.tensor(clamped_scale / lambda_max, dtype=torch.float32))

        if trainable:
            self.raw_alpha = nn.Parameter(init_raw_alpha)
        else:
            self.register_buffer("raw_alpha", init_raw_alpha)
        self.fixed_scale = float(scale)

    def set_fixed_scale(self, scale: float) -> None:
        """Updates the fixed scaling factor during evaluation."""
        self.fixed_scale = float(scale)
        clamped_scale = max(1e-4, min(scale, self.lambda_max - 1e-4))
        with torch.no_grad():
            self.raw_alpha.copy_(
                torch.logit(torch.tensor(clamped_scale / self.lambda_max, dtype=torch.float32))
            )

    def get_effective_scale(self) -> torch.Tensor:
        """Returns the effective scaling factor \sigma(\alpha) \in (0, \lambda_max)."""
        if self.trainable:
            return self.lambda_max * torch.sigmoid(self.raw_alpha)
        return torch.tensor(self.fixed_scale, device=self.raw_alpha.device, dtype=torch.float32)

    def compute_spatial_salience(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the standardized channel-wise variance mask W_perc in (-1, 1).

        Args:
            x: Visual embeddings of shape [Batch, Num_Tokens, Embed_Dim].

        Returns:
            w_perc: Bidirectional routing mask of shape [Batch, Num_Tokens, 1].
        """
        channel_var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        var_mean = torch.mean(channel_var, dim=1, keepdim=True)
        var_std = torch.sqrt(torch.var(channel_var, dim=1, keepdim=True, unbiased=False) + self.eps)
        norm_var = (channel_var - var_mean) / var_std
        return torch.tanh(norm_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies perceptual attention redistribution to spatial visual tokens (preserving [CLS]).

        Args:
            x: Input visual features of shape [Batch, 1 + Num_Patches, Embed_Dim].

        Returns:
            Refined visual features of the same shape.
        """
        eff_scale = self.get_effective_scale()
        if not self.trainable and eff_scale.item() <= 0.0:
            return x

        if self.use_variance_routing and x.size(1) > 1:
            cls_token = x[:, :1, :]
            spatial_tokens = x[:, 1:, :]
            w_perc = self.compute_spatial_salience(spatial_tokens)
            refined_spatial = spatial_tokens * (1.0 + eff_scale * w_perc)
            return torch.cat([cls_token, refined_spatial], dim=1)
        return x * (1.0 + eff_scale)

    @torch.no_grad()
    def extract_high_density_bbox(
        self, x: torch.Tensor, top_k_ratio: float = 0.15
    ) -> List[Tuple[float, float, float, float]]:
        """
        Extracts normalized bounding coordinates (x1, y1, x2, y2) of high-variance
        foreground regions without requiring an external object detector.
        """
        spatial_tokens = x[:, 1:, :] if x.size(1) > 1 else x
        w_perc = self.compute_spatial_salience(spatial_tokens).squeeze(-1)
        num_patches = w_perc.size(1)
        grid_size = int(num_patches ** 0.5)

        threshold = torch.quantile(w_perc, 1.0 - top_k_ratio, dim=-1, keepdim=True)
        salient_mask = (w_perc >= threshold).view(-1, grid_size, grid_size)

        bboxes: List[Tuple[float, float, float, float]] = []
        for b in range(salient_mask.size(0)):
            y_idx, x_idx = torch.where(salient_mask[b])
            if len(y_idx) == 0:
                bboxes.append((0.15, 0.15, 0.85, 0.85))
            else:
                x1 = max(0.0, float(x_idx.min().item()) / grid_size - 0.05)
                y1 = max(0.0, float(y_idx.min().item()) / grid_size - 0.05)
                x2 = min(1.0, float(x_idx.max().item() + 1) / grid_size + 0.05)
                y2 = min(1.0, float(y_idx.max().item() + 1) / grid_size + 0.05)
                bboxes.append((round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)))
        return bboxes


class BetaSemanticAdapter(nn.Module):
    """
    Module B: Beta-Semantic Entailment Adapter (Beta-SEA).
    A lightweight multi-head cross-attention adapter (~5M parameters) that aligns visual features
    with textual queries and serves as a zero-extra-parameter visual critic during inference.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        Projects visual features into the text-conditioned semantic manifold.

        Args:
            image_feats: Visual key/value tokens [Batch, Num_Patches, Embed_Dim].
            text_feats: Linguistic query tokens [Batch, Num_Queries, Embed_Dim].

        Returns:
            Refined cross-modal representations [Batch, Num_Queries, Embed_Dim].
        """
        q = text_feats
        k = v = image_feats
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        x = self.norm1(q + attn_out)
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        return x