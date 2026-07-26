"""Static-dynamic fusion variants.

All fusion modules share the interface:
    forward(z_s, z_d, static_rel, dyn_rel) -> {
        "z":     fused embedding [B, D]      (None for late_vote),
        "z_s":   projected static  [B, D],
        "z_d":   projected dynamic [B, D],
        "alpha": modality weights  [B, 3] or None  (order: static, dynamic, interaction),
    }

`reliability` is the proposed method: modality weights conditioned on both the
content embeddings and the raw reliability signals (packing evidence, trace
length, ...), plus an explicit static-dynamic interaction term and a
consistency score between the two views.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class _Base(nn.Module):
    def __init__(self, d_s: int, d_d: int, d_out: int):
        super().__init__()
        self.proj_s = nn.Sequential(nn.Linear(d_s, d_out), nn.LayerNorm(d_out))
        self.proj_d = nn.Sequential(nn.Linear(d_d, d_out), nn.LayerNorm(d_out))


class ConcatFusion(_Base):
    """Baseline: simple concatenation + MLP."""

    def __init__(self, d_s: int, d_d: int, d_out: int, hidden: int = 128):
        super().__init__(d_s, d_d, d_out)
        self.mix = _mlp([2 * d_out, hidden, d_out])

    def forward(self, z_s, z_d, static_rel=None, dyn_rel=None):
        zs, zd = self.proj_s(z_s), self.proj_d(z_d)
        return {"z": self.mix(torch.cat([zs, zd], -1)), "z_s": zs, "z_d": zd, "alpha": None}


class AttentionFusion(_Base):
    """Baseline: content-only attention weights over the two modalities."""

    def __init__(self, d_s: int, d_d: int, d_out: int, hidden: int = 128):
        super().__init__(d_s, d_d, d_out)
        self.score = _mlp([d_out, hidden, 1])

    def forward(self, z_s, z_d, static_rel=None, dyn_rel=None):
        zs, zd = self.proj_s(z_s), self.proj_d(z_d)
        w = torch.softmax(torch.cat([self.score(zs), self.score(zd)], -1), -1)  # [B, 2]
        z = w[:, :1] * zs + w[:, 1:] * zd
        alpha = torch.cat([w, torch.zeros_like(w[:, :1])], -1)
        return {"z": z, "z_s": zs, "z_d": zd, "alpha": alpha}


class LateVoteFusion(_Base):
    """Baseline: no fused embedding — the few-shot head averages per-modality distances."""

    def forward(self, z_s, z_d, static_rel=None, dyn_rel=None):
        return {"z": None, "z_s": self.proj_s(z_s), "z_d": self.proj_d(z_d), "alpha": None}


class ReliabilityFusion(_Base):
    """Proposed: reliability-aware weighting + explicit interaction term.

        z = a_s * z_s' + a_d * z_d' + a_sd * z_sd
    with [a_s, a_d, a_sd] = softmax of gates conditioned on (content, raw
    reliability signals, cross-view consistency).
    """

    def __init__(self, d_s: int, d_d: int, d_out: int, static_rel_dim: int, dyn_rel_dim: int, hidden: int = 128):
        super().__init__(d_s, d_d, d_out)
        self.rel_s = _mlp([static_rel_dim, hidden // 2, hidden // 2])
        self.rel_d = _mlp([dyn_rel_dim, hidden // 2, hidden // 2])
        self.inter = _mlp([3 * d_out, hidden, d_out])          # z_sd from [zs*zd ; zs ; zd]
        self.gate_s = _mlp([d_out + hidden // 2, hidden, 1])
        self.gate_d = _mlp([d_out + hidden // 2, hidden, 1])
        self.gate_sd = _mlp([d_out + hidden, hidden, 1])       # sees z_sd + both rel feats + consistency
        self.cons_proj = nn.Linear(1, hidden - hidden // 2 * 2) if hidden > hidden // 2 * 2 else None

    def forward(self, z_s, z_d, static_rel, dyn_rel):
        zs, zd = self.proj_s(z_s), self.proj_d(z_d)
        rs, rd = self.rel_s(static_rel), self.rel_d(dyn_rel)
        z_sd = torch.tanh(self.inter(torch.cat([zs * zd, zs, zd], -1)))
        cons = F.cosine_similarity(zs, zd, dim=-1, eps=1e-6).unsqueeze(-1)  # [B, 1]

        g_s = self.gate_s(torch.cat([zs, rs], -1))
        g_d = self.gate_d(torch.cat([zd, rd], -1))
        rel_all = torch.cat([rs, rd], -1)
        if self.cons_proj is not None:
            rel_all = torch.cat([rel_all, self.cons_proj(cons)], -1)
        g_sd = self.gate_sd(torch.cat([z_sd, rel_all], -1))

        alpha = torch.softmax(torch.cat([g_s, g_d, g_sd], -1), -1)  # [B, 3]
        z = alpha[:, 0:1] * zs + alpha[:, 1:2] * zd + alpha[:, 2:3] * z_sd
        return {"z": z, "z_s": zs, "z_d": zd, "alpha": alpha, "consistency": cons.squeeze(-1)}


def build_fusion(kind: str, d_s: int, d_d: int, d_out: int, static_rel_dim: int, dyn_rel_dim: int, hidden: int = 128) -> nn.Module:
    if kind == "concat":
        return ConcatFusion(d_s, d_d, d_out, hidden)
    if kind == "attention":
        return AttentionFusion(d_s, d_d, d_out, hidden)
    if kind == "late_vote":
        return LateVoteFusion(d_s, d_d, d_out)
    if kind == "reliability":
        return ReliabilityFusion(d_s, d_d, d_out, static_rel_dim, dyn_rel_dim, hidden)
    raise ValueError(f"unknown fusion kind: {kind}")
