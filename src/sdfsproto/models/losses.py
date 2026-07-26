"""Auxiliary losses: supervised contrastive, static-dynamic alignment, prototype separation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(z: torch.Tensor, y: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    """Supervised contrastive loss (Khosla et al.) on L2-normalized embeddings."""
    z = F.normalize(z, dim=-1)
    sim = z @ z.t() / temp                                   # [B, B]
    n = z.size(0)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = y.unsqueeze(0).eq(y.unsqueeze(1)) & ~eye           # positive pairs
    # log-softmax over all others
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    n_pos = pos.sum(1)
    valid = n_pos > 0
    if not valid.any():
        return z.new_zeros(())
    # NOTE: log_prob has -inf on the diagonal; use where() so -inf * 0 doesn't produce NaN
    contrib = torch.where(pos, log_prob, torch.zeros_like(log_prob))
    loss = -contrib.sum(1)[valid] / n_pos[valid]
    return loss.mean()


def align_loss(z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
    """Pull the static and dynamic views of the SAME sample together (cosine)."""
    return (1.0 - F.cosine_similarity(z_s, z_d, dim=-1, eps=1e-6)).mean()


def proto_sep_loss(protos: torch.Tensor, proto_class: torch.Tensor, margin: float = 4.0) -> torch.Tensor:
    """Hinge on pairwise distances between prototypes of DIFFERENT families."""
    if protos.size(0) < 2:
        return protos.new_zeros(())
    d = torch.cdist(protos, protos)                          # [P, P]
    diff = proto_class.unsqueeze(0).ne(proto_class.unsqueeze(1))
    triu = torch.triu(torch.ones_like(diff), diagonal=1).bool()
    mask = diff & triu
    if not mask.any():
        return protos.new_zeros(())
    return F.relu(margin - d[mask]).mean()
