"""Few-shot classification heads used as baselines against the prototype head.

All heads consume the same backbone embeddings (fair comparison):
    logits = head(z_support, support_y, z_query, n_way)   -> [Q, n_way]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def matching_logits(z_s: torch.Tensor, sy: torch.Tensor, z_q: torch.Tensor, n_way: int) -> torch.Tensor:
    """Matching Network: cosine attention over support, sum per class (log-prob logits)."""
    a = torch.softmax(F.normalize(z_q, dim=-1) @ F.normalize(z_s, dim=-1).t() * 10.0, dim=1)  # [Q, S]
    onehot = F.one_hot(sy, n_way).float()                                                      # [S, N]
    prob = (a @ onehot).clamp(min=1e-8)
    return prob.log()


def siamese_logits(z_s: torch.Tensor, sy: torch.Tensor, z_q: torch.Tensor, n_way: int) -> torch.Tensor:
    """Siamese-style: mean cosine similarity to each class's support samples."""
    sim = F.normalize(z_q, dim=-1) @ F.normalize(z_s, dim=-1).t()                              # [Q, S]
    out = z_q.new_zeros(z_q.size(0), n_way)
    for c in range(n_way):
        out[:, c] = sim[:, sy == c].mean(dim=1)
    return out * 10.0


class RelationModule(nn.Module):
    """Relation Network: learnable similarity g([z_q ; p_c]) on class-sum embeddings."""

    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z_s: torch.Tensor, sy: torch.Tensor, z_q: torch.Tensor, n_way: int) -> torch.Tensor:
        protos = torch.stack([z_s[sy == c].sum(0) for c in range(n_way)])                      # [N, D]
        q = z_q.unsqueeze(1).expand(-1, n_way, -1)                                             # [Q, N, D]
        p = protos.unsqueeze(0).expand(z_q.size(0), -1, -1)
        return self.net(torch.cat([q, p], dim=-1)).squeeze(-1)                                 # [Q, N]
