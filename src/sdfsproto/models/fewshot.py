"""Prototype construction (single & multi), distances, unknown detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _kmeans(x: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """Tiny k-means for sub-prototype discovery within one family. Returns [k, D] centers."""
    n = x.size(0)
    perm = torch.randperm(n, device=x.device)[:k]
    centers = x[perm].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(dim=1)                 # [n]
        for j in range(k):
            m = assign == j
            if m.any():
                centers[j] = x[m].mean(0)
    return centers


def build_prototypes(
    z: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    multi: bool = True,
    max_protos: int = 3,
    return_assign: bool = False,
):
    """Returns (protos [P, D], proto_class [P]) with P >= n_way.

    Single-prototype: class mean (standard ProtoNet).
    Multi-prototype:  up to `max_protos` k-means centers per class; a sample
    belongs to a family if it is close to ANY of its sub-prototypes.

    With return_assign=True also returns assign [len(z)]: the global
    sub-prototype index each support sample belongs to.
    """
    protos, cls = [], []
    assign = torch.full((z.size(0),), -1, device=z.device, dtype=torch.long)
    offset = 0
    for c in range(n_way):
        m = y == c
        zc = z[m]
        if multi and zc.size(0) >= 2:
            k = min(max_protos, zc.size(0))
            centers = _kmeans(zc, k)
        else:
            centers = zc.mean(0, keepdim=True)
        local = torch.cdist(zc, centers).argmin(dim=1)
        assign[m] = offset + local
        offset += centers.size(0)
        protos.append(centers)
        cls.extend([c] * centers.size(0))
    protos_t = torch.cat(protos, 0)
    cls_t = torch.tensor(cls, device=z.device, dtype=torch.long)
    if return_assign:
        return protos_t, cls_t, assign
    return protos_t, cls_t


def class_distances(
    query: torch.Tensor,
    protos: torch.Tensor,
    proto_class: torch.Tensor,
    n_way: int,
    metric: str = "euclidean",
) -> torch.Tensor:
    """d(x, c) = min_j d(f(x), p_c^j). Returns [Q, n_way]."""
    if metric == "euclidean":
        d = torch.cdist(query, protos)                                  # [Q, P]
    elif metric == "cosine":
        d = 1.0 - F.normalize(query, dim=-1) @ F.normalize(protos, dim=-1).t()
    else:
        raise ValueError(f"unknown metric: {metric}")
    out = query.new_full((query.size(0), n_way), float("inf"))
    for c in range(n_way):
        m = proto_class == c
        out[:, c] = d[:, m].min(dim=1).values
    return out


def predict_with_unknown(class_dist: torch.Tensor, tau: float | None) -> torch.Tensor:
    """argmin distance; -1 (unknown family) when even the nearest prototype is beyond tau."""
    pred = class_dist.argmin(dim=1)
    if tau is not None:
        pred = torch.where(class_dist.min(dim=1).values > tau, torch.full_like(pred, -1), pred)
    return pred
