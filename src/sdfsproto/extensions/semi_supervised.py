"""Retrieval-augmented semi-supervised prototype refinement (idea doc §11).

From a few labeled support samples, retrieve nearby unlabeled samples,
pseudo-label the confident ones and fold them into the prototypes:

    p_c = ( sum_{x in S_c} f(x) + beta * sum_j w_j f(x~_j) )
          / ( |S_c| + beta * sum_j w_j )

With multi-prototype families the formula is applied per sub-prototype:
each accepted pseudo sample updates only the sub-prototype it is nearest to.
Ambiguous samples (confidence below threshold) are discarded so they cannot
drag the prototype (§11 "loại bỏ mẫu mơ hồ").
"""

from __future__ import annotations

import torch

from ..models.fewshot import class_distances


@torch.no_grad()
def refine_prototypes(
    z_support: torch.Tensor,      # [S, D]
    assign: torch.Tensor,         # [S] global sub-prototype index per support sample
    protos: torch.Tensor,         # [P, D]
    proto_class: torch.Tensor,    # [P]
    z_unlabeled: torch.Tensor,    # [U, D]
    retrieve_k: int = 20,
    conf_threshold: float = 0.8,
    beta: float = 0.5,
    metric: str = "euclidean",
) -> tuple[torch.Tensor, dict]:
    """Returns (refined protos [P, D], stats)."""
    n_way = int(proto_class.max().item()) + 1
    if z_unlabeled.numel() == 0:
        return protos, {"n_pseudo": 0}

    cd = class_distances(z_unlabeled, protos, proto_class, n_way, metric)
    prob = torch.softmax(-cd, dim=1)
    conf, pred = prob.max(dim=1)

    # nearest sub-prototype (global index) for every unlabeled sample
    if metric == "euclidean":
        d_all = torch.cdist(z_unlabeled, protos)
    else:
        import torch.nn.functional as F
        d_all = 1.0 - F.normalize(z_unlabeled, dim=-1) @ F.normalize(protos, dim=-1).t()

    new_protos = protos.clone()
    n_pseudo = 0
    for c in range(n_way):
        # confident candidates predicted as family c, capped at retrieve_k by confidence
        cand = torch.nonzero((pred == c) & (conf >= conf_threshold), as_tuple=True)[0]
        if cand.numel() > retrieve_k:
            cand = cand[conf[cand].topk(retrieve_k).indices]
        for p in torch.nonzero(proto_class == c, as_tuple=True)[0]:
            members = z_support[assign == p]
            if cand.numel() > 0:
                # pseudo samples whose nearest sub-prototype of family c is p
                sub = proto_class == c
                nearest_sub = torch.nonzero(sub, as_tuple=True)[0][d_all[cand][:, sub].argmin(dim=1)]
                mine = cand[nearest_sub == p]
            else:
                mine = cand
            if mine.numel() == 0:
                if members.size(0) > 0:
                    new_protos[p] = members.mean(0)
                continue
            w = conf[mine]                                     # pseudo-label confidence weights
            num = members.sum(0) + beta * (w.unsqueeze(-1) * z_unlabeled[mine]).sum(0)
            den = members.size(0) + beta * w.sum()
            new_protos[p] = num / den.clamp(min=1e-8)
            n_pseudo += int(mine.numel())
    return new_protos, {"n_pseudo": n_pseudo}
