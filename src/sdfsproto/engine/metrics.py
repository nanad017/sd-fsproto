"""Episode metric aggregation: accuracy, macro-F1, unknown-detection AUROC, compactness."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score


class EpisodeMetrics:
    """Accumulates per-episode results; computes aggregate metrics at the end."""

    def __init__(self, tau: float | None = None):
        self.tau = tau
        self.accs: list[float] = []
        self.f1s: list[float] = []
        self.known_scores: list[np.ndarray] = []   # min-dist for known queries
        self.unknown_scores: list[np.ndarray] = [] # min-dist for unknown queries
        self.correct_dists: list[np.ndarray] = []  # dist to true class (compactness)
        self.family_f1: dict[str, list[float]] = {}

    def update(self, class_dist: torch.Tensor, qy: torch.Tensor, families: list[str] | None = None):
        cd = class_dist.cpu().numpy()
        y = qy.cpu().numpy()
        known = y >= 0
        pred = cd.argmin(axis=1)
        n_way = cd.shape[1]

        if known.any():
            yk, pk = y[known], pred[known]
            self.accs.append(float((yk == pk).mean()))
            per_class = f1_score(yk, pk, labels=list(range(n_way)), average=None, zero_division=0)
            self.f1s.append(float(per_class.mean()))
            if families is not None:
                for c, fam in enumerate(families):
                    if (yk == c).any():
                        self.family_f1.setdefault(fam, []).append(float(per_class[c]))
            self.known_scores.append(cd[known].min(axis=1))
            self.correct_dists.append(cd[known, yk])
        if (~known).any():
            self.unknown_scores.append(cd[~known].min(axis=1))

    def compute(self) -> dict:
        out = {
            "accuracy": float(np.mean(self.accs)) if self.accs else 0.0,
            "accuracy_ci95": float(1.96 * np.std(self.accs) / max(1, len(self.accs)) ** 0.5) if self.accs else 0.0,
            "macro_f1": float(np.mean(self.f1s)) if self.f1s else 0.0,
            "episodes": len(self.accs),
        }
        if self.correct_dists:
            out["proto_compactness"] = float(np.concatenate(self.correct_dists).mean())
        if self.known_scores and self.unknown_scores:
            ks = np.concatenate(self.known_scores)
            us = np.concatenate(self.unknown_scores)
            y_true = np.r_[np.ones_like(ks), np.zeros_like(us)]   # 1 = known
            score = -np.r_[ks, us]                                # smaller dist => more known
            out["unknown_auroc"] = float(roc_auc_score(y_true, score))
            if self.tau is not None:
                out["known_tpr_at_tau"] = float((ks <= self.tau).mean())
                out["unknown_fpr_at_tau"] = float((us <= self.tau).mean())
        if self.family_f1:
            out["family_f1"] = {f: float(np.mean(v)) for f, v in sorted(self.family_f1.items())}
        return out


def calibrate_tau(known_scores: np.ndarray, unknown_scores: np.ndarray | None = None,
                  q: float = 0.95) -> float:
    """Pick the unknown-detection threshold on validation data.

    With unknown scores available: maximize Youden's J (TPR_known - FPR_unknown)
    over candidate thresholds. Fallback: q-quantile of known distances.
    """
    if unknown_scores is None or len(unknown_scores) == 0:
        return float(np.quantile(known_scores, q))
    cands = np.unique(np.r_[known_scores, unknown_scores])
    best_tau, best_j = float(np.quantile(known_scores, q)), -1.0
    for t in cands:
        j = (known_scores <= t).mean() - (unknown_scores <= t).mean()
        if j > best_j:
            best_j, best_tau = j, float(t)
    return best_tau
