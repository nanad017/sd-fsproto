"""Classical ML baselines (RF / SVM / gradient boosting) under the SAME episodic
protocol: fit on the support set, predict the query set, aggregate over episodes.

Features are a fixed static vector (no learned encoder): metadata + reliability
signals + hashed import/string histograms + byte-value histogram.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

from ..data.dataset import EpisodeSampler, MalwareDataset

IMPORT_BINS = 64
STRING_BINS = 64
BYTE_BINS = 32


def static_feature_vector(sample: dict) -> np.ndarray:
    imp = np.bincount(sample["import_ids"] % IMPORT_BINS, minlength=IMPORT_BINS).astype(np.float32)
    imp /= max(1.0, imp.sum())
    str_ = np.bincount(sample["string_ids"] % STRING_BINS, minlength=STRING_BINS).astype(np.float32)
    str_ /= max(1.0, str_.sum())
    bytes_hist = np.bincount(sample["bytes"] // (256 // BYTE_BINS), minlength=BYTE_BINS).astype(np.float32)
    bytes_hist /= max(1.0, bytes_hist.sum())
    return np.concatenate([
        sample["metadata"].astype(np.float32),
        sample["static_rel"].astype(np.float32),
        imp, str_, bytes_hist,
    ])


def _make_clf(name: str):
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1)
    if name == "svm":
        return SVC(kernel="rbf", C=10.0, gamma="scale")
    if name == "gboost":  # stands in for XGBoost without the extra dependency
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(n_estimators=100, max_depth=4, verbosity=0)
        except ImportError:
            return GradientBoostingClassifier(n_estimators=100, max_depth=3)
    raise ValueError(f"unknown classical model: {name}")


def run_classical(
    data_root: str,
    clf_name: str,
    split: str = "test",
    n_way: int = 5,
    k_shot: int = 5,
    n_query: int = 10,
    episodes: int = 200,
    seed: int = 999,
) -> dict:
    ds = MalwareDataset(data_root)
    sampler = EpisodeSampler(ds, split, n_way, k_shot, n_query, n_unknown=0, seed=seed)
    feat_cache: dict[int, np.ndarray] = {}

    def feats(idx: list[int]) -> np.ndarray:
        out = []
        for i in idx:
            if i not in feat_cache:
                feat_cache[i] = static_feature_vector(ds[i])
            out.append(feat_cache[i])
        return np.stack(out)

    accs, f1s = [], []
    for _ in range(episodes):
        ep = sampler.sample()
        Xs, Xq = feats(ep["support_idx"]), feats(ep["query_idx"])
        ys, yq = np.array(ep["support_y"]), np.array(ep["query_y"])
        clf = _make_clf(clf_name)
        clf.fit(Xs, ys)
        pred = clf.predict(Xq)
        accs.append(float((pred == yq).mean()))
        f1s.append(float(f1_score(yq, pred, labels=list(range(n_way)), average="macro", zero_division=0)))
    return {
        "model": f"classical_{clf_name}",
        "accuracy": float(np.mean(accs)),
        "accuracy_ci95": float(1.96 * np.std(accs) / len(accs) ** 0.5),
        "macro_f1": float(np.mean(f1s)),
        "episodes": episodes,
    }
