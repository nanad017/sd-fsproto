"""Quantitative evidence for Claim 3: the fusion knows when a modality is unreliable.

Collects the per-sample fusion weights (alpha_static, alpha_dynamic, alpha_interaction)
and tests whether they shift in the direction the design predicts:

  * packed PE          -> alpha_static should DROP  (structure view is obfuscated)
  * incomplete trace   -> alpha_dynamic should DROP (sandbox saw nothing useful)
  * few imports        -> alpha_static should DROP  (import table stripped/hidden)

Each hypothesis is one-sided, stated before looking at the data. Reported with a
Mann-Whitney U test (no normality assumption) and Cliff's delta as effect size,
because a significant p on thousands of samples can still be a trivial difference.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import mannwhitneyu

from ..data.dataset import batch_to_device, collate
from ..data.schema import DYN_REL_DIM, STATIC_REL_DIM


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size in [-1, 1]. |d|: <0.15 negligible, <0.33 small,
    <0.47 medium, else large (Romano et al. thresholds)."""
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan")
    # rank-based computation instead of the O(n*m) pairwise loop
    order = np.argsort(np.concatenate([a, b]), kind="mergesort")
    ranks = np.empty(n_a + n_b, dtype=np.float64)
    ranks[order] = np.arange(1, n_a + n_b + 1)
    rank_sum_a = ranks[:n_a].sum()
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return float(2.0 * u_a / (n_a * n_b) - 1.0)


def effect_label(d: float) -> str:
    ad = abs(d)
    if np.isnan(d):
        return "n/a"
    return "negligible" if ad < 0.15 else "small" if ad < 0.33 else "medium" if ad < 0.47 else "large"


@torch.no_grad()
def collect_alpha(model, dataset, indices, device, max_bytes, max_seq, batch_size=32) -> dict:
    """Run samples through the model and gather alpha plus the reliability signals."""
    model.eval()
    alphas, s_rel, d_rel, fams, cons = [], [], [], [], []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        batch = batch_to_device(collate([dataset[i] for i in chunk], max_bytes, max_seq), device)
        out = model(batch)
        if out.get("alpha") is None:
            raise ValueError(
                "this run's fusion does not produce alpha — alpha analysis needs "
                "model.fusion.kind=reliability"
            )
        alphas.append(out["alpha"].cpu().numpy())
        if out.get("consistency") is not None:
            cons.append(out["consistency"].cpu().numpy())
        s_rel.append(batch["static_rel"].cpu().numpy())
        d_rel.append(batch["dyn_rel"].cpu().numpy())
        fams.append(batch["family"].cpu().numpy())
    return {
        "alpha": np.concatenate(alphas),                      # [N, 3]
        "static_rel": np.concatenate(s_rel),                  # [N, 8]
        "dyn_rel": np.concatenate(d_rel),                     # [N, 8]
        "family": np.concatenate(fams),
        "consistency": np.concatenate(cons) if cons else None,
        "n": sum(len(a) for a in alphas),
    }


# (name, group mask fn, which alpha column, expected direction, human-readable claim)
HYPOTHESES = [
    (
        "packed_vs_unpacked",
        lambda d: d["static_rel"][:, 5] > 0.5,
        0, "less",
        "PE bi packed => alpha_static thap hon",
    ),
    (
        "incomplete_vs_complete_trace",
        lambda d: d["dyn_rel"][:, 7] < 0.5,
        1, "less",
        "Trace khong hoan chinh => alpha_dynamic thap hon",
    ),
    (
        "few_imports_vs_many",
        lambda d: d["static_rel"][:, 3] < np.median(d["static_rel"][:, 3]),
        0, "less",
        "It import => alpha_static thap hon",
    ),
    (
        "short_trace_vs_long",
        lambda d: d["dyn_rel"][:, 0] < np.median(d["dyn_rel"][:, 0]),
        1, "less",
        "Trace ngan => alpha_dynamic thap hon",
    ),
]

ALPHA_NAMES = ["static", "dynamic", "interaction"]


def test_hypotheses(data: dict) -> list[dict]:
    """Run every pre-registered one-sided test. Groups smaller than 10 are skipped."""
    results = []
    for name, mask_fn, col, direction, claim in HYPOTHESES:
        mask = mask_fn(data)
        grp = data["alpha"][mask, col]
        ref = data["alpha"][~mask, col]
        row = {
            "test": name,
            "claim": claim,
            "alpha_component": ALPHA_NAMES[col],
            "n_group": int(mask.sum()),
            "n_reference": int((~mask).sum()),
        }
        if len(grp) < 10 or len(ref) < 10:
            row["status"] = "skipped (nhom qua nho)"
            results.append(row)
            continue
        u, p = mannwhitneyu(grp, ref, alternative=direction)
        d = cliffs_delta(grp, ref)
        row.update({
            "status": "ok",
            "mean_group": float(grp.mean()),
            "mean_reference": float(ref.mean()),
            "delta_mean": float(grp.mean() - ref.mean()),
            "p_value": float(p),
            "cliffs_delta": d,
            "effect": effect_label(d),
            "supported": bool(p < 0.05 and abs(d) >= 0.15),
        })
        results.append(row)
    return results


def summary_stats(data: dict) -> dict:
    a = data["alpha"]
    out = {
        "n_samples": int(data["n"]),
        "alpha_mean": {ALPHA_NAMES[i]: float(a[:, i].mean()) for i in range(3)},
        "alpha_std": {ALPHA_NAMES[i]: float(a[:, i].std()) for i in range(3)},
    }
    # A fusion that always picks the same modality proves nothing — flag it explicitly.
    dominant = a.mean(0).argmax()
    out["dominant_modality"] = ALPHA_NAMES[dominant]
    out["collapsed"] = bool(a[:, dominant].mean() > 0.9 and a[:, dominant].std() < 0.05)
    # correlation of each alpha against every reliability signal.
    # A constant signal (zero variance) has no defined correlation — report None
    # rather than letting numpy emit NaN and a divide warning.
    def safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
        if x.std() < 1e-12 or y.std() < 1e-12:
            return None
        return float(np.corrcoef(x, y)[0, 1])

    corr, constant = {}, []
    for j in range(STATIC_REL_DIM):
        if data["static_rel"][:, j].std() < 1e-12:
            constant.append(f"static_rel[{j}]")
    for j in range(DYN_REL_DIM):
        if data["dyn_rel"][:, j].std() < 1e-12:
            constant.append(f"dyn_rel[{j}]")
    for i in range(3):
        corr[ALPHA_NAMES[i]] = {
            **{f"static_rel[{j}]": safe_corr(a[:, i], data["static_rel"][:, j])
               for j in range(STATIC_REL_DIM)},
            **{f"dyn_rel[{j}]": safe_corr(a[:, i], data["dyn_rel"][:, j])
               for j in range(DYN_REL_DIM)},
        }
    out["correlations"] = corr
    out["constant_signals"] = constant
    return out


def format_report(stats: dict, tests: list[dict], family_names: list[str] | None = None) -> str:
    lines = ["# Phan tich alpha — bang chung cho Claim 3", ""]
    lines.append(f"So mau: {stats['n_samples']}")
    am, asd = stats["alpha_mean"], stats["alpha_std"]
    lines.append("")
    lines.append("| Thanh phan | Trung binh | Do lech chuan |")
    lines.append("|---|---|---|")
    for k in ALPHA_NAMES:
        lines.append(f"| alpha_{k} | {am[k]:.4f} | {asd[k]:.4f} |")
    lines.append("")
    if stats["collapsed"]:
        lines.append(f"> **CANH BAO: alpha da collapse ve modality '{stats['dominant_modality']}'** "
                     f"(trung binh > 0,9, phuong sai gan 0). Fusion khong thuc su thich ung theo "
                     f"tung mau => Claim 3 KHONG duoc chung minh. Can entropy regularization tren alpha.")
        lines.append("")

    lines.append("## Kiem dinh gia thuyet (mot phia, dat truoc khi xem du lieu)")
    lines.append("")
    lines.append("| Gia thuyet | n(nhom)/n(doi chung) | TB nhom | TB doi chung | p | Cliff's d | Muc do | Ket luan |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in tests:
        if t.get("status") != "ok":
            lines.append(f"| {t['claim']} | {t['n_group']}/{t['n_reference']} | — | — | — | — | — | {t.get('status')} |")
            continue
        verdict = "**ung ho**" if t["supported"] else "khong ung ho"
        p = f"{t['p_value']:.2e}" if t["p_value"] < 0.001 else f"{t['p_value']:.4f}"
        lines.append(
            f"| {t['claim']} | {t['n_group']}/{t['n_reference']} | {t['mean_group']:.4f} | "
            f"{t['mean_reference']:.4f} | {p} | {t['cliffs_delta']:+.3f} | {t['effect']} | {verdict} |"
        )
    lines.append("")
    lines.append("Ket luan can ca hai: p < 0,05 VA |Cliff's d| >= 0,15. "
                 "p nho tren hang nghin mau van co the la khac biet khong dang ke.")
    lines.append("")

    lines.append("## Tuong quan alpha voi tin hieu reliability (|r| >= 0,15)")
    lines.append("")
    for comp, cors in stats["correlations"].items():
        strong = {k: v for k, v in cors.items() if v is not None and abs(v) >= 0.15}
        if not strong:
            lines.append(f"- `alpha_{comp}`: khong co tuong quan nao dat nguong")
            continue
        top = sorted(strong.items(), key=lambda kv: -abs(kv[1]))[:5]
        lines.append(f"- `alpha_{comp}`: " + ", ".join(f"{k} = {v:+.3f}" for k, v in top))
    if stats.get("constant_signals"):
        lines.append("")
        lines.append(f"Tin hieu hang so (khong tinh duoc tuong quan): "
                     f"{', '.join(stats['constant_signals'])}")
    return "\n".join(lines)
