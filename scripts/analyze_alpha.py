#!/usr/bin/env python
"""Alpha analysis — quantitative evidence for Claim 3 (reliability-aware fusion).

Tests whether the learned fusion weights actually shift away from a modality when
that modality is unreliable (packed PE, incomplete sandbox trace, stripped imports).

Usage:
  python scripts/analyze_alpha.py --run runs/dataset_a [--split test] [--max-samples 4000]

Outputs <run>/alpha_analysis.md (table for the paper) and <run>/alpha_analysis.json.
Requires the run to use model.fusion.kind=reliability.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.config import Config
from sdfsproto.data.dataset import MalwareDataset
from sdfsproto.engine.alpha import collect_alpha, format_report, summary_stats, test_hypotheses
from sdfsproto.models.sdfsproto import build_model


def select_indices(ds: MalwareDataset, split: str | None, limit: int | None, seed: int) -> list[int]:
    """All labeled samples of a split (or of the whole dataset when split is None)."""
    mode = ds.index.get("split_mode", "family")
    idx = []
    for i, rec in enumerate(ds.samples):
        if not rec["labeled"]:
            continue
        if split:
            in_split = (rec["split"] == split) if mode == "sample" else (
                rec["family"] in ds.index["splits"][split])
            if not in_split:
                continue
        idx.append(i)
    if limit and len(idx) > limit:
        idx = list(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    return [int(i) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with best.pt + config.json")
    ap.add_argument("--split", default=None,
                    help="train/val/test; omit to use every labeled sample (more statistical power)")
    ap.add_argument("--max-samples", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run = Path(args.run)
    with open(run / "config.json", "r", encoding="utf-8") as f:
        cfg = Config._wrap(json.load(f))
    if cfg.model.fusion.kind != "reliability":
        raise SystemExit(f"run uses fusion.kind={cfg.model.fusion.kind}; alpha analysis needs "
                         f"'reliability'. Re-train with the proposed fusion.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MalwareDataset(cfg.data.root)
    model = build_model(cfg, ds.meta).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    idx = select_indices(ds, args.split, args.max_samples, args.seed)
    print(f"analyzing {len(idx)} samples (split={args.split or 'all'}) on {device}")

    data = collect_alpha(model, ds, idx, device,
                         cfg.model.static.bytes.max_len, cfg.model.dynamic.max_seq_len,
                         batch_size=args.batch_size)
    stats = summary_stats(data)
    tests = test_hypotheses(data)
    report = format_report(stats, tests)

    (run / "alpha_analysis.md").write_text(report + "\n", encoding="utf-8")
    with open(run / "alpha_analysis.json", "w", encoding="utf-8") as f:
        json.dump({"summary": stats, "tests": tests, "split": args.split,
                   "n_analyzed": len(idx)}, f, indent=2)
    print("\n" + report)
    print(f"\nsaved -> {run}/alpha_analysis.md, alpha_analysis.json")


if __name__ == "__main__":
    main()
