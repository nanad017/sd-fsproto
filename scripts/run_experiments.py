#!/usr/bin/env python
"""Run the full experiment matrix (idea doc §15-16) and aggregate results.

Usage:
  python scripts/run_experiments.py --config configs/default.yaml [--config configs/dummy_smoke.yaml]
                                    [--only 'proposed|fusion'] [--out results/exp1]

Each experiment = named set of dotted overrides on top of the base config.
Deep-learning runs go through Trainer (fit + test); classical baselines run
their own episodic protocol on the test split.
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.config import apply_overrides, load_config

# ---------------------------------------------------------------- experiment matrix
# Proposed model = semantic_seq dynamic backend + reliability fusion + multi-prototype.
EXPERIMENTS: list[tuple[str, list[str]]] = [
    # --- main results: 3 few-shot settings ---
    ("proposed_5w1s", ["data.k_shot=1"]),
    ("proposed_5w5s", []),
    ("proposed_10w5s", ["data.n_way=10"]),
    # --- dynamic backend comparison (full model) ---
    ("full_api_seq", ["model.dynamic.backend=api_seq"]),
    ("full_behavior_graph", ["model.dynamic.backend=behavior_graph"]),
    # --- static branch ablations (only meaningful when EMBER vectors exist) ---
    ("static_ember_only", ["model.static.branches=[ember]"]),
    ("static_ember_bytes", ["model.static.branches=[ember,bytes]"]),
    ("static_no_ember", ["model.static.branches=[bytes,image,imports,strings,metadata]"]),
    # --- modality ablations (5w5s) ---
    ("static_only", ["model.modality=static"]),
    ("dynamic_only_api_seq", ["model.modality=dynamic", "model.dynamic.backend=api_seq"]),
    ("dynamic_only_semantic", ["model.modality=dynamic", "model.dynamic.backend=semantic_seq"]),
    ("dynamic_only_graph", ["model.modality=dynamic", "model.dynamic.backend=behavior_graph"]),
    # --- fusion ablations ---
    ("fusion_concat", ["model.fusion.kind=concat"]),
    ("fusion_attention", ["model.fusion.kind=attention"]),
    ("fusion_late_vote", ["model.fusion.kind=late_vote"]),
    # --- prototype ablation ---
    ("single_proto", ["model.fewshot.multi_proto=false"]),
    # --- few-shot head baselines (same backbone) ---
    ("head_matching", ["model.fewshot.head=matching"]),
    ("head_relation", ["model.fewshot.head=relation"]),
    ("head_siamese", ["model.fewshot.head=siamese"]),
]

CLASSICAL = ["rf", "svm", "gboost"]


def fmt_row(name: str, m: dict) -> str:
    acc = f"{m.get('accuracy', 0):.4f} ± {m.get('accuracy_ci95', 0):.4f}"
    f1 = f"{m.get('macro_f1', 0):.4f}"
    auroc = f"{m['unknown_auroc']:.4f}" if "unknown_auroc" in m else "—"
    ms = f"{m['ms_per_query']:.1f}" if "ms_per_query" in m else "—"
    return f"| {name} | {acc} | {f1} | {auroc} | {ms} |"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--only", default=None, help="regex filter on experiment names")
    ap.add_argument("--out", default="results/latest")
    ap.add_argument("--skip-classical", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(args.only) if args.only else None
    results: dict[str, dict] = {}

    from sdfsproto.engine.trainer import Trainer

    for name, sets in EXPERIMENTS:
        if pat and not pat.search(name):
            continue
        print(f"\n=== {name} ===")
        cfg = load_config(*args.config)
        apply_overrides(cfg, sets + [f"run_dir={out_dir}/runs/{name}"])
        trainer = Trainer(cfg)
        trainer.fit()
        results[name] = trainer.test()
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if not args.skip_classical and (pat is None or pat.search("classical")):
        from sdfsproto.baselines.classical import run_classical
        base = load_config(*args.config)
        for clf in CLASSICAL:
            name = f"classical_{clf}"
            print(f"\n=== {name} ===")
            results[name] = run_classical(
                base.data.root, clf,
                n_way=base.data.n_way, k_shot=base.data.k_shot,
                n_query=base.data.n_query, episodes=base.data.test_episodes,
            )
            with open(out_dir / "results.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

    # ---- markdown table ----
    lines = [
        "| Model | Accuracy | Macro-F1 | Unknown AUROC | ms/query |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        lines.append(fmt_row(name, m))
    table = "\n".join(lines)
    with open(out_dir / "results.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print("\n" + table)
    print(f"\nSaved to {out_dir}/results.json and results.md")


if __name__ == "__main__":
    main()
