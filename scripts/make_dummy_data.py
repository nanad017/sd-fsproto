#!/usr/bin/env python
"""Generate the synthetic smoke-test dataset."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.data.dummy import generate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/dummy")
    ap.add_argument("--families", type=int, default=30)
    ap.add_argument("--min-samples", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=60)
    ap.add_argument("--unlabeled-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(a.root, a.families, a.min_samples, a.max_samples, a.unlabeled_frac, a.seed)
