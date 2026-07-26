#!/usr/bin/env python
"""Train + test SD-FSProto.

Usage:
  python scripts/train.py --config configs/default.yaml [--config configs/dummy_smoke.yaml]
                          [--set model.dynamic.backend=api_seq] [--set data.n_way=10] ...
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.config import apply_overrides, load_config
from sdfsproto.engine.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", required=True,
                    help="YAML config; repeat to deep-merge overrides")
    ap.add_argument("--set", action="append", default=[], dest="sets",
                    help="dotted override, e.g. model.dynamic.backend=api_seq")
    ap.add_argument("--no-test", action="store_true", help="skip final test evaluation")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    apply_overrides(cfg, args.sets)

    trainer = Trainer(cfg)
    fit_res = trainer.fit()
    print("fit:", fit_res)
    if not args.no_test:
        trainer.test()


if __name__ == "__main__":
    main()
