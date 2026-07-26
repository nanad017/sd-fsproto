#!/usr/bin/env python
"""Evaluate the two extensions on a trained checkpoint:

  1. Semi-supervised prototype refinement (§11): episodic eval on the test
     split with an unlabeled pool drawn from episode families (+ contamination
     from other families); compares accuracy before/after refinement.
  2. Drift-aware prototype update (§12): timeline simulation over test
     families; frozen prototypes vs adaptive PrototypeBank.

Usage:
  python scripts/eval_extensions.py --run runs/dummy_smoke [--episodes 100]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.config import Config
from sdfsproto.data.dataset import EpisodeSampler, MalwareDataset, batch_to_device, collate
from sdfsproto.extensions.drift import simulate_drift_adaptation
from sdfsproto.extensions.semi_supervised import refine_prototypes
from sdfsproto.models.fewshot import build_prototypes, class_distances
from sdfsproto.models.sdfsproto import build_model


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir containing best.pt + config.json")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--pool-size", type=int, default=60)
    ap.add_argument("--contamination", type=float, default=0.3,
                    help="fraction of the unlabeled pool drawn from OTHER families")
    ap.add_argument("--seed", type=int, default=555)
    args = ap.parse_args()

    run = Path(args.run)
    with open(run / "config.json", "r", encoding="utf-8") as f:
        cfg = Config._wrap(json.load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MalwareDataset(cfg.data.root)
    model = build_model(cfg, ds.meta).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    max_bytes = cfg.model.static.bytes.max_len
    max_seq = cfg.model.dynamic.max_seq_len

    def embed(idx: list[int]) -> torch.Tensor:
        b = batch_to_device(collate([ds[i] for i in idx], max_bytes, max_seq), device)
        return model(b)["z"]

    rng = np.random.default_rng(args.seed)
    d = cfg.data
    fs = cfg.model.fewshot
    ss = cfg.semi_supervised

    # ---------------- 1) semi-supervised refinement ----------------
    sampler = EpisodeSampler(ds, "test", d.n_way, d.k_shot, d.n_query, n_unknown=0, seed=args.seed)
    acc_base, acc_ref, n_pseudo_all = [], [], []
    for _ in range(args.episodes):
        ep = sampler.sample()
        used = set(ep["support_idx"]) | set(ep["query_idx"])
        # unlabeled pool: episode-family samples not used, + contamination from other families
        in_fam, out_fam = [], []
        for fam, idxs in sampler.fam_to_idx.items():
            for i in idxs:
                if i in used:
                    continue
                (in_fam if fam in ep["families"] else out_fam).append(i)
        n_out = int(args.pool_size * args.contamination)
        n_in = args.pool_size - n_out
        pool = list(rng.choice(in_fam, size=min(n_in, len(in_fam)), replace=False)) + \
               list(rng.choice(out_fam, size=min(n_out, len(out_fam)), replace=False))

        z_s = embed(ep["support_idx"])
        z_q = embed(ep["query_idx"])
        z_u = embed([int(i) for i in pool])
        sy = torch.tensor(ep["support_y"], device=device)
        qy = torch.tensor(ep["query_y"], device=device)

        protos, pcls, assign = build_prototypes(z_s, sy, d.n_way, fs.multi_proto, fs.max_protos,
                                                return_assign=True)
        cd0 = class_distances(z_q, protos, pcls, d.n_way, fs.metric)
        acc_base.append(float((cd0.argmin(1) == qy).float().mean()))

        protos2, stats = refine_prototypes(z_s, assign, protos, pcls, z_u,
                                           ss.retrieve_k, ss.conf_threshold, ss.beta, fs.metric)
        cd1 = class_distances(z_q, protos2, pcls, d.n_way, fs.metric)
        acc_ref.append(float((cd1.argmin(1) == qy).float().mean()))
        n_pseudo_all.append(stats["n_pseudo"])

    semi = {
        "acc_base": float(np.mean(acc_base)),
        "acc_refined": float(np.mean(acc_ref)),
        "delta": float(np.mean(acc_ref) - np.mean(acc_base)),
        "mean_pseudo_per_episode": float(np.mean(n_pseudo_all)),
        "episodes": args.episodes,
        "contamination": args.contamination,
    }
    print("semi-supervised:", json.dumps(semi, indent=2))

    # ---------------- 2) drift-aware adaptation ----------------
    drift_cfg = cfg.drift
    drift = simulate_drift_adaptation(
        embed, ds, ds.index["splits"]["test"],
        k_init=d.k_shot, chunk=10,
        gamma=drift_cfg.gamma, spawn_dist=drift_cfg.spawn_dist, max_protos=drift_cfg.max_protos,
    )
    print("drift:", json.dumps({k: v for k, v in drift.items() if not isinstance(v, list)}, indent=2))

    with open(run / "extensions.json", "w", encoding="utf-8") as f:
        json.dump({"semi_supervised": semi, "drift": drift}, f, indent=2)
    print(f"saved to {run}/extensions.json")


if __name__ == "__main__":
    main()
