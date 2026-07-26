"""Episodic trainer: fit / evaluate / test with tau calibration and unknown detection."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import EpisodeSampler, MalwareDataset, batch_to_device, collate
from ..models.sdfsproto import build_model, episode_forward
from .metrics import EpisodeMetrics, calibrate_tau


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = resolve_device(cfg.device)
        self.run_dir = Path(cfg.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg.to_dict(), f, indent=2)

        self.ds = MalwareDataset(cfg.data.root)
        self.model = build_model(cfg, self.ds.meta).to(self.device)
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        if cfg.train.scheduler == "cosine":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=cfg.train.epochs)
        else:
            self.sched = None
        self.max_bytes = cfg.model.static.bytes.max_len
        self.max_seq = cfg.model.dynamic.max_seq_len

    # ---------- helpers ----------

    def _load_episode(self, ep: dict):
        sb = collate([self.ds[i] for i in ep["support_idx"]], self.max_bytes, self.max_seq)
        qb = collate([self.ds[i] for i in ep["query_idx"]], self.max_bytes, self.max_seq)
        sy = torch.tensor(ep["support_y"], dtype=torch.long)
        qy = torch.tensor(ep["query_y"], dtype=torch.long)
        return (
            batch_to_device(sb, self.device),
            sy.to(self.device),
            batch_to_device(qb, self.device),
            qy.to(self.device),
        )

    # ---------- train ----------

    def train_epoch(self, epoch: int) -> dict:
        d = self.cfg.data
        sampler = EpisodeSampler(
            self.ds, "train", d.n_way, d.k_shot, d.n_query, n_unknown=0,
            seed=self.cfg.seed * 10_000 + epoch,
            split_mode=d.get("split_mode"), exclude_families=d.get("exclude_families"),
        )
        self.model.train()
        losses, accs = [], []
        for it in range(d.episodes_per_epoch):
            sb, sy, qb, qy = self._load_episode(sampler.sample())
            out = episode_forward(self.model, sb, sy, qb, qy, self.cfg, train=True)
            self.opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            if self.cfg.train.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.grad_clip)
            self.opt.step()
            losses.append(float(out["loss"]))
            pred = out["class_dist"].argmin(dim=1)
            accs.append(float((pred == qy).float().mean()))
            if (it + 1) % self.cfg.train.log_every == 0:
                print(f"  ep {epoch} it {it+1}/{d.episodes_per_epoch} "
                      f"loss {np.mean(losses[-self.cfg.train.log_every:]):.4f} "
                      f"acc {np.mean(accs[-self.cfg.train.log_every:]):.4f}")
        return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}

    # ---------- evaluate ----------

    @torch.no_grad()
    def evaluate(self, split: str, episodes: int, n_unknown: int = 0,
                 tau: float | None = None, seed: int = 1234,
                 family_detail: bool = False) -> tuple[dict, EpisodeMetrics]:
        d = self.cfg.data
        sampler = EpisodeSampler(self.ds, split, d.n_way, d.k_shot, d.n_query,
                                 n_unknown=n_unknown, seed=seed,
                                 split_mode=d.get("split_mode"),
                                 exclude_families=d.get("exclude_families"))
        self.model.eval()
        mm = EpisodeMetrics(tau)
        alphas, t_query, n_query_total = [], 0.0, 0
        for _ in range(episodes):
            ep = sampler.sample()
            sb, sy, qb, qy = self._load_episode(ep)
            t0 = time.perf_counter()
            out = episode_forward(self.model, sb, sy, qb, qy, self.cfg, train=False)
            t_query += time.perf_counter() - t0
            n_query_total += int(qy.numel())
            mm.update(out["class_dist"], qy, ep["families"] if family_detail else None)
            if out["alpha_q"] is not None:
                alphas.append(out["alpha_q"].mean(0).cpu().numpy())
        res = mm.compute()
        res["ms_per_query"] = 1000.0 * t_query / max(1, n_query_total)
        if alphas:
            a = np.stack(alphas).mean(0)
            res["alpha_mean"] = {"static": float(a[0]), "dynamic": float(a[1]), "interaction": float(a[2])}
        return res, mm

    # ---------- fit / test ----------

    def fit(self) -> dict:
        best_acc, best_epoch, patience = -1.0, -1, 0
        history = []
        for epoch in range(1, self.cfg.train.epochs + 1):
            tr = self.train_epoch(epoch)
            val, _ = self.evaluate("val", self.cfg.data.val_episodes, n_unknown=0)
            if self.sched:
                self.sched.step()
            history.append({"epoch": epoch, **{f"train_{k}": v for k, v in tr.items()},
                            "val_acc": val["accuracy"], "val_f1": val["macro_f1"]})
            print(f"epoch {epoch}: train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | "
                  f"val acc {val['accuracy']:.4f} f1 {val['macro_f1']:.4f}")
            if val["accuracy"] > best_acc:
                best_acc, best_epoch, patience = val["accuracy"], epoch, 0
                torch.save({"model": self.model.state_dict(), "cfg": self.cfg.to_dict(),
                            "epoch": epoch, "val_acc": best_acc}, self.run_dir / "best.pt")
            else:
                patience += 1
                if patience >= self.cfg.train.early_stop_patience:
                    print(f"early stop at epoch {epoch} (best {best_acc:.4f} @ {best_epoch})")
                    break
        with open(self.run_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return {"best_val_acc": best_acc, "best_epoch": best_epoch}

    def test(self) -> dict:
        ckpt_path = self.run_dir / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])

        d = self.cfg.data
        n_unknown = d.get("n_unknown", 5)
        # 1) calibrate tau on validation episodes (with unknowns present)
        _, mm_val = self.evaluate("val", d.val_episodes, n_unknown=n_unknown, seed=777)
        tau = self.cfg.model.fewshot.tau
        if tau is None and mm_val.known_scores:
            unk = np.concatenate(mm_val.unknown_scores) if mm_val.unknown_scores else None
            tau = calibrate_tau(np.concatenate(mm_val.known_scores), unk)
        # 2) final test with unknown queries
        res, _ = self.evaluate("test", d.test_episodes, n_unknown=n_unknown,
                               tau=tau, seed=999, family_detail=True)
        res["tau"] = tau
        with open(self.run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
        print("test:", json.dumps({k: v for k, v in res.items() if k != "family_f1"}, indent=2))
        return res
