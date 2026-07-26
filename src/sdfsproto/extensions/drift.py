"""Drift-aware prototype maintenance (idea doc §12).

    p_c^t = (1 - gamma) * p_c^{t-1} + gamma * z_mean_c^t

If a new batch of a family clusters far from every existing sub-prototype,
a NEW sub-prototype is spawned instead of dragging the old ones — the family
evolves without forgetting its old structure.
"""

from __future__ import annotations

import torch


class PrototypeBank:
    """Per-family list of sub-prototypes with EMA updates and spawning."""

    def __init__(self, gamma: float = 0.3, spawn_dist: float = 6.0, max_protos: int = 5):
        self.gamma = gamma
        self.spawn_dist = spawn_dist          # default; can be overridden per family
        self.max_protos = max_protos
        self.protos: dict[int, list[torch.Tensor]] = {}
        self.spawn: dict[int, float] = {}

    def init_family(self, family: int, protos: torch.Tensor, spawn_dist: float | None = None) -> None:
        self.protos[family] = [p.clone() for p in protos]
        self.spawn[family] = float(spawn_dist) if spawn_dist is not None else self.spawn_dist

    def as_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(protos [P, D], proto_family [P]) over all families in the bank."""
        ps, fams = [], []
        for f, plist in sorted(self.protos.items()):
            ps.extend(plist)
            fams.extend([f] * len(plist))
        return torch.stack(ps), torch.tensor(fams, dtype=torch.long, device=ps[0].device)

    def update(self, family: int, z_new: torch.Tensor) -> dict:
        """Fold a batch of new embeddings of `family` into its prototypes.

        Each sample EMA-updates its nearest sub-prototype; samples farther than
        spawn_dist from every sub-prototype form a candidate cluster whose mean
        becomes a new sub-prototype (capped at max_protos).
        """
        plist = self.protos[family]
        P = torch.stack(plist)                                  # [M, D]
        d = torch.cdist(z_new, P)                               # [B, M]
        dmin, nearest = d.min(dim=1)
        far = dmin > self.spawn.get(family, self.spawn_dist)

        spawned = 0
        for j in range(len(plist)):
            m = (~far) & (nearest == j)
            if m.any():
                z_bar = z_new[m].mean(0)
                plist[j] = (1 - self.gamma) * plist[j] + self.gamma * z_bar
        # Spawn only when a CLUSTER of samples sits outside the family's spread —
        # a lone outlier (e.g. one evasive run) must not create a sub-prototype.
        min_cluster = max(3, int(0.2 * z_new.size(0)))
        if int(far.sum()) >= min_cluster and len(plist) < self.max_protos:
            plist.append(z_new[far].mean(0).clone())
            spawned = 1
        self.protos[family] = plist
        return {"n_far": int(far.sum()), "spawned": spawned, "n_protos": len(plist)}


@torch.no_grad()
def simulate_drift_adaptation(
    embed_fn,
    dataset,
    families: list[str],
    k_init: int = 5,
    chunk: int = 10,
    gamma: float = 0.3,
    spawn_dist: float | str = "auto",
    max_protos: int = 5,
) -> dict:
    """Timeline evaluation: for each family, sort samples by timestamp, build the
    initial prototype from the first k samples, then stream the rest in chunks.

    Reports accuracy over time for (a) frozen prototypes vs (b) the drift-aware
    bank that adapts after each chunk (chunk labels revealed post-hoc, i.e. the
    'analyst labels new samples' loop of §13 step 7).

    embed_fn: callable(list of sample indices) -> [n, D] embeddings.
    """
    fam_id = {f: i for i, f in enumerate(families)}
    per_fam: dict[int, list[int]] = {i: [] for i in fam_id.values()}
    for i, rec in enumerate(dataset.samples):
        if rec["family"] in fam_id and rec["labeled"]:
            per_fam[fam_id[rec["family"]]].append(i)
    for f in per_fam:
        per_fam[f].sort(key=lambda i: dataset.samples[i]["timestamp"])

    auto = spawn_dist == "auto"
    bank = PrototypeBank(gamma, 6.0 if auto else float(spawn_dist), max_protos)
    frozen: dict[int, torch.Tensor] = {}
    streams: dict[int, list[int]] = {}
    intra: list[float] = []
    inits: dict[int, torch.Tensor] = {}
    for f, idx in per_fam.items():
        if len(idx) < k_init + chunk:
            continue
        z0 = embed_fn(idx[:k_init])
        inits[f] = z0
        frozen[f] = z0.mean(0)
        streams[f] = idx[k_init:]
        if z0.size(0) >= 2:
            intra.append(float((z0 - z0.mean(0)).norm(dim=1).mean()))
    # "auto": spawn only when a batch lands clearly outside the family's own spread
    global_spawn = 3.0 * (sum(intra) / len(intra)) if (auto and intra) else None
    for f, z0 in inits.items():
        sd = None
        if auto:
            own = float((z0 - z0.mean(0)).norm(dim=1).mean()) if z0.size(0) >= 2 else 0.0
            sd = max(3.0 * own, global_spawn or 0.0)
        bank.init_family(f, z0.mean(0, keepdim=True), spawn_dist=sd)

    n_steps = max(len(s) // chunk for s in streams.values())
    acc_frozen, acc_adaptive = [], []
    for step in range(n_steps):
        correct_f = correct_a = total = 0
        for f, stream in streams.items():
            batch = stream[step * chunk : (step + 1) * chunk]
            if not batch:
                continue
            z = embed_fn(batch)
            # frozen classification
            Pf = torch.stack(list(frozen.values()))
            fams_f = list(frozen.keys())
            pred_f = [fams_f[j] for j in torch.cdist(z, Pf).argmin(dim=1).tolist()]
            # adaptive classification
            Pa, fam_a = bank.as_tensors()
            pred_a = fam_a[torch.cdist(z, Pa).argmin(dim=1)].tolist()
            correct_f += sum(p == f for p in pred_f)
            correct_a += sum(p == f for p in pred_a)
            total += len(batch)
            bank.update(f, z)  # analyst labels arrive after classification
        if total:
            acc_frozen.append(correct_f / total)
            acc_adaptive.append(correct_a / total)
    return {
        "steps": n_steps,
        "acc_frozen": acc_frozen,
        "acc_adaptive": acc_adaptive,
        "mean_frozen": float(sum(acc_frozen) / max(1, len(acc_frozen))),
        "mean_adaptive": float(sum(acc_adaptive) / max(1, len(acc_adaptive))),
        "final_protos": {f: len(p) for f, p in bank.protos.items()},
    }
