"""SD-FSProto full model: static encoder + dynamic encoder + fusion, and the
episodic forward pass producing all four loss terms."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dynamic_encoders import build_dynamic_encoder
from .fewshot import build_prototypes, class_distances
from .fusion import build_fusion
from .losses import align_loss, proto_sep_loss, supcon_loss
from .static_encoders import StaticEncoder


class SDFSProto(nn.Module):
    """modality: 'both' (default) | 'static' | 'dynamic' (single-view ablations)."""

    def __init__(self, model_cfg):
        super().__init__()
        self.cfg = model_cfg
        self.modality = model_cfg.get("modality", "both")
        d_out = model_cfg.embed_dim

        self.static_enc = StaticEncoder(model_cfg.static) if self.modality != "dynamic" else None
        self.dynamic_enc = build_dynamic_encoder(model_cfg.dynamic) if self.modality != "static" else None

        # Few-shot head: proto (ours) | matching | relation | siamese (baselines)
        self.head_kind = model_cfg.fewshot.get("head", "proto")
        if self.head_kind == "relation":
            from ..baselines.heads import RelationModule
            self.relation_head = RelationModule(d_out)

        if self.modality == "both":
            f = model_cfg.fusion
            self.fusion = build_fusion(
                f.kind, model_cfg.static.out_dim, model_cfg.dynamic.out_dim,
                d_out, f.static_rel_dim, f.dyn_rel_dim, f.get("hidden", 128),
            )
            self.head = None
        else:
            in_dim = model_cfg.static.out_dim if self.modality == "static" else model_cfg.dynamic.out_dim
            self.fusion = None
            self.head = nn.Sequential(nn.Linear(in_dim, d_out), nn.LayerNorm(d_out))

    def forward(self, batch: dict) -> dict:
        if self.modality == "static":
            return {"z": self.head(self.static_enc(batch)), "z_s": None, "z_d": None, "alpha": None}
        if self.modality == "dynamic":
            return {"z": self.head(self.dynamic_enc(batch)), "z_s": None, "z_d": None, "alpha": None}
        z_s = self.static_enc(batch)
        z_d = self.dynamic_enc(batch)
        return self.fusion(z_s, z_d, batch["static_rel"], batch["dyn_rel"])


def build_model(cfg, data_meta: dict | None = None) -> SDFSProto:
    """Instantiate from full config; data meta.json (vocab sizes etc.) overrides cfg."""
    model_cfg = copy.deepcopy(cfg.model)
    if data_meta:
        dyn, st = model_cfg.dynamic, model_cfg.static
        dyn.api_vocab = data_meta.get("api_vocab", dyn.api_vocab)
        dyn.sem_vocab = data_meta.get("sem_vocab", dyn.sem_vocab)
        dyn.graph.feat_hash = data_meta.get("graph_feat_hash", dyn.graph.feat_hash)
        st.imports.hash_dim = data_meta.get("import_hash", st.imports.hash_dim)
        st.strings.hash_dim = data_meta.get("string_hash", st.strings.hash_dim)
        st.metadata.in_dim = data_meta.get("meta_dim", st.metadata.in_dim)
        st.ember.in_dim = data_meta.get("ember_dim", st.ember.in_dim)
    return SDFSProto(model_cfg)


def episode_forward(model: SDFSProto, sb, sy, qb, qy, cfg, train: bool = True) -> dict:
    """One episode: build prototypes from support, classify queries, compute losses.

    Queries with label -1 are open-set distractors: excluded from all losses,
    included in the returned distance matrix for unknown-detection metrics.
    """
    n_way = int(sy.max().item()) + 1
    fs = cfg.model.fewshot
    out_s, out_q = model(sb), model(qb)
    known = qy >= 0

    head = getattr(model, "head_kind", "proto")
    if head != "proto":
        # Baseline heads (Matching / Relation / Siamese) on the same backbone embeddings
        from ..baselines.heads import matching_logits, siamese_logits
        if out_s["z"] is None:
            raise ValueError("baseline heads require a fused embedding (not late_vote)")
        if head == "matching":
            logits_b = matching_logits(out_s["z"], sy, out_q["z"], n_way)
        elif head == "siamese":
            logits_b = siamese_logits(out_s["z"], sy, out_q["z"], n_way)
        elif head == "relation":
            logits_b = model.relation_head(out_s["z"], sy, out_q["z"], n_way)
        else:
            raise ValueError(f"unknown head: {head}")
        cd, protos, pcls = -logits_b, None, None
        z_support, z_query = out_s["z"], out_q["z"]
        late = False
    elif out_s["z"] is None:
        # late_vote fusion: average per-modality distances
        late = True
        cd, protos, pcls = None, None, None
        for key in ("z_s", "z_d"):
            p, pc = build_prototypes(out_s[key], sy, n_way, fs.multi_proto, fs.max_protos)
            d = class_distances(out_q[key], p, pc, n_way, fs.metric)
            cd = d if cd is None else cd + d
        cd = cd / 2.0
        z_support, z_query = out_s["z_s"] + out_s["z_d"], out_q["z_s"] + out_q["z_d"]
    else:
        late = False
        protos, pcls = build_prototypes(out_s["z"], sy, n_way, fs.multi_proto, fs.max_protos)
        cd = class_distances(out_q["z"], protos, pcls, n_way, fs.metric)
        z_support, z_query = out_s["z"], out_q["z"]

    logits = -cd
    loss_cls = F.cross_entropy(logits[known], qy[known]) if known.any() else logits.new_zeros(())

    lc = cfg.loss
    z_all = torch.cat([z_support, z_query[known]], 0)
    y_all = torch.cat([sy, qy[known]], 0)
    loss_con = supcon_loss(z_all, y_all, lc.temp_con) if lc.lambda_con > 0 else logits.new_zeros(())

    if lc.lambda_align > 0 and out_s["z_s"] is not None and not late:
        zs_all = torch.cat([out_s["z_s"], out_q["z_s"][known]], 0)
        zd_all = torch.cat([out_s["z_d"], out_q["z_d"][known]], 0)
        loss_align = align_loss(zs_all, zd_all)
    else:
        loss_align = logits.new_zeros(())

    if lc.lambda_sep > 0 and protos is not None:
        loss_sep = proto_sep_loss(protos, pcls, lc.margin_sep)
    else:
        loss_sep = logits.new_zeros(())

    loss = loss_cls + lc.lambda_con * loss_con + lc.lambda_align * loss_align + lc.lambda_sep * loss_sep
    return {
        "loss": loss,
        "loss_cls": loss_cls.detach(),
        "loss_con": loss_con.detach(),
        "loss_align": loss_align.detach(),
        "loss_sep": loss_sep.detach(),
        "class_dist": cd.detach(),
        "query_y": qy,
        "alpha_q": None if out_q.get("alpha") is None else out_q["alpha"].detach(),
    }
