"""Dynamic view: 4 plug-in backends sharing one interface.

  1. api_seq        Transformer over API-name tokens
  2. semantic_seq   Transformer over API<SEMANTIC_TAG> tokens
  3. behavior_graph typed message-passing GNN over the behavior graph
                    (pure torch, no torch-geometric dependency)
  4. api_set        order-free set encoder — for datasets whose behaviour report
                    is a SET of resolved APIs with no call sequence (Avast-CTU).
                    Using a sequence model there would pretend an order exists.

All backends: forward(batch) -> z_d [B, out_dim].
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeqTransformer(nn.Module):
    """Transformer encoder over a padded token sequence (0 = pad)."""

    def __init__(self, vocab: int, d_model: int, n_heads: int, n_layers: int,
                 max_len: int, out_dim: int, dropout: float = 0.1, input_key: str = "api_ids"):
        super().__init__()
        self.input_key = input_key
        self.embed = nn.Embedding(vocab + 1, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len + 1, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.fc = nn.Linear(d_model, out_dim)
        self.max_len = max_len

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch[self.input_key][:, : self.max_len]  # [B, T], 0 = pad
        mask = x.eq(0)
        # Guard against fully-empty rows (would NaN in attention): give them one pad token unmasked
        empty = mask.all(dim=1)
        if empty.any():
            mask[empty, 0] = False
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos(pos)
        h = self.encoder(h, src_key_padding_mask=mask)
        keep = (~mask).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # masked mean
        return self.fc(pooled)


class BehaviorGraphEncoder(nn.Module):
    """Typed message passing over the behavior graph.

    Node state: type embedding + hashed-name embedding.
    Message:    MLP([h_src ; edge-type embedding]) aggregated at dst (mean).
    Update:     GRUCell. Readout: per-graph masked mean + max -> fc.
    """

    def __init__(self, node_types: int, edge_types: int, feat_hash: int,
                 hidden: int, layers: int, out_dim: int):
        super().__init__()
        self.type_emb = nn.Embedding(node_types, hidden)
        self.feat_emb = nn.Embedding(feat_hash, hidden)
        self.edge_emb = nn.Embedding(edge_types, hidden)
        self.msg = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.update = nn.GRUCell(hidden, hidden)
        self.layers = layers
        self.readout = nn.Linear(2 * hidden, out_dim)
        self.hidden = hidden

    def forward(self, batch: dict) -> torch.Tensor:
        h = self.type_emb(batch["node_type"]) + self.feat_emb(batch["node_feat"])  # [V, H]
        src, dst, et = batch["edge_src"], batch["edge_dst"], batch["edge_type"]
        n_nodes = h.size(0)
        for _ in range(self.layers):
            if src.numel() > 0:
                m = self.msg(torch.cat([h[src], self.edge_emb(et)], dim=-1))  # [E, H]
                agg = torch.zeros(n_nodes, self.hidden, device=h.device, dtype=h.dtype)
                agg.index_add_(0, dst, m)
                deg = torch.zeros(n_nodes, device=h.device, dtype=h.dtype)
                deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
                agg = agg / deg.clamp(min=1.0).unsqueeze(-1)
            else:
                agg = torch.zeros_like(h)
            h = self.update(agg, h)
        # per-graph readout
        g = batch["node_batch"]
        n_graphs = int(batch["num_graphs"])
        mean = torch.zeros(n_graphs, self.hidden, device=h.device, dtype=h.dtype)
        mean.index_add_(0, g, h)
        cnt = torch.zeros(n_graphs, device=h.device, dtype=h.dtype)
        cnt.index_add_(0, g, torch.ones_like(g, dtype=h.dtype))
        mean = mean / cnt.clamp(min=1.0).unsqueeze(-1)
        mx = torch.full((n_graphs, self.hidden), torch.finfo(h.dtype).min, device=h.device, dtype=h.dtype)
        mx.index_reduce_(0, g, h, reduce="amax", include_self=True)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        return self.readout(torch.cat([mean, mx], dim=-1))


class ApiSetEncoder(nn.Module):
    """Order-free encoder over a SET of resolved APIs plus behaviour context tags.

    For reports that list which APIs were resolved but not in which order
    (Avast-CTU `behavior.summary.resolved_apis`). Mean-pools API embeddings, then
    concatenates a multi-hot vector of the semantic context tags derived from the
    registry/file/command lists — those carry the intent the API set alone misses.
    """

    def __init__(self, vocab: int, embed_dim: int, n_tags: int, hidden: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.bag = nn.EmbeddingBag(vocab, embed_dim, mode="mean")
        self.n_tags = n_tags
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim + n_tags),
            nn.Linear(embed_dim + n_tags, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        ids, offsets = batch["api_set_ids"], batch["api_set_ids_offsets"]
        if ids.numel() == 0:                       # every sample had an empty API set
            e = torch.zeros(offsets.size(0), self.bag.embedding_dim, device=offsets.device)
        else:
            e = self.bag(ids, offsets)
        tags = batch.get("context_tags")
        if tags is None:
            tags = e.new_zeros(e.size(0), self.n_tags)
        return self.net(torch.cat([e, tags.to(e.dtype)], dim=-1))


def build_dynamic_encoder(cfg) -> nn.Module:
    """cfg is the `model.dynamic` section."""
    kind = cfg.backend
    if kind == "api_seq":
        return SeqTransformer(cfg.api_vocab, cfg.d_model, cfg.n_heads, cfg.n_layers,
                              cfg.max_seq_len, cfg.out_dim, cfg.dropout, input_key="api_ids")
    if kind == "semantic_seq":
        return SeqTransformer(cfg.sem_vocab, cfg.d_model, cfg.n_heads, cfg.n_layers,
                              cfg.max_seq_len, cfg.out_dim, cfg.dropout, input_key="sem_ids")
    if kind == "behavior_graph":
        g = cfg.graph
        return BehaviorGraphEncoder(g.node_types, g.edge_types, g.feat_hash,
                                    g.hidden, g.layers, cfg.out_dim)
    if kind == "api_set":
        s = cfg.api_set
        return ApiSetEncoder(s.vocab, s.embed_dim, s.n_tags, s.hidden, cfg.out_dim, cfg.dropout)
    raise ValueError(f"unknown dynamic backend: {kind}")
