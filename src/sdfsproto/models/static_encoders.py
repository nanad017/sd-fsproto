"""Static view: 4 branches (bytes/MalConv, image CNN, imports+strings, metadata) -> z_s."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MalConvBranch(nn.Module):
    """Gated convolution over raw bytes (Raff et al., MalConv). Input: [B, L] with 0 = pad."""

    def __init__(self, embed_dim: int = 8, channels: int = 128, kernel: int = 512, stride: int = 512, out_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(257, embed_dim, padding_idx=0)  # 256 byte values + pad
        self.conv = nn.Conv1d(embed_dim, channels, kernel, stride=stride)
        self.gate = nn.Conv1d(embed_dim, channels, kernel, stride=stride)
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.conv.kernel_size[0]:  # guard: pad short inputs to one window
            x = F.pad(x, (0, self.conv.kernel_size[0] - x.size(1)))
        e = self.embed(x).transpose(1, 2)  # [B, E, L]
        h = self.conv(e) * torch.sigmoid(self.gate(e))
        h = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.fc(h)


class ImageBranch(nn.Module):
    """Small CNN over the grayscale binary image. Input: [B, 1, H, W] in [0,1]."""

    def __init__(self, channels: list[int] | None = None, out_dim: int = 128):
        super().__init__()
        channels = channels or [32, 64, 128]
        layers, c_in = [], 1
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(c_in, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


class BagBranch(nn.Module):
    """Order-free hashed token sets (imports or strings) via EmbeddingBag."""

    def __init__(self, hash_dim: int = 4096, embed_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.bag = nn.EmbeddingBag(hash_dim, embed_dim, mode="mean")
        self.fc = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.ReLU(inplace=True), nn.Linear(out_dim, out_dim))

    def forward(self, flat_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        if flat_ids.numel() == 0:  # a batch where every sample has an empty set
            e = torch.zeros(offsets.size(0), self.bag.embedding_dim, device=offsets.device)
        else:
            e = self.bag(flat_ids, offsets)
        return self.fc(e)


class EmberBranch(nn.Module):
    """Pre-computed EMBER-style feature vector -> MLP.

    EMBER packs byte/entropy histograms, string stats, header, section and hashed
    import/export features into one fixed-length vector (2,381 dims in EMBER2018;
    2,568 in the 2024 release). It therefore SUBSUMES the metadata and
    import/string branches — enable those together with `ember` only to measure
    the redundancy, not in the proposed model.

    What EMBER does NOT capture: byte ORDER (it stores histograms) and 2-D layout.
    That is exactly what the MalConv and image branches contribute alongside it.

    Counts in the vector are heavily skewed, so a log1p transform is applied
    before normalization (`log1p: true`, the default).
    """

    def __init__(self, in_dim: int = 2568, hidden: list[int] | None = None,
                 out_dim: int = 128, dropout: float = 0.1, log1p: bool = True):
        super().__init__()
        hidden = hidden or [1024, 512]
        self.log1p = log1p
        self.norm_in = nn.LayerNorm(in_dim)
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.log1p:
            # signed log1p: EMBER holds counts (>=0) but also a few signed stats
            x = torch.sign(x) * torch.log1p(x.abs())
        return self.net(self.norm_in(x))


class MetadataBranch(nn.Module):
    """PE header/section numeric features -> MLP."""

    def __init__(self, in_dim: int = 32, hidden: list[int] | None = None, out_dim: int = 128):
        super().__init__()
        hidden = hidden or [128, 128]
        layers: list[nn.Module] = [nn.LayerNorm(in_dim)]
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticEncoder(nn.Module):
    """Runs enabled branches and aggregates them into z_s.

    cfg is the `model.static` section. Branch ablations are driven by
    cfg.branches (e.g. ["bytes"], ["metadata", "imports"]).
    """

    BRANCHES = ("ember", "bytes", "image", "imports", "strings", "metadata")

    def __init__(self, cfg):
        super().__init__()
        self.enabled = [b for b in cfg.branches if b in self.BRANCHES]
        if not self.enabled:
            raise ValueError("static encoder: no branches enabled")
        d = cfg.out_dim
        self.branches = nn.ModuleDict()
        if "ember" in self.enabled:
            e = cfg.ember
            self.branches["ember"] = EmberBranch(
                e.in_dim, list(e.hidden), d, e.get("dropout", 0.1), e.get("log1p", True)
            )
        if "bytes" in self.enabled:
            b = cfg.bytes
            self.branches["bytes"] = MalConvBranch(b.embed_dim, b.channels, b.kernel, b.stride, d)
        if "image" in self.enabled:
            self.branches["image"] = ImageBranch(list(cfg.image.channels), d)
        if "imports" in self.enabled:
            self.branches["imports"] = BagBranch(cfg.imports.hash_dim, cfg.imports.embed_dim, d)
        if "strings" in self.enabled:
            self.branches["strings"] = BagBranch(cfg.strings.hash_dim, cfg.strings.embed_dim, d)
        if "metadata" in self.enabled:
            m = cfg.metadata
            self.branches["metadata"] = MetadataBranch(m.in_dim, list(m.hidden), d)
        self.mix = nn.Sequential(
            nn.LayerNorm(d * len(self.enabled)),
            nn.Linear(d * len(self.enabled), d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )
        self.out_dim = d

    def forward(self, batch: dict) -> torch.Tensor:
        hs = []
        for name in self.enabled:
            mod = self.branches[name]
            if name == "ember":
                hs.append(mod(batch["ember"]))
            elif name == "bytes":
                hs.append(mod(batch["bytes"]))
            elif name == "image":
                hs.append(mod(batch["image"]))
            elif name == "imports":
                hs.append(mod(batch["import_ids"], batch["import_ids_offsets"]))
            elif name == "strings":
                hs.append(mod(batch["string_ids"], batch["string_ids_offsets"]))
            elif name == "metadata":
                hs.append(mod(batch["metadata"]))
        return self.mix(torch.cat(hs, dim=-1))
