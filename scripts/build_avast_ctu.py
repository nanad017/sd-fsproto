#!/usr/bin/env python
"""Build a dataset from the Avast-CTU Public CAPE Dataset (reduced reports).

  python scripts/build_avast_ctu.py \
      --reports /path/to/reduced_reports \
      --labels  /path/to/public_labels.csv \
      --out     data/avast

The paper's split is MANDATORY and time-based: train < 2019-08-01, test >= that.
A random split inflates static-only accuracy from ~63% to >95% — it measures
nothing about generalisation. This script therefore ignores any requested random
split and always cuts on the date; validation is carved from the LATEST portion
of train, so val also sits ahead of train in time.
"""

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.data import schema
from sdfsproto.extract.avast_ctu import (
    API_SET_VOCAB,
    GRAPH_FEAT_HASH,
    IMPORT_HASH,
    STRING_HASH,
    extract_avast_ctu,
)
from sdfsproto.extract.cape_report import N_TAGS

CUTOFF = "2019-08-01"


def parse_date(s: str) -> float:
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s[: len(fmt) + 4], fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def load_labels(path: str) -> dict[str, dict]:
    """Accept flexible column naming: sha256/hash, classification/family, date, type."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{path}: rỗng")
    cols = {c.lower().strip(): c for c in rows[0]}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    c_sha = pick("sha256", "sha_256", "hash", "id")
    c_fam = pick("classification", "family", "label")
    c_date = pick("date", "first_seen", "timestamp")
    c_type = pick("type", "malware_type")
    if not c_sha or not c_fam:
        raise SystemExit(f"{path}: thiếu cột sha256 hoặc classification (có: {list(rows[0])})")
    if not c_date:
        print("CẢNH BÁO: không có cột date -> không chia được theo thời gian, "
              "kết quả sẽ bị thổi phồng. Xem mục 5 của đặc tả dataset.")

    out = {}
    for r in rows:
        sha = str(r[c_sha]).strip().lower()
        out[sha] = {
            "family": str(r[c_fam]).strip(),
            "date": str(r[c_date]).strip() if c_date else "",
            "type": str(r[c_type]).strip() if c_type else "",
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True, help="thư mục chứa <sha256>.json (reduced)")
    ap.add_argument("--labels", required=True, help="CSV: sha256, classification, date[, type]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", default=CUTOFF, help=f"mốc chia train/test (mặc định {CUTOFF})")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="cắt từ PHẦN MỚI NHẤT của train")
    ap.add_argument("--limit", type=int, help="chỉ xử lý N mẫu đầu (để thử nhanh)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels)
    cutoff_ts = parse_date(args.cutoff)

    files = sorted(Path(args.reports).glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"không thấy *.json trong {args.reports}")
    print(f"{len(files)} report, {len(labels)} nhãn")

    index = {"families": [], "splits": {}, "samples": [], "split_mode": "sample"}
    no_label, errors, no_date = 0, [], 0

    for n, fp in enumerate(files, 1):
        sha = fp.stem.lower()
        lab = labels.get(sha)
        if lab is None:
            no_label += 1
            continue
        try:
            sample = extract_avast_ctu(fp)
            np.savez_compressed(out / "samples" / f"{sha}.npz", **sample)
        except Exception:
            errors.append(sha)
            traceback.print_exc()
            continue

        ts = parse_date(lab["date"]) if lab["date"] else 0.0
        if ts == 0.0:
            no_date += 1
        fam = lab["family"]
        if fam not in index["families"]:
            index["families"].append(fam)
        index["samples"].append({
            "id": sha, "family": fam, "labeled": True, "timestamp": ts,
            "has_dynamic": True, "type": lab["type"],
            "split": "test" if ts >= cutoff_ts else "train",
        })
        if n % 2000 == 0:
            print(f"  ... {n}/{len(files)}", flush=True)

    # val = newest slice of train, per family, so val sits ahead of train in time
    by_fam: dict[str, list[int]] = {}
    for i, rec in enumerate(index["samples"]):
        if rec["split"] == "train":
            by_fam.setdefault(rec["family"], []).append(i)
    for idxs in by_fam.values():
        idxs.sort(key=lambda i: index["samples"][i]["timestamp"])
        for i in idxs[int(len(idxs) * (1 - args.val_frac)):]:
            index["samples"][i]["split"] = "val"

    meta = {
        "api_vocab": API_SET_VOCAB,
        "sem_vocab": API_SET_VOCAB * N_TAGS + N_TAGS,
        "api_set_vocab": API_SET_VOCAB,
        "n_context_tags": N_TAGS,
        "import_hash": IMPORT_HASH,
        "string_hash": STRING_HASH,
        "graph_feat_hash": GRAPH_FEAT_HASH,
        "image_size": 64,
        "bytes_len": 8192,
        "meta_dim": schema.META_DIM,
        "ember_dim": schema.EMBER_DIM,
        "synthetic": False,
        "source": "avast-ctu reduced reports",
        # honest capability flags for the paper
        "no_api_order": True,        # resolved_apis is a SET
        "no_call_arguments": True,
        "no_process_tree": True,
        "no_network_branch": True,
        "no_raw_binary": True,       # ember/bytes/image are zero-filled
        "degraded_semantics": True,
        "split_rule": f"time, cutoff {args.cutoff}",
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    from collections import Counter
    c = Counter((r["family"], r["split"]) for r in index["samples"])
    print(f"\n{len(index['samples'])} mẫu -> {out}")
    print(f"không có nhãn: {no_label} | lỗi: {len(errors)} | không có date: {no_date}")
    print(f"\nchia theo thời gian, mốc {args.cutoff}:")
    print(f"  {'family':14s} {'train':>7s} {'val':>7s} {'test':>7s}")
    for fam in sorted(index["families"]):
        print(f"  {fam:14s} {c[(fam,'train')]:7d} {c[(fam,'val')]:7d} {c[(fam,'test')]:7d}")
    tot = Counter(r["split"] for r in index["samples"])
    print(f"  {'TỔNG':14s} {tot['train']:7d} {tot['val']:7d} {tot['test']:7d}")
    print("\nDùng: --config configs/default.yaml --config configs/cpu_light.yaml "
          "--config configs/avast_ctu.yaml")


if __name__ == "__main__":
    main()
