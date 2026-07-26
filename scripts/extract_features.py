#!/usr/bin/env python
"""Build the processed dataset from real PE files + CAPE v2 reports.

Expected inputs:
  --pe-dir      directory of PE files named <id> (e.g. sha256)
  --report-dir  directory of CAPE reports named <id>.json (or <id>/report.json)
  --labels      CSV with header: id,family,timestamp[,labeled]
                timestamp = unix epoch (first-seen); labeled defaults to 1
  --out         output dataset root (samples/*.npz + index.json + meta.json)
  --val-frac / --test-frac   family-disjoint split fractions

Samples missing either the PE file or the report are skipped and listed.
"""

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.data import schema
from sdfsproto.extract import cape_report, pe_static


def find_report(report_dir: Path, sid: str) -> Path | None:
    for cand in (report_dir / f"{sid}.json", report_dir / sid / "report.json",
                 report_dir / sid / "reports" / "report.json"):
        if cand.exists():
            return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pe-dir", required=True)
    ap.add_argument("--report-dir", help="omit together with --static-only")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--static-only", action="store_true",
                    help="skip sandbox reports; dynamic view is filled with zeros")
    ap.add_argument("--split-mode", choices=["family", "sample"], default="family",
                    help="family = few-shot protocol (default); sample = few-family datasets")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.static_only and not args.report_dir:
        ap.error("--report-dir is required unless --static-only is given")

    pe_dir, out = Path(args.pe_dir), Path(args.out)
    report_dir = Path(args.report_dir) if args.report_dir else None
    (out / "samples").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.labels, newline="", encoding="utf-8")))
    index = {"families": [], "splits": {}, "samples": [], "split_mode": args.split_mode}
    skipped, errors, no_report = [], [], []

    for n, row in enumerate(rows, 1):
        sid = row["id"].strip()
        fam = row["family"].strip()
        pe_path = pe_dir / sid
        if not pe_path.exists():
            skipped.append(sid)
            continue
        rp = None if args.static_only else find_report(report_dir, sid)
        if not args.static_only and rp is None:
            skipped.append(sid)
            continue
        try:
            sample = pe_static.extract_pe(pe_path)
            if rp is None:
                sample.update(cape_report.empty_dynamic())
                no_report.append(sid)
            else:
                sample.update(cape_report.extract_cape(rp))
            np.savez_compressed(out / "samples" / f"{sid}.npz", **sample)
        except Exception:
            errors.append(sid)
            traceback.print_exc()
            continue
        if fam not in index["families"]:
            index["families"].append(fam)
        index["samples"].append({
            "id": sid,
            "family": fam,
            "labeled": bool(int(row.get("labeled", "1") or "1")),
            "timestamp": float(row.get("timestamp", "0") or "0"),
            "has_dynamic": rp is not None,
        })
        if n % 500 == 0:
            print(f"  ... {n}/{len(rows)} processed", flush=True)

    rng = np.random.default_rng(args.seed)
    fams = list(index["families"])
    if args.split_mode == "family":
        order = rng.permutation(len(fams))
        n_te = max(1, int(len(fams) * args.test_frac))
        n_va = max(1, int(len(fams) * args.val_frac))
        index["splits"] = {
            "test": [fams[i] for i in order[:n_te]],
            "val": [fams[i] for i in order[n_te : n_te + n_va]],
            "train": [fams[i] for i in order[n_te + n_va :]],
        }
        if len(fams) < 10:
            print(f"\nWARNING: only {len(fams)} families — a family-disjoint split this small "
                  f"cannot support few-shot evaluation. Consider --split-mode sample and do NOT "
                  f"report these numbers as few-shot results.")
    else:
        # sample-disjoint: stratify within each family so all families appear in every split
        by_fam: dict[str, list[int]] = {}
        for i, rec in enumerate(index["samples"]):
            by_fam.setdefault(rec["family"], []).append(i)
        for fam_name, idxs in by_fam.items():
            perm = rng.permutation(idxs)
            n_te = int(len(perm) * args.test_frac)
            n_va = int(len(perm) * args.val_frac)
            for j, i in enumerate(perm):
                index["samples"][i]["split"] = (
                    "test" if j < n_te else "val" if j < n_te + n_va else "train"
                )
        index["splits"] = {s: fams for s in ("train", "val", "test")}

    meta = {
        "api_vocab": cape_report.API_VOCAB,
        "sem_vocab": cape_report.SEM_VOCAB,
        "import_hash": pe_static.IMPORT_HASH,
        "string_hash": pe_static.STRING_HASH,
        "graph_feat_hash": cape_report.GRAPH_FEAT_HASH,
        "image_size": pe_static.IMG,
        "bytes_len": pe_static.BYTES_LEN,
        "meta_dim": schema.META_DIM,
        "synthetic": False,
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    n_ok = len(index["samples"])
    print(f"\nextracted {n_ok} samples, {len(fams)} families -> {out}  (split_mode={args.split_mode})")
    print(f"skipped (missing pe/report): {len(skipped)}; extraction errors: {len(errors)}; "
          f"without dynamic: {len(no_report)}")
    if rows:
        print(f"success rate: {100.0 * n_ok / len(rows):.1f}%")
    if skipped[:10]:
        print("  e.g. skipped:", skipped[:10])

    # per-family counts: the first thing to sanity-check before training
    counts: dict[str, int] = {}
    for rec in index["samples"]:
        counts[rec["family"]] = counts.get(rec["family"], 0) + 1
    print("per-family sample counts:")
    for fam_name, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fam_name:24s} {c}")


if __name__ == "__main__":
    main()
