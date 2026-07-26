#!/usr/bin/env python
"""Build data/A from the owner's pre-extracted CAPE features + EMBER vectors.

Inputs
  --cape-root   directory holding raw/<split>/<family>/<sha256>.json
  --ember       parquet/csv with a sha256 column + the EMBER feature columns
                (or a single column holding the vector as a list)
  --pe-dir      OPTIONAL: directory of PE files named <sha256>. When given, the
                MalConv byte branch and the grayscale image branch get real data;
                without it only the EMBER branch carries static information.
  --out         output dataset root

The split recorded in each JSON's meta is preserved verbatim — the dataset owner
fixed train/test and it must not be re-shuffled. Because that split is
sample-disjoint (every family in both halves), split_mode is forced to "sample":
numbers from this dataset are NOT few-shot results.

A val split is carved out of TRAIN only (stratified per family), never from test.
"""

import argparse
import glob
import json
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sdfsproto.data import schema
from sdfsproto.extract import cape_features
from sdfsproto.extract.cape_features import SEM_VOCAB
from sdfsproto.extract.cape_report import API_VOCAB, GRAPH_FEAT_HASH


def load_ember_npy(vec_path: str, id_path: str | None) -> tuple[dict[str, np.ndarray], int]:
    """Load an EMBER matrix from .npy plus the sha256 list that indexes its rows.

    Accepts several shapes because .npy carries no column names:
      * plain float matrix [N, D] + separate id file (.npy / .txt / .json / .csv)
      * structured array with a sha256-like field
      * dict-style .npy saved with allow_pickle ({sha256: vector})
    """
    arr = np.load(vec_path, allow_pickle=True)

    if arr.dtype == object and arr.ndim == 0:            # dict saved via np.save
        obj = arr.item()
        if isinstance(obj, dict):
            vecs = {str(k).lower(): np.asarray(v, dtype=np.float32) for k, v in obj.items()}
            return vecs, len(next(iter(vecs.values())))
        raise SystemExit(f"{vec_path}: .npy chứa object không phải dict")

    if arr.dtype.names:                                   # structured array
        names = list(arr.dtype.names)
        sha_field = next((n for n in names if n.lower() in ("sha256", "sha_256", "hash", "id")), None)
        if sha_field is None:
            raise SystemExit(f"{vec_path}: structured array không có field sha256 (fields: {names})")
        feat_fields = [n for n in names if n != sha_field]
        mat = np.stack([arr[n].astype(np.float32) for n in feat_fields], axis=1)
        vecs = {str(s).lower(): mat[i] for i, s in enumerate(arr[sha_field])}
        return vecs, mat.shape[1]

    if arr.ndim != 2:
        raise SystemExit(f"{vec_path}: cần ma trận 2 chiều [N, D], nhận được shape {arr.shape}")
    if id_path is None:
        raise SystemExit(
            f"{vec_path} là ma trận {arr.shape} không có sha256.\n"
            f"Cần thêm --ember-ids <file> chứa {arr.shape[0]} sha256 theo ĐÚNG thứ tự dòng "
            f"(.npy / .txt mỗi dòng một hash / .json list / .csv có cột sha256)."
        )

    p = Path(id_path)
    if p.suffix == ".npy":
        ids = [str(x) for x in np.load(p, allow_pickle=True).ravel()]
    elif p.suffix == ".json":
        raw = json.load(open(p, encoding="utf-8"))
        ids = [str(x) for x in (raw if isinstance(raw, list) else list(raw))]
    elif p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p)
        col = next((c for c in df.columns if c.lower() in ("sha256", "sha_256", "hash", "id")), df.columns[0])
        ids = [str(x) for x in df[col]]
    else:
        ids = [ln.strip() for ln in open(p, encoding="utf-8") if ln.strip()]

    if len(ids) != arr.shape[0]:
        raise SystemExit(f"lệch số dòng: vector {arr.shape[0]} vs id {len(ids)}")
    return {s.lower(): arr[i].astype(np.float32) for i, s in enumerate(ids)}, arr.shape[1]


def load_ember(path: str, id_path: str | None = None) -> tuple[dict[str, np.ndarray], int]:
    """Return {sha256: vector} and the vector dimension. Supports .npy / .parquet / .csv."""
    if path.endswith(".npy"):
        vecs, dim = load_ember_npy(path, id_path)
        print(f"EMBER: {len(vecs)} vector, {dim} chiều  (.npy)")
        return vecs, dim

    import pandas as pd

    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    sha_col = next((c for c in df.columns if c.lower() in ("sha256", "sha_256", "hash")), None)
    if sha_col is None:
        raise SystemExit(f"{path}: no sha256 column found (columns: {list(df.columns)[:10]}...)")

    feat_cols = [c for c in df.columns if c != sha_col]
    # single column holding a list/array per row?
    if len(feat_cols) == 1 and isinstance(df[feat_cols[0]].iloc[0], (list, np.ndarray)):
        vecs = {str(s).lower(): np.asarray(v, dtype=np.float32)
                for s, v in zip(df[sha_col], df[feat_cols[0]])}
    else:
        numeric = df[feat_cols].select_dtypes(include="number")
        if numeric.shape[1] != len(feat_cols):
            dropped = set(feat_cols) - set(numeric.columns)
            print(f"  bỏ {len(dropped)} cột không phải số: {sorted(dropped)[:8]}")
        arr = numeric.to_numpy(dtype=np.float32)
        vecs = {str(s).lower(): arr[i] for i, s in enumerate(df[sha_col])}
    dim = len(next(iter(vecs.values())))
    print(f"EMBER: {len(vecs)} vector, {dim} chiều")
    return vecs, dim


def load_pe_static(pe_path: Path) -> dict:
    from sdfsproto.extract.pe_static import extract_pe
    return extract_pe(pe_path)


def find_raw_report(root: Path, task_id, sha: str) -> Path | None:
    """Locate a raw CAPE report. Task-id layout is CAPE's own storage; sha layout
    covers reports copied out into a flat directory."""
    for cand in (
        root / str(task_id) / "reports" / "report.json",
        root / str(task_id) / "report.json",
        root / f"{sha}.json",
        root / sha / "reports" / "report.json",
    ):
        if cand.exists():
            return cand
    return None


def empty_static(ember_dim: int, bytes_len: int, img: int) -> dict:
    return {
        "bytes": np.zeros(bytes_len, dtype=np.uint8),
        "image": np.zeros((img, img), dtype=np.uint8),
        "import_ids": np.zeros(0, dtype=np.int64),
        "string_ids": np.zeros(0, dtype=np.int64),
        "metadata": np.zeros(schema.META_DIM, dtype=np.float32),
        "static_rel": np.zeros(schema.STATIC_REL_DIM, dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cape-root", required=True, help="dir containing raw/<split>/<family>/*.json")
    ap.add_argument("--raw-report-dir",
                    help="OPTIONAL nhưng nên dùng: thư mục raw CAPE report.json "
                         "(vd /opt/CAPEv2/storage/analyses). Có nó thì Backend 2 chạy ĐÚNG "
                         "thiết kế với arguments per-call, thay vì chế độ suy giảm.")
    ap.add_argument("--ember", help=".npy / .parquet / .csv chứa vector EMBER")
    ap.add_argument("--ember-ids", help="với .npy dạng ma trận: file sha256 theo đúng thứ tự dòng")
    ap.add_argument("--pe-dir", help="optional: PE files named <sha256>, enables bytes+image branches")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bytes-len", type=int, default=8192)
    ap.add_argument("--image-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15, help="carved out of TRAIN only")
    ap.add_argument("--exclude-family", action="append", default=[],
                    help="repeatable, e.g. --exclude-family Benign")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    pe_dir = Path(args.pe_dir) if args.pe_dir else None

    ember, ember_dim = (load_ember(args.ember, args.ember_ids) if args.ember
                        else ({}, schema.EMBER_DIM))

    files = sorted(glob.glob(str(Path(args.cape_root) / "raw" / "*" / "*" / "*.json")))
    if not files:
        raise SystemExit(f"không tìm thấy JSON nào ở {args.cape_root}/raw/<split>/<family>/*.json")
    print(f"tìm thấy {len(files)} file JSON")

    excluded = {f.lower() for f in args.exclude_family}
    index = {"families": [], "splits": {}, "samples": [], "split_mode": "sample"}
    n_no_ember, n_no_pe, n_not_ran, errors, skipped_fam = 0, 0, 0, [], 0

    raw_dir = Path(args.raw_report_dir) if args.raw_report_dir else None
    n_from_raw = 0

    for n, fp in enumerate(files, 1):
        try:
            dyn = cape_features.extract_cape_features(fp)
            meta = dyn.pop("_meta")
            sha = str(meta.get("sha256", Path(fp).stem)).lower()
            fam = str(meta.get("family", "unknown"))

            # Prefer the raw report when available: it carries per-call arguments,
            # which the pre-extracted format drops. Falls back silently per sample.
            if raw_dir is not None:
                rp = find_raw_report(raw_dir, meta.get("cape_task_id"), sha)
                if rp is not None:
                    try:
                        from sdfsproto.extract.cape_report import extract_cape
                        dyn = extract_cape(rp)
                        n_from_raw += 1
                    except Exception:
                        pass  # malformed raw report -> keep the extracted-format version
            if fam.lower() in excluded:
                skipped_fam += 1
                continue

            sample = {}
            if pe_dir and (pe_dir / sha).exists():
                sample.update(load_pe_static(pe_dir / sha))
            else:
                n_no_pe += 1
                sample.update(empty_static(ember_dim, args.bytes_len, args.image_size))
            sample["bytes"] = sample["bytes"][: args.bytes_len]

            if sha in ember:
                sample["ember"] = ember[sha].astype(np.float32)
            else:
                n_no_ember += 1
                sample["ember"] = np.zeros(ember_dim, dtype=np.float32)

            sample.update(dyn)
            if not meta.get("ran_ok"):
                n_not_ran += 1

            np.savez_compressed(out / "samples" / f"{sha}.npz", **sample)
            if fam not in index["families"]:
                index["families"].append(fam)
            index["samples"].append({
                "id": sha,
                "family": fam,
                "labeled": True,
                "timestamp": float(meta.get("timestamp", 0) or 0),
                "has_dynamic": bool(meta.get("ran_ok")),
                "split": str(meta.get("split", "train")),
                "ran_ok": bool(meta.get("ran_ok")),
            })
        except Exception:
            errors.append(fp)
            traceback.print_exc()
        if n % 500 == 0:
            print(f"  ... {n}/{len(files)}", flush=True)

    # carve val out of TRAIN only, stratified per family — test stays untouched
    rng = np.random.default_rng(args.seed)
    by_fam: dict[str, list[int]] = {}
    for i, rec in enumerate(index["samples"]):
        if rec["split"] == "train":
            by_fam.setdefault(rec["family"], []).append(i)
    for idxs in by_fam.values():
        perm = rng.permutation(idxs)
        for j in perm[: int(len(perm) * args.val_frac)]:
            index["samples"][j]["split"] = "val"

    fams = index["families"]
    index["splits"] = {s: fams for s in ("train", "val", "test")}
    meta_out = {
        "api_vocab": API_VOCAB,
        "sem_vocab": SEM_VOCAB,
        "graph_feat_hash": GRAPH_FEAT_HASH,
        "import_hash": 4096,
        "string_hash": 4096,
        "image_size": args.image_size,
        "bytes_len": args.bytes_len,
        "meta_dim": schema.META_DIM,
        "ember_dim": ember_dim,
        "synthetic": False,
        # degraded only when NO sample got its per-call arguments back from a raw report
        "degraded_semantics": n_from_raw == 0,
        "n_from_raw_report": n_from_raw,
        "source": ("cape_features + raw reports" if n_from_raw else "cape_features (pre-extracted)"),
    }
    with open(out / "index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)
    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)

    from collections import Counter
    n_ok = len(index["samples"])
    print(f"\n{n_ok} mẫu -> {out}")
    print(f"lỗi: {len(errors)} | thiếu EMBER: {n_no_ember} | thiếu PE: {n_no_pe} | "
          f"ran_ok=False: {n_not_ran} | bỏ theo family: {skipped_fam}")
    if raw_dir is not None:
        print(f"đọc từ raw report (CÓ arguments): {n_from_raw}/{n_ok}"
              + ("  -> Backend 2 chạy ĐÚNG thiết kế" if n_from_raw == n_ok
                 else f"  -> {n_ok - n_from_raw} mẫu vẫn suy giảm"))
    else:
        print("KHÔNG dùng raw report -> Backend 2 chạy chế độ suy giảm "
              "(không có arguments per-call). Thêm --raw-report-dir nếu raw còn.")
    print("\nsố mẫu theo family x split:")
    c = Counter((r["family"], r["split"]) for r in index["samples"])
    for fam in sorted(fams):
        row = " ".join(f"{s}={c[(fam, s)]:5d}" for s in ("train", "val", "test"))
        print(f"  {fam:14s} {row}")
    if n_no_pe == n_ok:
        print("\nLƯU Ý: không có file PE -> nhánh bytes/image toàn 0. "
              "Dùng --config configs/ember.yaml --set model.static.branches='[ember]'")
    if n_not_ran:
        print(f"\nLƯU Ý: {n_not_ran} mẫu ran_ok=False (trace rỗng). Chúng được GIỮ LẠI vì "
              "dyn_rel[7]=0 chính là tín hiệu cho reliability fusion. Loại bằng "
              "--exclude-family nếu muốn lặp lại quy trình nhị phân của bạn.")


if __name__ == "__main__":
    main()
