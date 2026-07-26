"""Static feature extraction from PE files (no execution).

Produces the static half of the sample schema: bytes, image, import_ids,
string_ids, metadata (32 dims), static_rel (8 signals).

NOTE: written against the pefile API; validate on the real dataset when it
arrives (packed/corrupt PEs must fall through gracefully, never crash a batch).
"""

from __future__ import annotations

import hashlib
import re
import traceback
from pathlib import Path

import numpy as np

from ..data import schema

IMPORT_HASH = 4096
STRING_HASH = 4096
BYTES_LEN = 65536
IMG = 64
MAX_STRINGS = 500
STRING_RE = re.compile(rb"[\x20-\x7e]{5,}")


def _h(s: str, mod: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8], "little") % mod


def _entropy(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _to_image(raw: bytes, out: int = IMG) -> np.ndarray:
    """Nataraj-style grayscale image: fixed width by file size, nearest resize."""
    n = len(raw)
    width = 32
    for limit, w in [(10_240, 32), (30_720, 64), (61_440, 128), (102_400, 256),
                     (204_800, 384), (512_000, 512), (1_048_576, 768)]:
        if n <= limit:
            width = w
            break
    else:
        width = 1024
    arr = np.frombuffer(raw, dtype=np.uint8)
    h = max(1, len(arr) // width)
    arr = arr[: h * width].reshape(h, width)
    ri = np.linspace(0, h - 1, out).astype(int)
    ci = np.linspace(0, width - 1, out).astype(int)
    return arr[np.ix_(ri, ci)].copy()


def extract_pe(path: str | Path) -> dict:
    """Returns the static part of a sample dict. Never raises on malformed PEs;
    degraded fields are simply empty/zero and reflected in static_rel."""
    import pefile  # optional dependency: pip install sdfsproto[extract]

    raw_all = Path(path).read_bytes()
    raw = np.frombuffer(raw_all[:BYTES_LEN], dtype=np.uint8).copy()
    image = _to_image(raw_all)
    filesize = len(raw_all)

    imports: list[int] = []
    meta = np.zeros(schema.META_DIM, dtype=np.float32)
    sec_entropies: list[float] = []
    frac_exec = frac_write = 0.0
    parse_ok = True
    try:
        pe = pefile.PE(data=raw_all, fast_load=False)
    except Exception:
        pe = None
        parse_ok = False

    n_dlls = n_funcs = n_exports = 0
    if pe is not None:
        try:
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", []) or []:
                dll = (entry.dll or b"").decode("latin-1").lower().rsplit(".", 1)[0]
                n_dlls += 1
                for imp in entry.imports:
                    name = imp.name.decode("latin-1") if imp.name else f"ord{imp.ordinal}"
                    imports.append(_h(f"{dll}:{name.lower()}", IMPORT_HASH))
                    n_funcs += 1
            n_exports = len(getattr(getattr(pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) or [])
            secs = pe.sections or []
            for s in secs:
                data = np.frombuffer(s.get_data() or b"", dtype=np.uint8)
                sec_entropies.append(_entropy(data))
            n_sec = max(1, len(secs))
            frac_exec = sum(bool(s.Characteristics & 0x20000000) for s in secs) / n_sec
            frac_write = sum(bool(s.Characteristics & 0x80000000) for s in secs) / n_sec

            oh, fh = pe.OPTIONAL_HEADER, pe.FILE_HEADER
            dd = oh.DATA_DIRECTORY
            first = secs[0] if secs else None
            meta[:] = [
                np.log1p(filesize) / 25.0,
                (fh.TimeDateStamp % 2**31) / 2**31,
                len(secs) / 20.0,
                np.log1p(oh.SizeOfCode) / 25.0,
                np.log1p(oh.SizeOfImage) / 25.0,
                oh.AddressOfEntryPoint / max(1, oh.SizeOfImage),
                oh.Subsystem / 20.0,
                bin(oh.DllCharacteristics).count("1") / 16.0,
                bin(fh.Characteristics).count("1") / 16.0,
                float(fh.Machine == 0x8664),
                n_dlls / 50.0,
                np.log1p(n_funcs) / 8.0,
                np.log1p(n_exports) / 8.0,
                float(len(dd) > 2 and dd[2].Size > 0),   # resources
                float(len(dd) > 9 and dd[9].Size > 0),   # TLS
                float(len(dd) > 6 and dd[6].Size > 0),   # debug
                float(len(dd) > 5 and dd[5].Size > 0),   # relocs
                float(len(dd) > 4 and dd[4].Size > 0),   # authenticode
                (filesize - pe.get_overlay_data_start_offset() if pe.get_overlay_data_start_offset() else 0) / max(1, filesize),
                (np.mean(sec_entropies) if sec_entropies else 0) / 8.0,
                (np.max(sec_entropies) if sec_entropies else 0) / 8.0,
                (np.min(sec_entropies) if sec_entropies else 0) / 8.0,
                frac_exec,
                frac_write,
                (np.mean([e > 7.2 for e in sec_entropies]) if sec_entropies else 0.0),
                (first.SizeOfRawData / max(1, filesize)) if first else 0.0,
                0.0,  # rsrc ratio (filled below if present)
                min(5.0, (first.Misc_VirtualSize / max(1, first.SizeOfRawData)) if first else 0.0) / 5.0,
                0.0,  # n_strings (filled after string extraction)
                0.0,  # mean string len (filled after)
                float(oh.CheckSum == 0),
                float(bool(first) and first.VirtualAddress <= oh.AddressOfEntryPoint
                      < first.VirtualAddress + max(first.Misc_VirtualSize, first.SizeOfRawData)),
            ]
            for s in secs:
                if b".rsrc" in s.Name:
                    meta[26] = s.SizeOfRawData / max(1, filesize)
        except Exception:
            import warnings
            warnings.warn(f"PE metadata extraction failed for {path}: {traceback.format_exc(limit=2)}")
            parse_ok = False

    found = STRING_RE.findall(raw_all[:2_097_152])[:MAX_STRINGS]
    string_ids = [_h(s.decode("latin-1").lower(), STRING_HASH) for s in found]
    meta[28] = np.log1p(len(found)) / 10.0
    meta[29] = (np.mean([len(s) for s in found]) if found else 0.0) / 20.0

    ent_all = _entropy(np.frombuffer(raw_all, dtype=np.uint8))
    max_ent = max(sec_entropies) if sec_entropies else ent_all
    frac_high = float(np.mean([e > 7.2 for e in sec_entropies])) if sec_entropies else float(ent_all > 7.2)
    packed = float(max_ent > 7.2 and n_funcs < 15)
    import_anomaly = float(n_funcs < 10 and filesize > 20_000) if parse_ok else 1.0
    static_rel = np.array(
        [
            ent_all / 8.0,
            max_ent / 8.0,
            frac_high,
            np.log1p(n_funcs) / 8.0,
            np.log1p(len(found)) / 10.0,
            packed,
            np.log1p(filesize) / 25.0,
            import_anomaly,
        ],
        dtype=np.float32,
    )

    return {
        "bytes": raw,
        "image": image.astype(np.uint8),
        "import_ids": np.array(sorted(set(imports)), dtype=np.int64),
        "string_ids": np.array(string_ids, dtype=np.int64),
        "metadata": meta,
        "static_rel": static_rel,
    }
