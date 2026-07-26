"""Adapter for the Avast-CTU Public CAPE Dataset (reduced reports).

Reduced report layout:
    { "behavior": {"summary": {...}}, "static": {"pe": {...}} }

Everything in `behavior.summary` is a SET of strings — there is no call sequence,
no per-call arguments, no process tree and no network branch. The dynamic view is
therefore encoded with the `api_set` backend, and semantic intent is recovered
from the registry / file / command lists as multi-hot CONTEXT TAGS.

Fields the model would like but this dataset cannot provide (filled with zeros,
the corresponding branches must be disabled in the config):
    ember, bytes, image   — no raw PE binaries are distributed

Two dataset-specific hazards handled here:
  * numbers are stored as STRINGS ("0x00001000", "5.83") — parsed defensively;
  * per-run random filenames (`...\\Temp\\FFFF450D574E5E5706FB.exe`) are collapsed
    to their PATTERN before hashing, otherwise the model memorises noise instead
    of a family trait.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..data import schema
from .cape_report import (
    ANTI_APIS,
    CRYPTO_APIS,
    EXEC_EXT_RE,
    INJECT_APIS,
    IP_RE,
    N_TAGS,
    RUN_KEY_RE,
    SERVICES_RE,
    SHELL_RE,
    SYSTEM_DIR_RE,
    TAG_ANTI_ANALYSIS,
    TAG_CRYPTO,
    TAG_HTTP,
    TAG_MUTEX,
    TAG_PERSIST_TASK,
    TAG_PROC_INJECT,
    TAG_RUN_KEY,
    TAG_SERVICE_CREATE,
    TAG_SERVICE_REG,
    TAG_SHELL_EXEC,
    TAG_SYSTEM_DIR_WRITE,
    TAG_TEMP_FILE,
    TAG_USER_WRITABLE_EXEC,
    TASK_RE,
    USER_WRITABLE_RE,
)

API_SET_VOCAB = 8192
IMPORT_HASH = 4096
STRING_HASH = 4096
GRAPH_FEAT_HASH = 4096

# Behaviour keys that may or may not be present — never assume they exist.
FILE_KEYS = ("files", "read_files", "write_files", "delete_files")
REG_KEYS = ("keys", "read_keys", "write_keys", "delete_keys")

_HEXNAME_RE = re.compile(r"\b[0-9a-f]{8,}\b", re.I)
_DIGITS_RE = re.compile(r"\d+")
_USERDIR_RE = re.compile(r"c:\\users\\[^\\]+", re.I)
_GUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)


def _h(s: str, mod: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8], "little") % mod


def normalize_path(s: str) -> str:
    """Collapse per-run randomness so the same family artifact hashes identically.

    `C:\\Users\\comp\\AppData\\Local\\Temp\\FFFF450D574E5E5706FB.exe`
      -> `%userdir%\\appdata\\local\\temp\\%hex%.exe`
    The random name is noise; the PATTERN (a long-hex name dropped in Temp) is the
    family trait worth learning.
    """
    v = str(s).lower()
    v = _USERDIR_RE.sub("%userdir%", v)
    v = _GUID_RE.sub("%guid%", v)
    v = _HEXNAME_RE.sub("%hex%", v)
    v = _DIGITS_RE.sub("%d", v)
    return v[:160]


def _as_list(d: dict, key: str) -> list[str]:
    v = d.get(key)
    return [str(x) for x in v if x] if isinstance(v, list) else []


def _hexint(v, default: int = 0) -> int:
    try:
        s = str(v).strip()
        return int(s, 16) if s.lower().startswith("0x") else int(s)
    except (TypeError, ValueError):
        return default


def _f(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------- dynamic

def detect_context_tags(summary: dict) -> set[int]:
    """Recover argument-level intent from the aggregate string sets."""
    tags: set[int] = set()
    regs = [x for k in REG_KEYS for x in _as_list(summary, k)]
    files = [x for k in FILE_KEYS for x in _as_list(summary, k)]
    cmds = _as_list(summary, "executed_commands") + _as_list(summary, "command_line")
    apis = [a.lower() for a in _as_list(summary, "resolved_apis")]

    for k in regs:
        if RUN_KEY_RE.search(k):
            tags.add(TAG_RUN_KEY)
        if SERVICES_RE.search(k):
            tags.add(TAG_SERVICE_REG)
    for f in files:
        if USER_WRITABLE_RE.search(f) and EXEC_EXT_RE.search(f):
            tags.add(TAG_USER_WRITABLE_EXEC)
        if "temp" in f.lower():
            tags.add(TAG_TEMP_FILE)
        if SYSTEM_DIR_RE.search(f) and EXEC_EXT_RE.search(f):
            tags.add(TAG_SYSTEM_DIR_WRITE)
    for c in cmds:
        if SHELL_RE.search(c):
            tags.add(TAG_SHELL_EXEC)
        if TASK_RE.search(c):
            tags.add(TAG_PERSIST_TASK)
    if _as_list(summary, "mutexes"):
        tags.add(TAG_MUTEX)
    if _as_list(summary, "started_services"):
        tags.add(TAG_SERVICE_CREATE)
    # API-name-only tags stay reliable without arguments
    for a in apis:
        fn = a.rsplit(".", 1)[-1]
        if fn in INJECT_APIS:
            tags.add(TAG_PROC_INJECT)
        if fn in CRYPTO_APIS:
            tags.add(TAG_CRYPTO)
        if fn in ANTI_APIS:
            tags.add(TAG_ANTI_ANALYSIS)
        if any(w in fn for w in ("internetopen", "httpsend", "winhttp", "urldownload")):
            tags.add(TAG_HTTP)
    return tags


def _dynamic(summary: dict) -> dict:
    apis = _as_list(summary, "resolved_apis")
    api_set_ids = np.array(sorted({_h(a.lower(), API_SET_VOCAB) for a in apis}), dtype=np.int64)

    tags = detect_context_tags(summary)
    ctx = np.zeros(N_TAGS, dtype=np.float32)
    for t in tags:
        ctx[t] = 1.0

    files = [x for k in FILE_KEYS for x in _as_list(summary, k)]
    regs = [x for k in REG_KEYS for x in _as_list(summary, k)]
    muts = _as_list(summary, "mutexes")
    cmds = _as_list(summary, "executed_commands")
    dlls = {a.split(".")[0].lower() for a in apis if "." in a}

    # ---- behaviour graph: root process + API nodes + object nodes.
    # No process tree and no ordering exist here, so there are no spawns/next edges.
    node_type = [schema.NODE_TYPE_ID["process"]]
    node_feat = [0]
    edges: list[tuple[int, int, int]] = []
    seen: dict[str, int] = {}

    def add(key: str, ntype: str, feat: int, etype: str) -> None:
        if key in seen:
            return
        seen[key] = len(node_type)
        node_type.append(schema.NODE_TYPE_ID[ntype])
        node_feat.append(feat)
        edges.append((0, seen[key], schema.EDGE_TYPE_ID[etype]))

    for a in apis[:200]:
        add(f"api:{a.lower()}", "api", _h(a.lower(), GRAPH_FEAT_HASH), "calls")
    for f in files[:60]:
        n = normalize_path(f)
        add(f"file:{n}", "file", _h(n, GRAPH_FEAT_HASH), "writes")
    for k in regs[:60]:
        n = normalize_path(k)
        add(f"reg:{n}", "registry", _h(n, GRAPH_FEAT_HASH), "writes")
    for m in muts[:20]:
        n = normalize_path(m)
        add(f"mtx:{n}", "mutex", _h(n, GRAPH_FEAT_HASH), "targets")
    for c in cmds[:20]:
        n = normalize_path(c)
        add(f"cmd:{n}", "process", _h(n, GRAPH_FEAT_HASH), "spawns")

    edge_arr = np.array(edges, dtype=np.int64).reshape(-1, 3)

    n_api = len(apis)
    dyn_rel = np.array([
        np.log1p(n_api) / 10.0,
        len(dlls) / max(1, n_api),            # DLL spread; no call counts exist
        0.0,                                  # run duration: not in the reduced report
        np.log1p(len(files)) / 8.0,
        np.log1p(len(regs)) / 8.0,
        0.0,                                  # network: the whole branch is absent
        1.0 if TAG_ANTI_ANALYSIS in tags else 0.0,
        1.0 if (apis or files or regs or muts or cmds) else 0.0,   # any behaviour at all
    ], dtype=np.float32)

    # api_ids / sem_ids kept so sequence backends still run, but the order is
    # ARBITRARY (sorted set). meta.json records no_api_order=true.
    api_ids = api_set_ids.copy()
    sem_ids = np.array([API_SET_VOCAB * N_TAGS + t for t in sorted(tags)]
                       + [int(a) * N_TAGS for a in api_set_ids], dtype=np.int64)

    return {
        "api_set_ids": api_set_ids,
        "context_tags": ctx,
        "api_ids": api_ids,
        "sem_ids": sem_ids,
        "node_type": np.array(node_type, dtype=np.int64),
        "node_feat": np.array(node_feat, dtype=np.int64),
        "edge_src": edge_arr[:, 0],
        "edge_dst": edge_arr[:, 1],
        "edge_type": edge_arr[:, 2],
        "dyn_rel": dyn_rel,
    }


# ---------------------------------------------------------------------- static

def _static(pe: dict) -> dict:
    sections = pe.get("sections") or []
    ents = [_f(s.get("entropy")) for s in sections if s.get("entropy") is not None]
    imports = pe.get("imports") or []
    resources = pe.get("resources") or []

    import_ids: list[int] = []
    n_funcs = 0
    for entry in imports:
        dll = str(entry.get("dll", "")).lower().rsplit(".", 1)[0]
        for imp in (entry.get("imports") or []):
            name = str(imp.get("name") or imp.get("address") or "")
            import_ids.append(_h(f"{dll}:{name.lower()}", IMPORT_HASH))
            n_funcs += 1
    # imphash is a strong family fingerprint — keep it as its own token
    if pe.get("imphash"):
        import_ids.append(_h(f"imphash:{pe['imphash']}", IMPORT_HASH))

    # no printable-string dump exists; use the string-like static fields instead
    string_ids = [_h(f"sec:{str(s.get('name','')).lower()}", STRING_HASH) for s in sections]
    string_ids += [_h(f"res:{str(r.get('name','')).lower()}", STRING_HASH) for r in resources]
    string_ids += [_h(f"lang:{str(r.get('language','')).lower()}", STRING_HASH) for r in resources]
    if pe.get("pdbpath"):
        string_ids.append(_h(f"pdb:{normalize_path(pe['pdbpath'])}", STRING_HASH))

    rep_ck, act_ck = _hexint(pe.get("reported_checksum")), _hexint(pe.get("actual_checksum"))
    overlay = pe.get("overlay") or {}
    ov_size = _hexint(overlay.get("size")) if isinstance(overlay, dict) else 0
    peid = pe.get("peid_signatures")
    packed_flag = 1.0 if peid else 0.0
    raw_total = sum(_hexint(s.get("size_of_data")) for s in sections)
    virt_total = sum(_hexint(s.get("virtual_size")) for s in sections)

    year = 0.0
    ts = str(pe.get("timestamp") or "")
    if len(ts) >= 4 and ts[:4].isdigit():
        year = (int(ts[:4]) - 1990) / 50.0     # forged timestamps are themselves a signal

    exec_secs = sum(1 for s in sections if "EXECUTE" in str(s.get("characteristics", "")))
    write_secs = sum(1 for s in sections if "WRITE" in str(s.get("characteristics", "")))
    n_sec = max(1, len(sections))

    meta = np.zeros(schema.META_DIM, dtype=np.float32)
    meta[:26] = [
        len(sections) / 20.0,
        (np.mean(ents) if ents else 0) / 8.0,
        (np.max(ents) if ents else 0) / 8.0,
        (np.min(ents) if ents else 0) / 8.0,
        (np.std(ents) if ents else 0) / 4.0,
        float(np.mean([e > 7.2 for e in ents])) if ents else 0.0,
        _f(pe.get("imported_dll_count")) / 50.0,
        np.log1p(n_funcs) / 8.0,
        np.log1p(len(pe.get("exports") or [])) / 8.0,
        1.0 if (rep_ck and rep_ck != act_ck) else 0.0,
        1.0 if rep_ck == 0 else 0.0,
        np.log1p(ov_size) / 20.0,
        1.0 if ov_size else 0.0,
        np.log1p(len(resources)) / 6.0,
        (np.mean([_f(r.get("entropy")) for r in resources]) if resources else 0) / 8.0,
        1.0 if pe.get("pdbpath") else 0.0,
        packed_flag,
        1.0 if (pe.get("digital_signers") or []) else 0.0,
        1.0 if (pe.get("guest_signers") or {}).get("aux_valid") else 0.0,
        _f(pe.get("osversion")) / 10.0,
        year,
        exec_secs / n_sec,
        write_secs / n_sec,
        np.log1p(raw_total) / 20.0,
        min(5.0, virt_total / max(1, raw_total)) / 5.0,
        1.0 if pe.get("icon_hash") else 0.0,
    ]

    static_rel = np.array([
        (np.mean(ents) if ents else 0) / 8.0,
        (np.max(ents) if ents else 0) / 8.0,
        float(np.mean([e > 7.2 for e in ents])) if ents else 0.0,
        np.log1p(n_funcs) / 8.0,
        np.log1p(len(string_ids)) / 10.0,
        packed_flag,                                  # PEiD: real packer ground truth
        np.log1p(raw_total) / 20.0,
        1.0 if n_funcs < 10 else 0.0,                 # import table stripped/hidden
    ], dtype=np.float32)

    return {
        "import_ids": np.array(sorted(set(import_ids)), dtype=np.int64),
        "string_ids": np.array(sorted(set(string_ids)), dtype=np.int64),
        "metadata": meta,
        "static_rel": static_rel,
    }


def extract_avast_ctu(path: str | Path, bytes_len: int = 8192, image: int = 64,
                      ember_dim: int = schema.EMBER_DIM) -> dict:
    """Read one reduced report and return a full sample dict for the .npz schema."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        rep = json.load(f)

    summary = ((rep.get("behavior") or {}).get("summary")) or {}
    pe = ((rep.get("static") or {}).get("pe")) or {}

    sample = _static(pe)
    sample.update(_dynamic(summary))
    # branches this dataset cannot feed — zeros, disable them in the config
    sample["bytes"] = np.zeros(bytes_len, dtype=np.uint8)
    sample["image"] = np.zeros((image, image), dtype=np.uint8)
    sample["ember"] = np.zeros(ember_dim, dtype=np.float32)
    return sample
