"""Adapter for pre-extracted CAPE feature JSON (`raw/<split>/<family>/<sha256>.json`).

This reads the ALREADY-EXTRACTED format documented by the dataset owner, not a raw
CAPE `report.json`. Use `cape_report.py` when raw reports are available.

## What is lost relative to raw reports

The extracted format stores `api.sequence` as API NAMES ONLY — no per-call arguments.
Backend 2 (semantic_seq) was designed around arguments: `RegSetValueExW` is benign or
malicious depending on whether it writes a Run key. Without them this adapter runs in
DEGRADED mode:

  * tags that depend only on the API name (injection, crypto, anti-analysis) are still
    assigned per call — those are recoverable;
  * tags that depend on arguments (run key, dropped executable, shell exec, HTTP) are
    detected from the AGGREGATE lists (`registry.write_keys`, `filesystem.write_files`,
    `exec.executed_commands`, `network.domains`) and emitted as CONTEXT TOKENS prefixed
    to the sequence — the sample is known to exhibit the behaviour, but not which call
    performed it.

`degraded_semantics=True` is recorded in meta.json so the paper can state this honestly.

## Sandbox artifacts

Paths listed by the dataset owner as sandbox instrumentation (wevtutil, SMaster*,
the random analyzer directory, the Python agent, INetSim's 192.168.122.1, ...) are
filtered out of every string-derived feature. Without this the model learns the
sandbox, not the malware.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..data import schema
from .cape_report import (
    API_VOCAB,
    GRAPH_FEAT_HASH,
    N_TAGS,
    TAG_ANTI_ANALYSIS,
    TAG_CRYPTO,
    TAG_HTTP,
    TAG_IP_DIRECT,
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
    EXEC_EXT_RE,
    IP_RE,
    RUN_KEY_RE,
    SERVICES_RE,
    SHELL_RE,
    SYSTEM_DIR_RE,
    TASK_RE,
    USER_WRITABLE_RE,
    _normalize_object,
    _tag_call,
)

# Context tokens live above the per-call token space so the two never collide.
SEM_VOCAB = API_VOCAB * N_TAGS + N_TAGS
MAX_SEQ_STORE = 20_000

# Sandbox instrumentation — must never become a feature (dataset owner section 5.3).
ARTIFACT_PATTERNS = [
    re.compile(r"\\wevtutil\.exe", re.I),
    re.compile(r"\\auditpol\.exe", re.I),
    re.compile(r"smaster(32|64)\.exe", re.I),
    re.compile(r"c:\\python\d+\\", re.I),
    re.compile(r"otelemetry\.pyw", re.I),
    re.compile(r"onedrivesetup\.exe", re.I),
    re.compile(r"\\net1?\.exe", re.I),
    re.compile(r"\\sc\.exe", re.I),
    re.compile(r"^c:\\[a-z0-9]{8,10}\\", re.I),   # random analyzer directory
    re.compile(r"192\.168\.122\.1"),               # INetSim sink
    re.compile(r"\\lsass\.exe", re.I),             # injected by the tlsdump module
]


def is_artifact(s: str) -> bool:
    return any(p.search(s) for p in ARTIFACT_PATTERNS)


def clean(items) -> list[str]:
    """Drop sandbox artifacts from a list of paths/names."""
    return [str(x) for x in (items or []) if x and not is_artifact(str(x))]


def _h(s: str, mod: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8], "little") % mod


# Aggregate-list -> context tag rules (the argument-dependent tags we cannot place per call)
def detect_context_tags(d: dict) -> set[int]:
    tags: set[int] = set()
    reg = clean(d.get("registry", {}).get("write_keys")) + clean(d.get("registry", {}).get("keys"))
    fs_w = clean(d.get("filesystem", {}).get("write_files"))
    cmds = clean(d.get("exec", {}).get("executed_commands"))
    svcs = clean(d.get("exec", {}).get("created_services")) + clean(d.get("exec", {}).get("started_services"))
    doms = clean(d.get("network", {}).get("domains"))
    hosts = clean(d.get("network", {}).get("hosts"))
    mutexes = clean(d.get("sync", {}).get("mutexes"))

    for k in reg:
        if RUN_KEY_RE.search(k):
            tags.add(TAG_RUN_KEY)
        if SERVICES_RE.search(k):
            tags.add(TAG_SERVICE_REG)
    for f in fs_w:
        if USER_WRITABLE_RE.search(f) and EXEC_EXT_RE.search(f):
            tags.add(TAG_USER_WRITABLE_EXEC)
        if "temp" in f.lower():
            tags.add(TAG_TEMP_FILE)
        if SYSTEM_DIR_RE.search(f):
            tags.add(TAG_SYSTEM_DIR_WRITE)
    for c in cmds:
        if SHELL_RE.search(c):
            tags.add(TAG_SHELL_EXEC)
        if TASK_RE.search(c):
            tags.add(TAG_PERSIST_TASK)
    if svcs:
        tags.add(TAG_SERVICE_CREATE)
    if mutexes:
        tags.add(TAG_MUTEX)
    for h in d.get("network", {}).get("http") or []:
        if isinstance(h, dict) and h.get("host") and not is_artifact(str(h.get("host"))):
            tags.add(TAG_HTTP)
    if doms:
        tags.add(TAG_HTTP)
    for h in hosts:
        if IP_RE.match(h.strip()):
            tags.add(TAG_IP_DIRECT)
    return tags


def _build_graph(d: dict, context_tags: set[int]):
    """Behavior graph from per-process info + aggregate object lists.

    Weaker than the raw-report graph: without per-call arguments we cannot say which
    API touched which object, so objects attach to their PROCESS rather than to a
    specific API node.
    """
    node_type: list[int] = []
    node_feat: list[int] = []
    edges: list[tuple[int, int, int]] = []
    node_of: dict[str, int] = {}

    def add(key: str, ntype: str, feat: int) -> int:
        if key not in node_of:
            node_of[key] = len(node_type)
            node_type.append(schema.NODE_TYPE_ID[ntype])
            node_feat.append(feat)
        return node_of[key]

    procs = d.get("api", {}).get("per_process") or []
    pid_node: dict[object, int] = {}
    for p in procs:
        name = str(p.get("name") or "proc").lower()
        if is_artifact(name):
            continue
        n = add(f"proc:{p.get('pid')}", "process", _h(_normalize_object("proc", name), GRAPH_FEAT_HASH))
        pid_node[p.get("pid")] = n
    # process tree edges
    for p in procs:
        child = pid_node.get(p.get("pid"))
        parent = pid_node.get(p.get("parent_id"))
        if child is not None and parent is not None and child != parent:
            edges.append((parent, child, schema.EDGE_TYPE_ID["spawns"]))
    # process -> api edges, plus api ordering within the stored prefix
    for p in procs:
        pn = pid_node.get(p.get("pid"))
        if pn is None:
            continue
        prev = None
        for api in (p.get("first_apis") or []):
            a = add(f"api:{p.get('pid')}:{str(api).lower()}", "api", _h(str(api).lower(), API_VOCAB))
            edges.append((pn, a, schema.EDGE_TYPE_ID["calls"]))
            if prev is not None and prev != a:
                edges.append((prev, a, schema.EDGE_TYPE_ID["next"]))
            prev = a

    root = pid_node.get(procs[0].get("pid")) if procs and pid_node else None
    if root is None:
        root = add("proc:none", "process", 0)

    # aggregate objects attach to the root process
    def attach(items, ntype: str, etype: str, limit: int = 40):
        for v in clean(items)[:limit]:
            key = _normalize_object(ntype, v[:200])
            n = add(key, ntype, _h(key, GRAPH_FEAT_HASH))
            edges.append((root, n, schema.EDGE_TYPE_ID[etype]))

    attach(d.get("filesystem", {}).get("write_files"), "file", "writes")
    attach(d.get("registry", {}).get("write_keys"), "registry", "writes")
    attach(d.get("network", {}).get("domains"), "network", "connects")
    attach(d.get("sync", {}).get("mutexes"), "mutex", "targets")
    attach(d.get("exec", {}).get("created_services"), "service", "targets")

    edge_arr = np.array(edges, dtype=np.int64).reshape(-1, 3)
    return (np.array(node_type, dtype=np.int64), np.array(node_feat, dtype=np.int64),
            edge_arr[:, 0], edge_arr[:, 1], edge_arr[:, 2])


def _dyn_rel(d: dict) -> np.ndarray:
    api, net, fsy, reg = d.get("api", {}), d.get("network", {}), d.get("filesystem", {}), d.get("registry", {})
    meta = d.get("meta", {})
    n_calls = int(api.get("n_api_calls", 0) or 0)
    n_uniq = int(api.get("n_unique_api", 0) or 0)
    # anti-analysis proxy: name-only tags we can still detect in the sequence
    anti = sum(1 for a in (api.get("counts") or {})
               if _tag_call(str(a).lower(), "", "") == TAG_ANTI_ANALYSIS)
    return np.array([
        np.log1p(n_calls) / 10.0,
        n_uniq / max(1, n_calls),
        min(1.0, float(meta.get("duration_sec", 0) or 0) / 200.0),
        np.log1p(int(fsy.get("n_files", 0) or 0)) / 8.0,
        np.log1p(int(reg.get("n_keys", 0) or 0)) / 8.0,
        np.log1p(int(net.get("n_domains", 0) or 0) + int(net.get("n_http", 0) or 0)) / 8.0,
        min(1.0, anti / 10.0),
        1.0 if meta.get("ran_ok") else 0.0,     # maps straight onto the dataset's ran_ok
    ], dtype=np.float32)


def extract_cape_features(path: str | Path, max_seq: int = MAX_SEQ_STORE) -> dict:
    """Read one pre-extracted JSON and return the dynamic half of the sample schema.

    `cape_verdict` is deliberately never read — it is another detector's conclusion,
    not observed behaviour (dataset owner section 2.11).
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        d = json.load(f)

    seq = [str(a) for a in (d.get("api", {}).get("sequence") or [])][:max_seq]
    api_ids = np.array([_h(a.lower(), API_VOCAB) for a in seq], dtype=np.int64)

    # per-call tags: only the ones inferable from the API name alone
    per_call = [_tag_call(a.lower(), "", "") for a in seq]
    sem = [int(aid) * N_TAGS + t for aid, t in zip(api_ids, per_call)]
    # argument-dependent behaviours become context tokens at the front of the sequence
    ctx = sorted(detect_context_tags(d))
    sem_ids = np.array([API_VOCAB * N_TAGS + t for t in ctx] + sem, dtype=np.int64)

    node_type, node_feat, e_src, e_dst, e_type = _build_graph(d, set(ctx))
    return {
        "api_ids": api_ids,
        "sem_ids": sem_ids,
        "node_type": node_type,
        "node_feat": node_feat,
        "edge_src": e_src,
        "edge_dst": e_dst,
        "edge_type": e_type,
        "dyn_rel": _dyn_rel(d),
        "_meta": d.get("meta", {}),
    }
