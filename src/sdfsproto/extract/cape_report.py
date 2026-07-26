"""Dynamic feature extraction from CAPE v2 report.json.

Produces the dynamic half of the sample schema:
  api_ids   (backend 1: API-name tokens)
  sem_ids   (backend 2: API<SEMANTIC_TAG> tokens - argument-aware)
  behavior graph arrays (backend 3)
  dyn_rel   (8 reliability signals)

Written against the standard CAPE v2 report layout:
  report["behavior"]["processes"][i]["calls"][j] = {api, category, arguments: [{name, value}], ...}
  report["behavior"]["summary"] = {files, keys, mutexes, ...}
  report["info"]["duration"], report["network"]

Validate against the real dataset's reports when they arrive — CAPE versions
differ slightly in field names.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ..data import schema

API_VOCAB = 512
N_TAGS = 16
SEM_VOCAB = API_VOCAB * N_TAGS
GRAPH_FEAT_HASH = 4096
MAX_CALLS = 2048

# Semantic tags: the SAME api is benign or malicious depending on its arguments.
TAG_OTHER = 0
TAG_RUN_KEY = 1
TAG_SERVICE_REG = 2
TAG_REG_WRITE = 3
TAG_USER_WRITABLE_EXEC = 4
TAG_TEMP_FILE = 5
TAG_SYSTEM_DIR_WRITE = 6
TAG_SHELL_EXEC = 7
TAG_PROC_INJECT = 8
TAG_HTTP = 9
TAG_IP_DIRECT = 10
TAG_MUTEX = 11
TAG_CRYPTO = 12
TAG_ANTI_ANALYSIS = 13
TAG_PERSIST_TASK = 14
TAG_SERVICE_CREATE = 15

RUN_KEY_RE = re.compile(r"\\(run|runonce|runservices)\b", re.I)
SERVICES_RE = re.compile(r"\\system\\(currentcontrolset|controlset\d+)\\services", re.I)
USER_WRITABLE_RE = re.compile(r"(appdata|%temp%|\\temp\\|programdata|\\users\\[^\\]+\\)", re.I)
EXEC_EXT_RE = re.compile(r"\.(exe|dll|scr|bat|cmd|ps1|vbs|js)\b", re.I)
SYSTEM_DIR_RE = re.compile(r"(\\windows\\|\\system32\\|\\syswow64\\)", re.I)
SHELL_RE = re.compile(r"(cmd\.exe|powershell|wscript|cscript|mshta|rundll32)", re.I)
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
INJECT_APIS = {"writeprocessmemory", "createremotethread", "ntmapviewofsection",
               "setthreadcontext", "queueuserapc", "ntqueueapcthread", "virtualallocex"}
CRYPTO_APIS = {"cryptencrypt", "cryptdecrypt", "cryptacquirecontexta", "cryptacquirecontextw",
               "bcryptencrypt", "bcryptdecrypt", "cryptgenkey", "crypthashdata"}
ANTI_APIS = {"isdebuggerpresent", "checkremotedebuggerpresent", "ntqueryinformationprocess",
             "getickcount", "gettickcount", "ntdelayexecution", "outputdebugstringa",
             "findwindowa", "findwindoww"}
TASK_RE = re.compile(r"(schtasks|taskschd|\\tasks\\)", re.I)


def _h(s: str, mod: int) -> int:
    return int.from_bytes(hashlib.md5(s.encode("utf-8", "ignore")).digest()[:8], "little") % mod


def _args_text(call: dict) -> str:
    parts = []
    for a in call.get("arguments", []) or []:
        v = a.get("value", "")
        parts.append(str(v) if not isinstance(v, (list, dict)) else json.dumps(v))
    return " ".join(parts)


def _tag_call(api_l: str, args: str, category: str) -> int:
    if api_l in INJECT_APIS:
        return TAG_PROC_INJECT
    if api_l in CRYPTO_APIS:
        return TAG_CRYPTO
    if api_l in ANTI_APIS:
        return TAG_ANTI_ANALYSIS
    if "mutex" in api_l or "mutant" in api_l:
        return TAG_MUTEX
    if TASK_RE.search(args):
        return TAG_PERSIST_TASK
    if api_l.startswith(("createservice", "openservice")) or "service" in category:
        return TAG_SERVICE_CREATE
    if api_l.startswith("reg") or category == "registry":
        if RUN_KEY_RE.search(args):
            return TAG_RUN_KEY
        if SERVICES_RE.search(args):
            return TAG_SERVICE_REG
        if "set" in api_l or "create" in api_l or "delete" in api_l:
            return TAG_REG_WRITE
        return TAG_OTHER
    if category in ("filesystem", "file") or api_l.startswith(("createfile", "writefile", "movefile", "copyfile", "deletefile", "ntcreatefile", "ntwritefile")):
        write_like = any(k in api_l for k in ("write", "create", "move", "copy", "delete"))
        if USER_WRITABLE_RE.search(args) and EXEC_EXT_RE.search(args):
            return TAG_USER_WRITABLE_EXEC
        if "temp" in args.lower():
            return TAG_TEMP_FILE
        if write_like and SYSTEM_DIR_RE.search(args):
            return TAG_SYSTEM_DIR_WRITE
        return TAG_OTHER
    if api_l.startswith(("createprocess", "shellexecute", "ntcreateuserprocess", "winexec")):
        return TAG_SHELL_EXEC if SHELL_RE.search(args) else TAG_OTHER
    if category in ("network", "socket") or api_l in ("connect", "internetopenurla", "internetopenurlw", "httpsendrequesta", "httpsendrequestw", "winhttpconnect", "getaddrinfo", "gethostbyname", "send", "urldownloadtofilew", "urldownloadtofilea"):
        low = args.lower()
        if "http" in low:
            return TAG_HTTP
        for tok in re.split(r"[\s,;]+", low):
            if IP_RE.match(tok):
                return TAG_IP_DIRECT
        return TAG_OTHER
    return TAG_OTHER


def _normalize_object(kind: str, value: str) -> str:
    """Collapse per-run noise so persistent artifacts hash identically across samples."""
    v = value.lower()
    v = re.sub(r"c:\\users\\[^\\]+", "%userdir%", v)
    v = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", "%guid%", v)
    v = re.sub(r"\d+", "%d", v)
    return f"{kind}:{v[:120]}"


def empty_dynamic() -> dict:
    """Valid all-zero dynamic view for samples with no sandbox report.

    Used by --static-only extraction. A single process node keeps the graph
    well-formed; dyn_rel is all zeros, which the reliability-aware fusion reads
    as "dynamic view carries no information" — the same signal it gets from a
    sample that evaded the sandbox.
    """
    return {
        "api_ids": np.zeros(0, dtype=np.int64),
        "sem_ids": np.zeros(0, dtype=np.int64),
        "node_type": np.array([schema.NODE_TYPE_ID["process"]], dtype=np.int64),
        "node_feat": np.zeros(1, dtype=np.int64),
        "edge_src": np.zeros(0, dtype=np.int64),
        "edge_dst": np.zeros(0, dtype=np.int64),
        "edge_type": np.zeros(0, dtype=np.int64),
        "dyn_rel": np.zeros(schema.DYN_REL_DIM, dtype=np.float32),
    }


def extract_cape(report_path: str | Path) -> dict:
    with open(report_path, "r", encoding="utf-8", errors="replace") as f:
        report = json.load(f)

    behavior = report.get("behavior", {}) or {}
    processes = behavior.get("processes", []) or []
    info = report.get("info", {}) or {}

    api_ids: list[int] = []
    sem_ids: list[int] = []
    tags_count = np.zeros(N_TAGS, dtype=np.int64)

    # ---- graph ----
    node_type: list[int] = []
    node_feat: list[int] = []
    edges: list[tuple[int, int, int]] = []
    node_of: dict[str, int] = {}

    def add_node(key: str, ntype: str, feat: int) -> int:
        if key not in node_of:
            node_of[key] = len(node_type)
            node_type.append(schema.NODE_TYPE_ID[ntype])
            node_feat.append(feat)
        return node_of[key]

    tag_to_node = {
        TAG_RUN_KEY: "registry", TAG_SERVICE_REG: "registry", TAG_REG_WRITE: "registry",
        TAG_USER_WRITABLE_EXEC: "file", TAG_TEMP_FILE: "file", TAG_SYSTEM_DIR_WRITE: "file",
        TAG_HTTP: "network", TAG_IP_DIRECT: "network",
        TAG_MUTEX: "mutex", TAG_SERVICE_CREATE: "service", TAG_PERSIST_TASK: "service",
    }

    n_calls_total = 0
    for proc in processes:
        pname = str(proc.get("process_name") or proc.get("module_path") or "proc").lower()
        p_node = add_node(f"proc:{proc.get('process_id', pname)}",
                          "process", _h(_normalize_object("proc", pname), GRAPH_FEAT_HASH))
        ppid = proc.get("parent_id")
        pkey = f"proc:{ppid}"
        if ppid is not None and pkey in node_of:
            edges.append((node_of[pkey], p_node, schema.EDGE_TYPE_ID["spawns"]))

        prev_api_node = None
        for call in (proc.get("calls", []) or [])[:MAX_CALLS]:
            api = str(call.get("api", "unknown"))
            api_l = api.lower()
            category = str(call.get("category", "")).lower()
            args = _args_text(call)
            aid = _h(api_l, API_VOCAB)
            tag = _tag_call(api_l, args, category)
            tags_count[tag] += 1
            api_ids.append(aid)
            sem_ids.append(aid * N_TAGS + tag)
            n_calls_total += 1

            a_node = add_node(f"api:{proc.get('process_id')}:{api_l}", "api", aid)
            edges.append((p_node, a_node, schema.EDGE_TYPE_ID["calls"]))
            if prev_api_node is not None and prev_api_node != a_node:
                edges.append((prev_api_node, a_node, schema.EDGE_TYPE_ID["next"]))
            prev_api_node = a_node

            kind = tag_to_node.get(tag)
            if kind:
                okey = _normalize_object(kind, args[:200])
                o_node = add_node(okey, kind, _h(okey, GRAPH_FEAT_HASH))
                et = "writes" if kind in ("file", "registry") else ("connects" if kind == "network" else "targets")
                edges.append((a_node, o_node, schema.EDGE_TYPE_ID[et]))

    if not node_type:  # completely empty trace: still emit a valid 1-node graph
        add_node("proc:none", "process", 0)

    summary = behavior.get("summary", {}) or {}
    n_files = len(summary.get("files", []) or []) or int(tags_count[[TAG_USER_WRITABLE_EXEC, TAG_TEMP_FILE, TAG_SYSTEM_DIR_WRITE]].sum())
    n_reg = len(summary.get("keys", []) or []) or int(tags_count[[TAG_RUN_KEY, TAG_SERVICE_REG, TAG_REG_WRITE]].sum())
    n_net = len((report.get("network", {}) or {}).get("hosts", []) or []) or int(tags_count[[TAG_HTTP, TAG_IP_DIRECT]].sum())
    duration = float(info.get("duration", 0.0) or 0.0)

    dyn_rel = np.array(
        [
            np.log1p(n_calls_total) / 10.0,
            (len(set(api_ids)) / max(1, len(api_ids))),
            min(1.0, duration / 120.0),
            np.log1p(n_files) / 8.0,
            np.log1p(n_reg) / 8.0,
            np.log1p(n_net) / 8.0,
            min(1.0, float(tags_count[TAG_ANTI_ANALYSIS]) / 10.0),
            1.0 if n_calls_total >= 20 else 0.0,
        ],
        dtype=np.float32,
    )

    edge_arr = np.array(edges, dtype=np.int64).reshape(-1, 3)
    return {
        "api_ids": np.array(api_ids, dtype=np.int64),
        "sem_ids": np.array(sem_ids, dtype=np.int64),
        "node_type": np.array(node_type, dtype=np.int64),
        "node_feat": np.array(node_feat, dtype=np.int64),
        "edge_src": edge_arr[:, 0],
        "edge_dst": edge_arr[:, 1],
        "edge_type": edge_arr[:, 2],
        "dyn_rel": dyn_rel,
    }
