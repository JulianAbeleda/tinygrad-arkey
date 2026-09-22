#!/usr/bin/env python3
"""B3 work item 1: aligned logical/physical CUDA decode DAG census (Route B3).

Captures BOTH frozen views of the real CUDA decode token (DEV=CUDA,
CUDA_GRAPH_STREAMS=1) in one lock-held run, then attaches duration weights from
the existing CUPTI node trace and produces a duration-weighted census.

Views
-----
LOGICAL:  pre-planner semantic view. One range-aware DepsTracker over the whole
          pre-plan linear (before memory_plan_rewrite), using real buffer
          identity (no arena aliasing). RAW/WAR/WAW edges only where semantic
          buffers actually conflict. Group assignment applied from the
          graph_split_rewrite admission observer (same call indexes).
PHYSICAL: post-planner frozen view. Mirrors CUDAGraph.new_node /
          _capture_construct exactly: per graph group, a FRESH DepsTracker over
          the group's calls, calling access_resources([b.base for b in bufs],
          outs) so arena slices collapse to full-arena ranges. The seam also
          wraps CUDAGraph._access_resources to record the runtime's ACTUAL
          frozen preds and verifies the mirror against them.

Modes
-----
--capture   Live GPU (run under flock). Installs the jit_lower +
            graph_split_rewrite seam (and the CUDAGraph._access_resources
            verifier), runs the decode harness for one 512-depth token
            (decode_runtime_overhead.main), selects the decode capture by
            group sizes, writes the aligned capture JSON.
--analyze   CPU-only. Reads the capture JSON and the CUPTI node trace sqlite
            (/tmp/b0_cuda_trace.sqlite by default), aligns trace nodes to DAG
            calls positionally (consecutive graphNodeIds), attaches per-node
            median durations, computes the duration-weighted census (per group
            and whole token, logical vs physical), writes the census JSON and
            prints a labeled report.
--selftest  Hermetic CPU self-test of the census math and edge-kind labeling.

No tinygrad runtime file is modified. Evidence classes: OBSERVED for what the
seam/trace produced directly, INFERRED for the interpretation.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, pathlib, sqlite3, subprocess, sys, tempfile
from typing import Any, Callable, Iterator

SCHEMA_CAPTURE = "tinygrad.cuda_route_aligned_census.capture.v1"
SCHEMA_CENSUS = "tinygrad.cuda_route_aligned_census.v1"
KINDS = ("RAW", "WAR", "WAW")
UNKNOWN = "UNKNOWN"
DEFAULT_TRACE = "/tmp/b0_cuda_trace.sqlite"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
DECODE_GROUP_SIZES = [32, 64, 128, 256, 512, 29]

try:
  from extra.llm_research.decode import dag_critical_path_sim as _sim
except ImportError:  # run directly as a script: script dir is on sys.path
  import dag_critical_path_sim as _sim  # type: ignore


def _atomic_json(path: str, payload: dict) -> None:
  p = pathlib.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=".%s." % p.name, suffix=".tmp", dir=p.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, p)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def _git_commit() -> str | None:
  try:
    return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                                   text=True).strip()
  except Exception:
    return None


def _driver_version() -> str | None:
  try:
    return subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                   text=True).strip().splitlines()[0]
  except Exception:
    return None


# ---------------------------------------------------------------------------
# Range-aware dependency walk with edge-kind labeling (mirrors jit.py usage)
# ---------------------------------------------------------------------------

class RecordingDepsTracker:
  """DepsTracker mirroring jit.py, labeling each returned edge RAW/WAR/WAW.

  State mutation is delegated to tinygrad.engine.jit.DepsTracker; this class
  only records the kind by scanning the same write/read range maps.
  """

  def __init__(self):
    from tinygrad.engine.jit import DepsTracker
    self._tracker = DepsTracker()
    self.edges: list[tuple[int, int, str]] = []
    # (dep, new, kind, base_key, overlap_start, overlap_end) per conflicting access.
    self.ranges: list[tuple[int, int, str, int, int, int]] = []
    self.base_nbytes: dict[int, int] = {}

  def access_resources(self, bufs: list[Any], write: list[int], new_dependency: int) -> list[Any]:
    kinds: dict[tuple[int, int], str] = {}
    for i, buf in enumerate(bufs):
      key = id(buf.base)
      s, e = buf.offset, buf.offset + buf.nbytes
      self.base_nbytes.setdefault(key, buf.base.nbytes)
      if i in write:
        for st, en, dep in self._tracker.w_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "WAW")
            self.ranges.append((int(dep), int(new_dependency), "WAW", key, max(st, s), min(en, e)))
        for st, en, dep in self._tracker.r_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "WAR")
            self.ranges.append((int(dep), int(new_dependency), "WAR", key, max(st, s), min(en, e)))
      else:
        for st, en, dep in self._tracker.w_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "RAW")
            self.ranges.append((int(dep), int(new_dependency), "RAW", key, max(st, s), min(en, e)))
    wait_nodes = self._tracker.access_resources(bufs, write, new_dependency)
    for dep in wait_nodes:
      self.edges.append((int(dep), int(new_dependency), kinds.get((id(dep), id(new_dependency)), UNKNOWN)))
    return wait_nodes


class _RecordingObserver:
  """GraphAdmissionObserver that records batch assignments and forwards to any existing observer."""

  def __init__(self, forward: Callable[[Any], None] | None):
    self.forward = forward
    self.records: list[Any] = []

  def __call__(self, event: Any) -> None:
    if self.forward is not None:
      self.forward(event)
    from tinygrad.engine.jit import GraphAdmissionObservation
    if isinstance(event, GraphAdmissionObservation):
      self.records.append(event)

  def bind_call(self, call_index: int, call: Any) -> None:
    if hasattr(self.forward, "bind_call"):
      self.forward.bind_call(call_index, call)


def _group_map(records: list[Any]) -> dict[int, Any]:
  """call_index -> group id: graph members use batch_index; direct calls get own ids."""
  m: dict[int, Any] = {}
  for r in records:
    if r.assignment == "graph":
      m[r.call_index] = r.batch_index
    elif r.assignment == "direct":
      m[r.call_index] = "direct-%d" % r.direct_call_index
    else:
      m[r.call_index] = None
  return m


def _group_sizes(group_map: dict[int, Any]) -> list[tuple[Any, int]]:
  """[(group_id, size)] in first-appearance order of group ids."""
  sizes: list[tuple[Any, int]] = []
  order: list[Any] = []
  counts: dict[Any, int] = {}
  for gid in group_map.values():
    if gid not in counts:
      counts[gid] = 0
      order.append(gid)
    counts[gid] += 1
  return [(gid, counts[gid]) for gid in order]


def _program_identity(ast: Any) -> dict:
  """Compiled-call identity: name + launch dims + write/read slots."""
  from tinygrad.uop.ops import Ops
  if ast.op is Ops.PROGRAM:
    return {"op": "PROGRAM", "name": str(ast.arg.name), "outs": tuple(int(x) for x in ast.arg.outs),
            "ins": tuple(int(x) for x in ast.arg.ins),
            "global_size": tuple(int(x) if isinstance(x, int) else float(x) for x in ast.arg.global_size),
            "local_size": tuple(int(x) if isinstance(x, int) else float(x) for x in (ast.arg.local_size or ())),
            "sig": (str(ast.arg.name),) + tuple(int(x) if isinstance(x, int) else float(x) for x in ast.arg.global_size) +
                   tuple(int(x) if isinstance(x, int) else float(x) for x in (ast.arg.local_size or ()))}
  if ast.op is Ops.COPY:
    return {"op": "COPY", "name": "copy", "outs": (0,), "ins": (1,),
            "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("copy", 1, 1, 1, 1, 1, 1)}
  return {"op": str(ast.op), "name": str(ast.op), "outs": (), "ins": (),
          "global_size": (), "local_size": (), "sig": (str(ast.op),)}


def _resolve_buffers(call: Any, input_uops: tuple[Any, ...]) -> tuple[list[Any], list[int], list[int]]:
  """Mirror GraphRunner buffer resolution without allocating: returns
  (buffers, write_idx, ins_idx) flattened across unwrap_multi device groups."""
  from tinygrad.engine.realize import get_call_outs_ins, unwrap_multi, resolve_params
  arg_uops = resolve_params(call, input_uops)
  outs, ins = get_call_outs_ins(call)
  all_bufs: list[Any] = []
  write_idx: list[int] = []
  ins_idx: list[int] = []
  for bufs, _device_vars in unwrap_multi(call, arg_uops):
    start = len(all_bufs)
    all_bufs.extend(b.buffer for b in bufs)
    write_idx.extend(start + i for i in outs)
    ins_idx.extend(start + i for i in ins)
  return all_bufs, write_idx, ins_idx


def build_logical_dag(pre_plan_linear: Any, post_plan_linear: Any, input_uops: tuple[Any, ...],
                      group_map: dict[int, Any], held_bufs: set[Any] | None = None) -> dict:
  """Logical view: pre-planner semantic RAW/WAR/WAW over real buffer identity.

  Also emits the logical buffer manifest (producer/consumers, lifetime, held
  status) and the logical-buffer -> arena placement pairing against the
  post-plan views of the same calls (arena ordinal, offset, byte interval).
  """
  from tinygrad.uop.ops import Ops
  from tinygrad.engine.realize import get_call_arg_uops, resolve_params
  if len(pre_plan_linear.src) != len(post_plan_linear.src):
    raise ValueError("pre-plan/post-plan call count mismatch: %d vs %d" %
                     (len(pre_plan_linear.src), len(post_plan_linear.src)))
  tracker = RecordingDepsTracker()
  nodes: list[dict] = []
  unknown: list[int] = []
  held_set = set(held_bufs or ())
  buf_meta: dict[int, dict] = {}
  arena_registry: dict[int, dict] = {}
  buf_placements: list[dict] = []  # (call_index, buf_key, arena_key, offset, size)
  for j, (pcall, dcall) in enumerate(zip(pre_plan_linear.src, post_plan_linear.src)):
    ast = dcall.src[0]
    if ast.op not in (Ops.PROGRAM, Ops.COPY):
      continue  # non-kernel calls (e.g. views) are not DAG nodes
    ident = _program_identity(ast)
    dep_unknown = False
    try:
      lbufs, write_idx, _ins = _resolve_buffers(pcall, input_uops)
      pbufs, _pw, _pi = _resolve_buffers(dcall, input_uops)
      flat_uops: list[Any] = resolve_params(pcall, input_uops)
      assert len(flat_uops) == len(lbufs), "single-device call expected for held detection"
      for pos, b in enumerate(lbufs):
        key = id(b.base)
        rec = buf_meta.setdefault(key, {"device": b.device, "nbytes": b.base.nbytes, "dtype": str(b.base.dtype),
                                        "first_call": j, "last_call": j, "producer": None, "consumers": []})
        rec["first_call"] = min(rec["first_call"], j)
        rec["last_call"] = max(rec["last_call"], j)
        rec["consumers"].append(j)
        if pos in write_idx and rec["producer"] is None:
          rec["producer"] = j
        if "held" not in rec and pos < len(flat_uops):
          u = flat_uops[pos]
          rec["held"] = bool(u in held_set or (u.op is not Ops.BUFFER and u.base in held_set))
          rec["held_reason"] = "held_bufs" if rec["held"] else "plannable"
        if pos < len(pbufs):
          av = pbufs[pos]
          akey = id(av.base)
          arena_registry.setdefault(akey, {"nbytes": av.base.nbytes, "ordinal": len(arena_registry)})
          buf_placements.append({"call": j, "buf_key": key, "arena_key": akey, "offset": av.offset, "size": av.nbytes})
      tracker.access_resources(lbufs, write_idx, j)
    except Exception:
      dep_unknown = True
      unknown.append(j)
    nodes.append({"call_index": j, "group_id": group_map.get(j), "name": ident["name"],
                  "op": ident["op"], "outs": ident["outs"], "ins": ident["ins"],
                  "global_size": ident["global_size"], "local_size": ident["local_size"],
                  "sig": ident["sig"], "dep_unknown": dep_unknown})
  # assign stable ordinals to logical buffers
  buf_ordinals: dict[int, int] = {k: i for i, k in enumerate(buf_meta)}
  manifest = []
  for key, rec in buf_meta.items():
    manifest.append({"ordinal": buf_ordinals[key], "device": rec["device"], "nbytes": rec["nbytes"],
                     "dtype": rec["dtype"], "first_call": rec["first_call"], "last_call": rec["last_call"],
                     "producer_call": rec["producer"], "consumer_calls": rec["consumers"],
                     "held": rec.get("held"), "held_reason": rec.get("held_reason", "plannable")})
  # logical-buffer -> arena placements (ordinals)
  arena_ordinals = {k: v["ordinal"] for k, v in arena_registry.items()}
  placements = [{"call": p["call"], "logical_buffer": buf_ordinals[p["buf_key"]],
                 "arena": arena_ordinals[p["arena_key"]], "offset": p["offset"], "size": p["size"]}
                for p in buf_placements if p["buf_key"] in buf_ordinals and p["arena_key"] in arena_ordinals]
  arenas = [{"ordinal": v["ordinal"], "nbytes": v["nbytes"], "key": str(k)} for k, v in arena_registry.items()]
  edges = []
  for f, t, k in tracker.edges:
    ranges = [r for r in tracker.ranges if r[0] == f and r[1] == t and r[2] == k]
    buf_ids = sorted({buf_ordinals.get(r[3], -1) for r in ranges} - {-1})
    edges.append({"from": f, "to": t, "kind": k,
                  "logical_buffers": buf_ids,
                  "ranges": [[r[4], r[5]] for r in ranges]})
  return {"nodes": nodes, "edges": edges, "unknown_dep_nodes": unknown,
          "buffer_manifest": manifest, "arena_manifest": arenas, "placements": placements}


def build_physical_dag(post_plan_linear: Any, input_uops: tuple[Any, ...], group_map: dict[int, Any]) -> dict:
  """Physical view: per-group frozen edges exactly as CUDAGraph builds them
  (fresh DepsTracker per group, base buffers -> full-arena ranges)."""
  from tinygrad.uop.ops import Ops
  groups: dict[Any, dict] = {}
  order: list[Any] = []
  trackers: dict[Any, RecordingDepsTracker] = {}
  for j, dcall in enumerate(post_plan_linear.src):
    gid = group_map.get(j)
    if gid is None:
      continue  # ignored/unknown call, not a graph member
    ast = dcall.src[0]
    if ast.op is Ops.SLICE:
      continue
    if ast.op not in (Ops.PROGRAM, Ops.COPY):
      continue
    if gid not in groups:
      groups[gid] = {"members": [], "edges": []}
      trackers[gid] = RecordingDepsTracker()
      order.append(gid)
    member = len(groups[gid]["members"])
    ident = _program_identity(ast)
    try:
      bufs, write_idx, _ins = _resolve_buffers(dcall, input_uops)
      bases = [b.base for b in bufs]
      trackers[gid].access_resources(bases, write_idx, member)
      dep_unknown = False
    except Exception:
      dep_unknown = True
    groups[gid]["members"].append({"member_index": member, "call_index": j, "name": ident["name"],
                                   "op": ident["op"], "outs": ident["outs"], "ins": ident["ins"],
                                   "global_size": ident["global_size"], "local_size": ident["local_size"],
                                   "sig": ident["sig"], "dep_unknown": dep_unknown})
  out_groups: list[dict] = []
  for gid in order:
    tr = trackers[gid]
    arena_registry: dict[int, dict] = {}
    for key, nbytes in tr.base_nbytes.items():
      arena_registry.setdefault(key, {"nbytes": nbytes, "ordinal": len(arena_registry)})
    edges = []
    for f, t, k in tr.edges:
      ranges = [r for r in tr.ranges if r[0] == f and r[1] == t and r[2] == k]
      arena_ids = sorted({arena_registry[r[3]]["ordinal"] for r in ranges if r[3] in arena_registry})
      edges.append({"from": f, "to": t, "kind": k,
                    "arena": arena_ids,
                    "ranges": [[r[4], r[5]] for r in ranges]})
    groups[gid]["edges"] = edges
    out_groups.append({"group_id": gid, "size": len(groups[gid]["members"]),
                       "nodes": groups[gid]["members"], "edges": edges,
                       "arena_manifest": [{"ordinal": v["ordinal"], "nbytes": v["nbytes"], "key": str(k)}
                                          for v in sorted(arena_registry.values(), key=lambda x: x["ordinal"])],
                       "unknown_dep_members": [n["member_index"] for n in groups[gid]["members"] if n["dep_unknown"]]})
  return {"groups": out_groups}


# ---------------------------------------------------------------------------
# CUDAGraph runtime verifier: record the actual frozen preds
# ---------------------------------------------------------------------------

def install_cudagraph_verifier(state: dict) -> Callable[[], None]:
  """Wrap CUDAGraph._access_resources to record per-instance frozen preds.

  Each invocation in the programmatic path corresponds to one graph call in
  call order; new_dependency is that call's graph node, and the returned deps
  are earlier graph nodes. This yields the exact pred lists CUDAGraph freezes.
  """
  from tinygrad.runtime.graph.cuda import CUDAGraph
  orig = CUDAGraph._access_resources
  state["cudagraph_instances"] = {}
  state["cudagraph_order"] = []
  counter = [0]

  def wrap(self, bufs, write, new_dependency):
    res = orig(self, bufs, write, new_dependency)
    key = id(self)
    if key not in state["cudagraph_instances"]:
      state["cudagraph_instances"][key] = {"n_calls": len(self.calls), "node2idx": {}, "preds": {}, "k": 0,
                                           "order": counter[0]}
      state["cudagraph_order"].append(key)
      counter[0] += 1
    st = state["cudagraph_instances"][key]
    k = st["k"]
    st["node2idx"][new_dependency] = k
    st["preds"][k] = [st["node2idx"][d] for d in res if d in st["node2idx"]]
    st["k"] += 1
    return res

  CUDAGraph._access_resources = wrap
  return lambda: setattr(CUDAGraph, "_access_resources", orig)


def runtime_preds_report(state: dict, group_sizes: list[int]) -> dict:
  """Map constructed CUDAGraph instances to decode groups by size+order suffix."""
  inst = state.get("cudagraph_instances", {})
  order = state.get("cudagraph_order", [])
  sizes = list(group_sizes)
  # decode instances are the LAST run of consecutive constructions whose sizes
  # equal the decode group sizes in order (all 6 constructed on first replay).
  matches: list[dict] = []
  for i in range(len(order) - len(sizes) + 1):
    if [inst[o]["n_calls"] for o in order[i:i + len(sizes)]] == sizes:
      matches.append(i)
  if not matches:
    return {"matched": False, "instances": len(order),
            "n_calls_list": [inst[o]["n_calls"] for o in order]}
  start = matches[-1]
  groups = []
  for gidx, o in enumerate(order[start:start + len(sizes)]):
    st = inst[o]
    groups.append({"group_id": gidx, "size": st["n_calls"],
                   "preds": [st["preds"].get(k, []) for k in range(st["n_calls"])]})
  return {"matched": True, "groups": groups}


# ---------------------------------------------------------------------------
# Seam (monkeypatches jit_lower + graph_split_rewrite; no runtime edits)
# ---------------------------------------------------------------------------

def install_seam(state: dict) -> Callable[[], None]:
  from tinygrad.engine import jit as tjit
  orig_lower, orig_split = tjit.jit_lower, tjit.graph_split_rewrite

  def wrapped_jit_lower(linear: Any, held_bufs: set[Any], input_uops: list[Any]) -> Any:
    state["input_uops"] = tuple(input_uops)
    state["pre_plan_linear"] = linear
    state["held_bufs"] = set(held_bufs)
    return orig_lower(linear, held_bufs, input_uops)

  def wrapped_graph_split(linear: Any, max_batch_size: int = 0, observer: Callable[[Any], None] | None = None) -> Any:
    rec = _RecordingObserver(observer)
    result = orig_split(linear, max_batch_size=max_batch_size, observer=rec)
    group_map = _group_map(rec.records)
    logical = build_logical_dag(state["pre_plan_linear"], linear, state["input_uops"] or (), group_map,
                                state.get("held_bufs"))
    physical = build_physical_dag(linear, state["input_uops"] or (), group_map)
    sizes = _group_sizes(group_map)
    state["captures"].append({"logical": logical, "physical": physical,
                              "group_sizes": sizes, "records": len(rec.records)})
    return result

  tjit.jit_lower = wrapped_jit_lower
  tjit.graph_split_rewrite = wrapped_graph_split

  def restore():
    tjit.jit_lower = orig_lower
    tjit.graph_split_rewrite = orig_split
  return restore


def _capture_group_sizes(cap: dict) -> list[int]:
  return [s for _gid, s in cap["group_sizes"] if not str(_gid).startswith("direct-")]


def select_decode_capture(captures: list[dict]) -> dict | None:
  for cap in captures:
    if _capture_group_sizes(cap) == DECODE_GROUP_SIZES:
      return cap
  for cap in captures:
    if len(_capture_group_sizes(cap)) == 6 and sum(_capture_group_sizes(cap)) == 1021:
      return cap
  if captures:
    return max(captures, key=lambda c: len(c["logical"]["nodes"]))
  return None


# ---------------------------------------------------------------------------
# CUPTI trace loading and alignment
# ---------------------------------------------------------------------------

def _trace_clusters(con: sqlite3.Connection, graph_id: int, gap_us: float = 20.0) -> list[list[dict]]:
  rows = list(con.execute(
    "select graphNodeId, shortName, gridX, gridY, gridZ, blockX, blockY, blockZ, start, end "
    "from CUPTI_ACTIVITY_KIND_KERNEL where graphId=? order by start", (graph_id,)))
  clusters: list[list[dict]] = []
  cur: list[dict] = []
  for r in rows:
    row = {"id": r[0], "name": r[1], "grid": (r[2], r[3], r[4]), "block": (r[5], r[6], r[7]),
           "start": r[8], "end": r[9], "duration": r[9] - r[8]}
    if cur and row["start"] - cur[-1]["end"] > gap_us * 1000:
      clusters.append(cur)
      cur = [row]
    else:
      cur.append(row)
  if cur:
    clusters.append(cur)
  return clusters


def _short_name(con: sqlite3.Connection, name_id: int) -> str:
  row = con.execute("select value from StringIds where id=?", (name_id,)).fetchone()
  return str(row[0]) if row else str(name_id)


def load_trace_durations(trace_path: str, group_sizes: list[int]) -> dict:
  """Per decode group: node-position -> median duration (us) over steady-state
  replays. Positions are node ids relative to the group's consecutive base."""
  con = sqlite3.connect(trace_path)
  out: dict[str, dict] = {"groups": {}, "source": trace_path}
  # decode graphIds: those whose replay clusters have exactly one of the sizes
  candidates = [r[0] for r in con.execute(
    "select distinct graphId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null and graphId != 0")]
  used_sizes: set[int] = set()
  for gid in sorted(candidates):
    clusters = _trace_clusters(con, gid)
    n = len(clusters[0]) if clusters else 0
    if n not in group_sizes or n in used_sizes:
      continue
    used_sizes.add(n)
    good = [c for c in clusters if len(c) == n][2:]  # drop warmup launches
    if len(good) < 3:
      continue
    base = min(c[0]["id"] for c in good)
    per_pos: dict[int, list[int]] = {}
    sig_at: dict[int, tuple] = {}
    for c in good:
      for row in c:
        pos = row["id"] - base
        per_pos.setdefault(pos, []).append(row["duration"])
        sig_at[pos] = (_short_name(con, row["name"]), row["grid"], row["block"])
    if len(per_pos) != n or any(len(per_pos[p]) < 3 for p in per_pos):
      continue
    durations = [sorted(per_pos[p])[len(per_pos[p]) // 2] / 1000.0 for p in range(n)]
    out["groups"][str(n)] = {"graph_id": gid, "base_id": base, "size": n, "replays": len(good),
                             "durations_us": durations,
                             "signatures": [list(sig_at[p]) for p in range(n)]}
  con.close()
  return out


def align_capture(capture: dict, trace: dict) -> dict:
  """Attach durations to logical and physical nodes; verify positional alignment."""
  duration_by_call: dict[int, float] = {}
  physical_durations: dict[str, list[float]] = {}
  alignment: list[dict] = []
  for group in capture["physical"]["groups"]:
    gid = str(group["group_id"])
    size = group["size"]
    tgroup = trace["groups"].get(str(size))
    if tgroup is None:
      alignment.append({"group_id": gid, "size": size, "aligned": False, "reason": "no trace group of this size"})
      continue
    durs = tgroup["durations_us"]
    sigs = tgroup["signatures"]
    mismatch = 0
    reasons: list[str] = []
    for m, node in enumerate(group["nodes"]):
      dag_sig = list(node["sig"])
      trace_sig = [str(sigs[m][0])] + list(sigs[m][1]) + list(sigs[m][2])
      if dag_sig != trace_sig:
        mismatch += 1
        if len(reasons) < 4:
          reasons.append("member %d: dag %s vs trace %s" % (m, dag_sig, trace_sig))
      duration_by_call[node["call_index"]] = durs[m]
    physical_durations[gid] = [durs[m] for m in range(size)]
    alignment.append({"group_id": gid, "size": size, "aligned": mismatch == 0,
                      "mismatched_positions": mismatch, "examples": reasons,
                      "method": "positional node-id base+member" if mismatch == 0 else "positional-with-mismatch"})
  for node in capture["logical"]["nodes"]:
    if node["call_index"] not in duration_by_call:
      # fall back to the group-local member position
      gid = node["group_id"]
      g = next((x for x in capture["physical"]["groups"] if str(x["group_id"]) == str(gid)), None)
      if g is not None:
        member = next((m for m, n in enumerate(g["nodes"]) if n["call_index"] == node["call_index"]), None)
        if member is not None and str(g["size"]) in trace["groups"]:
          duration_by_call[node["call_index"]] = trace["groups"][str(g["size"])]["durations_us"][member]
  return {"duration_by_call": duration_by_call, "physical_durations": physical_durations,
          "alignment": alignment,
          "aligned_nodes": sum(1 for a in alignment if a.get("aligned")),
          "total_groups": len(capture["physical"]["groups"])}


# ---------------------------------------------------------------------------
# Census metrics
# ---------------------------------------------------------------------------

def _sim_nodes(nodes: list[dict], durations: dict[int, float], edges: list[dict], idx: dict[int, int]) -> list[dict]:
  by_to: dict[int, list[int]] = {}
  for e in edges:
    by_to.setdefault(idx[e["to"]], []).append(idx[e["from"]])
  out = []
  for n in nodes:
    i = idx[n["call_index"]]
    out.append({"id": i, "name": n["name"], "duration": float(durations.get(n["call_index"], 0.0)),
                "deps": sorted(by_to.get(i, []))})
  return out


def _longest_chain(nodes: list[dict]) -> tuple[int, list[int]]:
  n = len(nodes)
  children: list[list[int]] = [[] for _ in range(n)]
  for node in nodes:
    for d in node["deps"]:
      children[d].append(node["id"])
  dp = [1] * n
  best = [1] * n
  for i in range(n - 1, -1, -1):
    for c in children[i]:
      if dp[i] < 1 + dp[c]:
        dp[i] = 1 + dp[c]
        best[i] = c
  root = max(range(n), key=lambda i: (dp[i], -i))
  chain = []
  cur = root
  while True:
    chain.append(cur)
    if best[cur] == cur:
      break
    cur = best[cur]
  return dp[root], chain


def _level_width(nodes: list[dict]) -> int:
  depth = {}
  for node in nodes:
    depth[node["id"]] = 1 + max((depth[d] for d in node["deps"]), default=0)
  counts: dict[int, int] = {}
  for d in depth.values():
    counts[d] = counts.get(d, 0) + 1
  return max(counts.values()) if counts else 0


def _ready_width_profile(nodes: list[dict], queues: int) -> dict:
  """Deterministic list schedule (same policy as dag_critical_path_sim) plus
  ready-set width tracking."""
  n = len(nodes)
  durs = [node["duration"] for node in nodes]
  children: list[list[int]] = [[] for _ in range(n)]
  for node in nodes:
    for d in node["deps"]:
      children[d].append(node["id"])
  need = [len(node["deps"]) for node in nodes]
  est = _sim.compute_est(nodes)
  tails = _sim.compute_tails(nodes)
  start = [0.0] * n
  end = [0.0] * n
  q_free = [0.0] * queues
  ready = [i for i in range(n) if need[i] == 0]
  pending = set(range(n))
  widths: dict[int, int] = {}
  max_width = 0
  while pending:
    if not ready:
      ready.append(min(pending))
    pick = max(ready, key=lambda i: (tails[i], -est[i], -i))
    q = min(range(queues), key=lambda j: (q_free[j], j))
    s = max(q_free[q], est[pick])
    start[pick] = s
    end[pick] = s + durs[pick]
    q_free[q] = end[pick]
    ready.remove(pick)
    pending.discard(pick)
    widths[len(ready)] = widths.get(len(ready), 0) + 1
    max_width = max(max_width, len(ready))
    for c in children[pick]:
      est[c] = max(est[c], end[pick])
      need[c] -= 1
      if need[c] == 0:
        ready.append(c)
  return {"max_ready": max_width, "histogram": {str(k): v for k, v in sorted(widths.items())},
          "span_us": round(max(end) if n else 0.0, 3)}


def _edge_kind_counts(edges: list[dict]) -> dict:
  return {k: sum(1 for e in edges if e["kind"] == k) for k in KINDS}


def _verdict(serialized_us: float, cp_us: float, sched2q_us: float, sched3q_us: float) -> dict:
  """Duration-weighted shape verdict. A strict chain has cp==serialized and
  2q/3q savings of 0. We call a view chain-shaped when the deterministic
  2-resource schedule cannot recover >= 5% of serialized time (the same
  >= 5% bar B2 used for overlap); otherwise duration-weighted independence
  exists. Threshold and mapping are INFERRED policy, not runtime guarantees."""
  denom = serialized_us or 0.0
  cp_save = (serialized_us - cp_us) / denom * 100.0 if denom else 0.0
  s2_save = (serialized_us - sched2q_us) / denom * 100.0 if denom else 0.0
  s3_save = (serialized_us - sched3q_us) / denom * 100.0 if denom else 0.0
  if s2_save >= 5.0 and cp_save >= 5.0:
    verdict = "INDEPENDENT"
  elif s2_save < 5.0:
    verdict = "CHAIN_SHAPED"
  else:
    verdict = "BORDERLINE"
  return {"verdict": verdict, "cp_saving_pct": round(cp_save, 3),
          "sched_2q_saving_pct": round(s2_save, 3), "sched_3q_saving_pct": round(s3_save, 3)}


def compute_view_census(nodes: list[dict], durations: dict[int, float], edges: list[dict],
                        idx: dict[int, int], label: str) -> dict:
  sim = _sim_nodes(nodes, durations, edges, idx)
  if sim:
    metrics = _sim.compute_metrics(sim)
  else:
    metrics = {"node_count": 0, "serialized_span_us": 0.0, "critical_path_us": 0.0,
               "schedule_2q_us": 0.0, "schedule_3q_us": 0.0, "savings_us": {}, "savings_pct": {},
               "node_classes": {}, "overlapping_classes": {}}
  chain_len, chain = _longest_chain(sim)
  roots = [node["id"] for node in sim if not node["deps"]]
  leaves = [i for i in range(len(sim)) if not any(i in node["deps"] for node in sim)]
  serial = metrics["serialized_span_us"]
  v = _verdict(serial, metrics["critical_path_us"], metrics["schedule_2q_us"], metrics["schedule_3q_us"])
  return {"view": label, "node_count": metrics["node_count"], "edge_count": len(edges),
          "edges_per_call": round(len(edges) / metrics["node_count"], 3) if metrics["node_count"] else 0.0,
          "edge_kinds": _edge_kind_counts(edges),
          "longest_chain_nodes": chain_len,
          "chain_ratio": round(chain_len / metrics["node_count"], 3) if metrics["node_count"] else 0.0,
          "critical_path_us": round(metrics["critical_path_us"], 3),
          "serialized_us": round(serial, 3),
          "schedule_2q_us": round(metrics["schedule_2q_us"], 3),
          "schedule_3q_us": round(metrics["schedule_3q_us"], 3),
          "ready_2q": _ready_width_profile(sim, 2),
          "ready_3q": _ready_width_profile(sim, 3),
          "roots": len(roots), "leaves": len(leaves), "max_level_width": _level_width(sim),
          "independent_branches_2q": _ready_width_profile(sim, 2)["max_ready"],
          "node_classes": metrics["node_classes"],
          "overlapping_classes_2q": metrics.get("overlapping_classes", {}).get("2q"),
          "verdict": v}


def compute_census(capture: dict, aligned: dict, trace: dict, route: dict) -> dict:
  logical = capture["logical"]
  physical = capture["physical"]
  durs = aligned["duration_by_call"]
  lg_idx = {n["call_index"]: i for i, n in enumerate(logical["nodes"])}
  groups: dict[str, dict] = {}
  cross_group: list[dict] = []
  for e in logical["edges"]:
    fn = logical["nodes"][lg_idx[e["from"]]]
    tn = logical["nodes"][lg_idx[e["to"]]]
    if fn["group_id"] != tn["group_id"]:
      cross_group.append({"from": e["from"], "to": e["to"], "kind": e["kind"],
                          "from_group": fn["group_id"], "to_group": tn["group_id"]})
  arena_by_key: dict[str, dict] = {}
  for a in logical.get("arena_manifest", []):
    arena_by_key[a["key"]] = {"logical_ordinal": a["ordinal"], "nbytes": a["nbytes"], "logical_buffers": []}
  for p in logical.get("placements", []):
    for a in logical.get("arena_manifest", []):
      if a["ordinal"] == p["arena"]:
        key = a["key"]
        if p["logical_buffer"] not in arena_by_key[key]["logical_buffers"]:
          arena_by_key[key]["logical_buffers"].append(p["logical_buffer"])
        break
  logical_edge_set = {(e["from"], e["to"]) for e in logical["edges"]}
  logical_edge_by_pair = {(e["from"], e["to"]): e for e in logical["edges"]}
  for g in physical["groups"]:
    gid = str(g["group_id"])
    members = g["nodes"]
    local_nodes = [{"call_index": m["call_index"], "group_id": gid, "name": m["name"],
                    "op": m["op"], "outs": m["outs"], "ins": m["ins"],
                    "global_size": m["global_size"], "local_size": m["local_size"],
                    "sig": m["sig"], "dep_unknown": m["dep_unknown"]} for m in members]
    l_idx = {m["call_index"]: i for i, m in enumerate(members)}
    # logical restriction to this group (edges whose endpoints are both members)
    l_edges = [e for e in logical["edges"]
               if e["from"] in l_idx and e["to"] in l_idx]
    durs_g = [float(durs.get(m["call_index"], 0.0)) for m in members]
    # physical edges are in local member-index space; logical edges in call space
    phys_nodes = [dict(n, call_index=m["member_index"]) for n, m in zip(local_nodes, members)]
    phys = compute_view_census(phys_nodes, {i: durs_g[i] for i in range(len(members))},
                               g["edges"], {i: i for i in range(len(members))}, "physical")
    logic = compute_view_census(local_nodes, durs, l_edges, l_idx, "logical")
    member_by_call = {m["call_index"]: m["member_index"] for m in members}
    attributed = []
    for e in g["edges"]:
      f_call = members[e["from"]]["call_index"]
      t_call = members[e["to"]]["call_index"]
      if (f_call, t_call) in logical_edge_set:
        source = "SEMANTIC"
        lgbufs = logical_edge_by_pair[(f_call, t_call)].get("logical_buffers", [])
      else:
        source = "PLANNER_ALIAS"
        lgbufs = []
      if members[e["from"]]["dep_unknown"] or members[e["to"]]["dep_unknown"]:
        source = UNKNOWN
    attributed.append({"from": e["from"], "to": e["to"], "kind": e["kind"], "source": source,
                       "arena": e.get("arena", []), "ranges": e.get("ranges", []),
                       "logical_buffers": lgbufs})
    sim = _sim_nodes(phys_nodes, {i: durs_g[i] for i in range(len(members))},
                     g["edges"], {i: i for i in range(len(members))})
    est = _sim.compute_est(sim) if sim else []
    ranked = []
    for e in attributed:
      if e["source"] != "PLANNER_ALIAS":
        continue
      slack = est[e["to"]] - (est[e["from"]] + durs_g[e["from"]]) if sim else 0.0
      ranked.append({"from": e["from"], "to": e["to"], "kind": e["kind"], "arena": e["arena"],
                     "ranges": e["ranges"], "logical_buffers": e["logical_buffers"],
                     "consumer_duration_us": round(durs_g[e["to"]], 3),
                     "on_critical_path": abs(slack) < 1e-6,
                     "slack_us": round(slack, 3)})
    ranked.sort(key=lambda r: (not r["on_critical_path"], -r["consumer_duration_us"], r["from"]))
    arena_ledger: dict[int, dict] = {}
    arena_ordinal_to_key = {a["ordinal"]: a["key"] for a in g.get("arena_manifest", [])}
    for r in ranked:
      for a_ord in r["arena"]:
        key = arena_ordinal_to_key.get(a_ord)
        if key is None or key not in arena_by_key:
          continue
        rec = arena_ledger.setdefault(a_ord, {"arena_key": key, "nbytes": arena_by_key[key]["nbytes"],
                                              "logical_buffers": list(arena_by_key[key]["logical_buffers"]),
                                              "on_critical_planner_edges": 0, "critical_consumer_us": 0.0})
        if r["on_critical_path"]:
          rec["on_critical_planner_edges"] += 1
          rec["critical_consumer_us"] = round(rec["critical_consumer_us"] + r["consumer_duration_us"], 3)
    groups[gid] = {"group_id": gid, "size": g["size"],
                   "physical": phys, "logical": logic,
                   "physical_preds_match_runtime": None,
                   "trace_span_us": None,
                   "attributed_edges": attributed,
                   "edge_sources": {"SEMANTIC": sum(1 for e in attributed if e["source"] == "SEMANTIC"),
                                    "PLANNER_ALIAS": sum(1 for e in attributed if e["source"] == "PLANNER_ALIAS"),
                                    UNKNOWN: sum(1 for e in attributed if e["source"] == UNKNOWN)},
                   "arena_manifest": g.get("arena_manifest", []),
                   "planner_added_edges_ranked": ranked[:20],
                   "planner_added_on_critical_path": sum(1 for r in ranked if r["on_critical_path"]),
                   "planner_added_cp_time_us": round(
                     sum(r["consumer_duration_us"] for r in ranked if r["on_critical_path"]), 3),
                   "arena_recovery_ledger": sorted(arena_ledger.values(),
                                                   key=lambda x: -x["critical_consumer_us"])[:10]}
    if "groups" in trace:
      tg = trace["groups"].get(str(g["size"]))
      if tg is not None:
        groups[gid]["trace_span_us"] = round(sum(tg["durations_us"]), 3)
  # whole-token logical with cross-group edges as real deps
  whole_logical = compute_view_census(logical["nodes"], durs, logical["edges"], lg_idx, "logical_whole_token")
  # operative whole-token: groups launch serially on the null stream, so the
  # realistic overlap budget is the SUM of per-group spans (not the merged DAG).
  sum_phys_serial = sum(groups[gid]["physical"]["serialized_us"] for gid in groups)
  sum_phys_cp = sum(groups[gid]["physical"]["critical_path_us"] for gid in groups)
  sum_phys_2q = sum(groups[gid]["physical"]["schedule_2q_us"] for gid in groups)
  sum_log_serial = sum(groups[gid]["logical"]["serialized_us"] for gid in groups)
  sum_log_cp = sum(groups[gid]["logical"]["critical_path_us"] for gid in groups)
  sum_log_2q = sum(groups[gid]["logical"]["schedule_2q_us"] for gid in groups)
  sum_log_3q = sum(groups[gid]["logical"]["schedule_3q_us"] for gid in groups)
  planner_delta_cp_us = round(sum_phys_cp - sum_log_cp, 3)
  whole_token = {
    "logical_merged": whole_logical,
    "physical_sum_serialized_us": round(sum_phys_serial, 3),
    "physical_sum_cp_us": round(sum_phys_cp, 3),
    "physical_sum_2q_us": round(sum_phys_2q, 3),
    "logical_sum_serialized_us": round(sum_log_serial, 3),
    "logical_sum_cp_us": round(sum_log_cp, 3),
    "logical_sum_2q_us": round(sum_log_2q, 3),
    "logical_sum_3q_us": round(sum_log_3q, 3),
    "planner_delta_cp_us": planner_delta_cp_us,
    "operative_physical_2q_saving_pct": round((sum_phys_serial - sum_phys_2q) / sum_phys_serial * 100, 3) if sum_phys_serial else 0.0,
    "operative_logical_2q_saving_pct": round((sum_log_serial - sum_log_2q) / sum_log_serial * 100, 3) if sum_log_serial else 0.0,
    "operative_logical_3q_saving_pct": round((sum_log_serial - sum_log_3q) / sum_log_serial * 100, 3) if sum_log_serial else 0.0,
  }
  wall_us = None
  if route.get("cuda_wall_ms_D"):
    wall_us = float(route["cuda_wall_ms_D"]) * 1000.0
    wall_source = "fresh harness D row"
  elif route.get("historical_cuda_wall_ms"):
    wall_us = float(route["historical_cuda_wall_ms"]) * 1000.0
    wall_source = "historical B0.2 anchor (6.3319 ms)"
  if wall_us:
    pct = planner_delta_cp_us / wall_us * 100.0
    if pct < 5.0:
      scale = "NOT_MECHANISM_SCALE"
    elif planner_delta_cp_us < 705.1:
      scale = "MECHANISM_SCALE_ONLY"
    elif planner_delta_cp_us < 705.1 + 1567.0:
      scale = "ROUTE_TAX_SCALE"
    else:
      scale = "PARITY_SCALE_THEORETICAL"
    whole_token["planner_delta_pct_of_cuda_wall"] = round(pct, 3)
    whole_token["scale_wall_source"] = wall_source
    whole_token["scale_classification"] = scale
  cross_kinds = {}
  for e in cross_group:
    cross_kinds[e["kind"]] = cross_kinds.get(e["kind"], 0) + 1
  unknown_total = len(logical["unknown_dep_nodes"]) + sum(
    len(g["unknown_dep_members"]) for g in physical["groups"])
  logical_par = whole_token["operative_logical_2q_saving_pct"]
  physical_par = whole_token["operative_physical_2q_saving_pct"]
  if unknown_total:
    attribution_verdict = "ATTRIBUTION_CONFOUNDED"
  elif logical_par < 5.0:
    attribution_verdict = "SEMANTIC_CHAIN / OVERLAP_LEVER_CLOSED"
  elif physical_par >= 5.0:
    attribution_verdict = "PLANNER_NOT_ROOT_CAUSE"
  elif whole_token.get("scale_classification") == "NOT_MECHANISM_SCALE":
    attribution_verdict = "PLANNER_EFFECT_NOT_SCALE"
  elif whole_token.get("scale_classification") == "MECHANISM_SCALE_ONLY":
    attribution_verdict = "PLANNER_CANDIDATE (mechanism scale, below route tax)"
  elif whole_token.get("scale_classification") in ("ROUTE_TAX_SCALE", "PARITY_SCALE_THEORETICAL"):
    attribution_verdict = "PLANNER_CANDIDATE"
  else:
    attribution_verdict = "PLANNER_CANDIDATE (unclassified scale)"
  resource_pairs = []
  char = {"flash": "compute", "gemv": "bandwidth", "kv": "bandwidth",
          "rmsnorm": "light", "residual": "light", "scatter": "light", "other": "unknown"}
  for gid in sorted(groups):
    pairs = groups[gid]["physical"]["overlapping_classes_2q"] or []
    for a, b, count in pairs:
      resource_pairs.append({"group": gid, "classes": [a, b], "count": count,
                             "character": [char.get(a, "unknown"), char.get(b, "unknown")]})
  return {"schema": SCHEMA_CENSUS, "route": route, "capture_schema": capture.get("schema"),
          "selected_group_sizes": _capture_group_sizes(capture),
          "logical": {"node_count": len(logical["nodes"]), "edge_count": len(logical["edges"]),
                      "edge_kinds": _edge_kind_counts(logical["edges"]),
                      "unknown_dep_nodes": logical["unknown_dep_nodes"]},
          "physical": {"group_count": len(physical["groups"]),
                       "group_sizes": [g["size"] for g in physical["groups"]],
                       "unknown_dep_members": sum(len(g["unknown_dep_members"]) for g in physical["groups"])},
          "per_group": groups,
          "whole_token": whole_token,
          "attribution_verdict": attribution_verdict,
          "resource_pairs_2q": resource_pairs,
          "cross_group": {"edge_count": len(cross_group), "by_kind": cross_kinds,
                          "note": "cross-group launches serialize on the null stream; no physical cross-group edges exist"},
          "alignment": aligned["alignment"],
          "trace": {"source": trace.get("source"), "groups": {k: {"graph_id": v["graph_id"], "size": v["size"],
                                                                 "replays": v["replays"]}
                                                             for k, v in trace.get("groups", {}).items()}},
          "evidence": {"edge_structures": "OBSERVED (seam mirror of CUDAGraph._access_resources; runtime-verified when physical_preds_match_runtime)",
                       "durations": "OBSERVED (CUPTI kernel rows, median over steady-state replays)",
                       "alignment": "OBSERVED positional when aligned; INFERRED when mismatches",
                       "edge_attribution": "SEMANTIC from aligned logical view; PLANNER_ALIAS otherwise; UNKNOWN on unresolvable endpoints",
                       "verdicts": "INFERRED (duration-weighted policy, >=5% 2-queue saving bar)",
                       "scale_classification": "DERIVED (planner_delta_cp vs fresh/anchored CUDA wall; route-tax anchors 705.1/1567.0 us historical)"}}


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def format_report(census: dict, capture_path: str, trace_path: str) -> str:
  lines = ["== B3.1 aligned logical/physical CUDA decode DAG census =="]
  lines.append("route: DEV=%s CUDA_GRAPH_STREAMS=%s | commit %s | driver %s | model %s" % (
    census["route"].get("DEV"), census["route"].get("CUDA_GRAPH_STREAMS"), census["route"].get("commit"),
    census["route"].get("driver"), census["route"].get("model")))
  lines.append("capture: %s | trace: %s" % (capture_path, trace_path))
  lines.append("group sizes: %s | logical nodes: %d | logical edges: %d (%s) | unknown deps: %d (must be 0)"
               % (census["selected_group_sizes"], census["logical"]["node_count"], census["logical"]["edge_count"],
                  census["logical"]["edge_kinds"], len(census["logical"]["unknown_dep_nodes"])))
  lines.append("physical: %d groups, sizes %s, unknown: %d" % (
    census["physical"]["group_count"], census["physical"]["group_sizes"], census["physical"]["unknown_dep_members"]))
  lines.append("")
  lines.append("per-group (logical | physical):")
  for gid in sorted(census["per_group"], key=lambda x: (isinstance(census["per_group"][x]["group_id"], str),
                                                        census["per_group"][x]["group_id"])):
    g = census["per_group"][gid]
    for view in ("logical", "physical"):
      m = g[view]
      lines.append("  group %s %-8s n=%-4d edges=%-4d edges/call=%.3f chain=%d/%-4d roots=%d leaves=%d width2q=%d "
                   "| serial=%.1fus cp=%.1fus 2q=%.1fus(%+.1f%%) | %s" % (
        gid, view, m["node_count"], m["edge_count"], m["edges_per_call"], m["longest_chain_nodes"],
        m["node_count"], m["roots"], m["leaves"], m["independent_branches_2q"], m["serialized_us"],
        m["critical_path_us"], m["schedule_2q_us"], m["verdict"]["sched_2q_saving_pct"], m["verdict"]["verdict"]))
    es = g["edge_sources"]
    lines.append("      runtime-verified preds: %s | edge sources: SEMANTIC %d, PLANNER_ALIAS %d, UNKNOWN %d | "
                 "planner-added on critical path: %d (%.1fus of consumer durations)" % (
                   "match" if g["physical_preds_match_runtime"] else "n/a",
                   es["SEMANTIC"], es["PLANNER_ALIAS"], es[UNKNOWN],
                   g["planner_added_on_critical_path"], g["planner_added_cp_time_us"]))
  lines.append("")
  lines.append("whole token (groups serialize on the null stream; overlap only WITHIN a group):")
  wt = census["whole_token"]
  lines.append("  physical sum: serialized %.1fus, 2-queue %.1fus (saving %.2f%%)" % (
    wt["physical_sum_serialized_us"], wt["physical_sum_2q_us"], wt["operative_physical_2q_saving_pct"]))
  lines.append("  logical sum:  serialized %.1fus, 2-queue %.1fus (saving %.2f%%), 3-queue %.1fus (saving %.2f%%)" % (
    wt["logical_sum_serialized_us"], wt["logical_sum_2q_us"], wt["operative_logical_2q_saving_pct"],
    wt["logical_sum_3q_us"], wt["operative_logical_3q_saving_pct"]))
  lm = wt["logical_merged"]
  lines.append("  logical merged (cross-group edges as real deps, hypothetical scheduler): serial %.1fus cp %.1fus "
               "2q %.1fus | verdict %s" % (lm["serialized_us"], lm["critical_path_us"], lm["schedule_2q_us"],
                                           lm["verdict"]["verdict"]))
  lines.append("  cross-group logical edges: %d %s" % (census["cross_group"]["edge_count"],
                                                       census["cross_group"]["by_kind"]))
  lines.append("  planner delta CP (physical-logical): %+.1fus (%s vs wall) | scale: %s | attribution: %s" % (
    wt.get("planner_delta_cp_us", 0.0),
    "%s" % (wt.get("planner_delta_pct_of_cuda_wall"), ) if "planner_delta_pct_of_cuda_wall" in wt else "n/a",
    wt.get("scale_classification", "n/a (no wall)"), census["attribution_verdict"]))
  if census["resource_pairs_2q"]:
    lines.append("  resource pairs (2-queue schedule, physical): %s" % (
      "; ".join("%s+%s x%d (%s)" % (r["classes"][0], r["classes"][1], r["count"],
                                    "+".join(r["character"])) for r in census["resource_pairs_2q"][:8])))
  lines.append("")
  lines.append("evidence classes:")
  lines.append("  OBSERVED: DAG edge structures (seam mirror + runtime _access_resources verifier), CUPTI durations, "
               "group sizes/kernel counts, positional alignment when mismatched_positions=0.")
  lines.append("  INFERRED: shape verdicts (duration-weighted >=5%% 2-queue bar), 2q/3q schedules as overlap potential "
               "(driver may not realize it), cross-group launch serialization.")
  lines.append("  DERIVED: planner_delta_cp, scale classification vs wall, planner-added edge ranking (tight-edge slack).")
  return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def main_capture(args: argparse.Namespace) -> int:
  import tinygrad.engine.jit as tjit
  from tinygrad.helpers import DEV  # noqa: F401  (DEV resolution is env-driven)
  state: dict = {"input_uops": None, "pre_plan_linear": None, "captures": [],
                 "cudagraph_instances": {}, "cudagraph_order": []}
  restore_seam = install_seam(state)
  restore_verifier = install_cudagraph_verifier(state)
  harness_out = str(pathlib.Path(args.out).with_suffix("")) + ".harness.json"
  try:
    import decode_runtime_overhead as dro
    argv = ["--model", args.model, "--ckpts", str(args.depth), "--max-context", str(args.max_context),
            "--nmeas", str(args.nmeas), "--reps", str(args.reps), "--warmup-decode", str(args.warmup_decode),
            "--chunk-size", str(args.chunk_size), "--out", harness_out]
    dro.main(argv)
  finally:
    restore_verifier()
    restore_seam()
  cap = select_decode_capture(state["captures"])
  if cap is None:
    sys.stderr.write("cuda_route_aligned_census: no decode capture found (seam fired %d times)\n" % len(state["captures"]))
    return 1
  route = {"DEV": os.environ.get("DEV", "?"), "CUDA_GRAPH_STREAMS": os.environ.get("CUDA_GRAPH_STREAMS", "1"),
           "commit": _git_commit(), "driver": _driver_version(), "model": args.model,
           "depth": args.depth, "nmeas": args.nmeas, "reps": args.reps, "warmup_decode": args.warmup_decode,
           "historical_cuda_wall_ms": 6.3319}
  try:
    with open(harness_out, encoding="utf-8") as f:
      hrows = json.load(f).get("rows") or []
    if hrows:
      route["cuda_wall_ms_D"] = hrows[0].get("wall_ms_D")
      route["cuda_wall_ms_W"] = hrows[0].get("wall_ms_W")
      route["tok_s_D"] = hrows[0].get("tok_s_D_diagnostic")
      route["harness_routes"] = hrows[0].get("routes")
  except Exception:
    pass
  payload = {"schema": SCHEMA_CAPTURE, "route": route,
             "seam_fired": len(state["captures"]),
             "selected_group_sizes": _capture_group_sizes(cap),
             "capture": cap,
             "runtime_preds": runtime_preds_report(state, _capture_group_sizes(cap))}
  # verify the mirror against the runtime-frozen preds
  if payload["runtime_preds"].get("matched"):
    for g in payload["runtime_preds"]["groups"]:
      gid = str(g["group_id"])
      mirror = cap["physical"]["groups"][g["group_id"]]
      mirror_preds = [[] for _ in range(len(mirror["nodes"]))]
      for e in mirror["edges"]:
        mirror_preds[e["to"]].append(e["from"])
      mirror_preds = [sorted(x) for x in mirror_preds]
      g["matches_mirror"] = mirror_preds == g["preds"]
      for pg in cap["physical"]["groups"]:
        if str(pg["group_id"]) == gid:
          pg["matches_runtime"] = g["matches_mirror"]
  _atomic_json(args.out, payload)
  sys.stdout.write("== B3.1 CUDA aligned census: --capture ==\n")
  sys.stdout.write("seam fired %d times; selected capture with group sizes %s\n"
                   % (len(state["captures"]), _capture_group_sizes(cap)))
  sys.stdout.write("logical nodes %d edges %d; physical groups %d\n" % (
    len(cap["logical"]["nodes"]), len(cap["logical"]["edges"]), len(cap["physical"]["groups"])))
  sys.stdout.write("runtime preds matched: %s\n" % payload["runtime_preds"].get("matched"))
  sys.stdout.write("wrote %s\n" % args.out)
  return 0


def main_analyze(args: argparse.Namespace) -> int:
  with open(args.capture, encoding="utf-8") as f:
    payload = json.load(f)
  capture = payload["capture"]
  trace = load_trace_durations(args.trace, _capture_group_sizes(capture))
  if not trace["groups"]:
    sys.stderr.write("cuda_route_aligned_census: no usable decode groups in trace %s\n" % args.trace)
    return 1
  aligned = align_capture(capture, trace)
  census = compute_census(capture, aligned, trace, payload["route"])
  # carry runtime verification into the census per-group rows
  if payload.get("runtime_preds", {}).get("matched"):
    for g in payload["runtime_preds"]["groups"]:
      gid = str(g["group_id"])
      if gid in census["per_group"]:
        census["per_group"][gid]["physical_preds_match_runtime"] = bool(g.get("matches_mirror"))
  _atomic_json(args.out, census)
  sys.stdout.write(format_report(census, args.capture, args.trace))
  sys.stdout.write("wrote %s\n" % args.out)
  return 0


def run_selftest(out: str | None) -> dict:
  """Hermetic checks: edge-kind labeling and the census math on a known DAG."""
  from tinygrad.dtype import dtypes
  from tinygrad.device import Buffer

  # 1. RecordingDepsTracker kinds on a tiny buffer script
  tracker = RecordingDepsTracker()
  b0 = Buffer("CPU", 4, dtypes.float32)
  b1 = Buffer("CPU", 4, dtypes.float32)
  b2 = Buffer("CPU", 4, dtypes.float32)
  # call 0: writes b0
  tracker.access_resources([b0], [0], 0)
  # call 1: reads b0, writes b1
  tracker.access_resources([b0, b1], [1], 1)
  # call 2: writes b0 (over the earlier write), reads b1
  tracker.access_resources([b0, b1], [0], 2)
  kinds = {(f, t): k for f, t, k in tracker.edges}
  assert kinds.get((0, 1)) == "RAW", kinds
  assert kinds.get((0, 2)) == "WAW", kinds
  assert kinds.get((1, 2)) == "WAR", kinds
  assert (1, 2) not in {e[:2] for e in tracker.edges if e[2] == "RAW"}

  # 2. census math on the known 7-node 2-group DAG
  dag = {
    "logical": {
      "nodes": [
        {"call_index": 0, "group_id": 0, "name": "a_w0", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_w0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 1, "group_id": 0, "name": "a_r1", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_r1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 2, "group_id": 0, "name": "a_w2", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_w2", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 3, "group_id": 1, "name": "b_r3", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r3", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 4, "group_id": 1, "name": "b_r4", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r4", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 5, "group_id": 1, "name": "b_w5", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_w5", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 6, "group_id": 1, "name": "b_r6", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r6", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
      ],
      "edges": [
        {"from": 0, "to": 1, "kind": "RAW"}, {"from": 0, "to": 2, "kind": "WAW"},
        {"from": 0, "to": 5, "kind": "WAW"}, {"from": 1, "to": 3, "kind": "RAW"},
        {"from": 1, "to": 4, "kind": "RAW"}, {"from": 1, "to": 5, "kind": "WAR"},
        {"from": 1, "to": 6, "kind": "RAW"}, {"from": 3, "to": 6, "kind": "RAW"},
      ],
      "unknown_dep_nodes": [],
    },
    "physical": {
      "groups": [
        {"group_id": 0, "size": 3, "nodes": [
          {"member_index": 0, "call_index": 0, "name": "a_w0", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_w0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 1, "call_index": 1, "name": "a_r1", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_r1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 2, "call_index": 2, "name": "a_w2", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a_w2", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        ], "edges": [{"from": 0, "to": 1, "kind": "RAW"}, {"from": 0, "to": 2, "kind": "WAW"}],
          "unknown_dep_members": []},
        {"group_id": 1, "size": 4, "nodes": [
          {"member_index": 0, "call_index": 3, "name": "b_r3", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r3", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 1, "call_index": 4, "name": "b_r4", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r4", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 2, "call_index": 5, "name": "b_w5", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_w5", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 3, "call_index": 6, "name": "b_r6", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b_r6", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        ], "edges": [{"from": 0, "to": 3, "kind": "RAW"}, {"from": 2, "to": 3, "kind": "WAR"}],
          "unknown_dep_members": []},
      ],
    },
  }
  durs = {0: 10.0, 1: 20.0, 2: 15.0, 3: 8.0, 4: 12.0, 5: 6.0, 6: 10.0}
  aligned = {"duration_by_call": durs, "physical_durations": {"0": [10.0, 20.0, 15.0], "1": [8.0, 12.0, 6.0, 10.0]},
             "alignment": [], "aligned_nodes": 0, "total_groups": 2}
  census = compute_census(capture={"schema": SCHEMA_CAPTURE, "logical": dag["logical"], "physical": dag["physical"]},
                          aligned=aligned, trace={"source": "synthetic", "groups": {}},
                          route={"DEV": "CUDA", "CUDA_GRAPH_STREAMS": "1", "commit": "selftest", "driver": "selftest",
                                 "model": "synthetic"})
  assert census["logical"]["node_count"] == 7 and census["logical"]["edge_count"] == 8
  g0 = census["per_group"]["0"]["logical"]
  assert g0["serialized_us"] == 45.0 and g0["critical_path_us"] == 30.0, g0
  assert g0["longest_chain_nodes"] == 3 and g0["node_count"] == 3
  wt = census["whole_token"]
  assert wt["logical_sum_serialized_us"] == 81.0, wt
  assert wt["logical_sum_2q_us"] == 56.0, wt
  assert wt["logical_sum_3q_us"] == 48.0, wt
  assert census["cross_group"]["edge_count"] == 5, census["cross_group"]
  assert census["cross_group"]["by_kind"] == {"RAW": 3, "WAR": 1, "WAW": 1}
  # attribution: group 1 physical has (5,6) WAR absent logically -> PLANNER_ALIAS
  es1 = census["per_group"]["1"]["edge_sources"]
  assert es1 == {"SEMANTIC": 1, "PLANNER_ALIAS": 1, UNKNOWN: 0}, es1
  es0 = census["per_group"]["0"]["edge_sources"]
  assert es0 == {"SEMANTIC": 2, "PLANNER_ALIAS": 0, UNKNOWN: 0}, es0
  ranked = census["per_group"]["1"]["planner_added_edges_ranked"]
  assert ranked and ranked[0]["from"] == 5 and ranked[0]["to"] == 6 and ranked[0]["kind"] == "WAR"

  # 3. two independent logical chains that planning collapses into one physical
  # chain (the B2 probe scenario): planner-added WAW/RAW edges must be classified
  # PLANNER_ALIAS and the physical view must be chain-shaped while logical is not.
  dag2 = {
    "logical": {
      "nodes": [
        {"call_index": 0, "group_id": 0, "name": "a0", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 1, "group_id": 0, "name": "a1", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 2, "group_id": 0, "name": "b0", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        {"call_index": 3, "group_id": 0, "name": "b1", "op": "PROGRAM", "outs": (0,), "ins": (),
         "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
      ],
      "edges": [{"from": 0, "to": 1, "kind": "RAW"}, {"from": 2, "to": 3, "kind": "RAW"}],
      "unknown_dep_nodes": [],
    },
    "physical": {
      "groups": [
        {"group_id": 0, "size": 4, "nodes": [
          {"member_index": 0, "call_index": 0, "name": "a0", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 1, "call_index": 1, "name": "a1", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("a1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 2, "call_index": 2, "name": "b0", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b0", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
          {"member_index": 3, "call_index": 3, "name": "b1", "op": "PROGRAM", "outs": (0,), "ins": (),
           "global_size": (1, 1, 1), "local_size": (1, 1, 1), "sig": ("b1", 1, 1, 1, 1, 1, 1), "dep_unknown": False},
        ], "edges": [{"from": 0, "to": 1, "kind": "RAW"}, {"from": 0, "to": 2, "kind": "WAW"},
                     {"from": 1, "to": 2, "kind": "WAW"}, {"from": 2, "to": 3, "kind": "RAW"}],
          "unknown_dep_members": []},
      ],
    },
  }
  durs2 = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}
  aligned2 = {"duration_by_call": durs2, "physical_durations": {"0": [10.0, 20.0, 30.0, 40.0]},
              "alignment": [], "aligned_nodes": 0, "total_groups": 1}
  census2 = compute_census(capture={"schema": SCHEMA_CAPTURE, "logical": dag2["logical"], "physical": dag2["physical"]},
                           aligned=aligned2, trace={"source": "synthetic", "groups": {}},
                           route={"DEV": "CUDA", "CUDA_GRAPH_STREAMS": "1", "commit": "selftest",
                                  "driver": "selftest", "model": "synthetic"})
  g0l = census2["per_group"]["0"]["logical"]
  g0p = census2["per_group"]["0"]["physical"]
  assert g0l["roots"] == 2 and g0l["verdict"]["verdict"] == "INDEPENDENT", g0l
  assert g0p["chain_ratio"] == 1.0 and g0p["verdict"]["verdict"] == "CHAIN_SHAPED", g0p
  assert census2["per_group"]["0"]["edge_sources"] == {"SEMANTIC": 2, "PLANNER_ALIAS": 2, UNKNOWN: 0}
  assert census2["per_group"]["0"]["planner_added_on_critical_path"] >= 1
  if out is not None:
    _atomic_json(out, census)
  return census


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument("--capture", action="store_true", help="live GPU capture (run under flock)")
  mode.add_argument("--analyze", action="store_true", help="CPU-only census from capture + trace")
  mode.add_argument("--selftest", action="store_true", help="hermetic CPU self-test")
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--nmeas", type=int, default=2)
  ap.add_argument("--reps", type=int, default=1)
  ap.add_argument("--warmup-decode", type=int, default=3)
  ap.add_argument("--chunk-size", type=int, default=32)
  ap.add_argument("--out", default="/tmp/b3_cuda_route_aligned_census.json")
  ap.add_argument("--trace", default=DEFAULT_TRACE)
  ap.add_argument("--capture-json", default="/tmp/b3_cuda_route_aligned_capture.json",
                  help="capture JSON for --analyze")
  args = ap.parse_args(argv)
  try:
    if args.capture:
      args.out = args.out or "/tmp/b3_cuda_route_aligned_capture.json"
      return main_capture(args)
    if args.analyze:
      args.capture = args.capture_json
      return main_analyze(args)
    if args.selftest:
      census = run_selftest(args.out)
      sys.stdout.write("== B3.1 aligned census selftest: PASS ==\n")
      sys.stdout.write("logical n=%d edges=%d cross=%d | sum serialized %.1f 2q %.1f 3q %.1f\n" % (
        census["logical"]["node_count"], census["logical"]["edge_count"], census["cross_group"]["edge_count"],
        census["whole_token"]["logical_sum_serialized_us"], census["whole_token"]["logical_sum_2q_us"],
        census["whole_token"]["logical_sum_3q_us"]))
      return 0
  except Exception as exc:
    sys.stderr.write("cuda_route_aligned_census: %s\n" % exc)
    return 1
  return 1


if __name__ == "__main__":
  sys.exit(main())
