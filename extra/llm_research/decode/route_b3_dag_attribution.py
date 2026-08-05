#!/usr/bin/env python3
"""Route B3.1: aligned logical/physical decode DAG attribution tooling (CPU).

Implements the B3 exhaustive execution scope section 8 (aligned
logical/physical DAG tooling). It answers, for a fixed call sequence:

- what the logical dependency DAG is (one range-aware tracker over logical
  buffer ranges, planner reuse disabled);
- what the physical dependency DAG is (same calls over planner arena ranges,
  from the placement manifest);
- which physical edges are SEMANTIC (already present logically) versus
  PLANNER_ALIAS (introduced by arena reuse), with exact arena/range/buffer
  attribution;
- DAG metrics (serialized span, duration-weighted critical path, 2/3-queue
  schedules, per-group rows) for both arms plus the planner-added delta.

Everything here is CPU-only and hermetic. No GPU is required. The live capture
seam (B3.2) will feed this tool real decode arms; this module's core operates
on the small CallRecord model so the attribution logic is testable without
constructing UOps.

Edge semantics deliberately mirror `full_token_dag_capture.RecordingDepsTracker`
(which delegates to `tinygrad.engine.jit.DepsTracker`): range-aware RAW/WAR/WAW
with one edge per (dep, new) pair and first-kind-wins priority. A parity test
against RecordingDepsTracker is part of the hermetic suite.

Usage:
  python3 extra/llm_research/decode/route_b3_dag_attribution.py --synthetic
"""
from __future__ import annotations

import argparse, contextlib, hashlib, json, os, pathlib, sys
from dataclasses import dataclass, field
from typing import Any

from extra.llm_research.decode.full_token_dag_capture import (
  FullTokenDagError, RecordingDepsTracker, attach_summary, restrict_dag, to_sim_nodes, validate_schema,
)

SCHEMA = "tinygrad.route_b3.dag_attribution.v1"
KINDS = ("RAW", "WAR", "WAW")
UNKNOWN = "UNKNOWN"
SEMANTIC = "SEMANTIC"
PLANNER_ALIAS = "PLANNER_ALIAS"


class B3AttributionError(ValueError):
  pass


DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


@dataclass
class CallAccess:
  """One buffer access within a call, on logical ranges."""
  buf: str
  offset: int
  nbytes: int
  write: bool
  logical_bufs: list[str] | None = None


@dataclass
class CallRecord:
  """Offline call model: stable identity fields plus range accesses."""
  index: int
  name: str
  accesses: list[CallAccess]
  duration_us: float = 0.0
  group: str | int | None = None
  identity: dict[str, Any] = field(default_factory=dict)
  unknown: bool = False

def attach_compiled_descriptors(calls:list[CallRecord], descriptors:dict[int,dict[str,Any]]) -> list[dict[str,Any]]:
  """Observational occurrence join; rejects missing/extra/partial descriptors."""
  required={"binary_sha256","grid","block","registers_per_thread","static_smem_bytes","dynamic_smem_bytes","local_mem_bytes"}
  ids={c.index for c in calls}
  if set(descriptors) != ids: raise B3AttributionError("compiled descriptor occurrence set does not exactly match calls")
  out=[]
  for c in calls:
    d=descriptors[c.index]
    if set(d) != required or not isinstance(d["binary_sha256"],str) or len(d["binary_sha256"]) != 64:
      raise B3AttributionError(f"call {c.index}: malformed compiled descriptor")
    if not all(isinstance(d[k],list) and len(d[k]) == 3 for k in ("grid","block")) or not all(isinstance(d[k],int) and d[k] >= 0 for k in required-{"binary_sha256","grid","block"}):
      raise B3AttributionError(f"call {c.index}: incomplete compiled resource tuple")
    out.append({"index":c.index,"name":c.name,"identity":c.identity,"descriptor":d})
  return out


# ---------------------------------------------------------------------------
# Planner manifest collection (live seam; exercised hermetically)
# ---------------------------------------------------------------------------

class PlannerManifestCollector:
  """Collector installed via tinygrad.schedule.memory._memory_manifest_collectors.

  The collector receives the placement evidence after planning, before buffer
  substitution, so its presence cannot change the planned linear. Buffer
  identity is a stable (device, dtype, logical bytes, ordinal) signature, never
  a pointer. With NO_MEMORY_PLANNER the arena view is unobserved and entries
  carry placement=None (dependency observation comes from the logical walk).
  """

  def __init__(self) -> None:
    self.manifest: dict[str, dict[str, Any]] = {}
    self.arena_labels: dict[Any, str] = {}
    self.arena_labels_by_uop_id: dict[int, str] = {}
    self._ordinals: dict[tuple[str, str, int], int] = {}

  def _buf_id(self, buf: Any) -> str:
    dev = buf.device if isinstance(buf.device, str) else ",".join(sorted(buf.device))
    key = (dev, str(buf.dtype), int(buf.arg))
    self._ordinals[key] = self._ordinals.get(key, 0) + 1
    return "buf:%s:%s:%d:%d" % (dev, buf.dtype, int(buf.arg), self._ordinals[key])

  def _arena_id(self, key: Any, arenas: dict[Any, Any]) -> str:
    if key in self.arena_labels:
      return self.arena_labels[key]
    # Lane key is (device, copy_flag); only the device participates in the
    # label so arena identities resolve from allocated Buffers (which do not
    # carry the copy flag). Same-device same-size lanes collide and fail
    # closed as UNKNOWN rather than misattributing.
    dev = str(key[0]) if isinstance(key, tuple) else str(key)
    if arenas is None or key not in arenas:
      label = "arena:%s" % dev
    else:
      label = "arena:%s:%d" % (dev, int(arenas[key].arg))
    self.arena_labels[key] = label
    return label

  def __call__(self, linear: Any, held_bufs: set[Any], arenas: dict[Any, Any] | None,
               offsets: dict[Any, int], nbytes: dict[Any, int],
               first: dict[Any, int], last: dict[Any, int]) -> None:
    from tinygrad.uop.ops import Ops
    copy_bufs: set[Any] = set()
    for si in linear.src:
      if si.src[0].op is Ops.COPY:
        copy_bufs.update(_linear_bufs(si))
    for buf, offset in offsets.items():
      bid = self._buf_id(buf)
      key = (buf.device, 1 if buf in copy_bufs else 0)
      placed = arenas is not None and key in arenas
      if placed:
        self.arena_labels_by_uop_id[id(arenas[key])] = self._arena_id(key, arenas)
      self.manifest[bid] = {
        "arena": self._arena_id(key, arenas) if placed else None,
        "offset": int(offset) if placed else None,
        "aligned_nbytes": int(nbytes[buf]),
        "logical_nbytes": int(buf.arg * buf.dtype.itemsize),
        "dtype": str(buf.dtype),
        "device": buf.device if isinstance(buf.device, str) else ",".join(sorted(buf.device)),
        "first_call": int(first.get(buf, -1)),
        "last_call": int(last.get(buf, -1)),
        "held": buf in held_bufs,
        "placement": "observed" if placed else "unobserved",
      }


def _linear_bufs(call: Any) -> list[Any]:
  from tinygrad.uop.ops import Ops
  from tinygrad.schedule.memory import _collect_bufs
  return [b for s in call.src[1:] for b in _collect_bufs(s)]


def collect_manifest(linear: Any, held_bufs: set[Any]) -> PlannerManifestCollector:
  """Run memory planning's collector call once with a fresh collector installed."""
  from tinygrad.schedule import memory as tmem
  collector = PlannerManifestCollector()
  token = tmem._memory_manifest_collectors.set((collector,))
  try:
    tmem.memory_plan_rewrite(linear, held_bufs)
  finally:
    tmem._memory_manifest_collectors.reset(token)
  return collector


# ---------------------------------------------------------------------------
# Range-aware edge builder (logical and physical arms)
# ---------------------------------------------------------------------------

def build_edges(calls: list[CallRecord], manifest: dict[str, dict[str, Any]] | None = None) -> list[dict]:
  """One range-aware edge per (dep, new) pair, kind priority WAW > WAR > RAW.

  With manifest=None the accesses are walked on logical ranges and each edge
  carries the logical buffer ids. With a manifest, accesses are mapped onto
  arena ranges (base = arena label, offset = arena offset + logical offset) and
  each edge carries the logical buffers whose reuse produced it. A buffer
  missing from the manifest marks the call UNKNOWN (fail closed).
  """
  writes: dict[str, list[tuple[int, int, int, str]]] = {}
  reads: dict[str, list[tuple[int, int, int, str]]] = {}
  edges: dict[tuple[int, int], dict[str, Any]] = {}

  def physical(acc: CallAccess) -> tuple[str, int, int] | None:
    entry = (manifest or {}).get(acc.buf)
    if entry is None or entry.get("offset") is None:
      return None
    arena = entry["arena"]
    start = int(entry["offset"]) + acc.offset
    return (arena, start, start + acc.nbytes)

  for call in calls:
    unknown = call.unknown
    for acc in call.accesses:
      if manifest is not None:
        pr = physical(acc)
        if pr is None:
          unknown = True
          continue
        base, start, end = pr
      else:
        base, start, end = acc.buf, acc.offset, acc.offset + acc.nbytes
      if acc.write:
        for ws, we, dep, prior_buf, prior_logical in writes.get(base, []):
          if ws < end and start < we:
            edge = edges.setdefault((dep, call.index), {"from": dep, "to": call.index, "kind": "WAW", "overlaps": []})
            edge["overlaps"].append({"base": base, "range": [max(ws, start), min(we, end)],
                                     "logical_bufs": sorted(set(prior_logical) | set(acc.logical_bufs or [acc.buf]))})
        for rs, re, dep, prior_buf, prior_logical in reads.get(base, []):
          if rs < end and start < re:
            edge = edges.setdefault((dep, call.index), {"from": dep, "to": call.index, "kind": "WAR", "overlaps": []})
            edge["overlaps"].append({"base": base, "range": [max(rs, start), min(re, end)],
                                     "logical_bufs": sorted(set(prior_logical) | set(acc.logical_bufs or [acc.buf]))})
        writes.setdefault(base, []).append((start, end, call.index, acc.buf, acc.logical_bufs or [acc.buf]))
      else:
        for ws, we, dep, prior_buf, prior_logical in writes.get(base, []):
          if ws < end and start < we:
            edge = edges.setdefault((dep, call.index), {"from": dep, "to": call.index, "kind": "RAW", "overlaps": []})
            edge["overlaps"].append({"base": base, "range": [max(ws, start), min(we, end)],
                                     "logical_bufs": sorted(set(prior_logical) | set(acc.logical_bufs or [acc.buf]))})
        reads.setdefault(base, []).append((start, end, call.index, acc.buf, acc.logical_bufs or [acc.buf]))
    if unknown:
      calls[call.index].unknown = True
  return list(edges.values())


def to_dag(calls: list[CallRecord], edges: list[dict]) -> dict:
  """Assemble a full_token_dag.v1 object with stable identities attached."""
  nodes = []
  for call in calls:
    identity = {
      "ordered_call_index": call.index,
      "operation_kind": call.name,
      "program_identity": call.identity.get("program_identity", call.name),
      "outputs": call.identity.get("outputs"),
      "inputs": call.identity.get("inputs"),
      "launch": call.identity.get("launch"),
      "semantic": call.identity.get("semantic"),
      "group": str(call.group) if call.group is not None else None,
      "position_in_group": call.identity.get("position_in_group"),
    }
    identity = {k: v for k, v in identity.items() if v is not None}
    metadata: dict[str, Any] = {"identity_sha256": _sha256(canonical(identity))}
    if call.unknown:
      metadata["deps_status"] = UNKNOWN
    nodes.append({"id": call.index, "name": call.name, "duration_us": round(call.duration_us, 3),
                  "group_id": call.group if call.group is not None else "unassigned-%d" % call.index,
                  "metadata": metadata})
  dag_edges = []
  ids = {n["id"] for n in nodes}
  for e in edges:
    from_group = next((n["group_id"] for n in nodes if n["id"] == e["from"]), None)
    to_group = next((n["group_id"] for n in nodes if n["id"] == e["to"]), None)
    dag_edges.append({"from": e["from"], "to": e["to"], "kind": e["kind"],
                      "crosses_group": from_group != to_group and from_group is not None and to_group is not None})
  return {"schema": "tinygrad.full_token_dag.v1", "nodes": nodes, "edges": dag_edges}


def check_alignment(logical_calls: list[CallRecord], physical_calls: list[CallRecord]) -> dict[str, Any]:
  """Ordered stable-signature alignment gate. Any mismatch is ALIGNMENT_CONFOUNDED."""
  def signatures(calls: list[CallRecord]) -> list[dict]:
    out = []
    for call in calls:
      identity = {
        "ordered_call_index": call.index,
        "operation_kind": call.name,
        "program_identity": call.identity.get("program_identity", call.name),
        "outputs": call.identity.get("outputs"),
        "inputs": call.identity.get("inputs"),
        "launch": call.identity.get("launch"),
      }
      out.append({k: v for k, v in identity.items() if v is not None})
    return out
  a, b = signatures(logical_calls), signatures(physical_calls)
  if a != b:
    for i, (x, y) in enumerate(zip(a, b)):
      if x != y:
        return {"aligned": False, "reason": "ALIGNMENT_CONFOUNDED",
                "detail": "call %d signature differs: %r vs %r" % (i, x, y)}
    return {"aligned": False, "reason": "ALIGNMENT_CONFOUNDED",
            "detail": "call count differs: %d vs %d" % (len(a), len(b))}
  return {"aligned": True, "reason": None, "detail": None}


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------

def attribute_edges(logical_calls: list[CallRecord], physical_calls: list[CallRecord],
                    logical_edges: list[dict], physical_edges: list[dict]) -> list[dict]:
  logical_pairs = {(e["from"], e["to"]): e for e in logical_edges}
  logical_kinds = {(e["from"], e["to"]): e["kind"] for e in logical_edges}
  unknown_ids = {c.index for c in logical_calls if c.unknown} | {c.index for c in physical_calls if c.unknown}
  groups = {c.index: c.group for c in physical_calls}
  attributed = []
  for e in physical_edges:
    f, t = e["from"], e["to"]
    if f in unknown_ids or t in unknown_ids:
      source = UNKNOWN
    elif (f, t) in logical_pairs:
      source = SEMANTIC
    else:
      source = PLANNER_ALIAS
    overlap = e["overlaps"][0]
    attributed.append({
      "from": f, "to": t, "kind": e["kind"],
      "logical_kind": logical_kinds.get((f, t)),
      "source": source,
      "logical_buffer_ids": sorted({b for o in e["overlaps"] for b in o["logical_bufs"]}),
      "arena": overlap["base"],
      "range": overlap["range"],
      "crosses_group": groups.get(f) != groups.get(t) and groups.get(f) is not None and groups.get(t) is not None,
    })
  return attributed


def _class_of(name: str) -> str:
  if name.startswith("flash_"): return "flash"
  if name.startswith("q4k_"): return "gemv"
  if name.startswith("q6k_"): return "gemv"
  if name.startswith(("E_", "r_")): return "elementwise"
  if name.startswith("copy"): return "copy"
  return "other"


def _critical_path(dag: dict) -> float:
  return attach_summary(dag)["summary"]["critical_path_us"]


def _critical_path_chain(dag: dict) -> set[int]:
  """One duration-weighted critical path as a node-id set (for edge ranking)."""
  from extra.llm_research.decode.dag_critical_path_sim import compute_tails
  nodes = to_sim_nodes(dag)
  if not nodes or all(n["duration"] == 0 for n in nodes):
    return set()
  n = len(nodes)
  tails = compute_tails(nodes)
  est = [0.0] * n
  for i in range(n):
    for d in nodes[i]["deps"]:
      est[i] = max(est[i], est[d] + nodes[d]["duration"])
  end = [est[i] + nodes[i]["duration"] + tails[i] - nodes[i]["duration"] for i in range(n)]
  # Reconstruct a max-length chain by walking predecessors.
  chain: list[int] = []
  cur = max(range(n), key=lambda i: (est[i] + nodes[i]["duration"] + (tails[i] - nodes[i]["duration"]), i))
  while True:
    chain.append(cur)
    best = None
    for d in nodes[cur]["deps"]:
      if est[d] + nodes[d]["duration"] == est[cur] and (best is None or est[d] > est[best]):
        best = d
    if best is None:
      break
    cur = best
  return {dag["nodes"][i]["id"] for i in chain}


def _cp_without_edge(dag: dict, edge: dict) -> float:
  reduced = {"schema": dag["schema"], "nodes": dag["nodes"], "edges": [e for e in dag["edges"] if (e["from"], e["to"]) != (edge["from"], edge["to"])]}
  return _critical_path(reduced)


def compute_attribution_report(logical_calls: list[CallRecord], physical_calls: list[CallRecord],
                               manifest: dict[str, dict[str, Any]] | None = None,
                               logical_edges: list[dict] | None = None,
                               physical_edges: list[dict] | None = None) -> dict:
  """Full B3.1 report: both arms, attributed edges, metrics, ledgers."""
  if logical_edges is None:
    logical_edges = build_edges(logical_calls)
  if physical_edges is None:
    physical_edges = build_edges(physical_calls, manifest)
  alignment = check_alignment(logical_calls, physical_calls)
  if not alignment["aligned"]:
    raise B3AttributionError(alignment["detail"])
  logical_dag = attach_summary(to_dag(logical_calls, logical_edges))
  physical_dag = attach_summary(to_dag(physical_calls, physical_edges))
  attributed = attribute_edges(logical_calls, physical_calls, logical_edges, physical_edges)

  by_kind_source: dict[str, int] = {}
  for e in attributed:
    key = "%s/%s" % (e["kind"], e["source"])
    by_kind_source[key] = by_kind_source.get(key, 0) + 1

  planner_edges = [e for e in attributed if e["source"] == PLANNER_ALIAS]
  cp_chain = _critical_path_chain(physical_dag)
  ranked_candidates = planner_edges
  if cp_chain:
    on_chain = [e for e in planner_edges if e["from"] in cp_chain and e["to"] in cp_chain]
    if on_chain:
      ranked_candidates = on_chain
  ranked_candidates = sorted(ranked_candidates,
                             key=lambda e: -(e["range"][1] - e["range"][0]))[:256]
  ranked_edges = []
  for e in ranked_candidates:
    impact = _critical_path(physical_dag) - _cp_without_edge(physical_dag, e) if cp_chain else 0.0
    ranked_edges.append({**e, "cp_impact_us": round(impact, 3)})
  ranked_edges.sort(key=lambda e: e["cp_impact_us"], reverse=True)

  buf_bytes: dict[str, int] = {}
  for e in planner_edges:
    for oid in e["logical_buffer_ids"]:
      buf_bytes[oid] = buf_bytes.get(oid, 0) + max(0, e["range"][1] - e["range"][0])
  top_buffers = sorted(buf_bytes.items(), key=lambda kv: kv[1], reverse=True)

  pairs: dict[str, int] = {}
  by_name = {c.index: c.name for c in physical_calls}
  for e in attributed:
    key = "%s/%s" % (_class_of(by_name.get(e["from"], "?")), _class_of(by_name.get(e["to"], "?")))
    pairs[key] = pairs.get(key, 0) + 1

  unknown_ids = {c.index for c in logical_calls if c.unknown} | {c.index for c in physical_calls if c.unknown}
  lp = logical_dag["summary"]["critical_path_us"]
  pp = physical_dag["summary"]["critical_path_us"]
  return {
    "schema": SCHEMA,
    "arms": {"logical": logical_dag, "physical": physical_dag},
    "attributed_edges": attributed,
    "alignment": alignment,
    "manifest": manifest or {},
    "summary": {
      "node_count": len(logical_calls),
      "edge_count": len(attributed),
      "edges_by_kind_source": by_kind_source,
      "logical_critical_path_us": lp,
      "physical_critical_path_us": pp,
      "planner_delta_cp_us": round(pp - lp, 3),
      "planner_delta_cp_pct": round(100.0 * (pp - lp) / lp, 3) if lp else None,
      "logical_serialized_us": logical_dag["summary"]["serialized_us"],
      "physical_serialized_us": physical_dag["summary"]["serialized_us"],
      "logical_schedules": logical_dag["summary"]["schedules"],
      "physical_schedules": physical_dag["summary"]["schedules"],
      "top_planner_edges": ranked_edges[:20],
      "top_recoverable_buffers": [{"buf": b, "overlap_bytes": n} for b, n in top_buffers[:20]],
      "unknown_dep_node_count": len(unknown_ids),
      "resource_pairs": pairs,
      "per_group": physical_dag["summary"]["per_group"],
    },
  }


def validate_report(report: dict) -> None:
  if not isinstance(report, dict) or report.get("schema") != SCHEMA:
    raise B3AttributionError("report schema must be %r" % SCHEMA)
  for arm in ("logical", "physical"):
    if arm not in report.get("arms", {}):
      raise B3AttributionError("report.arms missing %r" % arm)
    validate_schema(report["arms"][arm])
  edges = report.get("attributed_edges")
  if not isinstance(edges, list):
    raise B3AttributionError("report.attributed_edges must be a list")
  ids = {n["id"] for n in report["arms"]["physical"]["nodes"]}
  for e in edges:
    if e.get("from") not in ids or e.get("to") not in ids:
      raise B3AttributionError("attributed edge endpoints must reference nodes: %r" % (e,))
    if e.get("kind") not in KINDS or e.get("source") not in (SEMANTIC, PLANNER_ALIAS, UNKNOWN):
      raise B3AttributionError("attributed edge kind/source invalid: %r" % (e,))
  if report.get("alignment", {}).get("aligned") is not True:
    raise B3AttributionError("alignment must PASS in a valid report")


def canonical(obj: Any) -> str:
  return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha256(text: str) -> str:
  return hashlib.sha256(text.encode()).hexdigest()


def emit_report(report: dict, path: str | None = None) -> str:
  validate_report(report)
  text = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if path is not None:
    pathlib.Path(path).write_text(text)
  return text


# ---------------------------------------------------------------------------
# Synthetic fixtures (hermetic)
# ---------------------------------------------------------------------------

def _call_name(call: Any) -> str:
  from tinygrad.uop.ops import Ops
  ast = call.src[0]
  if ast.op is Ops.PROGRAM:
    return str(ast.arg.name)
  if ast.op is Ops.SLICE:
    return "view"
  if ast.op is Ops.COPY:
    return "copy"
  if ast.op is Ops.CUSTOM_FUNCTION:
    return "custom-%s" % ast.arg
  return str(ast.op)


def _program_identity(call: Any) -> str | None:
  from tinygrad.uop.ops import Ops
  ast = call.src[0]
  if ast.op is Ops.PROGRAM:
    return "%s:%s" % (ast.arg.name, _sha256(canonical(repr(ast.arg))))
  return None


def _sink_out_indices(call: Any) -> tuple[int, ...] | None:
  """Derive written-buffer positions for pre-compile SINK calls.

  get_call_outs_ins returns () for Ops.SINK ASTs; the written buffers are the
  STORE targets inside the sink body. Used so the pre-compile linears (logical
  and physical arms) get correct write labels.
  """
  from tinygrad.uop.ops import Ops
  ast = call.src[0]
  if ast.op is not Ops.SINK:
    return None
  args = list(call.src[1:])

  def base(u: Any) -> Any:
    while u.op in (Ops.INDEX, Ops.MEMORY_SEMANTIC) and len(u.src):
      u = u.src[0]
    return u

  outs: list[int] = []
  for node in ast.backward_slice:
    target = base(node.src[0]) if node.op is Ops.STORE else None
    if target is not None and target in args:
      pos = args.index(target)
      if pos not in outs:
        outs.append(pos)
  return tuple(outs)


def build_call_records(linear: Any, input_uops: tuple[Any, ...], group_map: dict[int, Any],
                       manifest: dict[str, dict[str, Any]] | None = None,
                       arena_labels: dict[int, str] | None = None) -> list[CallRecord]:
  """Walk a linear into CallRecords with resolved accesses (live capture).

  On the pre-planning linear, buffer uops are the logical buffers themselves.
  On the post-planning linear they are arena slices; the manifest maps the
  physical intervals back to logical buffer ids for attribution.
  """
  from tinygrad.engine.realize import get_call_arg_uops, get_call_outs_ins, unwrap_multi, resolve_params
  from tinygrad.uop.ops import Ops
  records: list[CallRecord] = []
  for call_index, call in enumerate(linear.src):
    accesses: list[CallAccess] = []
    unknown = False
    try:
      arg_uops = resolve_params(call, input_uops)
      outs, _ins = get_call_outs_ins(call)
      if not outs:
        derived = _sink_out_indices(call)
        if derived is not None:
          outs = derived
      write_idx: list[int] = []
      flat = 0
      for bufs, _device_vars in unwrap_multi(call, arg_uops):
        start = flat
        write_idx.extend(start + i for i in outs)
        for u in bufs:
          logical_bufs = None
          is_buffer = not hasattr(u, "op")
          b = u if is_buffer else u.ensure_allocated()
          if manifest is not None and is_buffer:
            base = getattr(u, "base", None)
            if base is not None and base is not u:
              # Session-stable physical base identity; the logical-buffer
              # provenance below is best-effort enrichment, never a source of
              # dependency UNKNOWN (ranges on the base are exact either way).
              buf_id = "arena:%d" % id(base)
              dev = base.device if isinstance(base.device, str) else ",".join(sorted(base.device))
              label = "arena:%s:%d" % (dev, int(base.nbytes))
              byte_start = int(getattr(b, "offset", 0))
              byte_end = byte_start + int(b.nbytes)
              logical_bufs = [bid for bid, e in manifest.items()
                              if e.get("arena") == label and e.get("offset") is not None
                              and e["offset"] <= byte_start and byte_end <= e["offset"] + int(e["aligned_nbytes"])
                              and (e.get("first_call") is None or e["first_call"] <= call_index <= e.get("last_call", call_index))]
            else:
              buf_id = "logical:%d" % id(u)
          else:
            buf_id = "logical:%d" % id(u)
          accesses.append(CallAccess(buf_id, int(getattr(b, "offset", 0)), int(b.nbytes),
                                     flat in write_idx, logical_bufs))
          flat += 1
    except Exception:
      unknown = True
    records.append(CallRecord(
      call_index, _call_name(call), accesses,
      group=group_map.get(call_index),
      identity={"program_identity": _program_identity(call),
                "outputs": None, "inputs": None, "launch": None},
      unknown=unknown,
    ))
  return records


@contextlib.contextmanager
def capture_aligned_dags(harness: contextlib.AbstractContextManager | Any,
                         report_path: str | None = None) -> Any:
  """Single-process dual snapshot: pre-planning (logical) and post-planning
  (physical) linears plus the placement manifest, then the full attribution
  report. The seam wraps jit_lower (input_uops), memory_plan_rewrite (both
  linears), and graph_split_rewrite (group boundaries); no runtime file edit.
  """
  from tinygrad.engine import jit as tjit
  from tinygrad.schedule import memory as tmem
  from extra.llm_research.decode.full_token_dag_capture import _RecordingObserver, _group_map
  state: dict[str, Any] = {"input_uops": None, "pre": None, "post": None, "compiled": None, "group_map": None}
  collector = PlannerManifestCollector()
  token = tmem._memory_manifest_collectors.set((collector,))
  orig_lower, orig_rewrite, orig_compile, orig_split = tjit.jit_lower, tjit.memory_plan_rewrite, tjit.compile_linear, tjit.graph_split_rewrite

  def wrapped_jit_lower(linear: Any, held_bufs: set[Any], input_uops: list[Any]) -> Any:
    state["input_uops"] = tuple(input_uops)
    return orig_lower(linear, held_bufs, input_uops)

  def wrapped_rewrite(linear: Any, held_bufs: set[Any] | None = None) -> Any:
    state["pre"] = linear
    result = orig_rewrite(linear, held_bufs)
    state["post"] = result
    # Snapshot the manifest of THIS planning call (the collector accumulates
    # across the prefill and decode captures in one process).
    state["manifest"] = dict(collector.manifest)
    return result

  def wrapped_compile(linear: Any) -> Any:
    result = orig_compile(linear)
    state["compiled"] = result
    return result

  def wrapped_graph_split(linear: Any, max_batch_size: int = 0, observer: Any = None) -> Any:
    rec = _RecordingObserver(observer)
    result = orig_split(linear, max_batch_size=max_batch_size, observer=rec)
    state["group_map"] = _group_map(rec.records)
    return result

  tjit.jit_lower = wrapped_jit_lower
  tjit.memory_plan_rewrite = wrapped_rewrite
  tjit.compile_linear = wrapped_compile
  tjit.graph_split_rewrite = wrapped_graph_split
  try:
    harness()
    if state["pre"] is None or state["compiled"] is None:
      raise B3AttributionError("capture seam did not fire: jit capture was not reached")
    group_map = state["group_map"] or {}
    logical_calls = build_call_records(state["pre"], state["input_uops"] or (), group_map)
    manifest = state.get("manifest") or {}
    physical_calls = build_call_records(state["compiled"], state["input_uops"] or (), group_map,
                                        manifest, collector.arena_labels_by_uop_id)
    # Overlay write positions from the compiled PROGRAM metadata onto the
    # pre-planning logical arm (pre-compile SINK ASTs do not carry outs).
    if len(state["pre"].src) == len(state["compiled"].src):
      for i, compiled in enumerate(state["compiled"].src):
        ast = compiled.src[0]
        outs = tuple(getattr(ast.arg, "outs", ())) if hasattr(ast, "arg") else ()
        if outs:
          for pos in outs:
            if pos < len(logical_calls[i].accesses):
              logical_calls[i].accesses[pos].write = True
        # The logical arm names its kernels by the compiled identity so the
        # ordered stable signatures align between arms.
        logical_calls[i].name = physical_calls[i].name
        logical_calls[i].identity["program_identity"] = physical_calls[i].identity["program_identity"]
        if len(logical_calls[i].accesses) != len(physical_calls[i].accesses):
          logical_calls[i].unknown = physical_calls[i].unknown = True
    # Physical accesses are already arena-mapped, so the physical arm is built
    # without a manifest; attribution resolves logical ids from access metadata.
    logical_edges = build_edges(logical_calls)
    physical_edges = build_edges(physical_calls)
    report = compute_attribution_report(logical_calls, physical_calls, manifest,
                                        logical_edges=logical_edges, physical_edges=physical_edges)
    if report_path is not None:
      emit_report(report, report_path)
    yield report
  finally:
    tjit.jit_lower = orig_lower
    tjit.memory_plan_rewrite = orig_rewrite
    tjit.compile_linear = orig_compile
    tjit.graph_split_rewrite = orig_split
    tmem._memory_manifest_collectors.reset(token)

def build_attribution_fixture() -> tuple[list[CallRecord], dict[str, dict[str, Any]]]:
  """Two logically independent chains physically chained by arena reuse.

  Logical edges: 0->1 RAW, 1->2 RAW, 3->4 RAW, 4->5 RAW, 5->6 RAW, 6->7 RAW.
  Physical adds (all PLANNER_ALIAS): 0->3 WAW, 1->3 WAR, 0->4 RAW, 1->4 WAW,
  1->5 RAW, 2->4 WAR, 2->5 WAW, 2->6 RAW. Durations make chain 0 the logical
  critical path (35us) while the physical critical path runs
  0->1->3->4->5->6->7 = 63us.
  """
  calls = [
    CallRecord(0, "a_w0", [CallAccess("A", 0, 4096, True)], duration_us=10.0, group=0),
    CallRecord(1, "a_r1", [CallAccess("A", 0, 4096, False), CallAccess("X", 0, 512, True)], duration_us=20.0, group=0),
    CallRecord(2, "a_w2", [CallAccess("X", 0, 512, False), CallAccess("Y", 0, 512, True)], duration_us=5.0, group=0),
    CallRecord(3, "b_w3", [CallAccess("B", 0, 4096, True)], duration_us=8.0, group=1),
    CallRecord(4, "b_r4", [CallAccess("B", 0, 4096, False), CallAccess("C", 0, 512, True)], duration_us=12.0, group=1),
    CallRecord(5, "b_w5", [CallAccess("C", 0, 512, False), CallAccess("D", 0, 512, True)], duration_us=6.0, group=1),
    CallRecord(6, "b_r6", [CallAccess("D", 0, 512, False), CallAccess("E", 0, 1024, True)], duration_us=4.0, group=1),
    CallRecord(7, "b_r7", [CallAccess("E", 0, 1024, False)], duration_us=3.0, group=1),
  ]
  manifest = {
    "A": {"arena": "arena:0", "offset": 0, "aligned_nbytes": 4096, "logical_nbytes": 4096, "device": "CPU", "held": False, "first_call": 0, "last_call": 1},
    "X": {"arena": "arena:0", "offset": 4096, "aligned_nbytes": 512, "logical_nbytes": 512, "device": "CPU", "held": False, "first_call": 1, "last_call": 2},
    "Y": {"arena": "arena:0", "offset": 4608, "aligned_nbytes": 512, "logical_nbytes": 512, "device": "CPU", "held": False, "first_call": 2, "last_call": 2},
    "B": {"arena": "arena:0", "offset": 0, "aligned_nbytes": 4096, "logical_nbytes": 4096, "device": "CPU", "held": False, "first_call": 3, "last_call": 4},
    "C": {"arena": "arena:0", "offset": 4096, "aligned_nbytes": 512, "logical_nbytes": 512, "device": "CPU", "held": False, "first_call": 4, "last_call": 5},
    "D": {"arena": "arena:0", "offset": 4608, "aligned_nbytes": 512, "logical_nbytes": 512, "device": "CPU", "held": False, "first_call": 5, "last_call": 6},
    "E": {"arena": "arena:0", "offset": 5120, "aligned_nbytes": 1024, "logical_nbytes": 1024, "device": "CPU", "held": False, "first_call": 6, "last_call": 7},
  }
  for call in calls:
    for acc in call.accesses:
      call.identity.setdefault("inputs", []).append({"name": acc.buf, "dtype": "float32", "size": acc.nbytes})
  return calls, manifest


def build_partial_overlap_fixture() -> tuple[list[CallRecord], dict[str, dict[str, Any]]]:
  """Partially overlapping and adjacent non-overlapping physical ranges."""
  calls = [
    CallRecord(0, "p_w0", [CallAccess("P", 0, 2048, True)], duration_us=5.0, group=0),
    CallRecord(1, "p_w1", [CallAccess("Q", 0, 2048, True)], duration_us=5.0, group=0),
    CallRecord(2, "p_r2", [CallAccess("Q", 0, 2048, False)], duration_us=5.0, group=0),
    CallRecord(3, "p_w3", [CallAccess("R", 0, 1024, True)], duration_us=5.0, group=0),
    CallRecord(4, "p_w4", [CallAccess("S", 0, 1024, True)], duration_us=5.0, group=0),
  ]
  manifest = {
    "P": {"arena": "arena:0", "offset": 0, "aligned_nbytes": 2048, "logical_nbytes": 2048, "device": "CPU", "held": False, "first_call": 0, "last_call": 0},
    "Q": {"arena": "arena:0", "offset": 1024, "aligned_nbytes": 2048, "logical_nbytes": 2048, "device": "CPU", "held": False, "first_call": 1, "last_call": 2},
    "R": {"arena": "arena:0", "offset": 4096, "aligned_nbytes": 1024, "logical_nbytes": 1024, "device": "CPU", "held": False, "first_call": 3, "last_call": 3},
    "S": {"arena": "arena:0", "offset": 5120, "aligned_nbytes": 1024, "logical_nbytes": 1024, "device": "CPU", "held": False, "first_call": 4, "last_call": 4},
  }
  return calls, manifest


def build_duration_weighted_fixture() -> list[CallRecord]:
  """Many small nodes on one branch must not outweigh one expensive node."""
  calls = [CallRecord(i, "d%d" % i, [CallAccess("B%d" % i, 0, 512, True)], duration_us=1.0, group=0) for i in range(5)]
  calls.append(CallRecord(5, "expensive", [CallAccess("E", 0, 512, True)], duration_us=50.0, group=1))
  return calls


def run_synthetic(out: str | None = None) -> dict:
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  s = report["summary"]
  checks = []

  def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))
    if not ok:
      raise B3AttributionError("synthetic self-test failed: %s (%s)" % (name, detail))

  check("logical_cp_35", s["logical_critical_path_us"] == 35.0, "got %r" % s["logical_critical_path_us"])
  check("physical_cp_63", s["physical_critical_path_us"] == 63.0, "got %r" % s["physical_critical_path_us"])
  check("planner_delta_28", s["planner_delta_cp_us"] == 28.0, "got %r" % s["planner_delta_cp_us"])
  check("planner_alias_count", s["edges_by_kind_source"].get("RAW/PLANNER_ALIAS", 0) == 3
        and s["edges_by_kind_source"].get("WAW/PLANNER_ALIAS", 0) == 3
        and s["edges_by_kind_source"].get("WAR/PLANNER_ALIAS", 0) == 2,
        "got %r" % s["edges_by_kind_source"])
  if out is not None:
    emit_report(report, out)
  return report


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--synthetic", action="store_true")
  ap.add_argument("--capture-cuda", action="store_true", help="live CUDA d512 aligned capture (GPU, lock-held)")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--out", default=None)
  args = ap.parse_args()
  if args.synthetic:
    report = run_synthetic(args.out)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0
  if args.capture_cuda:
    from tinygrad.helpers import Context
    from tinygrad.llm.model import Transformer
    from tinygrad.device import Device

    def harness() -> None:
      model, _kv = Transformer.from_gguf(args.model, 4608)
      gen = model.generate([1] * args.depth, chunk_size=32, temperature=0.0)
      with Context(DEBUG=0):
        # Token 1 prefill, token 2-3 warm the stable rollout jits (cnt 0), token 4
        # captures: jit_lower + memory_plan_rewrite + graph_split_rewrite fire here.
        for _ in range(4):
          next(gen)
      Device["CUDA"].synchronize()

    with capture_aligned_dags(harness, report_path=args.out) as report:
      print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0
  ap.error("no mode selected (--synthetic | --capture-cuda)")
  return 2


if __name__ == "__main__":
  sys.exit(main())
