#!/usr/bin/env python3
"""Current-HEAD RMSNorm topology census for the d512 decode token.

This is measurement tooling only.  It reuses the full-token DAG capture seam
from full_token_dag_capture.py (one range-aware DepsTracker over the
pre-split linear, graph-group assignment from the admission observer) and
drives it with the same model/prompt construction as
nv_norm_native_wall_ab.py.  The selected arm is a fresh process:

  control   production graph at HEAD
  candidate attn/ffn/output RMSNorm sites set to native fp16 warp-reduce

HCQ_GRAPH_PROFILE_JSON lines supply kernel durations when PROFILE=1 is set.
The report separates profiled topology evidence from the wall brackets: the
durations here are topology attribution, never an unprofiled wall claim.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.llm_research.decode import full_token_dag_capture as ftc
from extra.llm_research.decode.nv_norm_native_wall_ab import (
  DEFAULT_MODEL, SITES, _configure_native,
)
from extra.llm_research.decode.nv_ffn_reduce_output_ab import _configure as _configure_ffn_fold
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

SCHEMA = "tinygrad.nv_rmsnorm_current_head_topology.v1"


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _git_commit() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
  except Exception:
    return None


def _node_names(dag: dict) -> list[str]:
  return [str(n.get("name", "node-%d" % n["id"])) for n in dag["nodes"]]


def _group_sizes(dag: dict) -> list[int]:
  counts: dict[str, int] = {}
  order: list[str] = []
  for node in dag["nodes"]:
    g = str(node.get("group_id"))
    if g not in counts:
      counts[g] = 0
      order.append(g)
    counts[g] += 1
  return [counts[g] for g in order]


def _anchor_ids(dag: dict) -> dict[str, list[int]]:
  """Locate the mandatory decode GEMV anchors by rendered program name."""
  anchors: dict[str, list[int]] = {"Q": [], "O": [], "gate_up": [], "down": [], "vocab": []}
  for node in dag["nodes"]:
    name = str(node.get("name", ""))
    if name.startswith("q4k_g3_lanemap_gemv_epi_resadd_"):
      anchors["O"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_w1w3fused16_"):
      anchors["gate_up"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_w1w3fused_"):
      anchors["gate_up"].append(node["id"])
    elif "_epi_ffnresadd_" in name or name.endswith("_epi_ffnresadd"):
      anchors["down"].append(node["id"])
    elif name == "q4k_warp_coop_q8_dp4a_partial_4096_4096":
      # Layers 1..35 promote their Q projection to the warp-coop q8 route;
      # layer 0 stays on the lanemap route.
      anchors["Q"].append(node["id"])
    elif name.startswith("q4k_g3_lanemap_gemv_") and name.endswith("_4096_4096"):
      anchors["Q"].append(node["id"])
    elif "vocab_scalar_reduce" in name:
      anchors["vocab"].append(node["id"])
  for kind, ids in anchors.items():
    ids.sort()
  return anchors


def _decode_signal(dag: dict) -> int:
  """Count decode-only program markers so the capture seam prefers the decode DAG."""
  markers = ("q4k_", "q6k_", "flash_block_", "decode_kv_rope_store")
  return sum(1 for name in _node_names(dag) if any(marker in name for marker in markers))


def _select_decode_dag(dags: list[dict]) -> dict:
  """Prefer the pre-split linear that contains decode GEMV/attention programs.

  ``capture_full_token_dag`` records both the prefill linear and the decode
  linear in one generation pass.  Its size-based selector lands on the
  prefill graph at current HEAD, so this driver overrides the selection step
  (measurement-only) without changing the shared capture module.
  """
  decode = [dag for dag in dags if _decode_signal(dag) > 0]
  if decode:
    return max(decode, key=lambda dag: (_decode_signal(dag), len(dag.get("nodes", []))))
  raise ftc.FullTokenDagError(
    "no pre-split linear contained decode programs; captured %d DAGs" % len(dags))


def _profile_window(records: list[dict], dag: dict) -> tuple[int, list[dict]]:
  """Return (record_index, records) for the last replay cycle matching the DAG groups."""
  groups = ftc.ordered_group_ids(dag)
  names = [[str(n.get("name", "")) for n in
            sorted((n for n in dag["nodes"] if n.get("group_id") == gid), key=lambda n: n["id"])]
           for gid in groups]
  sizes = [len(group) for group in names]
  run_len = len(groups)

  def _matches_names(start: int) -> bool:
    for gi, expected in enumerate(names):
      actual = [str(e.get("name", "")) for e in (records[start + gi].get("entries") or [])]
      if actual != expected:
        return False
    return True

  name_matches = [i for i in range(0, len(records) - run_len + 1) if _matches_names(i)]
  size_matches = [i for i in range(0, len(records) - run_len + 1)
                  if [len(r.get("entries") or []) for r in records[i:i + run_len]] == sizes]
  # PDL and other launch reordering can make graph finalizers flush out of
  # token order, so prefer exact per-group program-name sequences and fall
  # back to the legacy size-only match.
  matches = name_matches or size_matches
  if not matches:
    return -1, []
  return matches[-1], records[matches[-1]:matches[-1] + run_len]


def _attach_profile_times(dag: dict, window: list[dict]) -> dict:
  """Attach per-node GPU start/end times from the matched replay window."""
  if not window:
    raise ftc.FullTokenDagError("no profile replay window matched the selected decode DAG")
  groups = ftc.ordered_group_ids(dag)
  for gi, gid in enumerate(groups):
    members = sorted([n for n in dag["nodes"] if n.get("group_id") == gid], key=lambda n: n["id"])
    entries = window[gi].get("entries") or []
    for i, node in enumerate(members):
      if i >= len(entries):
        continue
      entry = entries[i]
      try:
        node["start_ns"] = float(entry.get("start", 0) or 0)
        node["end_ns"] = float(entry.get("end", 0) or 0)
      except (TypeError, ValueError):
        node.pop("start_ns", None)
        node.pop("end_ns", None)
  with_times = [n for n in dag["nodes"] if "start_ns" in n and "end_ns" in n]
  if with_times:
    token_start = min(float(n["start_ns"]) for n in with_times)
    for node in with_times:
      # HCQ profile records timestamp in microseconds, not nanoseconds.
      # The *_ns field names are kept for compatibility with older captures.
      node["start_us"] = round(float(node["start_ns"]) - token_start, 3)
      node["end_us"] = round(float(node["end_ns"]) - token_start, 3)
  return dag


def _profile_census(dag: dict, records: list[dict]) -> dict:
  """Classify the rendered token into the common support families."""
  names = _node_names(dag)
  by_kind: Counter = Counter()
  copy_names = []
  native_norm_names = []
  norm_names = []
  for name in names:
    low = name.lower()
    if low.startswith("copy") or "_copy" in low or low == "copy":
      by_kind["copy"] += 1
      copy_names.append(name)
    if "rmsnorm_native" in low or low.startswith("rmsnorm_native"):
      by_kind["native_rmsnorm"] += 1
      native_norm_names.append(name)
    elif "rmsnorm" in low:
      by_kind["other_rmsnorm"] += 1
      norm_names.append(name)
    if "flash" in low:
      by_kind["flash"] += 1
    if low.startswith("q4k") or low.startswith("q6k"):
      by_kind["gemv"] += 1
  return {
    "kernel_count": len(names),
    "graph_group_count": len(set(str(n.get("group_id")) for n in dag["nodes"])),
    "group_sizes": _group_sizes(dag),
    "kind_counts": dict(sorted(by_kind.items())),
    "copy_names": sorted(set(copy_names)),
    "native_rmsnorm_names": sorted(set(native_norm_names)),
    "other_rmsnorm_names": sorted(set(norm_names)),
    "profile_record_count": len(records),
  }


def _capture(model_path: str, arm: str, sites: tuple[str, ...], depth: int,
             max_context: int, profile_jsonl: str, candidate_route: str) -> tuple[dict, list[dict]]:
  model = _load(model_path, max_context)
  if arm == "candidate":
    if candidate_route == "native": _configure_native(model, sites)
    elif candidate_route == "ffn-fold": _configure_ffn_fold(model, "candidate")
    else: raise ValueError(f"unknown candidate route {candidate_route!r}")
  elif arm != "control":
    raise ValueError(f"unknown arm {arm!r}")

  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  pathlib.Path(profile_jsonl).unlink(missing_ok=True)

  def harness() -> None:
    from tinygrad import Device
    from tinygrad.helpers import Context
    try:
      with Context(DEBUG=0):
        next(gen)  # capture token
      with Context(DEBUG=0):
        next(gen)  # replay 1
      with Context(DEBUG=0):
        next(gen)  # replay 2 (measured)
      Device["NV"].synchronize()
      with Context(DEBUG=0):
        next(gen)  # flush: collects the measured replay timestamps
    finally:
      gen.close()

  dag: dict = {}
  original_select = ftc._select_dag
  ftc._select_dag = _select_decode_dag
  try:
    with ftc.capture_full_token_dag(harness) as captured:
      dag = captured
  finally:
    ftc._select_dag = original_select
  ftc._apply_profile_durations(dag, profile_jsonl)
  records = ftc.load_profile_records(profile_jsonl) if pathlib.Path(profile_jsonl).exists() else []
  window_start, window = _profile_window(records, dag)
  dag = _attach_profile_times(dag, window)
  # The capture helper validates the summary shape only when it writes the
  # file; call the same composer after durations are attached.
  dag = ftc.attach_summary(dag)
  dag["_profile_window_start"] = window_start
  dag["_profile_window_len"] = len(window)
  return dag, records


def _report(dag: dict, records: list[dict], arm: str, sites: tuple[str, ...],
            model_path: str, depth: int, max_context: int, profile_jsonl: str) -> dict:
  metrics = ftc.compute_dag_metrics(dag)
  names = _node_names(dag)
  timed = [n for n in dag["nodes"] if "start_us" in n and "end_us" in n]
  token_start = min(n["start_us"] for n in timed) if timed else None
  token_end = max(n["end_us"] for n in timed) if timed else None
  return {
    "schema": SCHEMA,
    "arm": arm,
    "sites": list(sites),
    "model": model_path,
    "depth": depth,
    "max_context": max_context,
    "commit": _git_commit(),
    "profile_jsonl": profile_jsonl,
    "node_count": metrics["node_count"],
    "edge_count": len(dag.get("edges", [])),
    "cross_group_edge_count": sum(1 for e in dag.get("edges", []) if e.get("crosses_group")),
    "name_digest": _sha256("\n".join(names).encode()),
    "profile_window_start": dag.get("_profile_window_start"),
    "profile_window_len": dag.get("_profile_window_len"),
    "timed_node_count": len(timed),
    "token_start_us": token_start,
    "token_end_us": token_end,
    "token_span_us": round((token_end - token_start), 3) if token_start is not None and token_end is not None else None,
    "summary": dag.get("summary"),
    "metrics": metrics,
    "anchor_ids": _anchor_ids(dag),
    "census": _profile_census(dag, records),
    "nodes": dag["nodes"],
    "edges": dag["edges"],
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", required=True, choices=("control", "candidate"))
  ap.add_argument("--sites", default="ffn")
  ap.add_argument("--candidate-route", choices=("native", "ffn-fold"), default="native")
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--profile-jsonl", default="/tmp/nv-rmsnorm-topology-profile.jsonl")
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(args.profile_jsonl)
  sites = tuple(s for s in args.sites.split(",") if s)
  for site in sites:
    if site not in SITES:
      raise SystemExit(f"unknown site {site!r}; choose from {SITES}")
  dag, records = _capture(args.model, args.arm, sites, args.depth, args.max_context, args.profile_jsonl,
                          args.candidate_route)
  report = _report(dag, records, args.arm, sites, args.model, args.depth,
                   args.max_context, args.profile_jsonl)
  report["candidate_route"] = args.candidate_route
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: report[k] for k in ("arm", "sites", "node_count", "edge_count",
                                           "cross_group_edge_count", "anchor_ids", "census")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
