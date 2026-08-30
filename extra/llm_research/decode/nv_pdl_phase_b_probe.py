#!/usr/bin/env python3
"""Phase B construction census for the tested native-NV PDL arm.

This probe measures what the current construction actually arms, without
changing any tinygrad runtime file.  It monkeypatches measurement seams in
process:

* ``HCQGraph.__init__/__call__/collect_timestamps`` and
  ``graph_profile_payload`` so each HCQ graph-profile JSONL row carries a
  graph id and an invocation id.  That fixes the known PDL bookkeeping bug
  where the final five profile records came from different replay cycles.
* ``NVComputeQueue.exec`` records construction order, queue index, and the
  local graph position of every kernel.
* ``ops_nv._nv_pdl_arm_pair`` records every actually armed QMD latch pair,
  mapped back to local producer/consumer positions.
* ``renderer.cuda._nv_pdl_body`` records the exact instruction position of
  each emitted ``griddepcontrol.wait`` and ``launch_dependents``.

Consumer grid starts come from the HCQ start/end timestamp signals for the
matched replay cycle.  Wait-exit timestamps require a probe slot inside the
consumer body; this decode probe records the static wait position and leaves
the runtime wait exit ``null`` rather than fabricating one.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, subprocess, sys, time
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

SCHEMA = "tinygrad.nv_pdl_phase_b.v1"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

PDL_ENV = {
  "NV_PDL_PRODUCER_PROGRAMS": "prefix:q4k_,prefix:q6k_",
  "NV_PDL_CONSUMER_PROGRAMS": "prefix:reduce_output_rmsnorm,prefix:E_,prefix:r_,prefix:flash_,prefix:rmsnorm_q8_1_llama_provider",
  "NV_PDL_TRIGGER_POSITION": "end",
}

PROBE = {
  "next_graph_id": 0,
  "current_graph_id": None,
  "collect_for": None,
  "collect_graph": None,
  "graphs": {},
  "graph_meta": {},
  "pdl_placements": {},
  "tokens": [],
  "per_token_host_us": [],
}


def _atomic_json(path: pathlib.Path, payload: dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(f".{path.name}.tmp")
  tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  tmp.replace(path)


def _git_commit() -> str:
  return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def _install_probes() -> None:
  import tinygrad.renderer.cuda as cuda
  import tinygrad.runtime.graph.hcq as hcq
  import tinygrad.runtime.ops_nv as ops_nv

  orig_init = hcq.HCQGraph.__init__
  orig_call = hcq.HCQGraph.__call__
  orig_collect = hcq.HCQGraph.collect_timestamps
  orig_del = hcq.HCQGraph.__del__
  orig_payload = hcq.graph_profile_payload
  orig_arm = ops_nv._nv_pdl_arm_pair
  orig_exec = ops_nv.NVComputeQueue.exec
  orig_body = cuda._nv_pdl_body

  def init(self, *args, **kwargs):
    gid = PROBE["next_graph_id"]
    PROBE["next_graph_id"] += 1
    PROBE["current_graph_id"] = gid
    self._pdl_probe_graph_id = gid
    PROBE["graphs"][gid] = {"execs": [], "arm_calls": [], "arm_attempts": [], "exec_index": -1}
    result = orig_init(self, *args, **kwargs)
    PROBE["graph_meta"][gid] = {
      "size": len(self.calls),
      "names": [rt.name if rt is not None else "<copy>" for rt in self.runtimes],
      "queue_count": len(next(iter(self.compute_queues.values()))) if self.compute_queues else 0,
    }
    return result

  def call(self, *args, **kwargs):
    gid = getattr(self, "_pdl_probe_graph_id", None)
    # collect_timestamps inside this invocation exports the previous cycle.
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    result = orig_call(self, *args, **kwargs)
    if gid in PROBE["graphs"]:
      PROBE["graphs"][gid].setdefault("invocations", []).append(self.kickoff_value)
    return result

  def collect(self):
    gid = getattr(self, "_pdl_probe_graph_id", None)
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    return orig_collect(self)

  def delete(self):
    gid = getattr(self, "_pdl_probe_graph_id", None)
    PROBE["collect_for"] = self.kickoff_value
    PROBE["collect_graph"] = gid
    return orig_del(self)

  def payload(entries, deps, sigs):
    result = orig_payload(entries, deps, sigs)
    result["graph_id"] = PROBE.get("collect_graph")
    result["invocation"] = PROBE.get("collect_for")
    return result

  def exec(self, prg, args_state, global_size, local_size):
    gid = PROBE.get("current_graph_id")
    if gid in PROBE["graphs"]:
      state = PROBE["graphs"][gid]
      state["exec_index"] += 1
      state["execs"].append({
        "j": state["exec_index"], "name": str(prg.name),
        "queue": int(getattr(self, "queue_idx", 0)),
      })
    result = orig_exec(self, prg, args_state, global_size, local_size)
    return result

  def arm(active_qmd, new_qmd, active_name, new_name):
    gid = PROBE.get("current_graph_id")
    matched_before = True
    result = orig_arm(active_qmd, new_qmd, active_name, new_name)
    if gid in PROBE["graphs"]:
      state = PROBE["graphs"][gid]
      queue = state["execs"][state["exec_index"]]["queue"] if state["execs"] else 0
      same_queue = [e for e in state["execs"] if e["queue"] == queue and e["j"] < state["exec_index"]]
      attempt = {
        "producer_name": str(active_name), "consumer_name": str(new_name),
        "producer_j": same_queue[-1]["j"] if same_queue else -1, "consumer_j": state["exec_index"],
        "queue": queue,
        "armed": bool(result),
      }
      state["arm_attempts"].append(attempt)
      if result:
        state["arm_calls"].append(attempt)
    return result

  def body(name, kernel):
    consumers = frozenset(x for x in os.environ.get("NV_PDL_CONSUMER_PROGRAMS", "").split(",") if x)
    producers = frozenset(x for x in os.environ.get("NV_PDL_PRODUCER_PROGRAMS", "").split(",") if x)
    is_consumer = cuda._nv_pdl_match(name, consumers)
    is_producer = cuda._nv_pdl_match(name, producers)
    result = orig_body(name, kernel)
    if is_consumer or is_producer:
      record = PROBE["pdl_placements"].setdefault(str(name), {"name": str(name)})
      if is_consumer:
        record["wait_index"] = 0
        record["wait_instruction"] = 'asm volatile("griddepcontrol.wait;");'
      if is_producer:
        trigger_pos = os.environ.get("NV_PDL_TRIGGER_POSITION", "end")
        record["trigger_index"] = 0 if trigger_pos == "start" else len(result) - 1
        record["trigger_instruction"] = 'asm volatile("griddepcontrol.launch_dependents;");'
    return result

  hcq.HCQGraph.__init__ = init
  hcq.HCQGraph.__call__ = call
  hcq.HCQGraph.collect_timestamps = collect
  hcq.HCQGraph.__del__ = delete
  hcq.graph_profile_payload = payload
  ops_nv.NVComputeQueue.exec = exec
  ops_nv._nv_pdl_arm_pair = arm
  cuda._nv_pdl_body = body


def _decode_marker(names: list[str]) -> bool:
  return any(("q4k_" in n or "q6k_" in n or "flash_" in n) for n in names)


def _select_decode_dag(dags: list[dict]) -> dict:
  from extra.llm_research.decode import full_token_dag_capture as ftc
  decode = [d for d in dags if _decode_marker([str(n.get("name", "")) for n in d.get("nodes", [])])]
  if not decode:
    raise RuntimeError("no decode DAG captured")
  return max(decode, key=lambda d: (
    sum(1 for n in d.get("nodes", []) if any(m in str(n.get("name", ""))
        for m in ("q4k_", "q6k_", "flash_"))),
    len(d.get("nodes", []))))


def _map_graphs_to_groups() -> tuple[dict[int, int], dict[int, int]]:
  """Map construction-ordered graph ids to DAG group ids by the known group sizes."""
  expected = [32, 64, 128, 195]
  candidates = [(gid, meta) for gid, meta in sorted(PROBE["graph_meta"].items())
                if meta["size"] in expected and _decode_marker(meta["names"])]
  if [m["size"] for _, m in candidates] != expected:
    raise RuntimeError(f"decode graph-size signature mismatch: {[m['size'] for _, m in candidates]}")
  graph_to_group = {gid: group for group, (gid, _) in enumerate(candidates)}
  group_to_graph = {group: gid for gid, group in graph_to_group.items()}
  return graph_to_group, group_to_graph


def _load_records(path: pathlib.Path) -> list[dict]:
  rows = []
  for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
      continue
    rows.append(json.loads(line))
  return rows


def _selected_records(records: list[dict], group_to_graph: dict[int, int]) -> dict[int, dict]:
  by_graph: dict[int, list[dict]] = defaultdict(list)
  for record in records:
    if record.get("graph_id") in group_to_graph.values():
      by_graph[int(record["graph_id"])].append(record)
  selected: dict[int, dict] = {}
  invocation_by_group: dict[int, int] = {}
  for group, gid in sorted(group_to_graph.items()):
    rows = by_graph.get(gid, [])
    invocations = sorted({int(r["invocation"]) for r in rows if r.get("invocation") is not None})
    if len(invocations) < 2:
      raise RuntimeError(f"graph {gid} has {len(invocations)} recorded invocations; need at least two")
    inv = invocations[-2]  # the last flush invocation exports this cycle
    candidates = [r for r in rows if int(r.get("invocation")) == inv]
    if len(candidates) != 1:
      raise RuntimeError(f"graph {gid} invocation {inv} has {len(candidates)} profile rows")
    selected[gid] = candidates[0]
    invocation_by_group[group] = inv
  return selected, invocation_by_group


def _capture(args: argparse.Namespace, profile_jsonl: pathlib.Path) -> dict:
  from tinygrad import Device
  from tinygrad.helpers import Context

  from extra.llm_research.decode import full_token_dag_capture as ftc
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt

  model = _load(args.model, args.max_context)
  gen = model.generate(_prompt(args.model, args.depth), chunk_size=32, temperature=0.0)
  profile_jsonl.unlink(missing_ok=True)

  def harness() -> None:
    dev = Device["NV"]
    for _ in range(args.tokens):
      started = time.perf_counter()
      PROBE["tokens"].append(int(next(gen)))
      PROBE["per_token_host_us"].append((time.perf_counter() - started) * 1e6)
    dev.synchronize()
    with Context(DEBUG=0):
      PROBE["tokens"].append(int(next(gen)))  # flush: exports the measured cycle
    dev.synchronize()

  dag: dict = {}
  original_select = ftc._select_dag
  ftc._select_dag = _select_decode_dag
  try:
    with ftc.capture_full_token_dag(harness) as captured:
      dag = captured
  finally:
    ftc._select_dag = original_select
    gen.close()
  return dag


def _build_report(args: argparse.Namespace, dag: dict, profile_jsonl: pathlib.Path) -> dict:
  graph_to_group, group_to_graph = _map_graphs_to_groups()
  records = _load_records(profile_jsonl)
  selected, invocation_by_group = _selected_records(records, group_to_graph)

  edge_ids = {(e["from"], e["to"]) for e in dag.get("edges", [])}
  by_group: dict[int, list[dict]] = defaultdict(list)
  for node in dag["nodes"]:
    by_group[int(node["group_id"])].append(node)

  nodes = []
  token_start = None
  raw_by_id: dict[int, tuple[float, float]] = {}
  for group, members in sorted(by_group.items()):
    members = sorted(members, key=lambda n: n["id"])
    gid = group_to_graph[group]
    entries = selected[gid]["entries"]
    if len(entries) != len(members):
      raise RuntimeError(f"graph {gid} has {len(entries)} entries for {len(members)} nodes")
    for idx, node in enumerate(members):
      entry = entries[idx]
      start = float(entry["start"])
      end = float(entry["end"])
      raw_by_id[int(node["id"])] = (start, end)
      token_start = start if token_start is None else min(token_start, start)

  for group, members in sorted(by_group.items()):
    for node in sorted(members, key=lambda n: n["id"]):
      start, end = raw_by_id[int(node["id"])]
      nodes.append({
        "id": int(node["id"]), "name": str(node.get("name", "")),
        "group_id": int(node["group_id"]),
        "start_us": round(start - token_start, 3),
        "end_us": round(end - token_start, 3),
      })

  node_by_id = {n["id"]: n for n in nodes}
  pairs = []
  for group, gid in sorted(group_to_graph.items()):
    members = sorted(by_group[group], key=lambda n: n["id"])
    ids = [int(n["id"]) for n in members]
    names = [str(n.get("name", "")) for n in members]
    state = PROBE["graphs"][gid]
    for attempt in state["arm_calls"]:
      pj, cj = int(attempt["producer_j"]), int(attempt["consumer_j"])
      if pj >= len(ids) or cj >= len(ids):
        raise RuntimeError(f"graph {gid} arm index out of range")
      producer, consumer = node_by_id[ids[pj]], node_by_id[ids[cj]]
      if attempt["producer_name"] != names[pj] or attempt["consumer_name"] != names[cj]:
        exec_names = [e["name"] for e in state["execs"]]
        raise RuntimeError(
          f"graph {gid} arm name mismatch at {pj}/{cj}: "
          f"attempt={attempt['producer_name']!r}->{attempt['consumer_name']!r}, "
          f"dag={names[pj]!r}->{names[cj]!r}, execs={exec_names[max(0, pj - 2):cj + 2]}")
      producer_end = producer["end_us"]
      consumer_start = consumer["start_us"]
      pairs.append({
        "producer_id": ids[pj], "consumer_id": ids[cj],
        "producer_name": attempt["producer_name"], "consumer_name": attempt["consumer_name"],
        "group": group, "queue": attempt["queue"],
        "has_data_edge": (ids[pj], ids[cj]) in edge_ids,
        "producer_start_us": producer["start_us"], "producer_end_us": producer_end,
        "consumer_start_us": consumer_start, "consumer_end_us": consumer["end_us"],
        "launch_shadow_us": round(consumer_start - producer["start_us"], 3),
        "overlap_us": round(producer_end - consumer_start, 3),
        "wait_exit_us": None,
        "wait_exit_note": "not instrumented in decode body; static wait position is consumer instruction 0",
      })

  assignments = []
  for group, gid in sorted(group_to_graph.items()):
    state = PROBE["graphs"][gid]
    members = sorted(by_group[group], key=lambda n: n["id"])
    for idx, node in enumerate(members):
      assignments.append({"id": int(node["id"]), "name": str(node.get("name", "")),
                          "group": group, "queue": state["execs"][idx]["queue"]})

  real = [p for p in pairs if p["has_data_edge"]]
  incidental = [p for p in pairs if not p["has_data_edge"]]
  token_blob = ",".join(map(str, PROBE["tokens"])).encode()

  def pair_bucket(rows):
    counts = Counter()
    for p in rows:
      counts[(p["producer_name"].split("_", 1)[0] if p["producer_name"].startswith(("q4k", "q6k")) else "other",
              p["consumer_name"].split("_", 1)[0])] += 1
    return {f"{a}->{b}": v for (a, b), v in sorted(counts.items())}

  return {
    "schema": SCHEMA,
    "arm": args.arm,
    "queues": args.queues,
    "model": args.model,
    "depth": args.depth,
    "max_context": args.max_context,
    "commit": _git_commit(),
    "env": {"PROFILE": os.environ.get("PROFILE", ""),
            "HCQ_NUM_COMPUTE": os.environ.get("HCQ_NUM_COMPUTE", ""),
            **({k: os.environ.get(k, "") for k in PDL_ENV} if args.arm == "candidate" else {})},
    "graph_group_mapping": {str(g): gid for g, gid in sorted(group_to_graph.items())},
    "invocation_by_group": {str(g): inv for g, inv in sorted(invocation_by_group.items())},
    "profile_records_total": len(records),
    "node_count": len(nodes),
    "timed_node_count": len(nodes),
    "token_span_us": round(max(n["end_us"] for n in nodes) - min(n["start_us"] for n in nodes), 3),
    "token_evidence": {
      "count": len(PROBE["tokens"]),
      "sha256": hashlib.sha256(token_blob).hexdigest(),
      "first_token_ids": PROBE["tokens"][:16],
    },
    "profiled_host_per_token_us": {
      "median": round(sorted(PROBE["per_token_host_us"])[len(PROBE["per_token_host_us"]) // 2], 3),
      "note": "PROFILE=1 instrumentation tax included; not an endpoint wall",
    },
    "armed_pairs": {
      "total": len(pairs), "data_edge": len(real), "incidental": len(incidental),
      "by_bucket": pair_bucket(pairs),
      "real_by_bucket": pair_bucket(real),
      "incidental_by_bucket": pair_bucket(incidental),
      "median_launch_shadow_us": round(sorted(p["launch_shadow_us"] for p in pairs)[len(pairs) // 2], 3) if pairs else None,
      "median_overlap_us": round(sorted(p["overlap_us"] for p in pairs)[len(pairs) // 2], 3) if pairs else None,
      "positive_overlap_count": sum(1 for p in pairs if p["overlap_us"] > 0),
    },
    "pdl_placements": dict(sorted(PROBE["pdl_placements"].items())),
    "nodes": nodes,
    "assignments": assignments,
    "pairs": pairs,
    "endpoint_wall_note": (
      "endpoint wall is the prior same-session nv-pdl-queue-theories bracket "
      "(2q -11.641 us, 1q -8.201 us); this record is the construction census"
    ),
  }


def main() -> int:
  global PDL_ENV
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", required=True, choices=("control", "candidate"))
  ap.add_argument("--queues", type=int, choices=(1, 2), default=2)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--tokens", type=int, default=8)
  ap.add_argument("--producer-programs", default=PDL_ENV["NV_PDL_PRODUCER_PROGRAMS"])
  ap.add_argument("--consumer-programs", default=PDL_ENV["NV_PDL_CONSUMER_PROGRAMS"])
  ap.add_argument("--trigger-position", choices=("start", "end"), default=PDL_ENV["NV_PDL_TRIGGER_POSITION"])
  ap.add_argument("--profile-jsonl", type=pathlib.Path, required=True)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  PDL_ENV = {
    "NV_PDL_PRODUCER_PROGRAMS": args.producer_programs,
    "NV_PDL_CONSUMER_PROGRAMS": args.consumer_programs,
    "NV_PDL_TRIGGER_POSITION": args.trigger_position,
  }

  for key in PDL_ENV:
    os.environ.pop(key, None)
  if args.arm == "candidate":
    os.environ.update(PDL_ENV)
  os.environ["DEV"] = "NV"
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(args.profile_jsonl)
  if args.queues == 1:
    os.environ["HCQ_NUM_COMPUTE"] = "1"
  else:
    os.environ["HCQ_NUM_COMPUTE"] = "2"

  _install_probes()
  dag = _capture(args, args.profile_jsonl)
  report = _build_report(args, dag, args.profile_jsonl)
  _atomic_json(args.out, report)
  print(json.dumps({
    "arm": args.arm, "queues": args.queues, "node_count": report["node_count"],
    "armed_pairs": report["armed_pairs"]["total"],
    "data_edge": report["armed_pairs"]["data_edge"],
    "incidental": report["armed_pairs"]["incidental"],
    "token_sha": report["token_evidence"]["sha256"],
    "token_span_us": report["token_span_us"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
