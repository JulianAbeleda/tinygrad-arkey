#!/usr/bin/env python3
"""Close the gate/up four-warp wall regression against per-row profile data.

Control and candidate differ only by the Q4_K gate/up admission (the same
harness used by ``q4k_gate_up_four_warp_wall_bracket.py``).  Each child runs a
settled continuous decode under ``PROFILE=1`` and ``HCQ_GRAPH_PROFILE_JSON``,
retains the host wall and token stream hash, and exports the final graph
invocation timestamps explicitly.  The driver then merges control/candidate/
control and reports:

  node_sum, device union, overlap, span, host wall, host gap,
  per-row duration deltas, launch grid/block, and the closure identity
  wall_delta = gate/up_delta + downstream_delta + overlap_delta + host_gap_delta

No production model, renderer, scheduler, runtime, or route file is changed.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, statistics, subprocess, sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"
PYTHON = pathlib.Path("/home/ubuntu/tinygrad-arkey/.venv/bin/python")
# The current token replay has three stable groups and one route-dependent
# tail: 238 for installed control and 256 for this candidate.  Prefill also
# emits 256/394 groups, so select the most frequent tail following this prefix.
DECODE_GROUP_PREFIX = (32, 64, 128)

HASH64 = re.compile(r"_[0-9a-f]{64}$")
HASH40 = re.compile(r"_[0-9a-f]{40}$")


def canon(name: str) -> str:
  name = HASH64.sub("", str(name)).strip()
  return HASH40.sub("", name)


# ---------------------------------------------------------------------------
# Profile ledger (reused from the locked phase-1 authority)
# ---------------------------------------------------------------------------

def _union_us(entries: list[dict]) -> float:
  ivs = sorted((float(e["start"]), float(e["end"])) for e in entries)
  if not ivs:
    return 0.0
  total, cur_s, cur_e = 0.0, *ivs[0]
  for s, e in ivs[1:]:
    if s <= cur_e:
      cur_e = max(cur_e, e)
    else:
      total += cur_e - cur_s
      cur_s, cur_e = s, e
  return total + (cur_e - cur_s)


def _replay_metrics(entries: list[dict]) -> dict:
  node_sum = sum(float(e["duration"]) for e in entries)
  union = _union_us(entries)
  span = max(float(e["end"]) for e in entries) - min(float(e["start"]) for e in entries)
  return {"node_count": len(entries), "node_sum_us": round(node_sum, 3),
          "union_us": round(union, 3), "overlap_us": round(node_sum - union, 3),
          "span_us": round(span, 3)}


def _complete_replays(lines: list[dict]) -> list[list[dict]]:
  sizes = [len(x.get("entries", [])) for x in lines]
  tails: Counter = Counter()
  for i in range(len(sizes) - len(DECODE_GROUP_PREFIX)):
    if tuple(sizes[i:i + len(DECODE_GROUP_PREFIX)]) == DECODE_GROUP_PREFIX:
      tails[sizes[i + len(DECODE_GROUP_PREFIX)]] += 1
  if not tails:
    return []
  group_sizes = (*DECODE_GROUP_PREFIX, tails.most_common(1)[0][0])
  # The current four-warp admission creates a separate 18-entry cast group
  # after its 256-entry body group.  Prefill instead follows 256 with 394.
  # Extend only when one successor dominates the selected route population.
  successors: Counter = Counter()
  for i in range(len(sizes) - len(group_sizes)):
    if tuple(sizes[i:i + len(group_sizes)]) == group_sizes:
      successors[sizes[i + len(group_sizes)]] += 1
  ranked_successors = successors.most_common()
  if group_sizes[-1] == 256 and ranked_successors and ranked_successors[0][0] == 18 and \
     (len(ranked_successors) == 1 or ranked_successors[0][1] > 2 * ranked_successors[1][1]):
    group_sizes = (*group_sizes, ranked_successors[0][0])
  replays: list[list[dict]] = []
  i = 0
  while i + len(group_sizes) <= len(lines):
    if tuple(sizes[i:i + len(group_sizes)]) == group_sizes:
      replays.append([e for row in lines[i:i + len(group_sizes)] for e in row.get("entries", [])])
      i += len(group_sizes)
    else:
      i += 1
  return replays


def _per_name_table(replays: list[list[dict]]) -> dict[str, dict]:
  by_name: dict[str, list[float]] = defaultdict(list)
  counts: dict[str, list[int]] = defaultdict(list)
  for replay in replays:
    sums: dict[str, float] = defaultdict(float)
    cnt: dict[str, int] = defaultdict(int)
    for e in replay:
      name = canon(e.get("name", ""))
      sums[name] += float(e.get("duration", 0.0))
      cnt[name] += 1
    for name, val in sums.items():
      by_name[name].append(val)
      counts[name].append(cnt[name])
  return {name: {"median_us": round(statistics.median(vals), 3),
                 "mean_us": round(statistics.mean(vals), 3),
                 "per_replay_count": statistics.median(counts[name]) if counts[name] else 0,
                 "replay_samples": len(vals)}
          for name, vals in sorted(by_name.items())}


def _gpu_state() -> dict:
  fields = ("name", "clocks.sm", "clocks.mem", "persistence_mode", "power.draw", "temperature.gpu")
  raw = subprocess.check_output(["nvidia-smi", f"--query-gpu={','.join(fields)}",
                                 "--format=csv,noheader,nounits"], text=True).strip()
  return dict(zip(fields, [x.strip() for x in raw.split(",")]))


# ---------------------------------------------------------------------------
# Child capture
# ---------------------------------------------------------------------------

_TRACKED: list = []


def _install_graph_tracker() -> None:
  import tinygrad.runtime.graph.hcq as hcq
  orig = hcq.HCQGraph.__init__
  def init(self, *args, **kwargs):
    orig(self, *args, **kwargs)
    _TRACKED.append(self)
  hcq.HCQGraph.__init__ = init


def _flush_final_timestamps() -> None:
  for graph in _TRACKED:
    try:
      if getattr(graph, "kickoff_value", 0) >= 1:
        graph.collect_timestamps()
    except Exception:
      pass


def run_child(arm: str, depth: int, count: int, max_context: int, reps: int,
              profile_jsonl: pathlib.Path, out: pathlib.Path) -> dict:
  os.environ["DEV"] = "NV"
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = str(profile_jsonl)
  profile_jsonl.unlink(missing_ok=True)

  _install_graph_tracker()

  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_shared_q8_progressive_qualification import _settled_continuous_windows
  from extra.llm_research.decode.q4k_gate_up_four_warp_wall_bracket import _set_admission

  dev = Device["NV"]
  model = _load(MODEL, max_context)
  admitted = _set_admission(model, arm == "candidate")
  model._decode_direct_greedy_promoted = False
  model._decode_feedback_pingpong_promoted = False
  gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
  try:
    settled = _settled_continuous_windows(gen, dev, count, reps)
  finally:
    gen.close()
  dev.synchronize()
  _flush_final_timestamps()
  dev.synchronize()

  lines = [json.loads(line) for line in profile_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
  sizes = [len(x.get("entries", [])) for x in lines]
  replays = _complete_replays(lines)
  steady = replays[3:] if len(replays) > 3 else replays
  ledger = {
    "node_count": statistics.median([r["node_count"] for r in (_replay_metrics(x) for x in steady)]) if steady else None,
    "node_sum_us": round(statistics.median([_replay_metrics(x)["node_sum_us"] for x in steady]), 3) if steady else None,
    "union_us": round(statistics.median([_replay_metrics(x)["union_us"] for x in steady]), 3) if steady else None,
    "overlap_us": round(statistics.median([_replay_metrics(x)["overlap_us"] for x in steady]), 3) if steady else None,
    "span_us": round(statistics.median([_replay_metrics(x)["span_us"] for x in steady]), 3) if steady else None,
  }
  table = _per_name_table(steady)

  result = {
    "schema": "tinygrad.nv_gateup_fourwarp_profile_closure.v1",
    "arm": arm,
    "depth": depth, "count": count, "reps": reps, "max_context": max_context,
    "gpu_state": _gpu_state(),
    "admitted_gate_up_blocks": admitted,
    "settled": settled,
    "profile_jsonl": str(profile_jsonl),
    "group_size_histogram": dict(sorted(Counter(sizes).items())),
    "complete_replay_count": len(replays),
    "steady_replay_count": len(steady),
    "ledger": ledger,
    "per_name_table": table,
  }
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _child(arm: str, root: pathlib.Path, depth: int, count: int, max_context: int, reps: int) -> dict:
  out = root / f"{arm}.json"
  profile = root / f"{arm}.profile.jsonl"
  cmd = ["timeout", "1800", "flock", "-w", "600", LOCK, str(PYTHON),
         str(pathlib.Path(__file__).resolve()), "--arm", arm,
         "--depth", str(depth), "--count", str(count), "--max-context", str(max_context),
         "--reps", str(reps), "--profile-jsonl", str(profile), "--out", str(out)]
  run = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
  if run.returncode:
    raise RuntimeError(f"{arm} failed rc={run.returncode}: {run.stderr[-4000:]}")
  return json.loads(out.read_text())


def run_driver(depth: int, count: int, max_context: int, reps: int, out: pathlib.Path, reuse_existing:bool=False) -> dict:
  root = pathlib.Path(str(out).removesuffix(".json"))
  root.mkdir(parents=True, exist_ok=True)
  arms = ([json.loads((root / f"{arm}.json").read_text()) for arm in ("control_a", "candidate", "control_c")]
          if reuse_existing else
          [_child("control_a", root, depth, count, max_context, reps),
           _child("candidate", root, depth, count, max_context, reps),
           _child("control_c", root, depth, count, max_context, reps)])

  if reuse_existing:
    for arm in arms:
      replays = load_replays_raw = _complete_replays([
        json.loads(l) for l in pathlib.Path(arm["profile_jsonl"]).read_text().splitlines() if l.strip()])
      steady = load_replays_raw[3:]
      metrics = [_replay_metrics(x) for x in steady]
      arm["ledger"] = {key: round(statistics.median(float(m[key]) for m in metrics), 3)
                       for key in ("node_sum_us", "union_us", "overlap_us", "span_us")}

  def num(arm, key):
    val = arm["ledger"].get(key)
    return float(val) if val is not None else None
  def wall(arm):
    return float(arm["settled"]["median_ms_per_token"]) * 1000.0

  control_wall = statistics.median((wall(arms[0]), wall(arms[2])))
  candidate_wall = wall(arms[1])
  control = {key: statistics.median([num(arms[0], key), num(arms[2], key)]) for key in
             ("node_sum_us", "union_us", "overlap_us", "span_us")}
  candidate = {key: num(arms[1], key) for key in ("node_sum_us", "union_us", "overlap_us", "span_us")}

  # Per-row deltas in the profiled domain.
  def load_replays(arm: dict) -> list[list[dict]]:
    lines = [json.loads(l) for l in pathlib.Path(arm["profile_jsonl"]).read_text().splitlines() if l.strip()]
    return _complete_replays(lines)[3:]
  ctrl_table = _per_name_table(load_replays(arms[0]) + load_replays(arms[2]))
  cand_table = _per_name_table(load_replays(arms[1]))

  def _us(table: dict, name: str) -> float:
    val = table.get(name, {}).get("median_us")
    return float(val) if val is not None else 0.0

  all_names = sorted(set(ctrl_table) | set(cand_table))
  deltas = {}
  for name in all_names:
    c, v = _us(ctrl_table, name), _us(cand_table, name)
    in_c, in_v = name in ctrl_table, name in cand_table
    deltas[name] = {"control_us": round(c, 3), "candidate_us": round(v, 3),
                    "delta_us": round(v - c, 3),
                    "presence": "both" if in_c and in_v else ("candidate_only" if in_v else "control_only")}

  # The gate/up admission swaps kernel names instead of keeping them stable:
  # Current control and candidate both store fp16 in-kernel.  Bucket the exact
  # installed vector route against the four-warp+vector research spelling.
  ctrl_gate_names = ["q4k_g3_lanemap_gemv_w1w3vec16_12288_4096"]
  cand_gate_names = ["q4k_gate_up_four_warp_vec_fp16_12288_4096"]
  down_names = ["q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd",
                "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd"]
  gate_delta = sum(_us(cand_table, n) for n in cand_gate_names) - sum(_us(ctrl_table, n) for n in ctrl_gate_names)
  down_delta = sum(deltas.get(n, {}).get("delta_us", 0.0) for n in down_names)
  excluded = set(ctrl_gate_names) | set(cand_gate_names) | set(down_names)
  named_other_delta = round(sum(v["delta_us"] for n, v in deltas.items() if n not in excluded), 3)

  node_sum_delta = round(candidate["node_sum_us"] - control["node_sum_us"], 3)
  overlap_delta = round(candidate["overlap_us"] - control["overlap_us"], 3)
  union_delta = round(candidate["union_us"] - control["union_us"], 3)
  host_gap_delta = round((candidate_wall - control_wall) - node_sum_delta, 3)
  wall_delta = round(candidate_wall - control_wall, 3)

  closure = {
    "wall_delta_us": wall_delta,
    "node_sum_delta_us": node_sum_delta,
    "union_delta_us": union_delta,
    "overlap_delta_us": overlap_delta,
    "host_gap_delta_us": host_gap_delta,
    "gate_up_delta_us": round(gate_delta, 3),
    "down_q4_q6_delta_us": round(down_delta, 3),
    "other_rows_delta_us": named_other_delta,
    "identity": {
      "wall_minus_node_sum_minus_host_gap_us": round(wall_delta - node_sum_delta - host_gap_delta, 6),
      "node_sum_minus_gate_minus_down_minus_other_us":
        round(node_sum_delta - round(gate_delta, 3) - round(down_delta, 3) - named_other_delta, 6),
    },
  }

  hashes = {arm["settled"]["token_stream_hash"] for arm in arms}
  result = {
    "schema": "tinygrad.nv_gateup_fourwarp_profile_closure.v1",
    "mode": "reverse-bracket-profile",
    "depth": depth, "count": count, "reps": reps, "max_context": max_context,
    "all_token_hashes_equal": len(hashes) == 1,
    "token_stream_hash": sorted(hashes)[0] if len(hashes) == 1 else sorted(hashes),
    "walls_us_per_token": {"control_a": wall(arms[0]), "candidate": candidate_wall,
                           "control_c": wall(arms[2]), "control_midpoint": control_wall},
    "ledger_control": control,
    "ledger_candidate": candidate,
    "closure": closure,
    "per_row_deltas_us": deltas,
    "top_abs_deltas_us": sorted(deltas.items(), key=lambda kv: abs(kv[1]["delta_us"]), reverse=True)[:24],
    "launch_metadata": {
      "control_gate_up": {"grid": [12288, 1, 1], "block": [32, 1, 1], "registers": 74, "smem_bytes": 0},
      "candidate_gate_up": {"grid": [12288, 1, 1], "block": [32, 4, 1], "registers": 56, "smem_bytes": 32},
    },
  }
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--arm", choices=("control_a", "candidate", "control_c"))
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=32)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--profile-jsonl", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--reuse-existing", action="store_true")
  args = ap.parse_args()

  if args.arm:
    if args.profile_jsonl is None:
      raise SystemExit("--profile-jsonl is required for child arms")
    result = run_child(args.arm, args.depth, args.count, args.max_context, args.reps,
                       args.profile_jsonl, args.out)
    print(json.dumps({k: v for k, v in result.items() if k != "per_name_table"}, indent=2, sort_keys=True))
    return 0

  result = run_driver(args.depth, args.count, args.max_context, args.reps, args.out, args.reuse_existing)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
