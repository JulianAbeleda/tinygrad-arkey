#!/usr/bin/env python3
"""Partition the installed wide-flash production-conditioning residual.

Capture the exact MAXC1024 programs and live buffers from layer 0, then time
the score program on native NV HCQ under progressively stronger conditioning:

  hot          score immediately after an untimed score replay
  completions  Q and KV completion producers between warmup and score
  qkv_prefix   Q/K/V projection bodies plus both completion producers
  stream-read  a 16--128 MiB read-only working-set sweep before score

Only the final score launch is timestamped. This is measurement tooling; it
does not alter a production route or policy.
"""
from __future__ import annotations

import argparse, json, os, pathlib, statistics, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from nv_r_residual_cache_dispatch_probe import _alloc, _make_queue
from nv_l2_eviction_decisive import _compile_stream_read

SCHEMA = "tinygrad.nv_flash_wide_conditioning_probe.v1"
MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TARGET = "flash_vec_llama_score_pv_32_128_8_widekv16"
QPROJ = "q4k_g3_lanemap_gemv_vec_4096_4096"
KPROJ = "q4k_g3_lanemap_gemv_vec_1024_4096"
VPROJ = "q6k_v_four_warp_fp16_direct_1024_4096"
QDONE = "reduce_output_rmsnorm_rope_32_128"
KVDONE = "reduce_output_rmsnorm_rope_kv_cache_8_128"
SELECTED = (TARGET, QPROJ, KPROJ, VPROJ, QDONE, KVDONE)

# Latest installed device-ledger reference. The probe reports but does not fit
# against this value; a fresh production census remains the endpoint authority.
PRODUCTION_SCORE_US = 6.184889
STREAM_MIB = (16, 32, 64, 96, 128)
STREAM_BLOCK = 256


def _capture_programs(depth:int, max_context:int, tokens:int, profile_jsonl:pathlib.Path) -> dict:
  import tinygrad.runtime.ops_nv as ops_nv
  from tinygrad import Device
  from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
  from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import _flush_final_timestamps, _install_graph_tracker

  captured, seen = {}, {}
  _install_graph_tracker()
  orig_call = ops_nv.NVProgram.__call__

  def patched_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1), vals=(), wait=False, timeout=None):
    name = self.name
    if name in SELECTED:
      seen[name] = seen.get(name, 0) + 1
      if name not in captured:
        captured[name] = {"program":self, "bufs":tuple(bufs), "vals":tuple(vals),
                          "global_size":tuple(global_size), "local_size":tuple(local_size),
                          "buf_sizes":[int(x.size) for x in bufs], "occurrence":seen[name]}
    return orig_call(self, *bufs, global_size=global_size, local_size=local_size,
                     vals=vals, wait=wait, timeout=timeout)

  ops_nv.NVProgram.__call__ = patched_call
  try:
    model = _load(MODEL, max_context)
    model._decode_direct_greedy_promoted = True
    model._decode_feedback_pingpong_promoted = True
    gen = model.generate(_prompt(MODEL, depth), chunk_size=32, temperature=0.0)
    try:
      produced = [int(next(gen)) for _ in range(tokens)]
      Device["NV"].synchronize()
    finally: gen.close()
    _flush_final_timestamps()
    Device["NV"].synchronize()
  finally: ops_nv.NVProgram.__call__ = orig_call
  captured["_tokens"] = produced
  captured["_seen"] = seen
  return captured


def _production_score(profile_jsonl:pathlib.Path) -> dict:
  lines = [json.loads(x) for x in profile_jsonl.read_text().splitlines() if x.strip()]
  # Direct-greedy ping-pong emits one decode replay as four graph groups whose
  # target populations are 3+5+12+16 = 36 layers. Group by that invariant
  # instead of the legacy non-greedy graph-size signature.
  replays, current = [], []
  for row in lines:
    current += [x for x in row.get("entries", []) if x.get("name") == TARGET]
    if len(current) == 36:
      replays.append(current)
      current = []
    elif len(current) > 36:
      raise RuntimeError(f"target population exceeded 36 while grouping {profile_jsonl}")
  steady = replays[3:] if len(replays) > 3 else replays
  per_replay, counts = [], []
  for replay in steady:
    vals = [float(x["duration"]) for x in replay]
    if vals:
      per_replay.append(sum(vals)/len(vals))
      counts.append(len(vals))
  if not per_replay: raise RuntimeError(f"no {TARGET} entries in {profile_jsonl}")
  return {"complete_replays":len(replays), "steady_replays":len(steady),
          "target_counts":counts, "per_replay_mean_us":[round(x, 6) for x in per_replay],
          "median_us":round(statistics.median(per_replay), 6)}


def _exec(q, rec):
  q.exec(rec["program"], rec["program"].fill_kernargs(rec["bufs"], vals=rec["vals"]),
         rec["global_size"], rec["local_size"])


def _measure(dev, target:dict, prefix:list[dict], conditioner, n:int, timeout_s:float) -> list[float]:
  out = []
  for _ in range(n):
    q = _make_queue(dev)
    # Standardize every sample from the same target-hot state.
    _exec(q, target)
    if conditioner is not None:
      prg, bufs, grid, block = conditioner
      q.exec(prg, prg.fill_kernargs(bufs), grid, block)
    for rec in prefix: _exec(q, rec)
    start, end = dev.new_signal(), dev.new_signal()
    q.timestamp(start)
    _exec(q, target)
    q.timestamp(end)
    q.signal(dev.timeline_signal, dev.next_timeline()).submit(dev)
    dev.synchronize(timeout=int(timeout_s * 1000))
    out.append(float(end.timestamp - start.timestamp))
  return out


def _summary(xs:list[float], warmup:int) -> dict:
  ys = xs[warmup:]
  return {"median_us":round(statistics.median(ys), 3), "mean_us":round(statistics.mean(ys), 3),
          "min_us":round(min(ys), 3), "max_us":round(max(ys), 3),
          "samples_us":[round(x, 3) for x in ys]}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--capture-tokens", type=int, default=8)
  ap.add_argument("--n", type=int, default=40)
  ap.add_argument("--warmup", type=int, default=8)
  ap.add_argument("--timeout-s", type=float, default=180.0)
  ap.add_argument("--profile-jsonl", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  profile_jsonl = args.profile_jsonl or pathlib.Path(str(args.out)+".profile.jsonl")
  profile_jsonl.parent.mkdir(parents=True, exist_ok=True)
  profile_jsonl.unlink(missing_ok=True)
  os.environ.update(DEV="NV", PROFILE="1", HCQ_GRAPH_PROFILE_JSON=str(profile_jsonl))

  captured = _capture_programs(args.depth, args.max_context, args.capture_tokens, profile_jsonl)
  missing = [x for x in SELECTED if x not in captured]
  if missing: raise RuntimeError(f"program capture missing {missing}; seen={captured.get('_seen')}")

  from tinygrad import Device
  from tinygrad.runtime.ops_nv import NVProgram
  from extra.llm_research.decode.qk_norm_rope_wall_bracket import _gpu_state

  dev = Device["NV"]
  production = _production_score(profile_jsonl)
  # Read-only streams evict by capacity without adding dirty-L2 writeback to
  # the score interval.
  stream_buf = _alloc(dev, max(STREAM_MIB) * 1024 * 1024)
  stream_out = _alloc(dev, 4)
  dev.allocator._copyin(stream_buf, memoryview(bytearray(stream_buf.size)))
  dev.allocator._copyin(stream_out, memoryview(bytearray(stream_out.size)))
  dev.synchronize()

  streams = {}
  for mib in STREAM_MIB:
    words = mib * 1024 * 1024 // 4
    tag = f"flash_condition_{mib}mib"
    prg = NVProgram(dev, f"nv_l2_stream_{tag}", _compile_stream_read(dev, words, tag))
    streams[f"stream_read_{mib}mib"] = (prg, (stream_buf, stream_out),
      ((words + STREAM_BLOCK - 1) // STREAM_BLOCK, 1, 1), (STREAM_BLOCK, 1, 1))

  target = captured[TARGET]
  qkv_prefix = [captured[QPROJ], captured[KPROJ], captured[VPROJ], captured[QDONE], captured[KVDONE]]
  definitions = {
    "hot_a": ([], None),
    "completions": ([captured[QDONE], captured[KVDONE]], None),
    "qkv_prefix": (qkv_prefix, None),
    **{name:([], cond) for name, cond in streams.items()},
    "stream_read_64mib_qkv_prefix": (qkv_prefix, streams["stream_read_64mib"]),
    "stream_read_96mib_qkv_prefix": (qkv_prefix, streams["stream_read_96mib"]),
    "stream_read_128mib_qkv_prefix": (qkv_prefix, streams["stream_read_128mib"]),
    "hot_c": ([], None),
  }
  rows = {}
  for arm, (prefix, conditioner) in definitions.items():
    rows[arm] = _summary(_measure(dev, target, prefix, conditioner, args.n, args.timeout_s), args.warmup)
    print(json.dumps({arm:rows[arm]}, sort_keys=True), flush=True)

  hot_mid = statistics.median((rows["hot_a"]["median_us"], rows["hot_c"]["median_us"]))
  production_residual = production["median_us"] - hot_mid
  prefix_recovery = rows["qkv_prefix"]["median_us"] - hot_mid
  conditioned_arms = (*streams, "stream_read_64mib_qkv_prefix", "stream_read_96mib_qkv_prefix",
                      "stream_read_128mib_qkv_prefix")
  nearest_stream = min(conditioned_arms, key=lambda x:abs(rows[x]["median_us"]-production["median_us"]))
  payload = {
    "schema":SCHEMA,
    "commit":subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
    "gpu_state":_gpu_state(),
    "shape":{"depth":args.depth, "max_context":args.max_context, "capture_tokens":args.capture_tokens},
    "target":TARGET,
    "target_buf_sizes":target["buf_sizes"],
    "target_vals":list(target["vals"]),
    "target_global_size":list(target["global_size"]),
    "target_local_size":list(target["local_size"]),
    "captured_occurrences":{x:captured[x]["occurrence"] for x in SELECTED},
    "capture_seen":captured["_seen"],
    "capture_tokens_out":captured["_tokens"],
    "method":"exact installed live programs/buffers; native NV HCQ; untimed hot score before every sample; only final score timestamped",
    "production_profile_jsonl":str(profile_jsonl),
    "fresh_production_score":production,
    "n_per_arm":args.n-args.warmup,
    "stream_read_mib":list(STREAM_MIB),
    "rows":rows,
    "reconciliation":{
      "hot_midpoint_us":round(hot_mid, 3),
      "retained_production_score_us":PRODUCTION_SCORE_US,
      "fresh_production_score_us":production["median_us"],
      "production_minus_hot_us":round(production_residual, 3),
      "completions_minus_hot_us":round(rows["completions"]["median_us"]-hot_mid, 3),
      "qkv_prefix_minus_hot_us":round(prefix_recovery, 3),
      "qkv_prefix_share_of_production_residual":round(prefix_recovery/production_residual, 3) if production_residual > 0 else None,
      "stream_read_minus_hot_us":{x:round(rows[x]["median_us"]-hot_mid, 3) for x in streams},
      "cold_qkv_prefix_minus_hot_us":{x:round(rows[x]["median_us"]-hot_mid, 3) for x in
        ("stream_read_64mib_qkv_prefix", "stream_read_96mib_qkv_prefix", "stream_read_128mib_qkv_prefix")},
      "nearest_stream_arm":nearest_stream,
      "nearest_stream_us":rows[nearest_stream]["median_us"],
      "nearest_stream_minus_production_us":round(rows[nearest_stream]["median_us"]-production["median_us"], 3),
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload["reconciliation"], indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
