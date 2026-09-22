#!/usr/bin/env python3
"""In-loop census for the Q4 FFN-down quad-u128-smem load style vs the installed scalar GEMV.

Scope: `docs/task_workflow/input/nv-q4-down-quad-re-census-20260813.md` (authoritative floor
`nv-gemv-core-deficit-correction-20260813.md`: llama Q4 FFN-down floor 19.23 us/node, installed
control 26.75 us in-loop). Question: does the quad geometry that won standalone
(q4kd_16row_128thr_u128_quad_xsmem, 11.43 us) survive the real decode loop at d512? The MC2 lesson
is that only the in-loop census counts (the gate/up quad regressed 23.2 -> 49.2 us in-loop), so a
standalone win is never treated as evidence here.

The candidate arm leases a small set of FFN-down blocks with `Q4KFFNDownQuadAdmission`
(decode_routes research-only hook); the installed scalar path is the control. Both arms run the
real `model.generate` decode loop at fixed depth d512. Per-kernel medians come from the DEBUG=2
census token(s) (house protocol: prime with DEBUG=0, capture with DEBUG=2), token sha256 + first
token come from the settled wall windows, and the reverse bracket runs control before/after
candidate with equal token hashes and candidate wall < control wall.

Protocol: Qwen3-8B-Q4_K_M, d512, `[1]*depth` prompt (house census prompt), count measured tokens per
window x reps settled windows, max_context 1024. Run single arms under
`flock -w 90 /tmp/gpu-bench.lock timeout 900 python3 ... --mode child --out <path>`; run the
`--mode bracket` driver WITHOUT an outer flock (each child acquires the lock itself).

Regimes: the per-kernel census runs under `JIT=2` (no CUDA-graph batching, the exact regime of the
house per-kernel census -- `DEBUG=2` prime-token trace; the decode is batched into graphs at JIT=1
today, which hides per-kernel rows). This reproduces the authoritative installed control in-loop
median (26.37-26.75 us). The wall bracket runs under `JIT=1` (production CUDA-graph decode, ~5.2
ms/token) so a real in-loop win on the leased blocks is resolvable; direct-exec wall is CPU-bound
(~52 ms/token) and would swamp the kernel delta. Both regimes produce the same greedy token stream.
The two phases run in one child: wall first, then all model JITs are reset and the census phase
re-captures under JIT=2.
"""
from __future__ import annotations

import argparse, contextlib, hashlib, io, json, os, pathlib, re, statistics, subprocess, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad.helpers import Context
from tinygrad.llm.decode_routes import Q4KFFNDownQuadAdmission
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _settled_high_side_filter, _validate_run_extent)

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
CONTROL_NAME = "q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288"
QUAD_NAME = "q4k_g3_lanemap_gemv_quad_epi_ffnresadd_4096_12288"
Q4_FFN_DOWN_INDICES = (4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29)
LLAMA_Q4_DOWN_FLOOR_US = 19.23
INSTALLED_CONTROL_US = 26.75
TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")


def _normalize(spec: str) -> tuple[int, ...]:
  if not spec.strip(): return ()
  values = tuple(int(x) for x in spec.split(","))
  if values != tuple(sorted(set(values))) or any(x not in Q4_FFN_DOWN_INDICES for x in values):
    raise ValueError(f"indices must be an increasing subset of {Q4_FFN_DOWN_INDICES}")
  return values


def _install_quad(model, indices: tuple[int, ...]) -> list[int]:
  from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear
  for block in model.blk:
    linear = getattr(block, "ffn_down", None)
    if hasattr(linear, "_q4k_ffn_down_quad_admission"): delattr(linear, "_q4k_ffn_down_quad_admission")
  for index in indices:
    linear = model.blk[index].ffn_down
    if not isinstance(linear, Q4KPrimitiveLinear) or linear.route_role != "ffn_down" or \
       (linear.out_features, linear.in_features) != (4096, 12288):
      raise RuntimeError(f"block {index} is not the exact Q4_K FFN-down production role")
    linear._q4k_ffn_down_quad_admission = Q4KFFNDownQuadAdmission(index)
  return list(indices)


def _reset_decode_jits(model) -> int:
  from tinygrad import TinyJit
  reset = 0
  for value in vars(model).values():
    candidates = value if isinstance(value, (tuple, list)) else (value,)
    for item in candidates:
      if isinstance(item, TinyJit):
        item.reset(); reset += 1
  return reset


def parse_census_log(text: str) -> dict[str, list[float]]:
  per: dict[str, list[float]] = {}
  for line in text.splitlines():
    m = TM_RE.match(line)
    if not m: continue
    us = float(m.group(2)) * (1e-3 if m.group(3) == "ms" else 1.0)
    per.setdefault(m.group(1), []).append(us)
  return per


def _row(us: list[float]) -> dict | None:
  if not us: return None
  return {"count": len(us), "median_us": round(statistics.median(us), 3),
          "min_us": round(min(us), 3), "max_us": round(max(us), 3)}


def census_child(args) -> dict:
  from tinygrad import Device
  from tinygrad.llm.model import Transformer
  import tinygrad.llm.model as tgm
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  model, kv = Transformer.from_gguf(args.model, args.max_context)
  indices = _install_quad(model, args.indices) if args.quad else []
  prompt = [1] * args.depth
  dev = Device[Device.DEFAULT]
  # Wall phase: production JIT=1 (CUDA-graph decode), settled continuous windows.
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  with Context(JIT=1, DEBUG=0):
    prelude = int(next(gen))
    warmup = [int(next(gen)) for _ in range(6)]
    dev.synchronize()
    clock = time.perf_counter_ns
    samples, hashes, timed_tokens = [], [], []
    for _ in range(args.reps):
      started = clock(); window = [int(next(gen)) for _ in range(args.count)]; dev.synchronize()
      samples.append((clock() - started) / args.count / 1e6)
      timed_tokens.extend(window)
      hashes.append(hashlib.sha256(",".join(map(str, window)).encode()).hexdigest())
  gen.close()
  high_limit, accepted, rejected = _settled_high_side_filter(samples)

  # Census phase: reset all model JITs so the decode re-captures under JIT=2 (direct exec,
  # per-kernel DEBUG=2 rows) at the same depth as the wall phase.
  model.reset_generation_state()
  jits_reset = _reset_decode_jits(model)
  gen2 = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  with Context(JIT=2):
    with Context(DEBUG=0):
      prelude2 = int(next(gen2))
      warmup2 = [int(next(gen2)) for _ in range(6)]
      dev.synchronize()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
      with Context(DEBUG=2):
        census_tokens = [int(next(gen2)) for _ in range(args.census_tokens)]
    census = parse_census_log(buf.getvalue())
  gen2.close()

  control = _row(census.get(CONTROL_NAME, []))
  quad = _row(census.get(QUAD_NAME, []))
  per_kernel_us = {k: _row(v) for k, v in sorted(census.items())}
  total_kernels = sum(len(v) for v in census.values())
  total_us = sum(statistics.median(v) * len(v) for v in census.values())
  out = {
    "schema": "tinygrad.nv_q4_down_quad_census.v1", "mode": "child",
    "quad": bool(args.quad), "indices": indices,
    "depth": args.depth, "count": args.count, "reps": args.reps,
    "census_tokens": args.census_tokens, "max_context": args.max_context,
    "jits_reset": jits_reset,
    "prelude_token": prelude, "warmup_tokens": warmup,
    "census_prelude_token": prelude2, "census_tokens_sampled": census_tokens,
    "census": {"kernels_per_token": total_kernels, "kernel_us_total": round(total_us, 1),
               "control": control, "quad": quad, "per_kernel_us": per_kernel_us},
    "wall": {"prelude_token": prelude, "samples_ms_per_token": samples,
             "accepted_samples_ms_per_token": accepted,
             "external_contention_high_limit_ms": high_limit,
             "rejected_high_samples_ms_per_token": rejected,
             "median_ms_per_token": statistics.median(accepted),
             "token_hashes": hashes,
             "token_stream_hash": hashlib.sha256(",".join(map(str, timed_tokens)).encode()).hexdigest(),
             "first_token": timed_tokens[0] if timed_tokens else None,
             "timed_token_count": len(timed_tokens)},
  }
  return out


def _child_command(args, quad: bool, out: pathlib.Path) -> list[str]:
  cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", "child",
         "--model", args.model, "--depth", str(args.depth), "--count", str(args.count),
         "--reps", str(args.reps), "--census-tokens", str(args.census_tokens),
         "--max-context", str(args.max_context), "--indices", ",".join(map(str, args.indices)),
         "--out", str(out)]
  if quad: cmd.append("--quad")
  return cmd


def bracket(args) -> dict:
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  rows: dict[str, dict] = {}
  for label, quad in (("control-a", False), ("candidate", True), ("control-b", False)):
    out = root / f"{label}.json"
    run = subprocess.run(["timeout", f"{args.timeout}s", "flock", "-w", str(args.lock_wait), args.lock,
                          *_child_command(args, quad, out)],
                         text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if run.returncode:
      raise RuntimeError(f"{label} failed rc={run.returncode}: {run.stderr[-4000:]}")
    rows[label] = json.loads(out.read_text())
  controls = (rows["control-a"]["wall"]["median_ms_per_token"], rows["control-b"]["wall"]["median_ms_per_token"])
  control_wall = statistics.median(controls)
  candidate_wall = rows["candidate"]["wall"]["median_ms_per_token"]
  control_medians = [r["census"]["control"]["median_us"] for r in rows.values() if r["census"]["control"]]
  control_median_us = statistics.median(control_medians)
  quad_row = rows["candidate"]["census"]["quad"]
  quad_median_us = quad_row["median_us"] if quad_row else None
  hashes = {r["wall"]["token_stream_hash"] for r in rows.values()}
  firsts = {r["wall"]["first_token"] for r in rows.values()}
  gates = {
    "token_sha256_equal": len(hashes) == 1,
    "first_token_equal": len(firsts) == 1,
    "quad_in_loop_present": quad_median_us is not None,
    "quad_beats_llama_floor_19_23us": quad_median_us is not None and quad_median_us < LLAMA_Q4_DOWN_FLOOR_US,
    "quad_beats_installed_26_75us": quad_median_us is not None and quad_median_us < INSTALLED_CONTROL_US,
    "quad_beats_measured_control": quad_median_us is not None and quad_median_us < control_median_us,
    "wall_candidate_below_control": candidate_wall < control_wall,
  }
  gates["pass"] = all(gates.values())
  return {
    "schema": "tinygrad.nv_q4_down_quad_census.v1", "mode": "bracket",
    "indices": list(args.indices), "depth": args.depth, "count": args.count, "reps": args.reps,
    "control_median_ms_per_token": control_wall, "candidate_median_ms_per_token": candidate_wall,
    "candidate_minus_control_ms_per_token": candidate_wall - control_wall,
    "control_per_kernel_median_us": round(control_median_us, 3),
    "quad_per_kernel_median_us": quad_median_us,
    "llama_q4_down_floor_us_per_node": LLAMA_Q4_DOWN_FLOOR_US,
    "installed_control_book_us_per_node": INSTALLED_CONTROL_US,
    "token_stream_hashes": sorted(hashes), "first_tokens": sorted(firsts),
    "gates": gates, "verdict": "PASS" if gates["pass"] else "NO_GO",
    "arms": {label: {"out": str(root / f"{label}.json")} for label in rows},
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("child", "bracket"), required=True)
  ap.add_argument("--model", default=MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--census-tokens", type=int, default=3)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--indices", default="4,8,13,19,25,29")
  ap.add_argument("--quad", action="store_true", help="child: attach the quad admission to --indices")
  ap.add_argument("--out", required=True)
  ap.add_argument("--timeout", type=int, default=900)
  ap.add_argument("--lock-wait", type=int, default=90)
  ap.add_argument("--lock", default="/tmp/gpu-bench.lock")
  args = ap.parse_args()
  args.indices = _normalize(args.indices)
  _validate_run_extent(args.depth, args.count, args.max_context, args.reps, True)
  if args.mode == "child":
    row = census_child(args)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(json.dumps(row))
    os._exit(0)
  result = bracket(args)
  pathlib.Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
