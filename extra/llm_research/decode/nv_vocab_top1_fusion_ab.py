#!/usr/bin/env python3
"""Fresh-process A/B harness for the fused vocab-head top-1 route.

This is the GPU-side sibling of ``nv_vocab_top1_fusion_cpu_microgate.py``.  The
control arm runs the production greedy LM-head argmax tail (four aux kernels);
the candidate arm installs ``_decode_vocab_top1_lease`` so ``forward_greedy``
uses the packed per-tile (max,index) epilogue plus the ordinary scheduler
cross-tile reduce.
The only correctness gate on the fused arm is the exact token stream, because
the fused route deliberately does not materialise the 151936-row logits.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, sys

from extra.llm_research.decode.nv_fusion_cost_model import reconcile_cost_prediction
from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _settled_continuous_windows, _validate_run_extent)


_TAIL_PREFIXES = ("E_1187_32_4", "r_32_4_1187", "r_128_16_8_1187", "r_16_8")
_TAIL_FAMILY = ("1187", "r_16_8")
_VOCAB_LEGACY_PREFIX = "q6k_gen_coop_151936_4096_inkernel"
_VOCAB_EPI_NAME = "q6k_gen_coop_151936_4096_inkernel_epi_vocabtop1"
_VOCAB_REDUCE_NAME = "q6k_vocab_top1_reduce_151936_4096"


# Predicted-wall-delta contract (nv_fusion_cost_model.py).  llama's vocab head is one fused
# mul_mat_vec_q kernel with no separate argmax tail; our control tail is the four-kernel chain
# (~54.5us, l4-vocab-substrate-fusion-measurement-record-20260803.md).  The naive launch model
# predicts the candidate removes that tail at the cost of the packed epilogue plus one tiny
# reduce and the held-copy barrier.  The wall bracket reconciles the measured delta against
# that envelope; a measured delta on the opposite side of zero CONTRADICTS the premise.
COST_PREDICTION = {
  "contract": "before implementing, derive the predicted wall delta from the llama reference shape plus per-launch arithmetic; the wall bracket then confirms it or explains the gap",
  "llama_reference": "one fused mul_mat_vec_q vocab kernel (~303.75us) with no separate argmax tail; our control tail is E_1187_32_4 + r_32_4_1187 + r_128_16_8_1187 + r_16_8 (~54.5us)",
  "arithmetic": {
    "formula": "point = added_epilogue_mass - removed_tail_mass; positive = candidate slower",
    "removed_tail_mass": "four tail kernels x control medians (~54.5us total)",
    "added_epilogue_mass": "packed per-tile epilogue extra + ordinary scheduler u64 cross-tile reduce + held-copy barrier",
    "launch_us_range": [1.0, 2.0],
    "envelope": "best case: tail fully removed at zero added cost; pessimistic: added mass at twice the launch floor",
  },
  "tolerance_us": 20.0,
  "unmodeled": ["in-kernel packed-key compare/reduce", "cross-token launch hiding", "held-copy barrier serialization"],
}

TAIL_CHAIN_REFERENCE_US = 54.5
ADDED_EPILOGUE_FLOOR_US = 4.5


def validate_cost_prediction(bracket: dict, control_census: dict | None = None,
                             candidate_census: dict | None = None) -> dict:
  """Predicted-wall-delta gate.  The measured bracket delta must confirm the llama-shaped
  arithmetic or explain the gap; a measured delta outside the envelope on the opposite side
  of zero CONTRADICTS the premise and fails the campaign closed."""
  point = ADDED_EPILOGUE_FLOOR_US - TAIL_CHAIN_REFERENCE_US
  lo = -TAIL_CHAIN_REFERENCE_US
  hi = 2.0 * ADDED_EPILOGUE_FLOOR_US - TAIL_CHAIN_REFERENCE_US
  measured_delta_us = bracket["candidate_minus_control_ms"] * 1000.0
  reconciliation = reconcile_cost_prediction(measured_delta_us,
    {"predicted_delta_us": point, "range_us": [lo, hi]},
    tolerance_us=COST_PREDICTION["tolerance_us"])
  return {"run": True, "result": "PASS" if reconciliation["result"] != "CONTRADICTED" else "FAIL",
          "contract": COST_PREDICTION,
          "prediction": {"predicted_delta_us": point, "range_us": [lo, hi]},
          "reconciliation": reconciliation, "measured_delta_us": measured_delta_us,
          "note": "positive measured delta = candidate slower; CONTRADICTED fails the campaign closed"}


def _install(model, lease:bool) -> None:
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = False
  if lease:
    model._decode_vocab_top1_lease = True
  else:
    if hasattr(model, "_decode_vocab_top1_lease"): delattr(model, "_decode_vocab_top1_lease")


def child(model_path:str, depth:int, count:int, max_context:int, lease:bool) -> dict:
  from tinygrad import Tensor, UOp
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  model = _load(model_path, max_context)
  _install(model, lease)
  gen = model.generate(_prompt(model_path, depth), chunk_size=32, temperature=0.0)
  try: prelude = int(next(gen))
  finally: gen.close()
  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, max_context - 1)
  census = GraphAdmissionCensus()
  tokens = []
  for i in range(count):
    with observe_graph_admissions(census):
      sampled = model(token, start_pos.bind(depth + 1 + i), temp, use_flash=False, greedy=True).realize()
      sampled_id = int(sampled.item())
    tokens.append(sampled_id)
    token = sampled
  programs = [r.program_name for r in census.records if r.program_name]
  vocab_epi = programs.count(_VOCAB_EPI_NAME)
  vocab_legacy = sum(p.startswith(_VOCAB_LEGACY_PREFIX) and p != _VOCAB_EPI_NAME for p in programs)
  vocab_reduce = programs.count(_VOCAB_REDUCE_NAME)
  tail = [p for p in programs if any(p.startswith(prefix) for prefix in _TAIL_PREFIXES)]
  tail_family = [p for p in programs if p != _VOCAB_EPI_NAME and p != _VOCAB_LEGACY_PREFIX
                 and p != _VOCAB_REDUCE_NAME and any(f in p for f in _TAIL_FAMILY)]
  topology = {
    "lease": lease,
    "vocab_top1_epi_count": vocab_epi,
    "legacy_vocab_count": vocab_legacy,
    "vocab_top1_reduce_count": vocab_reduce,
    "tail_programs": tail,
    "tail_program_count": len(tail),
    "tail_family_programs": tail_family,
    "tail_family_program_count": len(tail_family),
    "program_count": len(programs),
    "pass": (lease and vocab_epi == 1 and vocab_legacy == 0 and vocab_reduce == 0 and len(tail) == 0)
            or (not lease and vocab_epi == 0 and vocab_reduce == 0 and vocab_legacy == 1 and len(tail) == 4),
  }
  # Hard gate (lease arm only): the fused route must remove the ENTIRE argmax
  # tail, not rename it. A tail-family kernel (any E_*/r_* with a 1187 vocab
  # extent, or the r_16_8 winner reduce) after the vocab GEMV means the argmax
  # still runs as separate kernels; the wall cannot move while that chain
  # survives. Observed 2026-08-17: the scheduler lowers
  # packed_argmax_from_tile_keys into E_1187_16_4 + r_16_4_1187, which the old
  # prefix gate (E_1187_32_4 / r_32_4_1187 / ...) missed. The control arm
  # legitimately carries the 4-kernel legacy chain, so the zero-tail gate only
  # applies when the lease is installed.
  if lease and len(tail_family) != 0: topology["pass"] = False
  if not topology["pass"]: raise RuntimeError(f"vocab top-1 topology failed: {topology}")
  return {"schema": "tinygrad.nv_vocab_top1_fusion.v1", "mode": "child", "lease": lease,
          "depth": depth, "count": count, "prelude_token": prelude, "tokens": tokens,
          "tokens_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest(),
          "topology": topology, "program_names": programs}


def timing_child(model_path:str, depth:int, count:int, max_context:int, reps:int, lease:bool) -> dict:
  from tinygrad import Device
  model = _load(model_path, max_context)
  _install(model, lease)
  prompt = _prompt(model_path, depth)
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  try:
    settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], count, reps)
  finally:
    gen.close()
  return {"schema": "tinygrad.nv_vocab_top1_fusion.v1", "mode": "timing-child", "lease": lease,
          "depth": depth, "count": count, "reps": reps, **settled}


def _cmd(args, mode:str, lease:bool, out:pathlib.Path) -> list[str]:
  cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", mode, "--model", args.model,
         "--depth", str(args.depth), "--count", str(args.count), "--max-context", str(args.max_context),
         "--reps", str(args.reps), "--out", str(out)]
  if lease: cmd.append("--lease")
  return cmd


def _run_child(args, mode:str, lease:bool, out:pathlib.Path) -> dict:
  run = subprocess.run(["timeout", f"{args.timeout}s", "flock", "-w", str(args.lock_wait), args.lock,
                        *_cmd(args, mode, lease, out)],
                       text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode: raise RuntimeError(f"{mode} lease={lease} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def ab(args) -> dict:
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  control_out = root / "control.json"
  candidate_out = root / "candidate.json"
  control = _run_child(args, "child", False, control_out)
  candidate = _run_child(args, "child", True, candidate_out)
  if control["tokens_sha256"] != candidate["tokens_sha256"]:
    raise RuntimeError(f"token stream mismatch: control={control['tokens_sha256']} candidate={candidate['tokens_sha256']}")
  rows = []
  for seq, lease in enumerate((False, True, False)):
    out = root / f"timing-{seq}.json"
    rows.append(_run_child(args, "timing-child", lease, out))
  bracket_control = statistics.median((rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"]))
  candidate_ms = rows[1]["median_ms_per_token"]
  hashes = {row["token_stream_hash"] for row in rows}
  result = {"schema": "tinygrad.nv_vocab_top1_fusion.v1", "mode": "ab", "control_child": control,
          "candidate_child": candidate, "token_streams_equal": control["tokens_sha256"] == candidate["tokens_sha256"],
          "timing_arms": rows, "timing_token_hashes_equal": len(hashes) == 1,
          "control_bracket_median_ms": bracket_control, "candidate_ms": candidate_ms,
          "candidate_minus_control_ms": candidate_ms - bracket_control,
          "candidate_speedup_pct": (bracket_control / candidate_ms - 1) * 100,
          "verdict": "WALL_PASS" if len(hashes) == 1 and candidate_ms < bracket_control else "NO_GO_WALL"}
  result["cost_gate"] = validate_cost_prediction(result)
  return result


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("child", "timing-child", "ab"), required=True)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=8)
  ap.add_argument("--max-context", type=int, default=1024)
  ap.add_argument("--reps", type=int, default=3)
  ap.add_argument("--lease", action="store_true")
  ap.add_argument("--out", required=True)
  ap.add_argument("--timeout", type=int, default=1200)
  ap.add_argument("--lock-wait", type=int, default=300)
  ap.add_argument("--lock", default="/tmp/gpu-bench.lock")
  args = ap.parse_args()
  _validate_run_extent(args.depth, args.count, args.max_context, args.reps, args.mode == "timing-child")
  result = ab(args) if args.mode == "ab" else \
           timing_child(args.model, args.depth, args.count, args.max_context, args.reps, args.lease) if args.mode == "timing-child" else \
           child(args.model, args.depth, args.count, args.max_context, args.lease)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
