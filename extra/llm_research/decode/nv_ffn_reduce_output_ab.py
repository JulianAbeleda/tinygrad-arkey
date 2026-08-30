#!/usr/bin/env python3
"""Per-site FFN reduce-output RMSNorm A/B at HEAD (bounded-fusion cell 8.3).

The candidate arm opens only ``_decode_reduce_output_ffn_rmsnorm_promoted``
on the model and every block while leaving the production q/k route and the
callify substrate exactly as loaded.  The control arm leaves the production
default: q/k promoted, FFN site closed.  The inter-arm topology contract is
therefore confined to three families: the ordinary FFN-norm reduce/epilogue
pair drops 36 and 36, and the 1_4096 fused-body family grows by 36, for a
net -36 programs with zero new weight materializations.

Gate order is fixed: exact full-logit SHA-256 equality, then the exact
program-family delta, then control/candidate/control settled continuous
wall windows under the shared GPU bench lock.  The candidate books only at
+50 us/token against both bracketing controls with one token-stream hash.
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, pathlib, statistics, subprocess, sys
import numpy as np

from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  _settled_continuous_windows, _validate_run_extent)


SCHEMA = "tinygrad.nv_ffn_reduce_output_ab.v1"
LOGITS_SCHEMA = "tinygrad.nv_ffn_reduce_output_ab.logits.v1"
TIMING_SCHEMA = "tinygrad.nv_ffn_reduce_output_ab.timing.v1"
PROMOTION_US = 50.0

R16_PREFIX = "r_16_256_"
E_FFN_PREFIX = "E_32_32_4_f14a5cc0"
C6_PREFIX = "reduce_output_rmsnorm_1_4096"
Q_BODY_PREFIX = "reduce_output_rmsnorm_32_128"
K_BODY_PREFIX = "reduce_output_rmsnorm_8_128"


def _digest(a: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def _configure(model, arm: str) -> None:
  model._decode_direct_greedy_promoted = True
  model._decode_feedback_pingpong_promoted = False
  promoted = arm == "candidate"
  if arm not in ("control", "candidate"):
    raise ValueError(f"unknown arm {arm!r}")
  model._decode_reduce_output_ffn_rmsnorm_promoted = promoted
  for block in model.blk:
    block._decode_reduce_output_ffn_rmsnorm_promoted = promoted


def _gates(model) -> dict:
  return {
    "decode_direct_greedy_promoted": bool(getattr(model, "_decode_direct_greedy_promoted", False)),
    "decode_feedback_pingpong_promoted": bool(getattr(model, "_decode_feedback_pingpong_promoted", False)),
    "decode_callify_substrate_promoted": bool(getattr(model, "_decode_callify_substrate_promoted", False)),
    "reduce_output_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_rmsnorm_promoted", False)),
    "reduce_output_ffn_rmsnorm_promoted": bool(getattr(model, "_decode_reduce_output_ffn_rmsnorm_promoted", False)),
    "block_reduce_output_rmsnorm_promoted": [
      bool(getattr(block, "_decode_reduce_output_rmsnorm_promoted", False)) for block in model.blk],
    "block_reduce_output_ffn_rmsnorm_promoted": [
      bool(getattr(block, "_decode_reduce_output_ffn_rmsnorm_promoted", False)) for block in model.blk],
  }


def _family_counts(programs: list[str]) -> dict[str, int]:
  return {
    "r16_256": sum(name.startswith(R16_PREFIX) for name in programs),
    "e32_32_4_ffn": sum(name.startswith(E_FFN_PREFIX) for name in programs),
    "reduce_output_c6": sum(name.startswith(C6_PREFIX) for name in programs),
    "reduce_output_q": sum(name.startswith(Q_BODY_PREFIX) for name in programs),
    "reduce_output_k": sum(name.startswith(K_BODY_PREFIX) for name in programs),
    "weight_store": sum("weight_store" in name for name in programs),
  }


def _exact_topology_delta(control: dict, candidate: dict) -> dict:
  before = _family_counts(control["program_names"])
  after = _family_counts(candidate["program_names"])
  delta = {name: after[name] - before[name] for name in before}
  unchanged_names = set(control["program_names"]) | set(candidate["program_names"])
  changed = {name: candidate["program_names"].count(name) - control["program_names"].count(name)
             for name in sorted(unchanged_names)
             if candidate["program_names"].count(name) != control["program_names"].count(name)}
  expected = {
    "r16_256": -36,
    "e32_32_4_ffn": -36,
    "reduce_output_c6": 36,
    "reduce_output_q": 0,
    "reduce_output_k": 0,
    "weight_store": 0,
  }
  pass_ = all(delta[name] == expected[name] for name in expected) and len(changed) == 3
  return {"changed_program_counts": changed, "family_delta": delta, "expected_family_delta": expected,
          "no_other_program_delta": len(changed) == 3, "pass": pass_}


def child(args) -> None:
  from tinygrad import Tensor, UOp
  from tinygrad.helpers import Context
  from tinygrad.engine.jit import GraphAdmissionCensus, observe_graph_admissions
  model = _load(args.model, args.max_context)
  _configure(model, args.arm)
  gates = _gates(model)
  if args.arm == "candidate" and not all(gates["block_reduce_output_ffn_rmsnorm_promoted"]):
    raise RuntimeError("candidate arm requires the FFN knob on the model and every block")
  if args.arm == "control" and any(gates["block_reduce_output_ffn_rmsnorm_promoted"]):
    raise RuntimeError("control arm requires the FFN knob closed on every block")
  gen = model.generate(_prompt(args.model, args.depth), chunk_size=32, temperature=0.0)
  try:
    prelude = int(next(gen))
  finally:
    gen.close()
  token, temp = Tensor([[1]], dtype="int32").contiguous(), Tensor([0.0])
  start_pos = UOp.variable("start_pos", 0, args.max_context - 1)
  with Context(JIT=0):
    _, eager = model.forward_with_logits(token, start_pos.bind(args.depth), temp)
  if not np.isfinite(eager.numpy()).all():
    raise RuntimeError("eager full logits non-finite")
  census = GraphAdmissionCensus()
  logits, tokens = [], []
  for index in range(args.count):
    with observe_graph_admissions(census):
      sampled, full = model.decode_with_logits(token, start_pos.bind(args.depth + 1 + index), temp)
      sampled_id, full_np = int(sampled.item()), full.numpy()
      if sampled_id != int(full_np.reshape(-1).argmax()):
        raise RuntimeError(f"sample/logit binding mismatch at {index}")
    tokens.append(sampled_id)
    logits.append(full_np)
    token = sampled
  stacked = np.stack(logits)
  if not np.isfinite(stacked).all():
    raise RuntimeError("decode full logits non-finite")
  programs = [record.program_name for record in census.records if record.program_name]
  row = {
    "schema": LOGITS_SCHEMA, "arm": args.arm, "model": args.model, "depth": args.depth,
    "count": args.count, "max_context": args.max_context, "gates": gates,
    "prelude_token": prelude, "tokens": tokens,
    "tokens_sha256": hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest(),
    "logits_sha256": _digest(stacked), "shape": list(stacked.shape), "finite": True,
    "program_count": len(programs), "program_names": programs, "family_counts": _family_counts(programs),
  }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out.with_suffix(".npz"), logits=stacked)
  out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: row[k] for k in ("arm", "logits_sha256", "tokens_sha256", "program_count", "family_counts")}))
  os._exit(0)


def timing_child(args) -> None:
  from tinygrad import Device
  model = _load(args.model, args.max_context)
  _configure(model, args.arm)
  gates = _gates(model)
  if args.arm == "candidate" and not all(gates["block_reduce_output_ffn_rmsnorm_promoted"]):
    raise RuntimeError("candidate arm requires the FFN knob on the model and every block")
  if args.arm == "control" and any(gates["block_reduce_output_ffn_rmsnorm_promoted"]):
    raise RuntimeError("control arm requires the FFN knob closed on every block")
  gen = model.generate(_prompt(args.model, args.depth), chunk_size=32, temperature=0.0)
  try:
    settled = _settled_continuous_windows(gen, Device[Device.DEFAULT], args.count, args.reps)
  finally:
    gen.close()
  row = {"schema": TIMING_SCHEMA, "arm": args.arm, "gates": gates, "settled_continuous": True,
         "warmup_decode_calls": 6, "reps": args.reps, "tokens_per_rep": args.count, **settled}
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
  print(json.dumps(row))
  os._exit(0)


def _child_command(args, mode: str, arm: str, out: pathlib.Path) -> list[str]:
  return [sys.executable, str(pathlib.Path(__file__).resolve()), "--mode", mode, "--arm", arm,
          "--model", args.model, "--depth", str(args.depth), "--count", str(args.count),
          "--max-context", str(args.max_context), "--reps", str(args.reps), "--out", str(out)]


def _run_child(args, mode: str, arm: str, out: pathlib.Path) -> dict:
  run = subprocess.run(
    ["timeout", str(args.timeout), "flock", "-w", str(args.lock_wait), args.lock, *_child_command(args, mode, arm, out)],
    text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  if run.returncode:
    raise RuntimeError(f"{arm} {mode} failed rc={run.returncode}: {run.stderr[-5000:]}")
  return json.loads(out.read_text())


def qualify(args) -> dict:
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  control = _run_child(args, "child", "control", root / "control.json")
  candidate = _run_child(args, "child", "candidate", root / "candidate.json")
  control_arr = np.load((root / "control.json").with_suffix(".npz"))["logits"]
  candidate_arr = np.load((root / "candidate.json").with_suffix(".npz"))["logits"]
  exact = {
    "same_shape": control_arr.shape == candidate_arr.shape,
    "logits_sha256_equal": control["logits_sha256"] == candidate["logits_sha256"],
    "tokens_equal": control["tokens"] == candidate["tokens"],
    "bitwise_equal": bool(np.array_equal(control_arr, candidate_arr)),
  }
  topology = _exact_topology_delta(control, candidate)
  gate_pass = all(exact.values()) and topology["pass"]
  return {"schema": SCHEMA, "mode": "qualify", "children": {"control": control, "candidate": candidate},
          "exact_logits_gate": exact, "topology_gate": topology,
          "gate_pass": gate_pass, "verdict": "PASS" if gate_pass else "FAIL_CLOSED"}


def timing(args) -> dict:
  root = pathlib.Path(args.out).with_suffix("")
  root.mkdir(parents=True, exist_ok=True)
  rows = []
  for seq, arm in enumerate(("control", "candidate", "control")):
    rows.append(_run_child(args, "timing-child", arm, root / f"{arm}-{seq}.json"))
  controls = (rows[0]["median_ms_per_token"], rows[2]["median_ms_per_token"])
  candidate = rows[1]["median_ms_per_token"]
  hashes = {row["token_stream_hash"] for row in rows}
  candidate_deltas = (candidate - controls[0], candidate - controls[1])
  promotion = len(hashes) == 1 and all(delta <= -PROMOTION_US / 1000.0 for delta in candidate_deltas)
  return {"schema": SCHEMA, "mode": "timing", "arms": rows, "all_token_hashes_equal": len(hashes) == 1,
          "control_bracket_median_ms": statistics.median(controls), "candidate_ms": candidate,
          "candidate_minus_control_a_ms": candidate_deltas[0],
          "candidate_minus_control_b_ms": candidate_deltas[1],
          "candidate_minus_control_a_us": candidate_deltas[0] * 1000.0,
          "candidate_minus_control_b_us": candidate_deltas[1] * 1000.0,
          "candidate_minus_control_bracket_ms": candidate - statistics.median(controls),
          "candidate_minus_control_bracket_us": (candidate - statistics.median(controls)) * 1000.0,
          "promotion_us": PROMOTION_US, "promoted": promotion,
          "verdict": "WALL_PASS" if promotion else "NO_GO_WALL"}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--mode", choices=("child", "timing-child", "qualify", "timing"), required=True)
  ap.add_argument("--arm", choices=("control", "candidate"), default="control")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--count", type=int, default=8)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--reps", type=int, default=4)
  ap.add_argument("--out", required=True)
  ap.add_argument("--timeout", type=int, default=600)
  ap.add_argument("--lock-wait", type=int, default=90)
  ap.add_argument("--lock", default="/tmp/gpu-bench.lock")
  args = ap.parse_args()
  _validate_run_extent(args.depth, args.count, args.max_context, args.reps, args.mode in ("timing", "timing-child"))
  if args.mode == "child":
    child(args)
  if args.mode == "timing-child":
    timing_child(args)
  result = qualify(args) if args.mode == "qualify" else timing(args)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
