#!/usr/bin/env python3
"""Full-shape performance discriminator for a Q6_K/Q8_1 vocabulary head.

This is research-only.  It reuses the oracle-qualified four-warp Q6/Q8 direct
consumer, scales it to the production 151936x4096 vocabulary matrix, and
charges the Q8 provider in the candidate graph.  It does not install a model
route; recurrent-logit qualification remains mandatory after a timing pass.
"""
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.llm.shared_q8_attention import _emit_q8_provider
import extra.llm_research.decode.q6k_q8_warp_direct_microgate as q8direct

ROWS, K = 151936, 4096
Q8_WORDS = K // 4 + K // 32


def _program(family: str, name: str, emitter, shape: tuple[int, ...], dtype=dtypes.float32):
  return KernelProgram(family, name, KernelProgramProvenance.RESEARCH_ONLY, emitter,
                       output_spec=OutputSpec(shape, dtype))


def run(replays: int, reps: int) -> dict:
  dev = Device.DEFAULT
  if str(dev) != "NV": raise RuntimeError(f"DEV=NV required, got {dev}")

  # The emitter reads ROWS as a module global when it is invoked.
  q8direct.ROWS = ROWS
  control_spec = q6k_spec_for_role(ROWS, K, role="vocab", row_tile=2,
    use_coop=True, reduction="in_kernel", target="nv_sm120")
  control_program = _program("research.q6k_vocab_q8_fullshape", control_spec.kernel_name,
    emit_q6k_gemv_kernel(control_spec), (ROWS,))
  provider_program = _program("research.q6k_vocab_q8_fullshape", "q8_provider_4096",
    _emit_q8_provider(), (Q8_WORDS,), dtypes.uint32)
  candidate_program = _program("research.q6k_vocab_q8_fullshape",
    f"q6k_q8_warp_direct_{ROWS}_{K}", q8direct.emit_q6k_q8_warp_direct(), (ROWS,))

  # Deterministic zero input is sufficient for this timing-only discriminator;
  # the exact nonzero arithmetic/oracle gate belongs to the reused primitive.
  weights = Tensor.zeros((ROWS * (K // 256) * 105,), dtype=dtypes.uint16, device=dev).realize()
  x = Tensor.zeros((K,), dtype=dtypes.float16, device=dev).realize()

  @TinyJit
  def control(w, xx):
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      w, xx, program=control_program)

  @TinyJit
  def candidate(w, xx):
    packed = execute_research_program(Tensor.empty((Q8_WORDS,), dtype=dtypes.uint32, device=dev),
      xx, program=provider_program)
    # The direct emitter consumes payload and scales as two views of the same packet.
    return execute_research_program(Tensor.empty((ROWS,), dtype=dtypes.float32, device=dev),
      w, packed[:K//4], packed[K//4:].bitcast(dtypes.float16), program=candidate_program)

  control(weights, x).realize(); candidate(weights, x).realize()
  control_out = control(weights, x).realize(); candidate_out = candidate(weights, x).realize()
  Device[dev].synchronize()
  exact_zero = bool((control_out == candidate_out).all().item())
  if not exact_zero: raise RuntimeError("zero-input control/candidate mismatch")

  for _ in range(100): control(weights, x).realize(); candidate(weights, x).realize()
  Device[dev].synchronize()

  def one(fn) -> float:
    Device[dev].synchronize(); start = time.perf_counter_ns()
    for _ in range(replays): fn(weights, x).realize()
    Device[dev].synchronize()
    return (time.perf_counter_ns() - start) / 1e3 / replays

  a, b, c = [], [], []
  for _ in range(reps):
    a.append(one(control)); b.append(one(candidate)); c.append(one(control))
  paired = [bb - (aa + cc) / 2 for aa, bb, cc in zip(a, b, c)]
  control_mid = statistics.median([(aa + cc) / 2 for aa, cc in zip(a, c)])
  candidate_med = statistics.median(b)
  delta = statistics.median(paired)
  return {
    "schema": "tinygrad.nv_q6k_vocab_q8_fullshape_gate.v1",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "shape": {"rows": ROWS, "k": K, "weight_bytes": ROWS * (K // 256) * 210},
    "contract": {"control": "installed Q6_K FP16 in-kernel reduction",
      "candidate": "Q8 provider + four-warp Q6_K/Q8_1 direct output",
      "production_route_changed": False, "quality_credit": False},
    "correctness": {"zero_input_exact": exact_zero,
      "nonzero_primitive_authority": "q6k_q8_warp_direct_microgate"},
    "timing": {"replays": replays, "reps": reps, "control_a": a, "candidate_b": b,
      "control_c": c, "paired_candidate_minus_control": paired,
      "control_midpoint_median_us": control_mid, "candidate_median_us": candidate_med,
      "delta_us": delta, "gate": "PASS" if delta <= -2.0 else "STOP"},
    "verdict": "PERFORMANCE_PASS_NEEDS_RECURRENT_QUALITY" if delta <= -2.0 else "NO_GO_FULLSHAPE",
  }


def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--replays", type=int, default=32)
  ap.add_argument("--reps", type=int, default=9); ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args(); result = run(args.replays, args.reps)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
