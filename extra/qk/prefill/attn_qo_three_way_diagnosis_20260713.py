#!/usr/bin/env python3
"""Sequential, isolated attn_qo diagnosis: generated direct, LDS, and raw pipe.

This is diagnostic-only.  It does not register a route or alter selection.  A
candidate runs in a fresh spawned child, passes the shared guarded full-output
correctness lifecycle, and only then receives kernel-only timing iterations.
The parent performs an independent tiny-GPU health check after every child and
stops the sequence on the first fault.
"""
from __future__ import annotations

import argparse, hashlib, json, math, statistics
from typing import Any, Sequence

import numpy as np

from extra.qk.prefill.guarded_execution import GuardPolicy, run_guarded_execution
from extra.qk.prefill.host_safety_canary import tiny_device_health
from tinygrad.runtime.process_isolated import run_isolated

SHAPE = (512, 4096, 4096)
ROLE = "attn_qo"
CANDIDATES = ("direct_l2", "lds", "pipe")
PIPE_IDENTITY = hashlib.sha256(
  b"attn_qo:512x4096x4096:build_gemm_pipe:tm2:tn2:v2:symbolic-control:preassembled-stream").hexdigest()


def compile_pipe_program(*, shape: tuple[int, int, int] = SHAPE, target: str = "AMD:ISA") -> tuple[Any, dict[str, Any]]:
  """Package the existing shipping raw pipe emitter as a compile-only PROGRAM."""
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import Estimates, compile_linear
  from tinygrad.helpers import Context
  from tinygrad.uop.ops import Ops, ProgramInfo
  from extra.qk.prefill.executable_artifact_preparation import compile_transport_evidence
  from extra.qk.prefill.wmma import build_gemm_pipe
  from extra.qk.prefill_graph_gemm_route import preassembled_gemm_program
  from extra.qk.prefill_schedule_spec import describe_prefill_schedule

  m, n, k = shape
  spec = describe_prefill_schedule(n, k, role=ROLE)
  if spec.route_family != "pipe": raise RuntimeError(f"attn_qo did not resolve to pipe: {spec.route_family!r}")
  tm, tn = spec.pipe_tm, spec.pipe_tn
  bm, bn, threads, lds_bytes, name = tm * 16, tn * 16, 32, 1, spec.kernel_name
  if m % bm or n % bn or k % 32: raise RuntimeError("pipe schedule is not tile-divisible")
  insts = build_gemm_pipe(m, n, k, tm, tn)
  grid = (n // bn, m // bm, 1)

  def asm_kernel(a, b, c):
    return preassembled_gemm_program(a, b, c, insts=insts, lds_bytes=lds_bytes, grid=grid, threads=threads, name=name,
      estimates=Estimates(ops=m*n*k*2, mem=(m*k+n*k+m*n)*2))

  with Context(DEV=target):
    a, b, c = Tensor.empty(m, k, dtype=dtypes.half), Tensor.empty(n, k, dtype=dtypes.half), Tensor.empty(m, n, dtype=dtypes.half)
    compiled = compile_linear(Tensor.custom_kernel(a, b, c, fxn=asm_kernel)[2].schedule_linear())
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM and isinstance(u.arg, ProgramInfo)]
  if len(programs) != 1: raise RuntimeError(f"expected one pipe PROGRAM, found {len(programs)}")
  program = programs[0]
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  if any(marker in source.lower() for marker in ("ds_load", "ds_store", "s_barrier")):
    raise RuntimeError("pipe instruction stream unexpectedly contains LDS synchronization/traffic")
  schedule = {"tile_m": bm, "tile_n": bn, "tile_k": 16, "threads": threads,
              "waves_m": 1, "waves_n": 1, "buffer_count": 2, "lds_bytes": lds_bytes}
  evidence = compile_transport_evidence(program, transport="pipe", canonical_identity=PIPE_IDENTITY,
    schedule=schedule, surface={"strict_pure": False, "ops_ins_count": len(insts),
      "generator": "extra.qk.prefill.wmma.build_gemm_pipe", "lds_transport": False},
    runtime_binding={"profile": "8b", "role": ROLE,
      "shape": {"m": m, "n": n, "k": k}, "target": {"backend": "AMD", "arch": "gfx1100", "wave_size": 32}})
  evidence["argument_order"] = ["a", "b", "output"]
  return program, evidence


def _always_alive() -> bool: return True


def _build_bundle(candidate: str, shape: tuple[int, int, int] = SHAPE):
  from extra.qk.prefill.isolated_guarded_executor import build_tinygrad_bundle
  if candidate == "pipe":
    program, evidence = compile_pipe_program(shape=shape)
    order = ("a", "b", "output")
  else:
    from extra.qk.prefill.attn_qo_progressive_correctness_20260713 import compile_attn_qo_stage, argument_order_for
    prepared = compile_attn_qo_stage(transport=candidate, shape=shape)
    program, evidence, order = prepared["program"], prepared["compile_evidence"], argument_order_for(candidate)
  bundle = build_tinygrad_bundle(program=program, compile_evidence=evidence, device="AMD",
                                 argument_order=order, health=_always_alive)
  return bundle, program, evidence, order


def _child_correctness_canary(candidate: str, shape: tuple[int, int, int], seed: int) -> dict[str, Any]:
  """Compile and correctness-check one candidate without timing. Runs only in an isolated child."""
  from extra.qk.prefill.attn_qo_progressive_correctness_20260713 import attn_qo_reference_inputs
  bundle, _program, evidence, _order = _build_bundle(candidate, shape)
  a, b, reference = attn_qo_reference_inputs(shape, seed=seed)
  try:
    correctness = run_guarded_execution(executable=bundle.executable, inputs={"a": a, "b": b}, reference=reference,
      hooks=bundle.hooks, policy=GuardPolicy(timeout_seconds=30.0),
      identity={"candidate": candidate, "shape": list(shape)}, output_dtype=np.float16)
  finally:
    bundle.executable.close()
  return {"candidate": candidate, "shape": list(shape), "status": "passed" if correctness.get("passed") else "correctness_failed",
          "binary_sha256": evidence["binary_sha256"], "correctness": correctness}


def run_correctness_canary(*, candidate: str = "pipe", shape: tuple[int, int, int] = (32, 32, 96),
                           seed: int = 0x5150, timeout_seconds: float = 60.0) -> dict[str, Any]:
  """Parent-side isolated correctness canary with an independent post-child GPU health check."""
  if candidate not in CANDIDATES: raise ValueError(f"unsupported candidate {candidate!r}")
  child = run_isolated(_child_correctness_canary, args=(candidate, shape, seed), timeout_seconds=timeout_seconds,
                       terminate_grace_seconds=0.5, start_method="spawn")
  healthy = tiny_device_health(timeout_seconds=30.0)
  result = child.result if isinstance(child.result, dict) else None
  return {"candidate": candidate, "shape": list(shape),
          "child": {"status": child.status, "timed_out": child.timed_out, "error": child.error,
                    "elapsed_seconds": child.elapsed_seconds},
          "health_after": healthy, "result": result,
          "passed": child.status == "passed" and result is not None and result.get("status") == "passed" and healthy}


def _child_candidate_session(candidate: str, seed: int, warmups: int, rounds: int) -> dict[str, Any]:
  """Compile, correctness-gate, then time one candidate. Runs only in child."""
  from extra.qk.prefill.attn_qo_progressive_correctness_20260713 import attn_qo_reference_inputs
  from extra.qk.prefill.anchor_isa_resource_capture import capture_program

  bundle, program, evidence, order = _build_bundle(candidate)
  a, b, reference = attn_qo_reference_inputs(SHAPE, seed=seed)
  policy = GuardPolicy(timeout_seconds=30.0)
  correctness = run_guarded_execution(executable=bundle.executable, inputs={"a": a, "b": b}, reference=reference,
    hooks=bundle.hooks, policy=policy, identity={"candidate": candidate}, output_dtype=np.float16)
  if not correctness.get("passed"):
    bundle.executable.close()
    return {"candidate": candidate, "status": "correctness_failed", "correctness": correctness}

  # Fresh guarded buffers for timing.  Setup and readback are excluded from the
  # returned device timestamps; guards and full output are checked afterward.
  hooks, allocations = bundle.hooks, {}
  values = {"a": a, "b": b, "output": np.zeros(reference.shape, dtype=np.float16)}
  try:
    for name, value in values.items():
      allocations[name] = hooks.allocate(name, value, policy)
      hooks.upload(allocations[name], value)
    if not all(hooks.guards_intact(x) for x in allocations.values()): raise RuntimeError("pre-timing guard failure")
    for _ in range(warmups): hooks.dispatch(bundle.executable, allocations)
    samples_s = [float(hooks.dispatch(bundle.executable, allocations)) for _ in range(rounds)]
    if not all(math.isfinite(x) and x > 0 for x in samples_s): raise RuntimeError("invalid device timestamp")
    output = hooks.readback(allocations["output"])
    guards = all(hooks.guards_intact(x) for x in allocations.values())
    unchanged = np.array_equal(hooks.readback(allocations["a"]), a) and np.array_equal(hooks.readback(allocations["b"]), b)
    numerics = bool(np.allclose(output, reference, rtol=policy.rtol, atol=policy.atol))
    if not (guards and unchanged and numerics):
      raise RuntimeError(f"post-timing validation failed: guards={guards} unchanged={unchanged} numerics={numerics}")
  finally:
    for allocation in allocations.values():
      try: hooks.release(allocation)
      except Exception: pass
    bundle.executable.close()

  resource = capture_program(program, candidate_id=evidence["canonical_identity"],
                             route_id=f"diagnostic.{candidate}", expected_pure=candidate != "pipe")
  median_s = statistics.median(samples_s)
  return {"candidate": candidate, "status": "passed", "correctness": correctness,
    "compile": {"canonical_identity": evidence["canonical_identity"], "binary_sha256": evidence["binary_sha256"],
                "argument_order": list(order), "schedule": evidence.get("schedule"),
                "program": resource["program"], "resources": resource["resources"], "isa": resource["isa"]},
    "timing": {"warmups": warmups, "rounds": rounds, "samples_ms": [x*1e3 for x in samples_s],
               "median_ms": median_s*1e3, "min_ms": min(samples_s)*1e3,
               "median_tflops": 2*SHAPE[0]*SHAPE[1]*SHAPE[2]/median_s/1e12}}


def run_sequence(*, candidates: Sequence[str] = CANDIDATES, seed: int = 0x5150,
                 warmups: int = 5, rounds: int = 20, timeout_seconds: float = 180.0) -> dict[str, Any]:
  rows, stopped = [], None
  for candidate in candidates:
    if candidate not in CANDIDATES: raise ValueError(f"unsupported candidate {candidate!r}")
    child = run_isolated(_child_candidate_session, args=(candidate, seed, warmups, rounds),
                         timeout_seconds=timeout_seconds, terminate_grace_seconds=0.5, start_method="spawn")
    healthy = tiny_device_health(timeout_seconds=30.0)
    row = {"candidate": candidate, "child": {"status": child.status, "timed_out": child.timed_out,
      "error": child.error, "elapsed_seconds": child.elapsed_seconds}, "health_after": healthy,
      "result": child.result if isinstance(child.result, dict) else None}
    rows.append(row)
    if child.status != "passed" or not isinstance(child.result, dict) or child.result.get("status") != "passed" or not healthy:
      stopped = candidate
      break
  return {"schema": "attn-qo-three-way-diagnosis.v1", "shape": list(SHAPE), "execution": "strictly_sequential",
          "warmups": warmups, "rounds": rounds, "rows": rows, "stopped_at": stopped,
          "passed": stopped is None and len(rows) == len(candidates)}


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidates", default=",".join(CANDIDATES))
  parser.add_argument("--correctness-only", action="store_true", help="run one isolated candidate without timing")
  parser.add_argument("--shape", default="x".join(map(str, SHAPE)), help="MxNxK shape for --correctness-only")
  parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--rounds", type=int, default=20)
  parser.add_argument("--timeout", type=float, default=180.0)
  parser.add_argument("--out")
  args = parser.parse_args()
  candidates = tuple(x.strip() for x in args.candidates.split(",") if x.strip())
  if args.correctness_only:
    if len(candidates) != 1: parser.error("--correctness-only requires exactly one candidate")
    try: shape = tuple(int(x) for x in args.shape.lower().split("x"))
    except ValueError: parser.error("--shape must be MxNxK")
    if len(shape) != 3: parser.error("--shape must be MxNxK")
    report = run_correctness_canary(candidate=candidates[0], shape=shape, timeout_seconds=args.timeout)  # type: ignore[arg-type]
  else:
    report = run_sequence(candidates=candidates, warmups=args.warmups, rounds=args.rounds, timeout_seconds=args.timeout)
  text = json.dumps(report, indent=2) + "\n"
  if args.out:
    from pathlib import Path
    Path(args.out).write_text(text)
  print(text, end="")
  return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
