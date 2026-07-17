"""One-dispatch GPU adapter for the K256 LDS producer round-trip probe.

This is intentionally separate from the full MMQ numerical harness.  It uses
the existing five-buffer input generation, buffer manifest, and artifact
evidence helpers, but dispatches only the producer/LDS-export PROGRAM once.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any
import subprocess
import sys

import numpy as np

from extra.qk.mmq_llama_five_buffer_gpu_harness import (_artifact_evidence, _buffer_manifest,
  _random_q4_words, ROOT, SHAPE)


PROTOCOL = "tinygrad.mmq_llama_lds_roundtrip_gpu_harness.v1"
PASS = "MMQ_LLAMA_LDS_ROUNDTRIP_GPU_PASS"
BLOCKED = "MMQ_LLAMA_LDS_ROUNDTRIP_GPU_BLOCKED"


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": False,
          "verdict": BLOCKED, "blocker": reason, "evidence": evidence}


def _probe_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reuse the existing harness's deterministic Q4/Q8 input conventions."""
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference
  m, n, k = SHAPE
  words = _random_q4_words(n, k, 20260717)
  source = np.random.default_rng(20260718).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  return words, values, scales, sums


def _worker() -> dict[str, Any]:
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_runtime
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import Ops
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_lds_roundtrip_probe import (build_llama_lds_roundtrip_probe,
    compile_llama_lds_roundtrip_probe, expected_llama_lds_roundtrip, compare_llama_lds_roundtrip)

  words, values_np, scales_np, sums_np = _probe_inputs()
  probe = build_llama_lds_roundtrip_probe()
  compiled = compile_llama_lds_roundtrip_probe(probe)
  if not compiled.emitted or compiled.program is None:
    return _blocked("LDS round-trip compile did not produce PROGRAM")
  program = compiled.program
  if program.op is not Ops.PROGRAM:
    return _blocked("compile result is not an AMD PROGRAM", op=str(program.op))
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  globals_ = tuple(program.arg.globals)
  dispatch = {"globals": list(globals_), "global_size": list(program.arg.global_size),
              "local_size": list(program.arg.local_size or ()), "vals": list(program.arg.vals({}))}
  if globals_ != tuple(range(5)):
    return _blocked("AMD PROGRAM global ABI is not five slots", dispatch=dispatch, **artifact)
  if tuple(program.arg.global_size) != (1, 1, 1) or tuple(program.arg.local_size or ()) != (256, 1, 1):
    return _blocked("LDS probe launch geometry drifted", dispatch=dispatch, **artifact)

  device = Device["AMD"]
  out = Tensor.empty(128*128, dtype=dtypes.float32, device="AMD").realize()
  q4 = Tensor(words, device="AMD").realize()
  values = Tensor(values_np.reshape(-1), device="AMD").realize()
  scales = Tensor(scales_np.reshape(-1), device="AMD").realize()
  sums = Tensor(sums_np.reshape(-1), device="AMD").realize()
  buffers = (out.uop.buffer, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
  manifest = _buffer_manifest(("output", "q4", "q8_values", "q8_scales", "q8_original_sums"), buffers)
  kernarg: dict[str, Any] = {}
  try:
    runtime = get_runtime("AMD", program)
    original_fill = runtime.fill_kernargs
    def capture_kernargs(bufs, vals=()):
      state = original_fill(bufs, vals)
      kernarg.update({"va_addr": int(state.buf.va_addr), "size": int(state.buf.size)})
      return state
    runtime.fill_kernargs = capture_kernargs
    dispatch.update({"program_va": int(runtime.lib_gpu.va_addr), "program_entry": int(runtime.prog_addr),
                     "program_descriptor": int(runtime.aql_prog_addr)})
  except BaseException as exc:
    return _blocked("AMD runtime setup failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  try:
    runtime(*(buffers[g].get_buf("AMD") for g in globals_), global_size=program.arg.global_size,
            local_size=program.arg.local_size, vals=program.arg.vals({}), wait=True)
  except BaseException as exc:
    return _blocked("AMD LDS probe dispatch failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  try:
    got = out.numpy()
    expected = expected_llama_lds_roundtrip(words, values_np, scales_np, sums_np)
    comparison = compare_llama_lds_roundtrip(got, expected)
  except BaseException as exc:
    return _blocked("LDS probe output comparison failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  evidence = {"dispatch_performed": True, "dispatch_count": 1, "comparison": comparison,
              "dispatch": dispatch, "kernarg": kernarg, **artifact, **manifest}
  if not comparison["passed"]:
    return _blocked("LDS round-trip mismatch", **evidence)
  return {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": True, "verdict": PASS,
          "blocker": None, "evidence": evidence}


def run_amd_validation(*, timeout_seconds: float = 300.0,
                       python: str = sys.executable,
                       env: dict[str, str] | None = None) -> dict[str, Any]:
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  try:
    proc = subprocess.run([python, "-m", "extra.qk.mmq_llama_lds_roundtrip_gpu_harness", "--worker"],
                          cwd=ROOT, env=child_env, text=True, capture_output=True,
                          timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    return _blocked("AMD LDS probe compile/dispatch timed out", timeout_seconds=timeout_seconds)
  except OSError as exc:
    return _blocked(f"AMD worker could not start: {exc}")
  try:
    return json.loads(proc.stdout.splitlines()[-1])
  except (json.JSONDecodeError, IndexError) as exc:
    return _blocked("AMD worker returned invalid JSON", returncode=proc.returncode,
                    stdout=proc.stdout[-4000:], stderr=proc.stderr[-2000:], error=str(exc))


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  args = parser.parse_args()
  row = _worker() if args.worker else run_amd_validation()
  print(json.dumps(row, sort_keys=True))
  return 0 if row.get("passed") else 1


if __name__ == "__main__": raise SystemExit(main())
