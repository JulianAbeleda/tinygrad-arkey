"""Runtime adapter for the direct WMMA correction probe.

This is intentionally separate from the full five-buffer harness.  It binds
only the direct-fragment diagnostic's five buffers and reports artifact,
resource, launch, and numerical evidence in one JSON-safe dictionary.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from extra.qk.mmq_llama_wmma_correction_probe import (LOCAL_SIZE, build_wmma_consumer_probe,
  compile_wmma_consumer_probe)
from extra.qk.prefill.amd_native_program_resources import amd_native_program_resources


PROTOCOL = "tinygrad.mmq_llama_wmma_correction_gpu_harness.v1"
PASS = "MMQ_LLAMA_WMMA_CORRECTION_GPU_PASS"
BLOCKED = "MMQ_LLAMA_WMMA_CORRECTION_GPU_BLOCKED"
RTOL, ATOL = 3e-3, 3e-3


def _comparison(got: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
  got, reference = np.asarray(got), np.asarray(reference)
  result: dict[str, Any] = {"status": "mismatch", "rtol": RTOL, "atol": ATOL,
    "got_shape": list(got.shape), "reference_shape": list(reference.shape)}
  if got.shape != reference.shape:
    result.update({"mismatch_count": None, "nan_got": None, "inf_got": None})
    return result
  close = np.isclose(got, reference, rtol=RTOL, atol=ATOL, equal_nan=False)
  mismatch = ~close
  result.update({"status": "pass" if not mismatch.any() else "mismatch",
    "mismatch_count": int(mismatch.sum()), "nan_got": int(np.isnan(got).sum()),
    "inf_got": int(np.isinf(got).sum()), "nan_reference": int(np.isnan(reference).sum()),
    "inf_reference": int(np.isinf(reference).sum())})
  if mismatch.any():
    index = int(np.flatnonzero(mismatch)[0])
    result.update({"first_mismatch_index": list(np.unravel_index(index, got.shape)),
      "first_mismatch_got": float(got.flat[index]), "first_mismatch_reference": float(reference.flat[index])})
  finite = np.isfinite(got) & np.isfinite(reference)
  errors = np.abs(got[finite] - reference[finite])
  result.update({"joint_finite": int(finite.sum()),
    "max_abs_error": float(errors.max()) if errors.size else None,
    "mean_abs_error": float(errors.mean()) if errors.size else None})
  return result


def _manifest(names, buffers) -> dict[str, Any]:
  rows = []
  for slot, (name, buf) in enumerate(zip(names, buffers)):
    handle = buf.get_buf("AMD")
    va = getattr(handle, "va_addr", None)
    rows.append({"slot": slot, "name": name, "dtype": str(buf.dtype), "elements": int(buf.size),
      "nbytes": int(buf.nbytes), "va_addr": int(va) if va is not None else None,
      "va_end": int(va + buf.nbytes) if va is not None else None})
  return {"buffers": rows, "overlap_slots": []}


def run_wmma_consumer_gpu() -> dict[str, Any]:
  """Compile and dispatch once on AMD; return a structured result, never raise for runtime blockers."""
  evidence: dict[str, Any] = {"protocol": PROTOCOL, "local_size": list(LOCAL_SIZE), "workgroup_count": 1}
  try:
    from tinygrad import Tensor, dtypes
    from tinygrad.device import Device
    from tinygrad.engine.realize import get_runtime
    from tinygrad.uop.ops import Ops
    probe = build_wmma_consumer_probe()
    compiled = compile_wmma_consumer_probe(probe)
    program = compiled.program
    if program is None or program.op is not Ops.PROGRAM:
      return {**evidence, "verdict": BLOCKED, "passed": False, "blocker": "compile did not produce PROGRAM"}
    source = next(x.arg for x in program.src if x.op is Ops.SOURCE)
    binary = next(x.arg for x in program.src if x.op is Ops.BINARY)
    evidence.update({"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
      "binary_sha256": hashlib.sha256(binary).hexdigest(), "source_nbytes": len(source.encode()),
      "binary_nbytes": len(binary), "program_global_size": list(program.arg.global_size),
      "program_local_size": list(program.arg.local_size or ())})
    evidence["resources"] = amd_native_program_resources(program, target="AMD:ISA:gfx1100")
    device = Device["AMD"]
    out = Tensor.empty(16*16, dtype=dtypes.float32, device="AMD").realize()
    a = Tensor(probe.fixture.a_fragments.reshape(-1), device="AMD").realize()
    b = Tensor(probe.fixture.b_fragments.reshape(-1), device="AMD").realize()
    dm = Tensor(probe.fixture.dm.reshape(-1), device="AMD").realize()
    ds = Tensor(probe.fixture.ds.reshape(-1), device="AMD").realize()
    buffers = (out.uop.buffer, a.uop.buffer, b.uop.buffer, dm.uop.buffer, ds.uop.buffer)
    evidence["buffers"] = _manifest(("output", "a_fragments", "b_fragments", "dm", "ds"), buffers)
    runtime = get_runtime("AMD", program)
    runtime(*(buffers[g].get_buf("AMD") for g in program.arg.globals),
      global_size=program.arg.global_size, local_size=program.arg.local_size,
      vals=program.arg.vals({}), wait=True)
    got = out.numpy().reshape(16, 16)
    comparison = _comparison(got, probe.fixture.reference)
    return {**evidence, "verdict": PASS if comparison["status"] == "pass" else "MMQ_LLAMA_WMMA_CORRECTION_GPU_MISMATCH",
      "passed": comparison["status"] == "pass", "comparison": comparison}
  except BaseException as exc:
    return {**evidence, "verdict": BLOCKED, "passed": False, "blocker": "AMD probe dispatch failed",
      "exception": type(exc).__name__, "error": str(exc)}


__all__ = ["BLOCKED", "PASS", "PROTOCOL", "run_wmma_consumer_gpu"]
