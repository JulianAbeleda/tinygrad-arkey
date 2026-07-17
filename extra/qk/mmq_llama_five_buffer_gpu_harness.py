"""Fail-closed GPU validation harness for the generated five-buffer MMQ graph.

The full-grid graph is deliberately kept separate from the runtime route.  This
module is an evidence harness only: it builds one deterministic 128x128x256
case in a child interpreter, compiles the exact AMD PROGRAM, dispatches it once,
and compares the result with an independent NumPy DS4 reference.  A compiler
timeout or a missing PROGRAM is reported as a blocker; it is never interpreted
as a numerical pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


PROTOCOL = "tinygrad.mmq_llama_five_buffer_gpu_harness.v1"
PASS = "MMQ_LLAMA_FIVE_BUFFER_GPU_PASS"
BLOCKED = "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128, 256)
RTOL = 3e-3
ATOL = 3e-3


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": False,
          "verdict": BLOCKED, "blocker": reason, "evidence": evidence}


def _random_q4_words(n: int, k: int, seed: int) -> np.ndarray:
  """Independent finite Q4_K bytes (metadata/scales are not emitter-derived)."""
  if k % 256: raise ValueError("Q4_K probe requires K divisible by 256")
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(n, k // 256, 144), dtype=np.uint8)
  # Keep super-block scales finite and moderate while leaving all other bytes
  # random.  This mirrors the independent packed-byte tests without importing
  # a test fixture.
  raw[:, :, :4] = np.frombuffer(np.array([0.03125, 0.0078125], dtype="<f2").tobytes(), dtype=np.uint8)
  return np.ascontiguousarray(raw.reshape(-1).view(np.uint32))


def _bind_sink(sink, args):
  """Replace the five slot parameters in a generated sink with call placeholders."""
  from tinygrad.uop.ops import Ops
  params = sorted({u for u in sink.toposort() if u.op is Ops.PARAM}, key=lambda u: u.arg.slot)
  if [u.arg.slot for u in params] != list(range(5)):
    raise ValueError("full-grid PROGRAM must expose exactly ABI slots 0..4")
  if len(args) != 5: raise ValueError("full-grid call requires five buffers")
  return sink.substitute(dict(zip(params, args)), walk=True)


def _buffer_manifest(names, buffers, *, device: str = "AMD") -> dict[str, Any]:
  """Describe concrete argument mappings without reading or mutating device memory."""
  rows = []
  for slot, (name, buf) in enumerate(zip(names, buffers)):
    handle = buf.get_buf(device)
    va = getattr(handle, "va_addr", None)
    rows.append({"slot": slot, "name": name, "dtype": str(buf.dtype), "elements": int(buf.size),
                 "nbytes": int(buf.nbytes), "va_addr": int(va) if va is not None else None,
                 "va_end": int(va + buf.nbytes) if va is not None else None})
  overlaps = []
  for i, left in enumerate(rows):
    if left["va_addr"] is None: continue
    for right in rows[i+1:]:
      if right["va_addr"] is None: continue
      if left["va_addr"] < right["va_end"] and right["va_addr"] < left["va_end"]:
        overlaps.append((left["slot"], right["slot"]))
  return {"buffers": rows, "overlap_slots": overlaps}


def _json_number(value: Any) -> int | float | str | None:
  """Return a JSON-safe scalar while preserving non-finite diagnostics."""
  if value is None: return None
  value = value.item() if isinstance(value, np.generic) else value
  if isinstance(value, (int, np.integer)): return int(value)
  if isinstance(value, (float, np.floating)):
    if np.isnan(value): return "nan"
    if np.isposinf(value): return "inf"
    if np.isneginf(value): return "-inf"
    return float(value)
  return str(value)


def _numeric_comparison(got: np.ndarray, reference: np.ndarray, *, rtol: float = RTOL,
                        atol: float = ATOL) -> dict[str, Any]:
  """Compare output without raising, including finite/non-finite evidence."""
  got, reference = np.asarray(got), np.asarray(reference)
  result: dict[str, Any] = {
    "status": "mismatch", "rtol": float(rtol), "atol": float(atol),
    "got_shape": list(got.shape), "reference_shape": list(reference.shape),
    "got_size": int(got.size), "reference_size": int(reference.size),
  }
  if got.shape != reference.shape:
    result.update({"mismatch_count": None, "first_mismatch_index": None,
                   "first_mismatch_got": None, "first_mismatch_reference": None,
                   "nan_got": int(np.count_nonzero(np.isnan(got))) if np.issubdtype(got.dtype, np.number) else None,
                   "nan_reference": int(np.count_nonzero(np.isnan(reference))) if np.issubdtype(reference.dtype, np.number) else None,
                   "inf_got": int(np.count_nonzero(np.isinf(got))) if np.issubdtype(got.dtype, np.number) else None,
                   "inf_reference": int(np.count_nonzero(np.isinf(reference))) if np.issubdtype(reference.dtype, np.number) else None,
                   "joint_finite": 0, "max_abs_error": None, "mean_abs_error": None})
    return result

  got_num, ref_num = np.issubdtype(got.dtype, np.number), np.issubdtype(reference.dtype, np.number)
  if not (got_num and ref_num):
    close = got == reference
    finite = np.zeros(got.shape, dtype=bool)
    nan_got = nan_ref = inf_got = inf_ref = None
  else:
    close = np.isclose(got, reference, rtol=rtol, atol=atol, equal_nan=False)
    got_finite, ref_finite = np.isfinite(got), np.isfinite(reference)
    finite = got_finite & ref_finite
    nan_got, nan_ref = int(np.count_nonzero(np.isnan(got))), int(np.count_nonzero(np.isnan(reference)))
    inf_got, inf_ref = int(np.count_nonzero(np.isinf(got))), int(np.count_nonzero(np.isinf(reference)))
  mismatches = ~close
  mismatch_count = int(np.count_nonzero(mismatches))
  first_index = first_got = first_ref = None
  if mismatch_count:
    flat = int(np.flatnonzero(mismatches)[0])
    first_index = [int(x) for x in np.unravel_index(flat, got.shape)]
    first_got, first_ref = _json_number(got.flat[flat]), _json_number(reference.flat[flat])
  if got_num and ref_num:
    errors = np.abs(got[finite] - reference[finite])
    max_error = _json_number(np.max(errors)) if errors.size else None
    mean_error = _json_number(np.mean(errors)) if errors.size else None
    joint_finite = int(np.count_nonzero(finite))
  else:
    max_error = mean_error = None
    joint_finite = 0
  result.update({
    "status": "pass" if mismatch_count == 0 else "mismatch", "mismatch_count": mismatch_count,
    "first_mismatch_index": first_index, "first_mismatch_got": first_got,
    "first_mismatch_reference": first_ref, "joint_finite": joint_finite,
    "max_abs_error": max_error, "mean_abs_error": mean_error,
    "nan_got": nan_got, "nan_reference": nan_ref, "inf_got": inf_got, "inf_reference": inf_ref,
  })
  return result


def _artifact_evidence(program, metadata_parser) -> tuple[Any, Any, dict[str, Any]]:
  """Capture source/binary identity and resource metadata before dispatch comparison."""
  binary = next((u.arg for u in program.src if u.op.name == "BINARY"), None)
  source = next((u.arg for u in program.src if u.op.name == "SOURCE"), None)
  evidence: dict[str, Any] = {
    "source_sha256": hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None,
    "binary_sha256": hashlib.sha256(binary).hexdigest() if isinstance(binary, bytes) else None,
    "source_nbytes": len(source.encode()) if isinstance(source, str) else None,
    "binary_nbytes": len(binary) if isinstance(binary, bytes) else None,
  }
  if isinstance(binary, bytes):
    try:
      evidence["resources"] = metadata_parser(binary)
    except BaseException as exc:
      evidence["resources_error"] = f"{type(exc).__name__}: {exc}"
      # llvm-readelf/objdump do not accept tinygrad's deliberately minimal ELF,
      # but the native loader and descriptor parser are the runtime authority.
      try:
        from tinygrad.renderer.amd.elf import descriptor_register_counts, kernel_descriptor_from_elf
        from tinygrad.runtime.autogen import amdgpu_kd
        desc = kernel_descriptor_from_elf(binary)
        vgpr, sgpr = descriptor_register_counts(desc, is_cdna=False)
        wave32 = 1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT
        evidence["resources"] = {
          "authority": "native_elf_descriptor", "vgpr": int(vgpr), "sgpr": sgpr,
          "lds_bytes": int(desc.group_segment_fixed_size),
          "scratch_bytes": int(desc.private_segment_fixed_size),
          "kernarg_bytes": int(desc.kernarg_size),
          "wavefront_size": 32 if int(desc.kernel_code_properties) & wave32 else 64,
          "kernel_code_entry_byte_offset": int(desc.kernel_code_entry_byte_offset),
          "compute_pgm_rsrc1": int(desc.compute_pgm_rsrc1),
          "compute_pgm_rsrc2": int(desc.compute_pgm_rsrc2),
          "compute_pgm_rsrc3": int(desc.compute_pgm_rsrc3),
        }
      except BaseException as fallback_exc:
        evidence["resources_fallback_error"] = f"{type(fallback_exc).__name__}: {fallback_exc}"
  return binary, source, evidence


def _worker() -> dict[str, Any]:
  """Compile and dispatch the sole AMD case.  Called only in a child process."""
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_runtime
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import Ops
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel
  from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec,
    q4k_q8_1_mmq_ds4_tile_reference, q8_1_mmq_ds4_quantize_reference,
    Q8_1_MMQ_DS4_LAYOUT, Q4KQ81MMQTileSpec)

  m, n, k = SHAPE
  words_np = _random_q4_words(n, k, 20260717)
  source_np = np.random.default_rng(20260718).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  ds4 = Q81MMQDS4Activation(values_np, scales_np, sums_np,
    Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  ref_spec = Q4KQ81MMQTileSpec(role="five_buffer_gpu_probe", m=m, n=n, k=k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, ref_spec)

  kernel = build_llama_five_buffer_full_kernel(m, n, k)
  compiled = compile_llama_five_buffer_full_kernel(kernel)
  if not compiled.emitted or compiled.program is None:
    return _blocked("full-grid compile did not produce PROGRAM", reported_blocker=compiled.blocker)
  program = compiled.program
  if program.op is not Ops.PROGRAM:
    return _blocked("compile result is not an AMD PROGRAM", op=str(program.op))
  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  if programs != [program]:
    return _blocked("expected exactly one PROGRAM", program_count=len(programs))

  # Capture the exact artifact identity/resources before any runtime work or
  # numerical comparison, so a mismatch remains reproducible and auditable.
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)

  # Materialize all buffers before dispatch.  The sink uses the exact five
  # parameter slots; runtime invocation avoids a second custom_kernel compile.
  device = Device["AMD"]
  out = Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
  q4 = Tensor(words_np, device="AMD").realize()
  values = Tensor(values_np.reshape(-1), device="AMD").realize()
  scales = Tensor(scales_np.reshape(-1), device="AMD").realize()
  sums = Tensor(sums_np.reshape(-1), device="AMD").realize()
  buffers = (out.uop.buffer, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
  manifest = _buffer_manifest(("output", "q4", "q8_values", "q8_scales", "q8_original_sums"), buffers)
  # ProgramInfo.globals stores integer indices into the call buffer tuple
  # (the same convention used by engine.realize.exec_kernel), not PARAM UOps.
  globals_ = tuple(program.arg.globals)
  if len(globals_) != 5 or any(g not in range(5) for g in globals_):
    return _blocked("AMD PROGRAM global ABI is not the expected five slots",
                    globals=list(globals_), program_global_size=list(program.arg.global_size),
                    program_local_size=list(program.arg.local_size or ()), **artifact, **manifest)
  dispatch = {"globals": list(globals_), "global_size": list(program.arg.global_size),
              "local_size": list(program.arg.local_size or ()), "vals": list(program.arg.vals({}))}
  # Capture the concrete kernarg allocation used by AMDProgram.__call__.  This
  # is diagnostic-only: the wrapper delegates to the normal allocator and
  # returns the same ArgsState, but lets a structured MMU blocker distinguish
  # a bad generated data pointer from a fault while reading the argument block.
  kernarg = {}
  try:
    runtime = get_runtime("AMD", program)
  except BaseException as exc:
    return _blocked("AMD runtime setup failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  fill_kernargs = runtime.fill_kernargs
  def capture_kernargs(bufs, vals=()):
    state = fill_kernargs(bufs, vals)
    kernarg.update({"va_addr": int(state.buf.va_addr), "size": int(state.buf.size)})
    return state
  runtime.fill_kernargs = capture_kernargs
  dispatch.update({"program_va": int(runtime.lib_gpu.va_addr),
                   "program_entry": int(runtime.prog_addr),
                   "program_descriptor": int(runtime.aql_prog_addr)})
  try:
    runtime(*(buffers[g].get_buf("AMD") for g in globals_),
            global_size=program.arg.global_size, local_size=program.arg.local_size,
            vals=program.arg.vals({}), wait=True)
  except BaseException as exc:
    return _blocked("AMD dispatch failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  try:
    got = out.numpy().reshape(m, n)
  except BaseException as exc:
    return _blocked("AMD output read failed", exception=type(exc).__name__, error=str(exc),
                    dispatch=dispatch, kernarg=kernarg, **artifact, **manifest)
  comparison = _numeric_comparison(got, reference)
  evidence = {"dispatch_performed": True, "full_output_compared": True,
              "global_size": list(program.arg.global_size), "local_size": list(program.arg.local_size),
              "dispatch": dispatch, "kernarg": kernarg, "comparison": comparison,
              "comparator_status": comparison["status"],
              "max_abs_error": comparison["max_abs_error"], "mean_abs_error": comparison["mean_abs_error"],
              **artifact, **manifest}
  if comparison["status"] != "pass":
    return _blocked("numeric output mismatch", **evidence)
  return {"protocol": PROTOCOL, "shape": [m, n, k], "passed": True, "verdict": PASS,
          "blocker": None, "evidence": evidence}


def run_amd_validation(*, timeout_seconds: float = 300.0,
                       python: str = sys.executable,
                       env: dict[str, str] | None = None) -> dict[str, Any]:
  """Run compilation/dispatch in an isolated child with a hard deadline."""
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  try:
    # ``__name__`` is ``__main__`` when this file is invoked with ``-m`` and
    # cannot be resolved by a child interpreter.  Use the importable module
    # path explicitly so the isolated worker can always start.
    proc = subprocess.run([python, "-m", "extra.qk.mmq_llama_five_buffer_gpu_harness", "--worker"], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    return _blocked("AMD full-grid compile/dispatch timed out", timeout_seconds=timeout_seconds)
  except OSError as exc:
    return _blocked(f"AMD worker could not start: {exc}")
  try:
    row = json.loads(proc.stdout.splitlines()[-1])
  except (json.JSONDecodeError, IndexError) as exc:
    return _blocked("AMD worker failed" if proc.returncode else f"AMD worker returned invalid JSON: {exc}",
                    returncode=proc.returncode, stdout=proc.stdout[-4000:], stderr=proc.stderr[-2000:])
  # The worker intentionally exits nonzero for a structured BLOCKED verdict;
  # preserve that evidence instead of replacing it with a generic failure.
  return row


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  args = parser.parse_args()
  if not args.worker:
    print(json.dumps(run_amd_validation(), indent=2, sort_keys=True))
    return 0
  try: row = _worker()
  except BaseException as exc:
    row = _blocked("AMD worker exception", exception=type(exc).__name__, error=str(exc))
  print(json.dumps(row, sort_keys=True))
  return 0 if row.get("passed") else 1


if __name__ == "__main__":
  raise SystemExit(main())
