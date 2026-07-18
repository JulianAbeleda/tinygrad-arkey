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
FULL_GRID_BACKEND_ID = "q4k_q8_1_mmq_amd_isa_full_grid_v0"
ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128, 256)
K_TILED_PROBE_SHAPE = (128, 128, 512)
TARGET_ROLE_PROBE_SHAPE = (512, 17408, 5120)
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
  # llama's MMQ producer packs ``make_half2(d, sum)`` into the staged DS4
  # records.  The ABI buffers remain fp32, but the consumer observes the
  # fp16-rounded metadata after its half2 LDS round-trip.  Build the oracle
  # from that same representation; comparing against the pre-rounding fp32
  # arrays would report deterministic arithmetic differences (and falsely
  # block an otherwise exact kernel).
  scales_ref = scales_np.astype(np.float16).astype(np.float32)
  sums_ref = sums_np.astype(np.float16).astype(np.float32)
  ds4 = Q81MMQDS4Activation(values_np, scales_ref, sums_ref,
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


def run_full_grid_r5_benchmark(*, warmups: int = 0, rounds: int = 1) -> dict[str, Any]:
  """Run the emitted 128x128x256 probe against direct-packed in one session.

  This is intentionally a bounded R5 experiment, not a production route.  It
  reuses one compiled full-grid PROGRAM and the same deterministic Q4/Q8
  buffers for warmups and rounds; direct-packed is timed after its own warmup
  on those same logical inputs.  The result carries source/binary/resource
  identity so machine-search can rank it without changing route selection.
  """
  if warmups < 0 or rounds <= 0: raise ValueError("warmups must be non-negative and rounds must be positive")
  import time
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_runtime
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import (
    build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel,
  )
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  from extra.qk.mmq_bounded_harness import _run_direct_packed
  m, n, k = SHAPE
  words_np = _random_q4_words(n, k, 20260717)
  source_np = np.random.default_rng(20260718).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  kernel = build_llama_five_buffer_full_kernel(m, n, k)
  compiled = compile_llama_five_buffer_full_kernel(kernel)
  if not compiled.emitted or compiled.program is None:
    return {"status": "BLOCKED", "exact_blocker": compiled.blocker or "full-grid compile did not emit PROGRAM"}
  program = compiled.program
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  out = Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
  q4 = Tensor(words_np, device="AMD").realize()
  values = Tensor(values_np.reshape(-1), device="AMD").realize()
  scales = Tensor(scales_np.reshape(-1), device="AMD").realize()
  sums = Tensor(sums_np.reshape(-1), device="AMD").realize()
  buffers = (out.uop.buffer, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
  runtime = get_runtime("AMD", program)
  args = tuple(buffers[g].get_buf("AMD") for g in program.arg.globals)
  def full_call():
    runtime(*args, global_size=program.arg.global_size, local_size=program.arg.local_size,
            vals=program.arg.vals({}), wait=True)
  # Always perform one untimed call to compile/cache any runtime launch path;
  # callers may request additional warmups, but timing must not include setup.
  for _ in range(max(1, warmups)): full_call()
  full_samples = []
  for _ in range(rounds):
    t0 = time.perf_counter(); full_call(); full_samples.append((time.perf_counter() - t0) * 1000.0)
  # Convert the exact DS4 values/scales to the row-major direct-packed ABI.
  xq = values_np.transpose(1, 0, 2).reshape(m, k)
  xscales = scales_np.transpose(1, 0, 2).reshape(m, k // 32)
  q4_bytes = words_np.view(np.uint8).reshape(n, k // 256, 144)
  for _ in range(max(1, warmups)): _run_direct_packed(q4_bytes, xq, xscales)
  direct_samples = []
  for _ in range(rounds):
    t0 = time.perf_counter(); _run_direct_packed(q4_bytes, xq, xscales); direct_samples.append((time.perf_counter() - t0) * 1000.0)
  # Use the same fp16 metadata round-trip as the authoritative worker.  The
  # benchmark is therefore independently useful evidence: correctness is not
  # inferred from a separate subprocess or from the direct comparator timing.
  scales_ref = scales_np.astype(np.float16).astype(np.float32)
  sums_ref = sums_np.astype(np.float16).astype(np.float32)
  ds4 = Q81MMQDS4Activation(values_np, scales_ref, sums_ref,
    Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  ref_spec = Q4KQ81MMQTileSpec(role="full_grid_r5", m=m, n=n, k=k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, ref_spec)
  got = out.numpy().reshape(m, n)
  comparison = _numeric_comparison(got, reference)
  correct = comparison["status"] == "pass"
  return {
    "status": "PASS" if correct else "BLOCKED", "bounded_only": True,
    "production_dispatch_changed": False, "default_route": "direct_packed",
    "exact_blocker": None if correct else "numeric output mismatch",
    "correctness": {"status": "PASS" if correct else "BLOCKED",
                     "authority": "full_grid_r5_same_session_reference",
                     "comparison": comparison},
    "timing": {"min_ms": min(full_samples), "median_ms": float(np.median(full_samples)), "samples_ms": full_samples,
                "comparator_status": comparison["status"],
                "direct_packed": {"status": "measured", "min_ms": min(direct_samples),
                                  "median_ms": float(np.median(direct_samples)), "samples_ms": direct_samples}},
    "artifacts": {**artifact, "backend_id": FULL_GRID_BACKEND_ID,
                   "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
                   "same_session_timing": True, "no_fallback": True},
    "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
    "source_sha256": artifact.get("source_sha256"), "binary_sha256": artifact.get("binary_sha256"),
    "resources": artifact.get("resources"),
    "same_session_timing": True, "no_fallback": True,
  }


def run_full_grid_k_tiled_probe(*, warmups: int = 1, rounds: int = 1) -> dict[str, Any]:
  """Prove two K=256 launches accumulate exactly for a 128x128x512 tile.

  This is an R7 adapter probe only.  The first launch uses the existing
  full-grid sink; the second uses its explicit ``accumulate`` variant and
  loads prior FP32 output before each owner store.  Inputs are repacked to
  tile-local Q4/Q8 buffers exactly as a production K-tiler would need to do.
  """
  if warmups < 0 or rounds <= 0: raise ValueError("warmups must be non-negative and rounds must be positive")
  import time
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import (
    build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel,
  )
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  m, n, k = K_TILED_PROBE_SHAPE
  words_np = _random_q4_words(n, k, 20260719)
  source_np = np.random.default_rng(20260720).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  # Compile one spill-free overwrite kernel and use fresh partial output for
  # each K epoch.  Accumulation is deliberately delegated to tinygrad's
  # elementwise path; this avoids a second full-grid sink with a global LOAD
  # on every WMMA store (which itself is allocator-bound).
  compiled = compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(128, 128, 256))
  if not compiled.emitted or compiled.program is None:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_k_tiled_probe.v1", "status": "BLOCKED",
            "shape": [m, n, k], "exact_blocker": compiled.blocker or "K-tiled kernel did not emit"}
  program = compiled.program
  artifacts, launch_samples = [], []
  def run_epochs(*, timed: bool = False):
    accum = Tensor.zeros(m * n, dtype=dtypes.float32, device="AMD").realize()
    elapsed = 0.0
    for epoch in range(k // 256):
      partial = Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
      q4_chunk = np.ascontiguousarray(words_np.view(np.uint8).reshape(n, k // 256, 144)[:, epoch:epoch+1, :].reshape(-1).view(np.uint32))
      values_chunk = np.ascontiguousarray(values_np[epoch*2:(epoch+1)*2].reshape(-1))
      scales_chunk = np.ascontiguousarray(scales_np[epoch*2:(epoch+1)*2].reshape(-1))
      sums_chunk = np.ascontiguousarray(sums_np[epoch*2:(epoch+1)*2].reshape(-1))
      q4 = Tensor(q4_chunk, device="AMD").realize()
      values = Tensor(values_chunk, device="AMD").realize()
      scales = Tensor(scales_chunk, device="AMD").realize()
      sums = Tensor(sums_chunk, device="AMD").realize()
      buffers = (partial.uop.buffer, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
      runtime = get_runtime("AMD", program)
      args = tuple(buffers[g].get_buf("AMD") for g in program.arg.globals)
      t0 = time.perf_counter()
      runtime(*args, global_size=program.arg.global_size, local_size=program.arg.local_size,
              vals=program.arg.vals({}), wait=True)
      accum = (accum + partial).realize()
      elapsed += (time.perf_counter() - t0) * 1000.0
    return accum, elapsed

  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  for epoch in range(k // 256):
    artifacts.append({"epoch": epoch, **artifact, "backend_id": FULL_GRID_BACKEND_ID,
                      "accumulation": "tinygrad_elementwise_add",
                      "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str)})
  for _ in range(max(1, warmups)):
    run_epochs()
  for _ in range(rounds):
    _, elapsed = run_epochs(timed=True)
    launch_samples.append(elapsed)
  accum, _ = run_epochs()
  scales_ref = scales_np.astype(np.float16).astype(np.float32)
  sums_ref = sums_np.astype(np.float16).astype(np.float32)
  ds4 = Q81MMQDS4Activation(values_np, scales_ref, sums_ref,
    Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  ref_spec = Q4KQ81MMQTileSpec(role="full_grid_k_tiled_probe", m=m, n=n, k=k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, ref_spec)
  comparison = _numeric_comparison(accum.numpy().reshape(m, n), reference)
  passed = comparison["status"] == "pass"
  return {
    "schema": "tinygrad.mmq_q4k_q8_1_full_grid_k_tiled_probe.v1", "status": "PASS" if passed else "BLOCKED",
    "shape": [m, n, k], "bounded_only": True, "production_dispatch_changed": False,
    "default_route": "direct_packed", "exact_blocker": None if passed else "numeric output mismatch",
    "correctness": {"status": "PASS" if passed else "BLOCKED", "comparison": comparison,
                     "authority": "same_session_fp16_rounded_ds4_reference"},
    "timing": {"samples_ms": launch_samples, "min_ms": min(launch_samples),
                "median_ms": float(np.median(launch_samples)), "k_epoch_launches": k // 256,
                "accumulation": "tinygrad_elementwise_add"},
    "artifacts": artifacts, "distinct_binary_identity": all(x["distinct_binary_identity"] for x in artifacts),
    "same_session_timing": True, "no_fallback": True,
  }


def run_full_grid_k_tiled_probe_isolated(*, timeout_seconds: float = 360.0,
                                          warmups: int = 1, rounds: int = 1) -> dict[str, Any]:
  """Run the K-tiled probe behind a hard child-process deadline."""
  if timeout_seconds <= 0: return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_k_tiled_probe.v1",
                                  "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  code = ("import json; from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_k_tiled_probe; "
          f"print(json.dumps(run_full_grid_k_tiled_probe(warmups={int(warmups)}, rounds={int(rounds)})))")
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_k_tiled_probe.v1", "status": "BLOCKED",
            "exact_blocker": "K-tiled overwrite/accumulate compile timed out",
            "timeout_seconds": timeout_seconds}
  try:
    result = json.loads(proc.stdout.strip().splitlines()[-1])
  except (IndexError, json.JSONDecodeError):
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_k_tiled_probe.v1", "status": "BLOCKED",
            "exact_blocker": "K-tiled child returned no structured result", "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:]}
  if proc.returncode != 0 and result.get("status") == "PASS":
    result.update({"status": "BLOCKED", "exact_blocker": "K-tiled child exited non-zero", "returncode": proc.returncode})
  return result


def run_full_grid_target_role_probe(*, warmups: int = 0, rounds: int = 1,
                                    epoch_limit: int | None = None,
                                    n_chunk_tiles: int | None = None,
                                    epoch_start: int = 0,
                                    host_accumulate: bool = False,
                                    per_epoch_check: bool = False,
                                    persistent_buffers: bool = False,
                                    preloaded_epochs: bool = False,
                                    sync_each_epoch: bool = False) -> dict[str, Any]:
  """Run the emitted K=256 program across the exact 14B ffn_gate_up shape.

  This remains bounded evidence: each epoch writes a fresh full-role partial
  and tinygrad performs the FP32 elementwise accumulation.  The final oracle
  comparison is full output, with no direct fallback.  Route admission still
  requires the surrounding role/health census.
  """
  if warmups < 0 or rounds <= 0: raise ValueError("warmups must be non-negative and rounds must be positive")
  import time
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import (
    build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel,
  )
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  m, n, k = TARGET_ROLE_PROBE_SHAPE
  total_epochs = k // 256
  if not 0 <= epoch_start < total_epochs: raise ValueError(f"epoch_start must be in [0,{total_epochs-1}]")
  if epoch_limit is None: epoch_limit = total_epochs - epoch_start
  if not 0 < epoch_limit <= total_epochs - epoch_start: raise ValueError(f"epoch_limit must be in [1,{total_epochs-epoch_start}]")
  total_n_tiles = n // 128
  if n_chunk_tiles is None: n_chunk_tiles = total_n_tiles
  if not 0 < n_chunk_tiles <= total_n_tiles: raise ValueError(f"n_chunk_tiles must be in [1,{total_n_tiles}]")
  if preloaded_epochs: persistent_buffers = True
  words_np = _random_q4_words(n, k, 20260721)
  source_np = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  compiled = compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(m, n, 256))
  if not compiled.emitted or compiled.program is None:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "shape": [m, n, k], "exact_blocker": compiled.blocker or "target role K=256 program did not emit"}
  program = compiled.program
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  q4_blocks = words_np.view(np.uint8).reshape(n, k // 256, 144)
  repack_evidence = {
    "q4_sha256": hashlib.sha256(q4_blocks.tobytes()).hexdigest(),
    "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
    "q8_values_sha256": hashlib.sha256(values_np.tobytes()).hexdigest(),
    "q8_scales_sha256": hashlib.sha256(scales_np.tobytes()).hexdigest(),
    "q8_sums_sha256": hashlib.sha256(sums_np.tobytes()).hexdigest(),
    "q8_layout": "q8_ds4[epoch, m, groups]",
  }
  reduction_evidence = {
    "source_revision": "ac4cddeb0dbd778f650bf568f6f08344a06abe3a",
    "owned_components": ["cooperative tile loop", "Q4_K tile_x staging", "Q8_1 tile_y two-panel lifecycle"],
    "source_anchors": ["mmq.cuh:mul_mat_q_process_tile", "mmq.cuh:load_tiles_q4_K"],
  }
  completed_epochs = 0
  epoch_checks = []
  # Optional lifecycle diagnostic: keep one allocation for every dispatch
  # argument and refresh its contents between epochs.  This deliberately
  # removes allocator address reuse from the target-health experiment; it does
  # not alter the generated program or route policy.
  persistent_partial = persistent_q4 = persistent_values = persistent_scales = persistent_sums = None

  def copyin_buffer(tensor, array) -> None:
    # Buffer.copyin is synchronous host-to-device staging for HCQ allocators;
    # copy the entire persistent allocation so no stale tail can be observed.
    buf = tensor.uop.buffer
    buf.get_buf("AMD")  # Tensor.realize materializes the UOp, but allocation is lazy.
    buf.copyin(memoryview(np.ascontiguousarray(array)))

  if persistent_buffers:
    persistent_partial = Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
    q4_capacity = (n * (k // 256) * 36) if preloaded_epochs else n_chunk_tiles * 128 * 36
    persistent_q4 = Tensor.empty(q4_capacity, dtype=dtypes.uint32, device="AMD").realize()
    value_records = (k // 128) if preloaded_epochs else 2
    persistent_values = Tensor.empty(value_records * m * 128, dtype=dtypes.int8, device="AMD").realize()
    persistent_scales = Tensor.empty(value_records * m * 4, dtype=dtypes.float32, device="AMD").realize()
    persistent_sums = Tensor.empty(value_records * m * 4, dtype=dtypes.float32, device="AMD").realize()
    if preloaded_epochs:
      copyin_buffer(persistent_q4, words_np.view(np.uint8).reshape(-1).view(np.uint32))
      copyin_buffer(persistent_values, values_np.reshape(-1))
      copyin_buffer(persistent_scales, scales_np.reshape(-1))
      copyin_buffer(persistent_sums, sums_np.reshape(-1))

  def run_epochs(*, timed: bool = False):
    nonlocal completed_epochs
    accum = None if host_accumulate else Tensor.zeros(m * n, dtype=dtypes.float32, device="AMD").realize()
    accum_host = np.zeros(m * n, dtype=np.float32) if host_accumulate else None
    elapsed = 0.0
    for epoch in range(epoch_start, epoch_start + epoch_limit):
      partial = persistent_partial if persistent_buffers else Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
      if persistent_buffers:
        if not preloaded_epochs:
          copyin_buffer(persistent_values, values_np[epoch*2:(epoch+1)*2].reshape(-1))
          copyin_buffer(persistent_scales, scales_np[epoch*2:(epoch+1)*2].reshape(-1))
          copyin_buffer(persistent_sums, sums_np[epoch*2:(epoch+1)*2].reshape(-1))
      else:
        values = Tensor(np.ascontiguousarray(values_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize()
        scales = Tensor(np.ascontiguousarray(scales_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize()
        sums = Tensor(np.ascontiguousarray(sums_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize()
      t0 = time.perf_counter()
      for n0 in range(0, n, n_chunk_tiles*128):
        n1 = min(n, n0 + n_chunk_tiles*128)
        tile_count = (n1 - n0) // 128
        q4_chunk = np.ascontiguousarray(q4_blocks[n0:n1, epoch:epoch+1, :].reshape(-1).view(np.uint32))
        if persistent_buffers:
          if preloaded_epochs:
            q4 = persistent_q4.uop.buffer.view(q4_chunk.size, dtypes.uint32,
              (epoch * n * 36 + n0 * 36) * dtypes.uint32.itemsize)
          else:
            q4_storage = np.zeros(n_chunk_tiles * 128 * 36, dtype=np.uint32)
            q4_storage[:q4_chunk.size] = q4_chunk
            copyin_buffer(persistent_q4, q4_storage)
        # Buffer views shift the destination and Q4 tile origins without changing
        # the compiled full-N stride; gidx0 then ranges only over this bounded chunk.
        out_view = partial.uop.buffer.view(m*n - n0, dtypes.float32, n0*dtypes.float32.itemsize)
        if persistent_buffers:
          if preloaded_epochs:
            values = persistent_values.uop.buffer.view(2*m*128, dtypes.int8,
              epoch * 2 * m * 128 * dtypes.int8.itemsize)
            scales = persistent_scales.uop.buffer.view(2*m*4, dtypes.float32,
              epoch * 2 * m * 4 * dtypes.float32.itemsize)
            sums = persistent_sums.uop.buffer.view(2*m*4, dtypes.float32,
              epoch * 2 * m * 4 * dtypes.float32.itemsize)
            buffers = (out_view, q4, values, scales, sums)
          else:
            buffers = (out_view, persistent_q4.uop.buffer, persistent_values.uop.buffer,
                       persistent_scales.uop.buffer, persistent_sums.uop.buffer)
        else:
          buffers = (out_view, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
        runtime = get_runtime("AMD", program)
        args = tuple(buffers[g].get_buf("AMD") for g in program.arg.globals)
        runtime(*args, global_size=(tile_count, m//128, 1), local_size=program.arg.local_size,
                vals=program.arg.vals({}), wait=True)
      partial_host = partial.numpy() if per_epoch_check else None
      if host_accumulate:
        if partial_host is None: partial_host = partial.numpy()
        accum_host += partial_host
      else:
        accum = (accum + partial).realize()
      if per_epoch_check:
        ep_scales = scales_np[epoch*2:(epoch+1)*2].astype(np.float16).astype(np.float32)
        ep_sums = sums_np[epoch*2:(epoch+1)*2].astype(np.float16).astype(np.float32)
        ep_ds4 = Q81MMQDS4Activation(values_np[epoch*2:(epoch+1)*2], ep_scales, ep_sums,
          Q81MMQDS4ActivationSpec(m=m, k=256, m_tile=m))
        ep_spec = Q4KQ81MMQTileSpec(role="target_epoch_check", m=m, n=n, k=256,
          m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
        ep_ref = q4k_q8_1_mmq_ds4_tile_reference(q4_blocks[:, epoch:epoch+1, :].reshape(-1).view(np.uint8), ep_ds4, ep_spec)
        epoch_checks.append({"epoch": epoch, **_numeric_comparison(partial_host.reshape(m, n), ep_ref)})
      if sync_each_epoch:
        # Optional lifecycle diagnostic: force the backend queue to drain at
        # the epoch boundary. This does not alter the generated kernel/route.
        from tinygrad.device import Device
        Device["AMD"].synchronize()
      elapsed += (time.perf_counter() - t0) * 1000.0
      completed_epochs += 1
    return (accum_host if host_accumulate else accum), elapsed
  try:
    for _ in range(warmups): run_epochs()
    samples = []
    for _ in range(rounds): accum, elapsed = run_epochs(timed=True); samples.append(elapsed)
  except BaseException as exc:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "shape": [m, n, k], "role": "ffn_gate_up", "bounded_only": True,
            "production_dispatch_changed": False, "default_route": "direct_packed",
            "exact_blocker": "target-role GPU dispatch failed or timed out",
            "exception": type(exc).__name__, "error": str(exc), "completed_epochs": completed_epochs,
            "artifacts": {**artifact, "backend_id": FULL_GRID_BACKEND_ID,
                          "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
            "no_fallback": True}, "epoch_checks": epoch_checks,
            "repack": repack_evidence, "reduction": reduction_evidence}
  compare_k = epoch_limit * 256
  compare_words = q4_blocks[:, epoch_start:epoch_start+epoch_limit, :].reshape(-1).view(np.uint8)
  compare_values = values_np[epoch_start*2:(epoch_start+epoch_limit)*2]
  compare_scales = scales_np[epoch_start*2:(epoch_start+epoch_limit)*2]
  compare_sums = sums_np[epoch_start*2:(epoch_start+epoch_limit)*2]
  scales_ref, sums_ref = compare_scales.astype(np.float16).astype(np.float32), compare_sums.astype(np.float16).astype(np.float32)
  ds4 = Q81MMQDS4Activation(compare_values, scales_ref, sums_ref, Q81MMQDS4ActivationSpec(m=m, k=compare_k, m_tile=m))
  spec = Q4KQ81MMQTileSpec(role="full_grid_target_role_probe", m=m, n=n, k=compare_k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(compare_words, ds4, spec)
  comparison = _numeric_comparison((accum if host_accumulate else accum.numpy()).reshape(m, n), reference)
  passed = comparison["status"] == "pass"
  return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "PASS" if passed else "BLOCKED",
          "shape": [m, n, k], "role": "ffn_gate_up", "bounded_only": True,
          "production_dispatch_changed": False, "default_route": "direct_packed",
          "exact_blocker": None if passed else "numeric output mismatch",
          "correctness": {"status": "PASS" if passed else "BLOCKED", "comparison": comparison,
                           "authority": "same_session_fp16_rounded_ds4_reference"},
          "timing": {"samples_ms": samples, "min_ms": min(samples), "median_ms": float(np.median(samples)),
                     "k_epoch_launches": epoch_limit, "total_k_epoch_launches": total_epochs,
                     "n_chunk_tiles": n_chunk_tiles,
                     "accumulation": "host_fp32_add" if host_accumulate else "tinygrad_elementwise_add",
                     "persistent_buffers": persistent_buffers,
                     "preloaded_epochs": preloaded_epochs,
                     "sync_each_epoch": sync_each_epoch,
                     "epoch_checks": epoch_checks},
          "artifacts": {**artifact, "backend_id": FULL_GRID_BACKEND_ID,
                        "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
                        "no_fallback": True, "same_session_timing": True},
          "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
          "same_session_timing": True, "no_fallback": True,
          "repack": repack_evidence, "reduction": reduction_evidence}


def run_full_grid_target_role_probe_isolated(*, timeout_seconds: float = 900.0,
                                              warmups: int = 0, rounds: int = 1,
                                              epoch_limit: int | None = None,
                                              n_chunk_tiles: int | None = None,
                                              epoch_start: int = 0,
                                              host_accumulate: bool = False,
                                              per_epoch_check: bool = False,
                                              persistent_buffers: bool = False,
                                              preloaded_epochs: bool = False,
                                              sync_each_epoch: bool = False) -> dict[str, Any]:
  if timeout_seconds <= 0: return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1",
                                  "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  code = ("import json; from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_target_role_probe; "
          f"print(json.dumps(run_full_grid_target_role_probe(warmups={int(warmups)}, rounds={int(rounds)}, "
          f"epoch_limit={repr(epoch_limit)}, n_chunk_tiles={repr(n_chunk_tiles)}, epoch_start={int(epoch_start)}, "
          f"host_accumulate={bool(host_accumulate)}, per_epoch_check={bool(per_epoch_check)}, "
          f"persistent_buffers={bool(persistent_buffers)}, preloaded_epochs={bool(preloaded_epochs)}, "
          f"sync_each_epoch={bool(sync_each_epoch)})), flush=True)")
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "exact_blocker": f"target-role compile/{epoch_limit if epoch_limit is not None else 'full'}-epoch dispatch timed out",
            "timeout_seconds": timeout_seconds}
  try: result = json.loads(proc.stdout.strip().splitlines()[-1])
  except (IndexError, json.JSONDecodeError):
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "exact_blocker": "target-role child returned no structured result", "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-1000:],
            "diagnostic": {"epoch_limit": epoch_limit, "n_chunk_tiles": n_chunk_tiles,
                           "epoch_start": epoch_start, "host_accumulate": host_accumulate,
                           "per_epoch_check": per_epoch_check, "persistent_buffers": persistent_buffers,
                           "preloaded_epochs": preloaded_epochs, "sync_each_epoch": sync_each_epoch}}
  if proc.returncode != 0 and result.get("status") == "PASS":
    result.update({"status": "BLOCKED", "exact_blocker": "target-role child exited non-zero", "returncode": proc.returncode})
  return result


def run_full_grid_shape_probe(*, m: int = 256, n: int = 256, k: int = 256) -> dict[str, Any]:
  """Single-epoch dispatch diagnostic for isolating grid/address failures."""
  if min(m, n, k) <= 0 or m % 128 or n % 128 or k != 256:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_shape_probe.v1", "status": "BLOCKED",
            "exact_blocker": "diagnostic requires M/N multiples of 128 and K=256", "shape": [m, n, k]}
  import time
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_runtime
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel
  from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec,
    Q4KQ81MMQTileSpec, Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference, q8_1_mmq_ds4_quantize_reference)
  words_np = _random_q4_words(n, k, 20260723)
  source_np = np.random.default_rng(20260724).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  compiled = compile_llama_five_buffer_full_kernel(build_llama_five_buffer_full_kernel(m, n, k))
  if not compiled.emitted or compiled.program is None:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_shape_probe.v1", "status": "BLOCKED",
            "shape": [m, n, k], "exact_blocker": compiled.blocker or "shape probe did not emit"}
  program = compiled.program
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  out = Tensor.empty(m*n, dtype=dtypes.float32, device="AMD").realize()
  q4 = Tensor(words_np, device="AMD").realize()
  values = Tensor(values_np.reshape(-1), device="AMD").realize()
  scales = Tensor(scales_np.reshape(-1), device="AMD").realize()
  sums = Tensor(sums_np.reshape(-1), device="AMD").realize()
  buffers = (out.uop.buffer, q4.uop.buffer, values.uop.buffer, scales.uop.buffer, sums.uop.buffer)
  runtime = get_runtime("AMD", program); args = tuple(buffers[g].get_buf("AMD") for g in program.arg.globals)
  try:
    t0 = time.perf_counter(); runtime(*args, global_size=program.arg.global_size, local_size=program.arg.local_size,
                                      vals=program.arg.vals({}), wait=True); elapsed = (time.perf_counter()-t0)*1000
    scales_ref, sums_ref = scales_np.astype(np.float16).astype(np.float32), sums_np.astype(np.float16).astype(np.float32)
    ds4 = Q81MMQDS4Activation(values_np, scales_ref, sums_ref, Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
    spec = Q4KQ81MMQTileSpec(role="full_grid_shape_probe", m=m, n=n, k=k, m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
    got = out.numpy().reshape(m,n)
    comparison = _numeric_comparison(got, q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, spec))
    tile_rows = []
    for tm in range(m // 128):
      for tn in range(n // 128):
        tile_spec = Q4KQ81MMQTileSpec(role="full_grid_shape_probe_tile", m=m, n=n, k=k,
          m0=tm*128, n0=tn*128, m_tile=128, n_tile=128, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
        tile_cmp = _numeric_comparison(got[tm*128:(tm+1)*128, tn*128:(tn+1)*128],
          q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, tile_spec))
        tile_rows.append({"tile_m": tm, "tile_n": tn, "mismatch_count": tile_cmp["mismatch_count"],
                          "max_abs_error": tile_cmp["max_abs_error"]})
  except BaseException as exc:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_shape_probe.v1", "status": "BLOCKED", "shape": [m,n,k],
            "exact_blocker": "shape probe dispatch failed or timed out", "exception": type(exc).__name__, "error": str(exc),
            "artifacts": {**artifact, "resources": artifact.get("resources"), "backend_id": FULL_GRID_BACKEND_ID}}
  return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_shape_probe.v1", "status": "PASS" if comparison["status"] == "pass" else "BLOCKED",
          "shape": [m,n,k], "comparison": comparison, "tile_rows": tile_rows,
          "global_size": list(program.arg.global_size), "local_size": list(program.arg.local_size or ()),
          "timing_ms": elapsed, "artifacts": {**artifact,
          "backend_id": FULL_GRID_BACKEND_ID, "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
          "no_fallback": True}}


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
