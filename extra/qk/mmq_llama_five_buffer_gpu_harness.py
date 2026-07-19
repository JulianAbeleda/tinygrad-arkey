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
import time
from typing import Any, Mapping

import numpy as np

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_EXACT_ROLE_SPEC, DEFAULT_INVENTORY, ExactRoleSpec, admit_exact_role_spec, exact_role_spec,
)

PROTOCOL = "tinygrad.mmq_llama_five_buffer_gpu_harness.v1"
PASS = "MMQ_LLAMA_FIVE_BUFFER_GPU_PASS"
BLOCKED = "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
FULL_GRID_BACKEND_ID = "q4k_q8_1_mmq_amd_isa_full_grid_v0"
ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128, 256)
K_TILED_PROBE_SHAPE = (128, 128, 512)
TARGET_ROLE_PROBE_SHAPE = DEFAULT_EXACT_ROLE_SPEC.shape
RTOL = 3e-3
ATOL = 3e-3
TARGET_IN_PLACE_ACCUMULATION = "target_in_place_fp32_add"
_TARGET_BUFFER_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
ATTN_QO_DIAGNOSTIC_GLOBAL_GRIDS = (
  (1, 4, 1), (8, 4, 1), (9, 4, 1), (16, 4, 1), (32, 4, 1), (40, 4, 1),
)


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": False,
          "verdict": BLOCKED, "blocker": reason, "evidence": evidence}


def _validate_attn_qo_diagnostic_global_grid(
    role_spec: ExactRoleSpec, requested: tuple[int, ...] | list[int] | None,
    ) -> tuple[int, int, int] | None:
  """Admit only the explicit attn_qo fault-localization ladder."""
  if requested is None: return None
  if role_spec.role != "attn_qo":
    raise ValueError("bounded diagnostic global grid is allowlisted only for role 'attn_qo'")
  if not isinstance(requested, (tuple, list)) or len(requested) != 3 or \
     any(not isinstance(value, int) or isinstance(value, bool) for value in requested):
    raise ValueError("bounded diagnostic global grid requires exactly three integer dimensions")
  grid = tuple(requested)
  if grid not in ATTN_QO_DIAGNOSTIC_GLOBAL_GRIDS:
    raise ValueError(
      f"bounded attn_qo diagnostic global grid must be one of {ATTN_QO_DIAGNOSTIC_GLOBAL_GRIDS}")
  return grid


def _apply_diagnostic_global_grid_to_target_calls(
    output: Any, selected_programs: tuple[Any, ...] | list[Any],
    bounded_grid: tuple[int, int, int],
    ) -> tuple[Any, dict[str, Any]]:
  """Reduce only selected CALL grids while preserving exact frozen PROGRAMs."""
  from tinygrad import Tensor
  from tinygrad.uop.ops import (
    DIAGNOSTIC_LAUNCH_AUTHORITY, DiagnosticCallInfo, Ops,
  )
  selected_programs = tuple(selected_programs)
  calls = [node for node in output.uop.toposort()
           if node.op is Ops.CALL and node.src[0] in selected_programs]
  if [call.src[0] for call in calls] != list(selected_programs):
    raise RuntimeError("bounded diagnostic grid did not find the exact ordered target CALL sequence")
  full_grids = tuple(tuple(program.arg.global_size) for program in selected_programs)
  local_sizes = tuple(tuple(program.arg.local_size or ()) for program in selected_programs)
  if not full_grids or any(grid != (40, 4, 1) for grid in full_grids):
    raise ValueError("bounded attn_qo diagnostic requires exact frozen full grids (40,4,1)")
  if any(len(local) != 3 for local in local_sizes):
    raise ValueError("bounded diagnostic requires concrete three-dimensional frozen local sizes")
  if any(requested > full for requested, full in zip(bounded_grid, full_grids[0])):
    raise ValueError("bounded diagnostic grid exceeds the frozen PROGRAM global grid")

  program_keys = tuple(program.key.hex() for program in selected_programs)
  binary_hashes = tuple(hashlib.sha256(next(
    source.arg for source in program.src if source.op is Ops.BINARY)).hexdigest()
    for program in selected_programs)
  replacements = {}
  for call in calls:
    original = call.arg
    diagnostic = DiagnosticCallInfo(
      original.grad_fxn, original.metadata, original.name, original.precompile,
      original.precompile_backward, original.memory_semantic_slots,
      bounded_grid, DIAGNOSTIC_LAUNCH_AUTHORITY)
    replacements[call] = call.replace(arg=diagnostic)
  bounded_output = Tensor(output.uop.substitute(replacements, walk=True))
  bounded_calls = [node for node in bounded_output.uop.toposort()
                   if node.op is Ops.CALL and node.src[0] in selected_programs]
  if [call.src[0] for call in bounded_calls] != list(selected_programs):
    raise RuntimeError("bounded diagnostic CALL rewrite changed the ordered frozen PROGRAM sequence")
  if any(call.arg.diagnostic_global_size != bounded_grid or
         call.arg.diagnostic_launch_authority != DIAGNOSTIC_LAUNCH_AUTHORITY
         for call in bounded_calls):
    raise RuntimeError("bounded diagnostic CALL rewrite lost its explicit launch authority")
  if tuple(program.key.hex() for program in selected_programs) != program_keys:
    raise RuntimeError("bounded diagnostic CALL rewrite changed a frozen PROGRAM key")
  if tuple(hashlib.sha256(next(
      source.arg for source in program.src if source.op is Ops.BINARY)).hexdigest()
      for program in selected_programs) != binary_hashes:
    raise RuntimeError("bounded diagnostic CALL rewrite changed a frozen binary")
  return bounded_output, {
    "schema": "tinygrad.mmq_attn_qo_bounded_global_grid.v1",
    "enabled": True, "research_only": True, "diagnostic_only": True,
    "production_promotion": False, "promotion_eligible": False,
    "c1_certification_claimed": False, "c1_certification_eligible": False,
    "reason": "launch-only fault localization is not full-grid correctness or provenance evidence",
    "role_allowlist": ["attn_qo"],
    "allowed_global_grid_ladder": [list(grid) for grid in ATTN_QO_DIAGNOSTIC_GLOBAL_GRIDS],
    "requested_global_grid": list(bounded_grid),
    "effective_bounded_global_grids": [list(bounded_grid) for _ in bounded_calls],
    "frozen_full_global_grids": [list(grid) for grid in full_grids],
    "frozen_local_sizes": [list(local) for local in local_sizes],
    "effective_local_sizes": [list(local) for local in local_sizes],
    "program_keys_before": list(program_keys), "program_keys_after": list(program_keys),
    "binary_sha256_before": list(binary_hashes), "binary_sha256_after": list(binary_hashes),
    "program_objects_preserved": all(
      bounded.src[0] is original.src[0] for original, bounded in zip(calls, bounded_calls)),
    "program_keys_preserved": True, "binary_identities_preserved": True,
    "local_sizes_preserved": True, "buffer_abi_preserved": True,
    "call_count": len(bounded_calls),
  }


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


def _pack_q4_epochs_contiguous(q4_blocks: np.ndarray) -> np.ndarray:
  """Pack ``[N, epoch, 144-byte]`` Q4 blocks for one contiguous view per epoch."""
  blocks = np.asarray(q4_blocks)
  if blocks.ndim != 3 or blocks.shape[2] != 144 or blocks.dtype != np.uint8:
    raise ValueError("Q4 preload requires uint8 blocks shaped [N, epoch, 144]")
  return np.ascontiguousarray(blocks.transpose(1, 0, 2)).reshape(-1).view(np.uint32)


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


def _zero_persistent_target_output(output: Any, zero_values: np.ndarray, copyin) -> Any:
  """Zero and return the one output allocation shared by every K epoch."""
  if output is None: raise RuntimeError("target in-place accumulation requires a persistent output")
  copyin(output, zero_values)
  return output


def _accumulate_target_role_epoch(partial: Any, accum: Any, accum_host: Any,
                                  partial_host: Any, *, mode: str) -> tuple[Any, Any]:
  """Advance one epoch without hiding a readback or a second accumulation kernel."""
  if mode == TARGET_IN_PLACE_ACCUMULATION:
    # ``partial`` is the persistent output and the target kernel has already
    # loaded/added/stored it in place.  Do not read it or launch another op.
    return partial, accum_host
  if mode == "host_fp32_add":
    if partial_host is None: partial_host = partial.numpy()
    accum_host += partial_host
    return accum, accum_host
  return (accum + partial).realize(), accum_host


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


def _exact_zero_comparison(got: np.ndarray) -> dict[str, Any]:
  """Require a diagnostic output region to remain exactly floating-point zero."""
  return _numeric_comparison(got, np.zeros_like(got), rtol=0.0, atol=0.0)


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


def _callable_class_name(value: Any) -> str:
  """Return the underlying class/function identity for a possibly-partial factory."""
  value = getattr(value, "func", value)
  typ = value if isinstance(value, type) else type(value)
  return f"{typ.__module__}.{typ.__qualname__}"


def _runtime_identity_evidence(device: Any, runtime: Any, binary_sha256: str | None) -> dict[str, Any]:
  """Describe the exact tinygrad AMD runtime/queue and uploaded program addresses."""
  lib = getattr(runtime, "lib_gpu", None)
  lib_va = getattr(lib, "va_addr", None)
  return {
    "amd_aql_env": os.environ.get("AMD_AQL"),
    "amd_aql_effective": bool(getattr(device, "is_aql", False)),
    "queue_mode": "AQL" if bool(getattr(device, "is_aql", False)) else "PM4",
    "queue_class": _callable_class_name(getattr(device, "hw_compute_queue_t", None)),
    "runtime_class": f"{type(runtime).__module__}.{type(runtime).__qualname__}",
    "binary_sha256": binary_sha256,
    "lib_va": int(lib_va) if lib_va is not None else None,
    "lib_nbytes": int(getattr(lib, "size", 0)) if lib is not None else None,
    # AMDProgram's executable entry and AQL descriptor are separate addresses
    # within the uploaded library image. Keep explicit names and compatibility
    # aliases so fault reports cannot confuse either with the allocation base.
    "program_va": int(getattr(runtime, "prog_addr")) if hasattr(runtime, "prog_addr") else None,
    "entry_va": int(getattr(runtime, "prog_addr")) if hasattr(runtime, "prog_addr") else None,
    "descriptor_va": int(getattr(runtime, "aql_prog_addr")) if hasattr(runtime, "aql_prog_addr") else None,
    "launch_count": 0,
    "launches": [],
  }


def _launch_buffer_evidence(names: tuple[str, ...], buffers: tuple[Any, ...],
                            globals_: tuple[int, ...]) -> list[dict[str, Any]]:
  """Map each runtime argument back to its tinygrad base allocation and view."""
  rows: list[dict[str, Any]] = []
  for call_index, slot in enumerate(globals_):
    buf = buffers[slot]
    handle, base = buf.get_buf("AMD"), buf.base
    base_handle = base.get_buf("AMD")
    view_va, base_va = getattr(handle, "va_addr", None), getattr(base_handle, "va_addr", None)
    offset = int(getattr(buf, "offset", 0))
    rows.append({
      "call_index": call_index, "slot": int(slot), "name": names[slot],
      "va": int(view_va) if view_va is not None else None,
      "base_va": int(base_va) if base_va is not None else None,
      "offset_bytes": offset, "nbytes": int(buf.nbytes), "base_nbytes": int(base.nbytes),
      "va_matches_base_offset": (int(view_va) == int(base_va) + offset)
        if view_va is not None and base_va is not None else None,
    })
  return rows


def _dispatch_with_runtime_evidence(runtime: Any, buffers: tuple[Any, ...], globals_: tuple[int, ...],
                                    *, global_size: tuple[int, int, int], local_size: tuple[int, int, int] | None,
                                    vals: tuple[Any, ...], runtime_evidence: dict[str, Any],
                                    context: Mapping[str, Any], wait: bool = True) -> Any:
  """Invoke the normal runtime once while recording its real kernarg allocation."""
  launch = {
    **dict(context), "global_size": list(global_size), "local_size": list(local_size or ()),
    "arguments": _launch_buffer_evidence(_TARGET_BUFFER_NAMES, buffers, globals_),
  }
  args = tuple(buffers[g].get_buf("AMD") for g in globals_)
  had_instance_fill = "fill_kernargs" in getattr(runtime, "__dict__", {})
  prior_instance_fill = getattr(runtime, "__dict__", {}).get("fill_kernargs")
  original_fill = runtime.fill_kernargs
  captured: dict[str, Any] = {}

  def capture_kernargs(bufs, values=(), kernargs=None):
    state = original_fill(bufs, values, kernargs)
    captured["state"] = state
    captured["bound_pointer_words"] = [int(getattr(buf, "va_addr")) for buf in state.bufs[:5]]
    return state

  runtime.fill_kernargs = capture_kernargs
  try:
    result = runtime(*args, global_size=global_size, local_size=local_size, vals=vals, wait=wait)
  finally:
    if had_instance_fill: runtime.fill_kernargs = prior_instance_fill
    else: delattr(runtime, "fill_kernargs")
    state = captured.get("state")
    if state is not None:
      launch["kernarg"] = {
        "va": int(state.buf.va_addr), "size": int(state.buf.size),
        "bound_pointer_words": captured.get("bound_pointer_words"),
      }
      try:
        pointer_words = list(state.buf.cpu_view().view(size=5 * 8, fmt="Q"))
        launch["kernarg"]["pointer_words"] = [int(x) for x in pointer_words]
        launch["kernarg"]["pointer_words_match_bound"] = (
          launch["kernarg"]["pointer_words"] == launch["kernarg"]["bound_pointer_words"])
      except BaseException as exc:
        launch["kernarg"]["pointer_words"] = None
        launch["kernarg"]["pointer_words_read_error"] = f"{type(exc).__name__}: {exc}"
    else:
      launch["kernarg"] = None
    runtime_evidence["launches"].append(launch)
    runtime_evidence["launch_count"] = len(runtime_evidence["launches"])
  return result


def _validated_child_env_overrides(overrides: Mapping[str, str] | None) -> dict[str, str]:
  """Keep the differential surface deliberately limited to PM4 versus AQL."""
  normalized = {} if overrides is None else {str(k): str(v) for k, v in overrides.items()}
  if set(normalized) - {"AMD_AQL"}:
    raise ValueError("child_env_overrides only permits AMD_AQL")
  if "AMD_AQL" in normalized and normalized["AMD_AQL"] not in {"0", "1"}:
    raise ValueError("AMD_AQL child override must be '0' or '1'")
  return normalized


def _validate_frozen_fixture(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
  """Fail closed unless every deterministic fixture field/hash is identical."""
  if dict(expected) != dict(actual): raise ValueError("runtime fixture differs from frozen bundle")


def _load_frozen_execution_binding(role_spec: ExactRoleSpec, frozen_bundle: str | Path, *,
                                   binding_loader=None):
  """Bind an admitted execution role to a reusable frozen PROGRAM, CPU-only."""
  if binding_loader is None:
    from extra.qk.prefill.frozen_exact_role_runtime import load_frozen_exact_role_binding
    binding_loader = load_frozen_exact_role_binding
  binding = binding_loader(role_spec, frozen_bundle)
  artifact, artifact_spec = binding.artifact, binding.artifact_role_spec
  manifest = artifact.manifest
  from extra.qk.mmq_frozen_target_artifact import FILE_NAMES as FROZEN_FILE_NAMES
  artifact_fixture_sha = manifest["files"][FROZEN_FILE_NAMES["fixture"]]["sha256"]
  relationship = ("same_role_exact_fixture" if artifact_spec == role_spec
                  else "distinct_full_role_shared_program_geometry")
  identity = {
    "path": str(Path(frozen_bundle).resolve()), "manifest_schema": manifest["schema"],
    "state": manifest["state"], "program_key": binding.program_key,
    # Compatibility fields continue to mean the retained artifact fixture.
    "fixture_schema": artifact.fixture.get("schema"), "fixture_sha256": artifact_fixture_sha,
    "serialized_program_sha256": manifest["artifacts"]["serialized_program_sha256"],
    "compile_performed": False, "requires_recompile": False,
    # Never conflate the PROGRAM donor's role fixture with the role being run.
    "artifact_role": artifact_spec.role, "artifact_full_role_shape": list(artifact_spec.shape),
    "artifact_fixture_schema": artifact.fixture.get("schema"),
    "artifact_fixture_sha256": artifact_fixture_sha,
    "execution_role": role_spec.role, "execution_full_role_shape": list(role_spec.shape),
    "program_shape": list(role_spec.program.shape), "program_grid": list(role_spec.program.grid),
    "shared_program_geometry": binding.shared_program_geometry,
    "fixture_relationship": relationship,
  }
  return binding, identity


def _validate_frozen_execution_fixture(binding, runtime_fixture: Mapping[str, Any],
                                       canonical_execution_fixture: Mapping[str, Any]) -> dict[str, Any]:
  """Validate the execution fixture without relabeling a distinct donor fixture."""
  _validate_frozen_fixture(canonical_execution_fixture, runtime_fixture)
  same_role = binding.artifact_role_spec == binding.role_spec
  if same_role: _validate_frozen_fixture(binding.artifact.fixture, runtime_fixture)
  encoded = json.dumps(dict(runtime_fixture), sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode()
  return {
    "artifact_role": binding.artifact_role_spec.role,
    "artifact_full_role_shape": list(binding.artifact_role_spec.shape),
    "execution_role": binding.role_spec.role,
    "execution_full_role_shape": list(binding.role_spec.shape),
    "relationship": ("same_role_exact_fixture" if same_role
                     else "distinct_full_role_shared_program_geometry"),
    "artifact_fixture_equals_execution_fixture": same_role,
    "execution_fixture_schema": runtime_fixture.get("schema"),
    "execution_fixture_canonical_sha256": hashlib.sha256(encoded).hexdigest(),
  }


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
                                    role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                    epoch_limit: int | None = None,
                                    n_chunk_tiles: int | None = None,
                                    epoch_start: int = 0,
                                    host_accumulate: bool = False,
                                    in_kernel_accumulate: bool = False,
                                    per_epoch_check: bool = False,
                                    persistent_buffers: bool = False,
                                    preloaded_epochs: bool = False,
                                    sync_each_epoch: bool = False,
                                    stable_metadata_staging: bool = False,
                                    stable_epoch_staging: bool = False,
                                    wait_each_dispatch: bool = True,
                                    frozen_bundle: str | Path | None = None) -> dict[str, Any]:
  """Run the emitted K=256 program across one admitted exact 14B Q4 role.

  By default each epoch writes a full-role partial and tinygrad performs the
  FP32 elementwise accumulation.  The opt-in target in-place mode instead
  binds one zeroed persistent output to every epoch and performs no external
  add or intermediate readback.  The final oracle comparison is full output,
  with no direct fallback.  Route admission still requires the surrounding
  role/health census.
  """
  role_spec = admit_exact_role_spec(role_spec)
  if warmups < 0 or rounds <= 0: raise ValueError("warmups must be non-negative and rounds must be positive")
  if stable_metadata_staging and not preloaded_epochs:
    raise ValueError("stable_metadata_staging requires preloaded_epochs")
  if stable_epoch_staging and not stable_metadata_staging:
    raise ValueError("stable_epoch_staging requires stable_metadata_staging")
  if not wait_each_dispatch and not (
      in_kernel_accumulate and persistent_buffers and preloaded_epochs and
      stable_metadata_staging and stable_epoch_staging and not per_epoch_check):
    raise ValueError("asynchronous epoch dispatch requires in-place accumulation, persistent preloaded buffers, "
                     "all-input fixed-VA staging, and no intermediate readback")
  if in_kernel_accumulate and host_accumulate:
    raise ValueError("in_kernel_accumulate and host_accumulate are mutually exclusive")
  if in_kernel_accumulate and per_epoch_check:
    raise ValueError("per_epoch_check is unsafe with in_kernel_accumulate because it performs intermediate readback")
  if in_kernel_accumulate and not (persistent_buffers or preloaded_epochs):
    raise ValueError("in_kernel_accumulate requires persistent_buffers")
  if frozen_bundle is not None and not in_kernel_accumulate:
    raise ValueError("frozen target bundle requires in_kernel_accumulate")
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
  m, n, k = role_spec.shape
  role_identity = {"role": role_spec.role, "shape": [m, n, k]}
  total_epochs = role_spec.epochs
  if not 0 <= epoch_start < total_epochs: raise ValueError(f"epoch_start must be in [0,{total_epochs-1}]")
  if epoch_limit is None: epoch_limit = total_epochs - epoch_start
  if not 0 < epoch_limit <= total_epochs - epoch_start: raise ValueError(f"epoch_limit must be in [1,{total_epochs-epoch_start}]")
  total_n_tiles = n // 128
  if n_chunk_tiles is None: n_chunk_tiles = total_n_tiles
  if not 0 < n_chunk_tiles <= total_n_tiles: raise ValueError(f"n_chunk_tiles must be in [1,{total_n_tiles}]")
  if preloaded_epochs: persistent_buffers = True
  accumulation_mode = (TARGET_IN_PLACE_ACCUMULATION if in_kernel_accumulate else
                       "host_fp32_add" if host_accumulate else "tinygrad_elementwise_add")
  frozen_identity: dict[str, Any] | None = None
  frozen_binding = None
  execution_fixture_identity: dict[str, Any] | None = None
  fixture_roles: dict[str, Any] | None = None
  if frozen_bundle is not None:
    try:
      frozen_binding, frozen_identity = _load_frozen_execution_binding(role_spec, frozen_bundle)
      loaded = frozen_binding.artifact
      manifest = loaded.manifest
      program = loaded.program
    except BaseException as exc:
      return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
              **role_identity, "exact_blocker": "frozen target bundle validation failed",
              "exception": type(exc).__name__, "error": str(exc), "accumulation": accumulation_mode,
              "compile_performed": False, "requires_recompile": False}
  else:
    compiled = compile_llama_five_buffer_full_kernel(
      build_llama_five_buffer_full_kernel(*role_spec.program.shape, accumulate=in_kernel_accumulate))
    if not compiled.emitted or compiled.program is None:
      return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
              **role_identity, "exact_blocker": compiled.blocker or "target role K=256 program did not emit",
              "accumulation": accumulation_mode}
    program = compiled.program
  words_np = _random_q4_words(n, k, 20260721)
  source_np = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  binary, source, artifact = _artifact_evidence(program, parse_amdgpu_metadata)
  artifact.update({"compile_performed": frozen_bundle is None, "requires_recompile": frozen_bundle is None})
  if frozen_identity is not None:
    if artifact.get("binary_sha256") != loaded.manifest.get("artifacts", {}).get("binary_sha256") or \
       artifact.get("source_sha256") != loaded.manifest.get("artifacts", {}).get("source_sha256"):
      return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
              **role_identity, "exact_blocker": "loaded PROGRAM identity differs from frozen manifest",
              "accumulation": accumulation_mode, "artifacts": artifact,
              "compile_performed": False, "requires_recompile": False}
    artifact["frozen_bundle"] = frozen_identity
  q4_blocks = words_np.view(np.uint8).reshape(n, k // 256, 144)
  q4_epoch_major = _pack_q4_epochs_contiguous(q4_blocks) if preloaded_epochs or frozen_identity is not None else None
  repack_evidence = {
    "q4_sha256": hashlib.sha256(q4_blocks.tobytes()).hexdigest(),
    "q4_layout": "q4_k_bytes[n, k_epoch, 144]",
    "q8_values_sha256": hashlib.sha256(values_np.tobytes()).hexdigest(),
    "q8_scales_sha256": hashlib.sha256(scales_np.tobytes()).hexdigest(),
    "q8_sums_sha256": hashlib.sha256(sums_np.tobytes()).hexdigest(),
    "q8_layout": "q8_ds4[epoch, m, groups]",
  }
  if q4_epoch_major is not None:
    repack_evidence.update({
      "q4_epoch_major_sha256": hashlib.sha256(q4_epoch_major.tobytes()).hexdigest(),
      "q4_epoch_major_layout": "q4_k_bytes[k_epoch, n, 144]",
      "q4_epoch_major_dtype": str(q4_epoch_major.dtype),
      "q4_epoch_major_elements": int(q4_epoch_major.size),
    })
  if frozen_identity is not None:
    from extra.qk.mmq_target_epoch_orchestrator import FIXTURE_SCHEMA, target_fixture_evidence
    execution_fixture_identity = {
      "schema": FIXTURE_SCHEMA, "role": role_spec.role, "shape": [m, n, k],
      "total_epochs": total_epochs, "seeds": {"q4": 20260721, "q8_source": 20260722},
      "repack": repack_evidence,
      "source_sha256": hashlib.sha256(np.ascontiguousarray(source_np).tobytes()).hexdigest(),
    }
    try:
      fixture_roles = _validate_frozen_execution_fixture(
        frozen_binding, execution_fixture_identity, target_fixture_evidence(role_spec=role_spec))
      frozen_identity.update(fixture_roles)
    except ValueError:
      return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
              **role_identity, "exact_blocker": "runtime execution fixture validation failed",
              "accumulation": accumulation_mode, "artifacts": artifact,
              "execution_fixture": execution_fixture_identity,
              "compile_performed": False, "requires_recompile": False}
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
  persistent_q4_stage = persistent_values_stage = None
  persistent_scales_stage = persistent_sums_stage = None
  target_output_zero = np.zeros(m * n, dtype=np.float32) if in_kernel_accumulate else None
  metadata_staging = {
    "mode": "fixed_va_gpu_sdma" if stable_metadata_staging else "preloaded_views",
    "source_preloaded": bool(preloaded_epochs),
    "transfer": "gpu_sdma" if stable_metadata_staging else None,
    "fixed_va": bool(stable_metadata_staging),
    "per_epoch_vas": [],
  }
  epoch_staging = {
    "mode": "all_inputs_fixed_va_gpu_sdma" if stable_epoch_staging else "q4_q8_value_preloaded_views",
    "source_preloaded": bool(preloaded_epochs),
    "transfer": "gpu_sdma" if stable_epoch_staging else None,
    "fixed_va": bool(stable_epoch_staging),
    "bytes_per_epoch": n * 36 * dtypes.uint32.itemsize + 2 * m * 128 * dtypes.int8.itemsize,
    "per_epoch_vas": [],
  }
  runtime_evidence: dict[str, Any] | None = None

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
    if stable_epoch_staging:
      # The live frozen runtime binds one persistent allocation for every ABI
      # slot. Mirror that contract here instead of changing Q4/Q8-value view
      # pointers between launches.
      persistent_q4_stage = Tensor.empty(n * 36, dtype=dtypes.uint32, device="AMD").realize()
      persistent_values_stage = Tensor.empty(2 * m * 128, dtype=dtypes.int8, device="AMD").realize()
    if stable_metadata_staging:
      # Keep full preloaded sources for the epoch sequence, but bind the
      # kernel to one stable one-epoch metadata allocation refreshed by SDMA.
      persistent_scales_stage = Tensor.empty(2 * m * 4, dtype=dtypes.float32, device="AMD").realize()
      persistent_sums_stage = Tensor.empty(2 * m * 4, dtype=dtypes.float32, device="AMD").realize()
    if preloaded_epochs:
      # ``_random_q4_words`` is N-major: [N, epoch, 144]. A Buffer view can
      # shift only one contiguous base, so preload epoch-major storage instead
      # of incorrectly treating the original N-major flattening as contiguous
      # [epoch, N, 144].
      if q4_epoch_major is None: raise RuntimeError("preloaded Q4 epoch-major storage was not prepared")
      copyin_buffer(persistent_q4, q4_epoch_major)
      copyin_buffer(persistent_values, values_np.reshape(-1))
      copyin_buffer(persistent_scales, scales_np.reshape(-1))
      copyin_buffer(persistent_sums, sums_np.reshape(-1))

  def run_epochs(*, timed: bool = False):
    nonlocal completed_epochs, runtime_evidence
    if in_kernel_accumulate:
      accum = _zero_persistent_target_output(persistent_partial, target_output_zero, copyin_buffer)
    else:
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
      if stable_epoch_staging:
        if persistent_q4_stage is None or persistent_values_stage is None or \
           persistent_q4 is None or persistent_values is None:
          raise RuntimeError("stable epoch staging buffers were not allocated")
        dev = Device["AMD"]
        allocator = dev.allocator
        src_q4 = persistent_q4.uop.buffer.view(
          n * 36, dtypes.uint32, epoch * n * 36 * dtypes.uint32.itemsize)
        src_values = persistent_values.uop.buffer.view(
          2 * m * 128, dtypes.int8, epoch * 2 * m * 128 * dtypes.int8.itemsize)
        dst_q4 = persistent_q4_stage.uop.buffer.get_buf("AMD")
        dst_values = persistent_values_stage.uop.buffer.get_buf("AMD")
        src_q4_buf, src_values_buf = src_q4.get_buf("AMD"), src_values.get_buf("AMD")
        if dev.hw_copy_queue_t is None or not hasattr(allocator, "_transfer"):
          raise RuntimeError("stable epoch staging requires an AMD SDMA transfer queue")
        allocator._transfer(dst_q4, src_q4_buf, src_q4.nbytes, src_dev=dev, dest_dev=dev)
        allocator._transfer(dst_values, src_values_buf, src_values.nbytes, src_dev=dev, dest_dev=dev)
        stage_q4_va, stage_values_va = int(dst_q4.va_addr), int(dst_values.va_addr)
        if epoch_staging["per_epoch_vas"] and (
            stage_q4_va != epoch_staging["per_epoch_vas"][0]["stage_q4_va"] or
            stage_values_va != epoch_staging["per_epoch_vas"][0]["stage_values_va"]):
          raise RuntimeError("stable epoch staging VA changed across epochs")
        entry = {
          "epoch": epoch,
          "source_q4_va": int(src_q4_buf.va_addr),
          "source_values_va": int(src_values_buf.va_addr),
          "stage_q4_va": stage_q4_va,
          "stage_values_va": stage_values_va,
        }
        if not any(x["epoch"] == epoch for x in epoch_staging["per_epoch_vas"]):
          epoch_staging["per_epoch_vas"].append(entry)
      if stable_metadata_staging:
        if persistent_scales_stage is None or persistent_sums_stage is None or persistent_scales is None or persistent_sums is None:
          raise RuntimeError("stable metadata staging buffers were not allocated")
        dev = Device["AMD"]
        allocator = dev.allocator
        src_scales = persistent_scales.uop.buffer.view(2*m*4, dtypes.float32,
          epoch * 2 * m * 4 * dtypes.float32.itemsize)
        src_sums = persistent_sums.uop.buffer.view(2*m*4, dtypes.float32,
          epoch * 2 * m * 4 * dtypes.float32.itemsize)
        dst_scales = persistent_scales_stage.uop.buffer.get_buf("AMD")
        dst_sums = persistent_sums_stage.uop.buffer.get_buf("AMD")
        src_scales_buf = src_scales.get_buf("AMD")
        src_sums_buf = src_sums.get_buf("AMD")
        if dev.hw_copy_queue_t is None or not hasattr(allocator, "_transfer"):
          raise RuntimeError("stable metadata staging requires an AMD SDMA transfer queue")
        # _transfer enqueues SDMA and signals the device timeline; the
        # subsequent HCQ compute launch waits on that timeline before reading.
        allocator._transfer(dst_scales, src_scales_buf, src_scales.nbytes, src_dev=dev, dest_dev=dev)
        allocator._transfer(dst_sums, src_sums_buf, src_sums.nbytes, src_dev=dev, dest_dev=dev)
        stage_scales_va, stage_sums_va = int(dst_scales.va_addr), int(dst_sums.va_addr)
        if metadata_staging["per_epoch_vas"] and (
            stage_scales_va != metadata_staging["per_epoch_vas"][0]["stage_scales_va"] or
            stage_sums_va != metadata_staging["per_epoch_vas"][0]["stage_sums_va"]):
          raise RuntimeError("stable metadata staging VA changed across epochs")
        entry = {
          "epoch": epoch,
          "source_scales_va": int(src_scales_buf.va_addr),
          "source_sums_va": int(src_sums_buf.va_addr),
          "stage_scales_va": stage_scales_va,
          "stage_sums_va": stage_sums_va,
        }
        if not any(x["epoch"] == epoch for x in metadata_staging["per_epoch_vas"]):
          metadata_staging["per_epoch_vas"].append(entry)
      for n0 in range(0, n, n_chunk_tiles*128):
        n1 = min(n, n0 + n_chunk_tiles*128)
        tile_count = (n1 - n0) // 128
        q4_chunk = np.ascontiguousarray(q4_blocks[n0:n1, epoch:epoch+1, :].reshape(-1).view(np.uint32))
        if persistent_buffers:
          if preloaded_epochs:
            q4_source = persistent_q4_stage.uop.buffer if stable_epoch_staging else persistent_q4.uop.buffer
            q4 = q4_source.view(q4_chunk.size, dtypes.uint32,
              n0 * 36 * dtypes.uint32.itemsize if stable_epoch_staging
              else (epoch * n * 36 + n0 * 36) * dtypes.uint32.itemsize)
          else:
            q4_storage = np.zeros(n_chunk_tiles * 128 * 36, dtype=np.uint32)
            q4_storage[:q4_chunk.size] = q4_chunk
            copyin_buffer(persistent_q4, q4_storage)
        # Buffer views shift the destination and Q4 tile origins without changing
        # the compiled full-N stride; gidx0 then ranges only over this bounded chunk.
        out_view = partial.uop.buffer.view(m*n - n0, dtypes.float32, n0*dtypes.float32.itemsize)
        if persistent_buffers:
          if preloaded_epochs:
            values = (persistent_values_stage.uop.buffer if stable_epoch_staging else
                      persistent_values.uop.buffer.view(2*m*128, dtypes.int8,
                        epoch * 2 * m * 128 * dtypes.int8.itemsize))
            if stable_metadata_staging:
              scales = persistent_scales_stage.uop.buffer
              sums = persistent_sums_stage.uop.buffer
            else:
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
        if runtime_evidence is None:
          runtime_evidence = _runtime_identity_evidence(Device["AMD"], runtime, artifact.get("binary_sha256"))
          runtime_evidence.update({
            "intermediate_readback": bool(per_epoch_check),
            "external_accumulation_add": accumulation_mode != TARGET_IN_PLACE_ACCUMULATION,
          })
        _dispatch_with_runtime_evidence(
          runtime, buffers, tuple(program.arg.globals),
          global_size=(tile_count, m//128, 1), local_size=program.arg.local_size,
          vals=tuple(program.arg.vals({})), runtime_evidence=runtime_evidence,
          context={"epoch": epoch, "n0": n0, "n1": n1, "tile_count": tile_count},
          wait=wait_each_dispatch)
      partial_host = partial.numpy() if per_epoch_check else None
      accum, accum_host = _accumulate_target_role_epoch(
        partial, accum, accum_host, partial_host, mode=accumulation_mode)
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
        Device["AMD"].synchronize()
      elapsed += (time.perf_counter() - t0) * 1000.0
      completed_epochs += 1
    if not wait_each_dispatch:
      # The asynchronous mode measures the complete submitted epoch chain,
      # not merely host enqueue latency. Same-device SDMA/compute timeline
      # dependencies protect each fixed staging buffer until its prior
      # consumer completes; this final drain is the sole host synchronization.
      t0 = time.perf_counter()
      Device["AMD"].synchronize()
      elapsed += (time.perf_counter() - t0) * 1000.0
    return (accum_host if host_accumulate else accum), elapsed
  try:
    for _ in range(warmups): run_epochs()
    samples = []
    for _ in range(rounds): accum, elapsed = run_epochs(timed=True); samples.append(elapsed)
  except BaseException as exc:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "shape": [m, n, k], "role": role_spec.role, "bounded_only": True,
            "production_dispatch_changed": False, "default_route": "direct_packed",
            "exact_blocker": "target-role GPU dispatch failed or timed out",
            "exception": type(exc).__name__, "error": str(exc), "completed_epochs": completed_epochs,
            "artifacts": {**artifact, "backend_id": FULL_GRID_BACKEND_ID,
                          "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
            "no_fallback": True, "accumulation": accumulation_mode}, "epoch_checks": epoch_checks,
            "accumulation": accumulation_mode,
            "repack": repack_evidence, "reduction": reduction_evidence,
            "metadata_staging": metadata_staging, "epoch_staging": epoch_staging,
            "runtime_evidence": runtime_evidence,
            "execution_fixture": execution_fixture_identity, "fixture_roles": fixture_roles,
            "compile_performed": frozen_bundle is None, "requires_recompile": frozen_bundle is None}
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
          "shape": [m, n, k], "role": role_spec.role, "bounded_only": True,
          "production_dispatch_changed": False, "default_route": "direct_packed",
          "accumulation": accumulation_mode,
          "exact_blocker": None if passed else "numeric output mismatch",
          "correctness": {"status": "PASS" if passed else "BLOCKED", "comparison": comparison,
                           "authority": "same_session_fp16_rounded_ds4_reference"},
          "timing": {"samples_ms": samples, "min_ms": min(samples), "median_ms": float(np.median(samples)),
                     "k_epoch_launches": epoch_limit, "total_k_epoch_launches": total_epochs,
                     "n_chunk_tiles": n_chunk_tiles,
                     "accumulation": accumulation_mode,
                     "persistent_buffers": persistent_buffers,
                     "preloaded_epochs": preloaded_epochs,
                     "sync_each_epoch": sync_each_epoch,
                     "wait_each_dispatch": wait_each_dispatch,
                     "stable_metadata_staging": stable_metadata_staging,
                     "stable_epoch_staging": stable_epoch_staging,
                     "metadata_staging": metadata_staging, "epoch_staging": epoch_staging,
                     "epoch_checks": epoch_checks},
          "artifacts": {**artifact, "backend_id": FULL_GRID_BACKEND_ID,
                        "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
                        "no_fallback": True, "same_session_timing": True,
                        "accumulation": accumulation_mode},
          "distinct_binary_identity": isinstance(binary, bytes) and isinstance(source, str),
          "same_session_timing": True, "no_fallback": True,
          "repack": repack_evidence, "reduction": reduction_evidence,
          "metadata_staging": metadata_staging, "epoch_staging": epoch_staging,
          "runtime_evidence": runtime_evidence,
          "execution_fixture": execution_fixture_identity, "fixture_roles": fixture_roles,
          "compile_performed": frozen_bundle is None, "requires_recompile": frozen_bundle is None}


_SCHEDULER_PREFIX_INPUT_NAMES = ("q4", "q8_values", "q8_scales", "q8_sums")
_SCHEDULER_PREFIX_CHANGE_SLOTS = (*_SCHEDULER_PREFIX_INPUT_NAMES, "all_except_q8_scales", "all")


def _decode_aql_kernel_dispatch_packet(packet: bytes) -> dict[str, Any]:
  """Decode the safety-critical fields of one concrete 64-byte AQL packet."""
  if len(packet) != 64: raise ValueError("AQL packet census requires exactly 64 bytes")
  import ctypes
  from tinygrad.runtime.autogen import hsa
  header = int.from_bytes(packet[:2], "little")
  field = lambda shift, width: (header >> shift) & ((1 << width) - 1)
  packet_type = field(hsa.HSA_PACKET_HEADER_TYPE, hsa.HSA_PACKET_HEADER_WIDTH_TYPE)
  if packet_type != hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH:
    return {"packet_type": packet_type, "kernel_dispatch": False}
  decoded = hsa.hsa_kernel_dispatch_packet_t.from_buffer_copy(packet)
  return {
    "packet_type": packet_type, "kernel_dispatch": True, "header": header,
    "barrier": bool(field(hsa.HSA_PACKET_HEADER_BARRIER, hsa.HSA_PACKET_HEADER_WIDTH_BARRIER)),
    "acquire_fence_scope": field(hsa.HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
                                 hsa.HSA_PACKET_HEADER_WIDTH_SCACQUIRE_FENCE_SCOPE),
    "release_fence_scope": field(hsa.HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE,
                                 hsa.HSA_PACKET_HEADER_WIDTH_SCRELEASE_FENCE_SCOPE),
    "system_fence_scope": hsa.HSA_FENCE_SCOPE_SYSTEM,
    "kernel_object": int(decoded.kernel_object),
    "kernarg_address": int(ctypes.cast(decoded.kernarg_address, ctypes.c_void_p).value or 0),
    "workgroup_size": [
      int(decoded.workgroup_size_x), int(decoded.workgroup_size_y), int(decoded.workgroup_size_z)],
    "grid_size": [int(decoded.grid_size_x), int(decoded.grid_size_y), int(decoded.grid_size_z)],
  }


def _aql_target_program_identity(prg: Any) -> dict[str, Any]:
  """Return the stable runtime identity used to distinguish target from producer kernels."""
  lib = getattr(prg, "lib", None)
  return {
    "function_name": getattr(prg, "name", None),
    "binary_sha256": hashlib.sha256(bytes(lib)).hexdigest()
      if isinstance(lib, (bytes, bytearray, memoryview)) else None,
  }


def _aql_runtime_lifecycle_evidence(prg: Any, args_state: Any) -> dict[str, Any]:
  """Audit uploaded code-object and kernarg allocation ranges without changing either."""
  from tinygrad.engine.realize import runtime_cache
  lib, kernarg = getattr(prg, "lib_gpu", None), args_state.buf
  lib_va, lib_nbytes = int(getattr(lib, "va_addr", 0)), int(getattr(lib, "size", 0))
  entry_va, descriptor_va = int(getattr(prg, "prog_addr", 0)), int(getattr(prg, "aql_prog_addr", 0))
  kernarg_va, kernarg_nbytes = int(kernarg.va_addr), int(kernarg.size)
  kernarg_payload_nbytes = int(getattr(prg, "kernargs_segment_size", 0))
  lib_end, kernarg_end = lib_va + lib_nbytes, kernarg_va + kernarg_nbytes
  argument_ranges = [(int(buf.va_addr), int(buf.va_addr)+int(buf.size)) for buf in args_state.bufs]
  cache_bindings = [{
    "program_key": key[0].hex(), "device": key[1],
  } for key, runtime in runtime_cache.items() if runtime is prg]
  checks = {
    "program_library_range_nonempty": lib_va > 0 and lib_nbytes > 0,
    "program_entry_in_library_range": lib_va <= entry_va < lib_end,
    "program_descriptor_in_library_range": lib_va <= descriptor_va < lib_end,
    "kernarg_payload_exactly_40_bytes": kernarg_payload_nbytes == 40,
    "kernarg_allocation_matches_payload": kernarg_nbytes == kernarg_payload_nbytes,
    "kernarg_64_byte_aligned": kernarg_va > 0 and kernarg_va % 64 == 0,
    "kernarg_does_not_overlap_program_library": kernarg_end <= lib_va or kernarg_va >= lib_end,
    "kernarg_does_not_overlap_argument_buffers":
      all(kernarg_end <= start or kernarg_va >= end for start, end in argument_ranges),
  }
  return {
    "runtime_object_id": id(prg), "runtime_cache_bindings": cache_bindings,
    "program_library_va": lib_va, "program_library_nbytes": lib_nbytes,
    "program_entry_va": entry_va, "program_entry_offset": entry_va-lib_va,
    "program_descriptor_va": descriptor_va, "program_descriptor_offset": descriptor_va-lib_va,
    "kernarg_va": kernarg_va, "kernarg_allocation_nbytes": kernarg_nbytes,
    "kernarg_payload_nbytes": kernarg_payload_nbytes, "checks": checks,
    "all_checks_pass": all(checks.values()),
  }


def _audit_target_aql_kernargs(qwords: list[int], prior_qwords: list[list[int]], *,
                               expected_vas: list[int] | None,
                               require_fixed_scale_va: bool,
                               require_all_five_vas_fixed: bool = False,
                               require_all_five_vas_distinct: bool = False) -> dict[str, bool]:
  """Validate the five target pointers before its containing doorbell is rung."""
  if len(qwords) != 5: raise ValueError("target AQL census requires exactly five kernarg Qwords")
  checks = {
    "five_qwords_nonzero": all(type(value) is int and value > 0 for value in qwords),
    "five_qwords_match_expected_vas": expected_vas is None or qwords == expected_vas,
    "output_va_fixed": not prior_qwords or qwords[0] == prior_qwords[0][0],
    "q8_scale_va_fixed": not require_fixed_scale_va or not prior_qwords or qwords[3] == prior_qwords[0][3],
    "all_five_vas_fixed": not require_all_five_vas_fixed or not prior_qwords or qwords == prior_qwords[0],
    "all_five_vas_distinct": not require_all_five_vas_distinct or len(set(qwords)) == 5,
  }
  return checks


class AQLPacketCensusRealizationError(RuntimeError):
  """Fallback wrapper when an underlying exception cannot carry census evidence."""
  def __init__(self, message: str, packet_census: Mapping[str, Any]):
    super().__init__(message)
    self.aql_packet_census = dict(packet_census)


def _aql_packet_census_from_exception(exc: BaseException) -> dict[str, Any] | None:
  census = getattr(exc, "aql_packet_census", None)
  return dict(census) if isinstance(census, Mapping) else None


class PM4DispatchCensusRealizationError(RuntimeError):
  """Fallback wrapper when an underlying exception cannot carry PM4 evidence."""
  def __init__(self, message: str, dispatch_census: Mapping[str, Any]):
    super().__init__(message)
    self.pm4_dispatch_census = dict(dispatch_census)


def _pm4_dispatch_census_from_exception(exc: BaseException) -> dict[str, Any] | None:
  census = getattr(exc, "pm4_dispatch_census", None)
  return dict(census) if isinstance(census, Mapping) else None


def _amd_dispatch_census_from_exception(exc: BaseException) -> dict[str, Any] | None:
  """Recover partial evidence from either native tinygrad AMD queue mode."""
  return _aql_packet_census_from_exception(exc) or _pm4_dispatch_census_from_exception(exc)


def _realize_outputs_together(output: Any, retained_outputs: tuple[Any, ...]) -> None:
  """Keep diagnostic companions live by making them outputs of one realization."""
  output.realize(*retained_outputs)


class FiveBufferPreparationError(RuntimeError):
  """Fallback wrapper when producer/initialization failure cannot carry evidence."""
  def __init__(self, message: str, preparation_phase: Mapping[str, Any]):
    super().__init__(message)
    self.preparation_phase = dict(preparation_phase)


def _preparation_phase_from_exception(exc: BaseException) -> dict[str, Any] | None:
  phase = getattr(exc, "preparation_phase", None)
  return dict(phase) if isinstance(phase, Mapping) else None


def _realize_and_synchronize_five_buffer_preparation(
    preparation_outputs: tuple[Any, ...]) -> dict[str, Any]:
  """Finish producer, uploads, and output zeroing before target instrumentation.

  ``preparation_outputs`` are the scheduler's exact five ABI tensors before
  target PROGRAM attachment. This uses normal Tensor realization and the
  device's synchronize path; it does not introduce another GPU launcher.
  """
  if not isinstance(preparation_outputs, tuple) or len(preparation_outputs) != 5:
    raise ValueError("five-buffer preparation requires the exact five ABI tensors")
  from tinygrad.device import Device
  from extra.qk.prefill.frozen_epoch_program_set_scheduler import PREPARATION_RECEIPT_SCHEMA
  dev = Device["AMD"]
  phase = {
    "schema": PREPARATION_RECEIPT_SCHEMA,
    "status": "PENDING", "phase": "producer_and_output_initialization",
    "target_dispatch_allowed": False,
    "realize": {"began": False, "returned": False},
    "synchronize": {"began": False, "returned": False, "failure": None},
    "allocations": [],
  }
  try:
    phase["realize"]["began"] = True
    _realize_outputs_together(preparation_outputs[0], preparation_outputs[1:])
    phase["realize"]["returned"] = True
    phase["synchronize"]["began"] = True
    dev.synchronize()
    phase["synchronize"]["returned"] = True
    allocations = []
    for slot, (name, tensor) in enumerate(zip(_TARGET_BUFFER_NAMES, preparation_outputs)):
      buffer = tensor.uop.buffer
      handle = buffer.get_buf("AMD")
      allocations.append({
        "slot": slot, "name": name, "va": int(handle.va_addr),
        "nbytes": int(buffer.nbytes), "allocation_nbytes": int(handle.size),
        "buffer_uop_key": tensor.uop.buf_uop.key.hex(),
      })
    phase["allocations"] = allocations
    checks = {
      "exact_five_slots": len(allocations) == 5,
      "all_vas_nonzero": all(row["va"] > 0 for row in allocations),
      "all_vas_distinct": len({row["va"] for row in allocations}) == 5,
      "all_extents_nonempty": all(row["nbytes"] > 0 for row in allocations),
      "all_allocations_cover_tensor_extents":
        all(row["allocation_nbytes"] >= row["nbytes"] for row in allocations),
    }
    phase.update({
      "checks": checks, "all_checks_pass": all(checks.values()),
      "target_dispatch_allowed": all(checks.values()),
      "status": "PASS" if all(checks.values()) else "ALLOCATION_REJECTED",
    })
    if not phase["target_dispatch_allowed"]:
      raise RuntimeError("five-buffer preparation allocation audit failed")
    return phase
  except BaseException as exc:
    if phase["synchronize"]["began"] and not phase["synchronize"]["returned"]:
      phase["synchronize"]["failure"] = f"{type(exc).__name__}: {exc}"
    if phase["status"] == "PENDING":
      phase["status"] = (
        "REALIZATION_ERROR" if not phase["realize"]["returned"] else
        "SYNCHRONIZATION_ERROR" if not phase["synchronize"]["returned"] else
        "ALLOCATION_AUDIT_ERROR")
    try: setattr(exc, "preparation_phase", phase)
    except (AttributeError, TypeError):
      raise FiveBufferPreparationError(str(exc), phase) from exc
    raise


def _retained_producer_tensors(produced_tiles: list[Any]) -> tuple[Any, ...]:
  retained = tuple(value for tile in produced_tiles for value in (tile.values, tile.scales, tile.sums))
  if len(retained) != len(produced_tiles) * 3 or len({id(value) for value in retained}) != len(retained):
    raise RuntimeError("producer diagnostic requires three distinct retained tensors per epoch")
  return retained


def _realize_with_aql_packet_census(output: Any, expected_vas: list[list[int]] | None = None, *,
                                    target_program_identity: Mapping[str, Any] | None = None,
                                    target_program_identities: tuple[Mapping[str, Any], ...] | None = None,
                                    target_program_keys: tuple[str, ...] | None = None,
                                    target_launch_dims: tuple[
                                      tuple[tuple[int, ...], tuple[int, ...]], ...] | None = None,
                                    target_dispatch_count: int | None = None,
                                    require_fixed_scale_va: bool = False,
                                    require_all_five_vas_fixed: bool = False,
                                    require_all_five_vas_distinct: bool = False,
                                    require_runtime_lifecycle: bool = False,
                                    retained_outputs: tuple[Any, ...] = ()) -> dict[str, Any]:
  """Realize while auditing target AQL packet/kernargs before each doorbell.

  Producer and scheduler-generated contiguous kernels are inventoried but are
  not interpreted as five-buffer calls. Only a runtime program matching the
  frozen target function and binary identity is subject to its ABI checks.
  """
  from tinygrad.device import Device
  dev = Device["AMD"]
  if not bool(getattr(dev, "is_aql", False)):
    _realize_outputs_together(output, retained_outputs)
    return {"enabled": False, "status": "NOT_APPLICABLE", "reason": "target queue is PM4",
            "retained_companion_output_count": len(retained_outputs)}
  if target_program_identity is not None and target_program_identities is not None:
    raise ValueError("AQL target census accepts one identity mode")
  ordered_identity_sequence = target_program_identities is not None
  if target_program_identities is not None:
    normalized_identities = tuple(dict(identity) for identity in target_program_identities)
    if not normalized_identities:
      raise ValueError("AQL target census ordered identity sequence is empty")
    target_dispatch_count = len(normalized_identities)
  elif target_program_identity is not None:
    normalized_identities = (dict(target_program_identity),)
  else:
    normalized_identities = None
  if normalized_identities is None:
    if expected_vas is None: raise ValueError("AQL target census requires target program identity")
    # Compatibility mode for the producer-free probe, whose only kernels are
    # the exact target calls and whose concrete VAs are known before realize.
    target_dispatch_count = len(expected_vas)
  elif any(identity.get("function_name") in (None, "") or identity.get("binary_sha256") in (None, "")
           for identity in normalized_identities):
    raise ValueError("AQL target census identity sequence is incomplete")
  if ordered_identity_sequence and len({
      (identity["function_name"], identity["binary_sha256"]) for identity in normalized_identities
  }) != len(normalized_identities):
    raise ValueError("AQL target census ordered identities must be distinct")
  if target_dispatch_count is None:
    target_dispatch_count = len(expected_vas) if expected_vas is not None else None
  if not isinstance(target_dispatch_count, int) or isinstance(target_dispatch_count, bool) or target_dispatch_count <= 0:
    raise ValueError("AQL target census requires a positive target dispatch count")
  if expected_vas is not None and len(expected_vas) != target_dispatch_count:
    raise ValueError("AQL target census expected VA rows differ from target dispatch count")
  if target_program_keys is not None and (
      len(target_program_keys) != target_dispatch_count or
      any(not isinstance(key, str) or len(key) != 64 for key in target_program_keys)):
    raise ValueError("AQL target census PROGRAM keys differ from the target dispatch sequence")
  if require_runtime_lifecycle and target_program_keys is None:
    raise ValueError("AQL runtime lifecycle census requires exact PROGRAM keys")
  if target_launch_dims is not None and (
      len(target_launch_dims) != target_dispatch_count or any(
        len(row) != 2 or len(row[0]) != 3 or len(row[1]) != 3 or
        any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for dims in row for value in dims)
        for row in target_launch_dims)):
    raise ValueError("AQL dispatch census launch dimensions differ from the target sequence")

  from tinygrad.runtime import ops_amd
  constructed, published, calls = [], [], []
  non_target_kernel_dispatch_count = compute_doorbell_count = 0
  synchronize = {"began": False, "returned": False, "failure": None}
  original_exec = ops_amd.AMDComputeAQLQueue.exec
  original_publish = ops_amd._publish_aql_packet
  original_doorbell = ops_amd.AMDQueueDesc.signal_doorbell

  def snapshot(status: str, exc: BaseException | None = None) -> dict[str, Any]:
    copied_calls = [{
      **row,
      "program_identity": dict(row["program_identity"]),
      "expected_program_identity": None if row["expected_program_identity"] is None
        else dict(row["expected_program_identity"]),
      "kernarg_qwords": list(row["kernarg_qwords"]),
      "expected_vas": None if row["expected_vas"] is None else list(row["expected_vas"]),
      "argument_buffers": [dict(argument) for argument in row["argument_buffers"]],
      "runtime_lifecycle": {
        **row["runtime_lifecycle"], "checks": dict(row["runtime_lifecycle"]["checks"])},
      "checks": dict(row["checks"]),
    } for row in calls]
    result = {
      "enabled": True, "status": status,
      "capture_point": "ring_slot_after_header_last_publication_and_kernargs_before_doorbell",
      "target_program_identity": dict(target_program_identity) if target_program_identity is not None else None,
      "target_program_identities": [dict(identity) for identity in normalized_identities]
        if target_program_identities is not None else None,
      "target_program_keys": list(target_program_keys) if target_program_keys is not None else None,
      "target_launch_dims": None if target_launch_dims is None else [
        [list(global_size), list(local_size)] for global_size, local_size in target_launch_dims],
      "target_call_count": len(copied_calls),
      "accepted_target_call_count": sum(row["accepted_before_doorbell"] for row in copied_calls),
      "non_target_kernel_dispatch_count": non_target_kernel_dispatch_count,
      "compute_doorbell_count": compute_doorbell_count, "require_fixed_scale_va": require_fixed_scale_va,
      "require_all_five_vas_fixed": require_all_five_vas_fixed,
      "require_all_five_vas_distinct": require_all_five_vas_distinct,
      "require_runtime_lifecycle": require_runtime_lifecycle,
      "retained_companion_output_count": len(retained_outputs),
      "pending_constructed_dispatch_count": len(constructed),
      "pending_published_packet_count": len(published),
      "synchronize": dict(synchronize),
      "call_count": len(copied_calls), "calls": copied_calls,
    }
    if exc is not None:
      result.update({
        "realization_exception": type(exc).__name__, "realization_error": str(exc),
        "all_accepted_target_calls_pass": all(row["all_checks_pass"] for row in copied_calls),
      })
    return result

  def audited_exec(queue: Any, prg: Any, args_state: Any, global_size: tuple[Any, ...],
                   local_size: tuple[Any, ...]) -> Any:
    result = original_exec(queue, prg, args_state, global_size, local_size)
    constructed.append({
      "kernel_object": int(prg.aql_prog_addr), "kernarg_address": int(args_state.buf.va_addr),
      "kernarg_buffer": args_state.buf,
      "argument_buffers": [{
        "slot": slot, "va": int(buf.va_addr), "size": int(buf.size),
      } for slot, buf in enumerate(args_state.bufs)],
      "runtime_lifecycle": _aql_runtime_lifecycle_evidence(prg, args_state),
      "program_identity": _aql_target_program_identity(prg),
      "global_size": tuple(int(value) for value in global_size),
      "local_size": tuple(int(value) for value in local_size),
    })
    return result

  def audited_publish(slot: Any, packet: bytes) -> None:
    original_publish(slot, packet)
    published_packet = bytes(slot.view(size=64, fmt='B')[:])
    if published_packet != packet:
      raise RuntimeError("AQL packet census found ring bytes differ from the packet published")
    published.append(published_packet)

  def audited_doorbell(queue_desc: Any, doorbell_dev: Any, doorbell_value: int | None = None) -> Any:
    nonlocal non_target_kernel_dispatch_count, compute_doorbell_count
    # AMDQueueDesc is shared by compute and SDMA queues. Only the default AQL
    # compute ring contains packets observed by audited_exec/audited_publish.
    if queue_desc is not doorbell_dev.compute_queue_desc(0):
      return original_doorbell(queue_desc, doorbell_dev, doorbell_value)
    compute_doorbell_count += 1
    kernel_packets = [row for packet in published
                      if (row:=_decode_aql_kernel_dispatch_packet(packet))["kernel_dispatch"]]
    if len(kernel_packets) != len(constructed):
      raise RuntimeError("AQL packet census did not find one published kernel packet per constructed dispatch")
    first_new_call = len(calls)
    for packet, built in zip(kernel_packets, constructed):
      target_function_names = ({identity["function_name"] for identity in normalized_identities}
                               if normalized_identities is not None else set())
      is_target = normalized_identities is None or \
                  built["program_identity"].get("function_name") in target_function_names
      if not is_target:
        non_target_kernel_dispatch_count += 1
        continue
      call_index = len(calls)
      if call_index >= target_dispatch_count:
        raise RuntimeError("AQL packet census observed more target dispatches than expected")
      kernarg_qwords = [int(value) for value in
                        built["kernarg_buffer"].cpu_view().view(size=40, fmt='Q')[:5]]
      argument_vas = [row["va"] for row in built["argument_buffers"]]
      expected_row = None if expected_vas is None else expected_vas[call_index]
      expected_identity = (None if normalized_identities is None else
                           normalized_identities[call_index] if ordered_identity_sequence else normalized_identities[0])
      expected_program_key = None if target_program_keys is None else target_program_keys[call_index]
      expected_launch = None if target_launch_dims is None else target_launch_dims[call_index]
      lifecycle = built["runtime_lifecycle"]
      kernarg_start, kernarg_end = lifecycle["kernarg_va"], \
        lifecycle["kernarg_va"] + lifecycle["kernarg_allocation_nbytes"]
      prior_kernarg_ranges = [
        (row["runtime_lifecycle"]["kernarg_va"],
         row["runtime_lifecycle"]["kernarg_va"] + row["runtime_lifecycle"]["kernarg_allocation_nbytes"])
        for row in calls]
      checks = {
        "barrier": packet["barrier"] is True,
        "acquire_system_fence": packet["acquire_fence_scope"] == packet["system_fence_scope"],
        "release_system_fence": packet["release_fence_scope"] == packet["system_fence_scope"],
        "kernel_object_matches_constructed": packet["kernel_object"] == built["kernel_object"],
        "kernarg_address_matches_constructed": packet["kernarg_address"] == built["kernarg_address"],
        "five_qwords_match_constructed_buffers": kernarg_qwords == argument_vas,
        "five_constructed_buffer_vas_distinct": len(set(argument_vas)) == 5,
        "ordered_program_identity_matches": expected_identity is None or
          built["program_identity"] == expected_identity,
        "dispatch_dimensions_match":
          expected_launch is None or (
            built["global_size"] == expected_launch[0] and
            built["local_size"] == expected_launch[1] and
            packet["workgroup_size"] == list(expected_launch[1]) and
            packet["grid_size"] == [
              global_dim * local_dim
              for global_dim, local_dim in zip(expected_launch[0], expected_launch[1])]),
        **({
          **lifecycle["checks"],
          "runtime_cache_exact_program_binding":
            lifecycle["runtime_cache_bindings"] == [{
              "program_key": expected_program_key, "device": "AMD"}],
          "runtime_object_distinct_from_prior_targets":
            all(lifecycle["runtime_object_id"] != row["runtime_lifecycle"]["runtime_object_id"] for row in calls),
          "kernarg_disjoint_from_prior_target_records":
            all(kernarg_end <= start or kernarg_start >= end for start, end in prior_kernarg_ranges),
        } if require_runtime_lifecycle else {}),
        **_audit_target_aql_kernargs(
          kernarg_qwords, [row["kernarg_qwords"] for row in calls],
          expected_vas=expected_row, require_fixed_scale_va=require_fixed_scale_va,
          require_all_five_vas_fixed=require_all_five_vas_fixed,
          require_all_five_vas_distinct=require_all_five_vas_distinct),
      }
      accepted_before_doorbell = all(checks.values())
      calls.append({
        "call": call_index, **{key: packet[key] for key in (
          "header", "barrier", "acquire_fence_scope", "release_fence_scope",
          "system_fence_scope", "kernel_object", "kernarg_address")},
        "program_identity": built["program_identity"], "expected_program_identity": expected_identity,
        "program_key": expected_program_key,
        "kernarg_qwords": kernarg_qwords, "expected_vas": expected_row,
        "argument_buffers": built["argument_buffers"],
        "runtime_lifecycle": built["runtime_lifecycle"],
        "checks": checks, "all_checks_pass": accepted_before_doorbell,
        "accepted_before_doorbell": accepted_before_doorbell,
        "target_exec_observed": True, "submit_began": False, "submit_returned": False,
        "global_size": list(built["global_size"]), "local_size": list(built["local_size"]),
        "expected_global_size": None if expected_launch is None else list(expected_launch[0]),
        "expected_local_size": None if expected_launch is None else list(expected_launch[1]),
        "packet_workgroup_size": list(packet["workgroup_size"]),
        "packet_grid_size": list(packet["grid_size"]),
      })
      if not accepted_before_doorbell:
        raise RuntimeError("AQL target packet census rejected dispatch before doorbell")
    published.clear()
    constructed.clear()
    for row in calls[first_new_call:]: row["submit_began"] = True
    result = original_doorbell(queue_desc, doorbell_dev, doorbell_value)
    for row in calls[first_new_call:]: row["submit_returned"] = True
    return result

  ops_amd.AMDComputeAQLQueue.exec = audited_exec
  ops_amd._publish_aql_packet = audited_publish
  ops_amd.AMDQueueDesc.signal_doorbell = audited_doorbell
  try:
    try:
      _realize_outputs_together(output, retained_outputs)
      synchronize["began"] = True
      dev.synchronize()
      synchronize["returned"] = True
    except BaseException as exc:
      if synchronize["began"] and not synchronize["returned"]:
        synchronize["failure"] = f"{type(exc).__name__}: {exc}"
      census = snapshot(
        "SYNCHRONIZATION_ERROR" if synchronize["began"] else "REALIZATION_ERROR", exc)
      try: setattr(exc, "aql_packet_census", census)
      except (AttributeError, TypeError):
        raise AQLPacketCensusRealizationError(str(exc), census) from exc
      raise
  finally:
    ops_amd.AMDComputeAQLQueue.exec = original_exec
    ops_amd._publish_aql_packet = original_publish
    ops_amd.AMDQueueDesc.signal_doorbell = original_doorbell
  if published or constructed or len(calls) != target_dispatch_count:
    raise RuntimeError("AQL packet census did not close the exact target dispatch count")
  return snapshot("PASS")


def _realize_with_pm4_dispatch_census(
    output: Any, *, target_program_identities: tuple[Mapping[str, Any], ...],
    target_program_keys: tuple[str, ...],
    target_launch_dims: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    require_all_five_vas_fixed: bool = False,
    require_all_five_vas_distinct: bool = False,
    retained_outputs: tuple[Any, ...] = ()) -> dict[str, Any]:
  """Realize through tinygrad PM4 while auditing every target before its doorbell.

  This observes the existing AMDComputeQueue path; it does not manufacture,
  submit, or reinterpret a second launcher. The exact PM4 dwords are retained
  by count and hash, while AMDProgram and CLikeArgsState remain the authority
  for the executable entry and five-pointer ABI.
  """
  from tinygrad.device import Device
  dev = Device["AMD"]
  if bool(getattr(dev, "is_aql", False)):
    _realize_outputs_together(output, retained_outputs)
    return {"enabled": False, "status": "NOT_APPLICABLE", "reason": "target queue is AQL",
            "retained_companion_output_count": len(retained_outputs)}
  identities = tuple(dict(identity) for identity in target_program_identities)
  count = len(identities)
  if count <= 0 or any(identity.get("function_name") in (None, "") or
                       identity.get("binary_sha256") in (None, "") for identity in identities):
    raise ValueError("PM4 dispatch census requires complete ordered target identities")
  if len({(row["function_name"], row["binary_sha256"]) for row in identities}) != count:
    raise ValueError("PM4 dispatch census ordered target identities must be distinct")
  if len(target_program_keys) != count or any(
      not isinstance(key, str) or len(key) != 64 for key in target_program_keys):
    raise ValueError("PM4 dispatch census PROGRAM keys differ from the target sequence")
  if len(target_launch_dims) != count or any(
      len(row) != 2 or len(row[0]) != 3 or len(row[1]) != 3 or
      any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
          for dims in row for value in dims)
      for row in target_launch_dims):
    raise ValueError("PM4 dispatch census launch dimensions differ from the target sequence")

  from tinygrad.runtime import ops_amd
  pending: dict[int, dict[str, Any]] = {}
  calls: list[dict[str, Any]] = []
  non_target_exec_count = non_target_submit_count = compute_submit_count = 0
  synchronize = {"began": False, "returned": False, "failure": None}
  target_function_names = {identity["function_name"] for identity in identities}
  original_exec = ops_amd.AMDComputeQueue.exec
  original_submit = ops_amd.AMDComputeQueue._submit

  def snapshot(status: str, exc: BaseException | None = None) -> dict[str, Any]:
    copied_calls = [{
      **row,
      "program_identity": dict(row["program_identity"]),
      "expected_program_identity": dict(row["expected_program_identity"]),
      "kernarg_qwords": list(row["kernarg_qwords"]),
      "argument_buffers": [dict(argument) for argument in row["argument_buffers"]],
      "runtime_lifecycle": {
        **row["runtime_lifecycle"], "checks": dict(row["runtime_lifecycle"]["checks"])},
      "global_size": list(row["global_size"]), "local_size": list(row["local_size"]),
      "expected_global_size": list(row["expected_global_size"]),
      "expected_local_size": list(row["expected_local_size"]),
      "checks": dict(row["checks"]),
    } for row in calls]
    result = {
      "enabled": True, "status": status, "queue_mode": "PM4",
      "capture_point": "AMDComputeQueue._submit_after_complete_command_construction_before_ring_copy_and_doorbell",
      "queue_contract": "HCQProgram_wait_memory_barrier_exec_signal_submit",
      "target_program_identities": [dict(identity) for identity in identities],
      "target_program_keys": list(target_program_keys),
      "target_call_count": len(copied_calls),
      "accepted_target_call_count": sum(row["accepted_before_doorbell"] for row in copied_calls),
      "non_target_exec_count": non_target_exec_count,
      "non_target_submit_count": non_target_submit_count,
      "compute_submit_count": compute_submit_count,
      "require_all_five_vas_fixed": require_all_five_vas_fixed,
      "require_all_five_vas_distinct": require_all_five_vas_distinct,
      "retained_companion_output_count": len(retained_outputs),
      "pending_target_queue_count": len(pending),
      "synchronize": dict(synchronize),
      "call_count": len(copied_calls), "calls": copied_calls,
    }
    if exc is not None:
      result.update({
        "realization_exception": type(exc).__name__, "realization_error": str(exc),
        "all_accepted_target_calls_pass": all(row["all_checks_pass"] for row in copied_calls),
      })
    return result

  def audited_exec(queue: Any, prg: Any, args_state: Any, global_size: tuple[Any, ...],
                   local_size: tuple[Any, ...]) -> Any:
    nonlocal non_target_exec_count
    before_exec_dword_count = len(queue._q)
    result = original_exec(queue, prg, args_state, global_size, local_size)
    identity = _aql_target_program_identity(prg)
    if identity.get("function_name") not in target_function_names:
      non_target_exec_count += 1
      return result
    if id(queue) in pending:
      raise RuntimeError("PM4 dispatch census found multiple target execs in one queue")
    pending[id(queue)] = {
      "queue": queue, "program": prg, "args_state": args_state,
      "program_identity": identity,
      "argument_buffers": [{
        "slot": slot, "va": int(buf.va_addr), "size": int(buf.size),
      } for slot, buf in enumerate(args_state.bufs)],
      "runtime_lifecycle": _aql_runtime_lifecycle_evidence(prg, args_state),
      "global_size": tuple(int(value) for value in global_size),
      "local_size": tuple(int(value) for value in local_size),
      "before_exec_dword_count": before_exec_dword_count,
      "after_exec_dword_count": len(queue._q),
    }
    return result

  def audited_submit(queue: Any, submit_dev: Any) -> Any:
    nonlocal non_target_submit_count, compute_submit_count
    built = pending.pop(id(queue), None)
    if built is None:
      non_target_submit_count += 1
      return original_submit(queue, submit_dev)
    compute_submit_count += 1
    call_index = len(calls)
    if call_index >= count:
      raise RuntimeError("PM4 dispatch census observed more target submits than expected")
    expected_identity = identities[call_index]
    expected_program_key = target_program_keys[call_index]
    expected_global, expected_local = target_launch_dims[call_index]
    args_state, lifecycle = built["args_state"], built["runtime_lifecycle"]
    kernarg_qwords = [
      int(value) for value in args_state.buf.cpu_view().view(size=40, fmt='Q')[:5]]
    argument_vas = [row["va"] for row in built["argument_buffers"]]
    kernarg_start = lifecycle["kernarg_va"]
    kernarg_end = kernarg_start + lifecycle["kernarg_allocation_nbytes"]
    prior_ranges = [
      (row["runtime_lifecycle"]["kernarg_va"],
       row["runtime_lifecycle"]["kernarg_va"] + row["runtime_lifecycle"]["kernarg_allocation_nbytes"])
      for row in calls]
    command_words = list(queue._q)
    concrete_command_words = all(
      isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xffffffff
      for value in command_words)
    command_bytes = b"".join(
      int(value).to_bytes(4, "little") for value in command_words) if concrete_command_words else b""
    checks = {
      "queue_class_is_native_pm4": type(queue) is ops_amd.AMDComputeQueue,
      "queue_device_matches_submit_device": queue.dev is submit_dev is dev,
      "device_reports_pm4": not bool(getattr(submit_dev, "is_aql", False)),
      "ordered_program_identity_matches": built["program_identity"] == expected_identity,
      **lifecycle["checks"],
      "runtime_cache_exact_program_binding":
        lifecycle["runtime_cache_bindings"] == [{
          "program_key": expected_program_key, "device": "AMD"}],
      "runtime_object_distinct_from_prior_targets":
        all(lifecycle["runtime_object_id"] != row["runtime_lifecycle"]["runtime_object_id"]
            for row in calls),
      "kernarg_disjoint_from_prior_target_records":
        all(kernarg_end <= start or kernarg_start >= end for start, end in prior_ranges),
      "five_qwords_match_constructed_buffers": kernarg_qwords == argument_vas,
      "five_constructed_buffer_vas_distinct": len(set(argument_vas)) == 5,
      "dispatch_dimensions_match":
        built["global_size"] == expected_global and built["local_size"] == expected_local,
      "wait_and_barrier_commands_precede_exec": built["before_exec_dword_count"] > 0,
      "exec_appended_pm4_commands":
        built["after_exec_dword_count"] > built["before_exec_dword_count"],
      "signal_commands_follow_exec": len(command_words) > built["after_exec_dword_count"],
      "pm4_command_words_concrete": concrete_command_words,
      "pm4_command_stream_nonempty": bool(command_bytes),
      **_audit_target_aql_kernargs(
        kernarg_qwords, [row["kernarg_qwords"] for row in calls],
        expected_vas=None, require_fixed_scale_va=False,
        require_all_five_vas_fixed=require_all_five_vas_fixed,
        require_all_five_vas_distinct=require_all_five_vas_distinct),
    }
    accepted = all(checks.values())
    calls.append({
      "call": call_index, "program_identity": built["program_identity"],
      "expected_program_identity": expected_identity,
      "program_key": expected_program_key,
      "kernarg_qwords": kernarg_qwords,
      "argument_buffers": built["argument_buffers"],
      "runtime_lifecycle": lifecycle,
      "global_size": built["global_size"], "local_size": built["local_size"],
      "expected_global_size": expected_global, "expected_local_size": expected_local,
      "pm4_dword_count": len(command_words),
      "pm4_sha256": hashlib.sha256(command_bytes).hexdigest() if command_bytes else None,
      "before_exec_dword_count": built["before_exec_dword_count"],
      "after_exec_dword_count": built["after_exec_dword_count"],
      "checks": checks, "all_checks_pass": accepted,
      "accepted_before_doorbell": accepted,
      "target_exec_observed": True, "submit_began": False, "submit_returned": False,
    })
    if not accepted:
      raise RuntimeError("PM4 target dispatch census rejected submit before doorbell")
    calls[-1]["submit_began"] = True
    result = original_submit(queue, submit_dev)
    calls[-1]["submit_returned"] = True
    return result

  ops_amd.AMDComputeQueue.exec = audited_exec
  ops_amd.AMDComputeQueue._submit = audited_submit
  try:
    try:
      _realize_outputs_together(output, retained_outputs)
      synchronize["began"] = True
      dev.synchronize()
      synchronize["returned"] = True
    except BaseException as exc:
      if synchronize["began"] and not synchronize["returned"]:
        synchronize["failure"] = f"{type(exc).__name__}: {exc}"
      census = snapshot(
        "SYNCHRONIZATION_ERROR" if synchronize["began"] else "REALIZATION_ERROR", exc)
      try: setattr(exc, "pm4_dispatch_census", census)
      except (AttributeError, TypeError):
        raise PM4DispatchCensusRealizationError(str(exc), census) from exc
      raise
  finally:
    ops_amd.AMDComputeQueue.exec = original_exec
    ops_amd.AMDComputeQueue._submit = original_submit
  if pending or len(calls) != count:
    raise RuntimeError("PM4 dispatch census did not close the exact target dispatch count")
  return snapshot("PASS")


def _realize_with_amd_dispatch_census(
    output: Any, *, target_program_identities: tuple[Mapping[str, Any], ...],
    target_program_keys: tuple[str, ...],
    target_launch_dims: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    require_all_five_vas_fixed: bool = False,
    require_all_five_vas_distinct: bool = False,
    retained_outputs: tuple[Any, ...] = ()) -> dict[str, Any]:
  """Select the existing tinygrad AMD queue without changing dispatch semantics."""
  from tinygrad.device import Device
  if bool(getattr(Device["AMD"], "is_aql", False)):
    return _realize_with_aql_packet_census(
      output, target_program_identities=target_program_identities,
      target_program_keys=target_program_keys, target_launch_dims=target_launch_dims,
      require_runtime_lifecycle=True,
      require_all_five_vas_fixed=require_all_five_vas_fixed,
      require_all_five_vas_distinct=require_all_five_vas_distinct,
      retained_outputs=retained_outputs)
  return _realize_with_pm4_dispatch_census(
    output, target_program_identities=target_program_identities,
    target_program_keys=target_program_keys, target_launch_dims=target_launch_dims,
    require_all_five_vas_fixed=require_all_five_vas_fixed,
    require_all_five_vas_distinct=require_all_five_vas_distinct,
    retained_outputs=retained_outputs)


def _scheduler_prefix_two_launches(address_mode: str, epoch_inputs: tuple[tuple[Any, ...], tuple[Any, ...]],
                                   change_slot: str = "all") -> tuple[tuple[Any, ...], tuple[Any, ...]]:
  """Choose the two exact scheduler calls without manufacturing a launcher."""
  if address_mode not in ("same", "changed"):
    raise ValueError("scheduler prefix-two address_mode must be 'same' or 'changed'")
  if change_slot not in _SCHEDULER_PREFIX_CHANGE_SLOTS:
    raise ValueError(f"scheduler prefix-two change_slot must be one of {_SCHEDULER_PREFIX_CHANGE_SLOTS}")
  if address_mode == "same" and change_slot != "all":
    raise ValueError("scheduler prefix-two same mode does not accept a per-slot change")
  if len(epoch_inputs) != 2 or any(len(row) != 4 for row in epoch_inputs):
    raise ValueError("scheduler prefix-two requires two complete four-input epochs")
  first, second = epoch_inputs
  if len({id(value) for value in first}) != 4 or len({id(value) for value in second}) != 4:
    raise ValueError("each scheduler prefix epoch requires four distinct input tensors")
  if any(left is right for left, right in zip(first, second)):
    raise ValueError("scheduler prefix epochs must own distinct input tensors")
  if address_mode == "same": return first, first
  changed = (set(_SCHEDULER_PREFIX_INPUT_NAMES) if change_slot == "all" else
             set(_SCHEDULER_PREFIX_INPUT_NAMES) - {"q8_scales"} if change_slot == "all_except_q8_scales" else
             {change_slot})
  return first, tuple(second[index] if name in changed else first[index]
                      for index, name in enumerate(_SCHEDULER_PREFIX_INPUT_NAMES))


def run_frozen_scheduler_prefix_two_probe(*, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                          frozen_bundle: str | Path,
                                          address_mode: str,
                                          change_slot: str = "all") -> dict[str, Any]:
  """Dispatch the frozen PROGRAM twice through the normal tinygrad scheduler.

  This is the producer-free counterpart to the existing fixed-VA target-role
  probe. All five ABI tensors are populated and realized before the two-call
  ``custom_kernel`` graph is built. ``same`` reuses epoch zero's four inputs.
  ``changed`` binds epoch one for either one selected input slot or all inputs,
  retaining epoch zero for every unselected input. Slot zero is the same
  in-place output allocation and its AFTER edge orders call 2.
  """
  role_spec = admit_exact_role_spec(role_spec)
  if role_spec.epochs < 2: raise ValueError("scheduler prefix-two requires at least two K256 epochs")

  from tinygrad import Tensor
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )

  binding, frozen_identity = _load_frozen_execution_binding(role_spec, frozen_bundle)
  program = binding.artifact.program
  m, n, k = role_spec.shape
  words_np = _random_q4_words(n, k, 20260721)
  source_np = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  q4_blocks = words_np.view(np.uint8).reshape(n, role_spec.epochs, 144)

  epoch_tensors = []
  for epoch in range(2):
    q4_np = np.ascontiguousarray(q4_blocks[:, epoch:epoch+1, :].reshape(-1).view(np.uint32))
    epoch_tensors.append((
      Tensor(q4_np, device="AMD").realize(),
      Tensor(np.ascontiguousarray(values_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize(),
      Tensor(np.ascontiguousarray(scales_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize(),
      Tensor(np.ascontiguousarray(sums_np[epoch*2:(epoch+1)*2].reshape(-1)), device="AMD").realize(),
    ))
  output_seed = Tensor(np.zeros(m * n, dtype=np.float32), device="AMD").realize()
  Device["AMD"].synchronize()
  launches = _scheduler_prefix_two_launches(address_mode, tuple(epoch_tensors), change_slot)

  def tensor_va(tensor: Any) -> int:
    return int(tensor.uop.buffer.get_buf("AMD").va_addr)

  launch_evidence = []
  output = output_seed
  for call_index, inputs in enumerate(launches):
    vas = [tensor_va(output_seed), *(tensor_va(value) for value in inputs)]
    input_source_epochs = [int(value is epoch_tensors[1][slot]) for slot, value in enumerate(inputs)]
    launch_evidence.append({
      "call": call_index, "input_source_epochs": input_source_epochs,
      "vas": vas,
      "arguments": [{"slot": slot, "name": name, "va": va,
                     **({"source_epoch": input_source_epochs[slot-1]} if slot else {})}
                    for slot, (name, va) in enumerate(zip(_TARGET_BUFFER_NAMES, vas))],
    })
    output = output.custom_kernel(
      *inputs, fxn=lambda *_buffers, program=program: program)[0]

  same_slots = [left == right for left, right in
                zip(launch_evidence[0]["vas"], launch_evidence[1]["vas"])]
  changed = set() if address_mode == "same" else (
    set(_SCHEDULER_PREFIX_INPUT_NAMES) if change_slot == "all" else
    set(_SCHEDULER_PREFIX_INPUT_NAMES) - {"q8_scales"} if change_slot == "all_except_q8_scales" else
    {change_slot})
  expected_same_slots = [True, *(name not in changed for name in _SCHEDULER_PREFIX_INPUT_NAMES)]
  if same_slots != expected_same_slots:
    raise RuntimeError("scheduler prefix-two concrete VA relationship differs from requested mode")
  calls = [u for u in output.uop.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
  if len(calls) != 2 or any(len(get_call_arg_uops(call)) != 5 for call in calls) or \
     calls[0] not in get_call_arg_uops(calls[1])[0].toposort():
    raise RuntimeError("scheduler prefix-two did not retain two ordered five-buffer PROGRAM calls")

  packet_census = _realize_with_aql_packet_census(output, [row["vas"] for row in launch_evidence])
  reference = np.zeros((m, n), dtype=np.float32)
  reference_input_source_epochs = []
  for inputs in launches:
    source_epochs = [int(value is epoch_tensors[1][slot]) for slot, value in enumerate(inputs)]
    reference_input_source_epochs.append(source_epochs)
    q4_epoch, values_epoch, scales_epoch, sums_epoch = source_epochs
    ep_scales = scales_np[scales_epoch*2:(scales_epoch+1)*2].astype(np.float16).astype(np.float32)
    ep_sums = sums_np[sums_epoch*2:(sums_epoch+1)*2].astype(np.float16).astype(np.float32)
    ep_ds4 = Q81MMQDS4Activation(
      values_np[values_epoch*2:(values_epoch+1)*2], ep_scales, ep_sums,
      Q81MMQDS4ActivationSpec(m=m, k=256, m_tile=m))
    ep_spec = Q4KQ81MMQTileSpec(
      role="frozen_scheduler_prefix_two", m=m, n=n, k=256, m_tile=m, n_tile=n,
      activation_layout=Q8_1_MMQ_DS4_LAYOUT)
    reference += q4k_q8_1_mmq_ds4_tile_reference(
      np.ascontiguousarray(q4_blocks[:, q4_epoch:q4_epoch+1, :]).reshape(-1), ep_ds4, ep_spec)
  comparison = _numeric_comparison(output.numpy().reshape(m, n), reference)
  passed = comparison["status"] == "pass"
  return {
    "schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1",
    "status": "PASS" if passed else "BLOCKED", "exact_blocker": None if passed else "numeric output mismatch",
    "research_only": True, "production_dispatch_changed": False, "default_route": "direct_packed",
    "role": role_spec.role, "shape": list(role_spec.shape), "prefix_epochs": [0, 1],
    "address_mode": address_mode, "change_slot": change_slot,
    "reference_input_source_epochs": reference_input_source_epochs,
    "dispatch": {
      "launcher": "tinygrad_scheduler", "mode": "lazy_ops_program_chain", "count": 2,
      "program_key": binding.program_key, "all_five_tensors_realized_before_graph_build": True,
      "program_calls_in_graph": len(calls), "slot_va_equal_between_calls": same_slots,
      "expected_slot_va_equal_between_calls": expected_same_slots, "launches": launch_evidence,
      "aql_packet_census": packet_census,
    },
    "correctness": {"status": "PASS" if passed else "BLOCKED", "comparison": comparison,
                    "authority": "same_session_fp16_rounded_ds4_reference"},
    "frozen_bundle": frozen_identity, "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def _producer_oracle_diagnostic(actual_values: np.ndarray, actual_scales: np.ndarray, actual_sums: np.ndarray,
                                oracle_values: np.ndarray, oracle_scales: np.ndarray,
                                oracle_sums: np.ndarray) -> dict[str, Any]:
  """Compare the realized scheduler producer with the independent NumPy quantizer."""
  arrays = (actual_values, actual_scales, actual_sums, oracle_values, oracle_scales, oracle_sums)
  if any(not isinstance(value, np.ndarray) for value in arrays):
    raise TypeError("producer diagnostic requires NumPy arrays")
  if actual_values.shape != oracle_values.shape or actual_scales.shape != oracle_scales.shape or \
     actual_sums.shape != oracle_sums.shape:
    raise ValueError("producer diagnostic shapes differ from NumPy oracle")
  qvalue_mismatch_count = int(np.count_nonzero(actual_values != oracle_values))
  scale_errors = np.abs(actual_scales.astype(np.float64) - oracle_scales.astype(np.float64))
  sum_errors = np.abs(actual_sums.astype(np.float64) - oracle_sums.astype(np.float64))
  actual_scales_half = actual_scales.astype(np.float16)
  oracle_scales_half = oracle_scales.astype(np.float16)
  actual_sums_half = actual_sums.astype(np.float16)
  oracle_sums_half = oracle_sums.astype(np.float16)
  raw_scale_mismatch_count = int(np.count_nonzero(actual_scales != oracle_scales))
  raw_sum_mismatch_count = int(np.count_nonzero(actual_sums != oracle_sums))
  target_half_scale_mismatch_count = int(np.count_nonzero(actual_scales_half != oracle_scales_half))
  target_half_sum_mismatch_count = int(np.count_nonzero(actual_sums_half != oracle_sums_half))
  exact = qvalue_mismatch_count == raw_scale_mismatch_count == raw_sum_mismatch_count == 0
  return {
    "status": "PASS" if exact else "PRODUCER_ORACLE_ROUNDING_DRIFT",
    "qvalue_mismatch_count": qvalue_mismatch_count,
    "raw_scale_mismatch_count": raw_scale_mismatch_count,
    "raw_sum_mismatch_count": raw_sum_mismatch_count,
    "max_scale_abs_error": float(np.max(scale_errors)) if scale_errors.size else 0.0,
    "max_sum_abs_error": float(np.max(sum_errors)) if sum_errors.size else 0.0,
    "target_half_scale_mismatch_count": target_half_scale_mismatch_count,
    "target_half_sum_mismatch_count": target_half_sum_mismatch_count,
    "exact_numpy_oracle_match": exact,
  }


def _producer_probe_status(consumer_comparison_status: str, producer_diagnostic_status: str
                           ) -> tuple[str, str | None]:
  if consumer_comparison_status != "pass":
    return "CONSUMER_MISMATCH", "target output differs from reference built from actual producer bytes"
  if producer_diagnostic_status != "PASS":
    return "PRODUCER_ORACLE_ROUNDING_DRIFT", None
  return "PASS", None


def run_frozen_scheduler_producer_prefix_probe(*, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                               frozen_bundle: str | Path,
                                               epoch_limit: int) -> dict[str, Any]:
  """Run a 1/2-epoch frozen scheduler prefix with the real physical Q8 producer."""
  role_spec = admit_exact_role_spec(role_spec)
  if not isinstance(epoch_limit, int) or isinstance(epoch_limit, bool) or epoch_limit not in (1, 2):
    raise ValueError("producer-backed scheduler prefix epoch_limit must be 1 or 2")
  if epoch_limit > role_spec.epochs:
    raise ValueError("producer-backed scheduler prefix exceeds the admitted role epoch count")

  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  from extra.qk.prefill.frozen_exact_role_scheduler import build_frozen_exact_q4k_schedule
  from extra.qk.q4k_q8_activation_producer import produce_physical_ds4_q8_1_tensor

  binding, frozen_identity = _load_frozen_execution_binding(role_spec, frozen_bundle)
  m, n, k = role_spec.shape
  words_np = _random_q4_words(n, k, 20260721)
  # Model activations arrive as fp16. Preserve that rounding in both the real
  # producer input and the independent NumPy reference.
  activation_np = np.random.default_rng(20260722).standard_normal((m, k), dtype=np.float32).astype(np.float16)
  packed_weight = Tensor(words_np, dtype=dtypes.uint32, device="AMD")
  activation = Tensor(activation_np, dtype=dtypes.float16, device="AMD")
  linear = SimpleNamespace(
    bias=None, out_features=n, in_features=k, q4k_storage=object(),
    prefill_packed_weight=lambda: packed_weight)
  produced_tiles = []
  def capture_producer(source, spec):
    tile = produce_physical_ds4_q8_1_tensor(source, spec)
    produced_tiles.append(tile)
    return tile
  schedule = build_frozen_exact_q4k_schedule(
    linear, activation, role_spec=role_spec, frozen_bundle=frozen_bundle,
    enabled=True, binding=binding, activation_producer=capture_producer,
    epoch_limit=epoch_limit, fixed_scale_stage=True)
  if schedule is None: raise RuntimeError("producer-backed frozen scheduler unexpectedly remained disabled")
  if len(produced_tiles) != epoch_limit:
    raise RuntimeError("producer-backed scheduler did not expose one physical Q8 tile per target epoch")
  output = schedule.output
  program = binding.artifact.program
  calls = [u for u in output.uop.toposort() if u.op is Ops.CALL and u.src[0] is program]
  if len(calls) != epoch_limit or any(len(get_call_arg_uops(call)) != 5 for call in calls):
    raise RuntimeError("producer-backed scheduler prefix lost its exact target call count or ABI")
  scale_buffer_keys = [get_call_arg_uops(call)[3].buf_uop.key.hex() for call in calls]
  if len(set(scale_buffer_keys)) != 1:
    raise RuntimeError("producer-backed scheduler prefix lost its fixed Q8 scale buffer identity")
  for previous, current in zip(calls, calls[1:]):
    if previous not in get_call_arg_uops(current)[0].toposort() or \
       previous not in get_call_arg_uops(current)[3].toposort():
      raise RuntimeError("producer-backed scheduler prefix lost output or Q8 scale ordering")

  target_identity = {
    "function_name": program.arg.function_name,
    "binary_sha256": binding.binary_sha256,
  }
  graph_evidence = {
    "program_calls": len(calls), "expected_program_calls": epoch_limit,
    "five_buffer_abi": True, "output_after_chain": True,
    "q8_scale_buffer_keys": scale_buffer_keys,
    "q8_scale_fixed_buffer_identity": len(set(scale_buffer_keys)) == 1,
  }
  retained_outputs = _retained_producer_tensors(produced_tiles)
  graph_evidence.update({
    "retained_producer_tensor_count": len(retained_outputs),
    "retained_producer_tensor_names_per_epoch": ["values", "scales", "sums"],
    "retained_as_companion_realization_outputs": True,
  })
  try:
    packet_census = _realize_with_aql_packet_census(
      output, target_program_identity=target_identity,
      target_dispatch_count=epoch_limit, require_fixed_scale_va=True,
      retained_outputs=retained_outputs)
  except BaseException as exc:
    packet_census = _aql_packet_census_from_exception(exc)
    return {
      "schema": "tinygrad.mmq_frozen_scheduler_producer_prefix_probe.v1",
      "status": "BLOCKED", "exact_blocker": "producer-backed target dispatch rejected before doorbell",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False, "default_route": "direct_packed",
      "role": role_spec.role, "shape": list(role_spec.shape), "epoch_limit": epoch_limit,
      "producer": "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
      "graph": graph_evidence, "target_program_identity": target_identity,
      **({"dispatch": {"aql_packet_census": packet_census}} if packet_census is not None else {}),
      "frozen_bundle": frozen_identity, "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  q4_blocks = words_np.view(np.uint8).reshape(n, role_spec.epochs, 144)
  activation_ref = activation_np.astype(np.float32)
  oracle_values, oracle_scales, oracle_sums = q8_1_mmq_ds4_quantize_reference(
    activation_ref[:, :epoch_limit*256])
  actual_values = np.concatenate([tile.values.numpy() for tile in produced_tiles], axis=0)
  actual_scales = np.concatenate([tile.scales.numpy() for tile in produced_tiles], axis=0)
  actual_sums = np.concatenate([tile.sums.numpy() for tile in produced_tiles], axis=0)
  producer_diagnostic = _producer_oracle_diagnostic(
    actual_values, actual_scales, actual_sums, oracle_values, oracle_scales, oracle_sums)

  q4_prefix = np.ascontiguousarray(q4_blocks[:, :epoch_limit, :]).reshape(-1)
  ref_spec = Q4KQ81MMQTileSpec(
    role="frozen_scheduler_producer_prefix", m=m, n=n, k=epoch_limit*256,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  # Consumer authority: use the exact realized producer bytes. The target
  # stages metadata through half2, so round only metadata through fp16 here.
  actual_ds4 = Q81MMQDS4Activation(
    actual_values,
    actual_scales.astype(np.float16).astype(np.float32),
    actual_sums.astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=epoch_limit*256, m_tile=m))
  consumer_reference = q4k_q8_1_mmq_ds4_tile_reference(q4_prefix, actual_ds4, ref_spec)
  got = output.numpy().reshape(m, n)
  consumer_comparison = _numeric_comparison(got, consumer_reference)
  # Historical/source oracle remains a separate diagnostic. It must not
  # relabel a correct consumer of the producer's actual bytes as incorrect.
  oracle_ds4 = Q81MMQDS4Activation(
    oracle_values,
    oracle_scales.astype(np.float16).astype(np.float32),
    oracle_sums.astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=epoch_limit*256, m_tile=m))
  source_oracle_reference = q4k_q8_1_mmq_ds4_tile_reference(q4_prefix, oracle_ds4, ref_spec)
  source_oracle_comparison = _numeric_comparison(got, source_oracle_reference)
  status, blocker = _producer_probe_status(consumer_comparison["status"], producer_diagnostic["status"])
  return {
    "schema": "tinygrad.mmq_frozen_scheduler_producer_prefix_probe.v1",
    "status": status, "exact_blocker": blocker,
    "research_only": True, "production_dispatch_changed": False, "default_route": "direct_packed",
    "role": role_spec.role, "shape": list(role_spec.shape), "epoch_limit": epoch_limit,
    "producer": "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
    "graph": graph_evidence,
    "dispatch": {
      "launcher": "tinygrad_scheduler", "mode": "lazy_ops_program_chain",
      "count": epoch_limit, "program_key": binding.program_key,
      "target_program_identity": target_identity, "aql_packet_census": packet_census,
    },
    "correctness": {
      "status": "PASS" if consumer_comparison["status"] == "pass" else "CONSUMER_MISMATCH",
      "comparison": consumer_comparison,
      "authority": "same_session_actual_producer_bytes_with_target_fp16_metadata_roundtrip",
    },
    "producer_diagnostic": {
      **producer_diagnostic,
      "source_oracle_output_comparison": source_oracle_comparison,
      "source_oracle": "extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
    },
    "scheduler_evidence": schedule.evidence,
    "frozen_bundle": frozen_identity, "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def _frozen_program_set_target_identities(binding: Any, prefix_epochs: int) -> tuple[dict[str, str], ...]:
  """Derive the ordered runtime identities from the exact retained v2 payloads."""
  programs, binaries = binding.artifact.programs[:prefix_epochs], binding.artifact.binaries[:prefix_epochs]
  if len(programs) != prefix_epochs or len(binaries) != prefix_epochs:
    raise ValueError("frozen v2 prefix does not contain all requested program payloads")
  identities = tuple({
    "function_name": program.arg.function_name,
    "binary_sha256": hashlib.sha256(binary).hexdigest(),
  } for program, binary in zip(programs, binaries))
  if any(not row["function_name"] or not row["binary_sha256"] for row in identities):
    raise ValueError("frozen v2 prefix contains an incomplete runtime identity")
  if len({(row["function_name"], row["binary_sha256"]) for row in identities}) != prefix_epochs:
    raise ValueError("frozen v2 prefix target identities are not distinct")
  return identities


class FrozenRuntimePreconstructionError(RuntimeError):
  """Fail-closed wrapper retaining partial exact-runtime preconstruction evidence."""
  def __init__(self, message: str, evidence: Mapping[str, Any]):
    super().__init__(message)
    self.runtime_preconstruction = dict(evidence)


def _runtime_preconstruction_device_snapshot(dev: Any) -> dict[str, int | None]:
  """Read lifecycle counters without submitting work or manufacturing a queue."""
  signal = getattr(dev, "timeline_signal", None)
  try: signal_value = int(signal.value) if signal is not None else None
  except BaseException: signal_value = None
  return {
    "timeline_value": int(getattr(dev, "timeline_value"))
      if isinstance(getattr(dev, "timeline_value", None), int) else None,
    "signal_value": signal_value,
    "prof_exec_counter": int(getattr(dev, "prof_exec_counter"))
      if isinstance(getattr(dev, "prof_exec_counter", None), int) else None,
  }


def _preconstruct_frozen_program_runtimes(
    programs: tuple[Any, ...], program_keys: tuple[str, ...],
    target_identities: tuple[Mapping[str, Any], ...], *, device: str = "AMD",
    ) -> dict[str, Any]:
  """Construct exact cached runtimes before realization through ``get_runtime``.

  This is a diagnostic lifecycle discriminator, not a launcher. AMDProgram
  remains responsible for native code allocation/upload/synchronization, and
  the scheduler later resolves the same objects from the ordinary runtime
  cache before submitting the unchanged PROGRAM chain.
  """
  count = len(programs)
  if count <= 0 or len(program_keys) != count or len(target_identities) != count:
    raise ValueError("runtime preconstruction requires equally sized nonempty PROGRAM identity sequences")
  if any(not isinstance(key, str) or len(key) != 64 for key in program_keys):
    raise ValueError("runtime preconstruction requires exact hexadecimal PROGRAM keys")
  identities = tuple(dict(identity) for identity in target_identities)
  if any(identity.get("function_name") in (None, "") or
         identity.get("binary_sha256") in (None, "") for identity in identities):
    raise ValueError("runtime preconstruction requires complete binary identities")
  if len(set(program_keys)) != count or len({
      (identity["function_name"], identity["binary_sha256"]) for identity in identities
  }) != count:
    raise ValueError("runtime preconstruction requires distinct ordered PROGRAM identities")
  if any(getattr(program, "key", b"").hex() != key
         for program, key in zip(programs, program_keys)):
    raise ValueError("runtime preconstruction PROGRAM keys differ from the retained sequence")

  from tinygrad.device import Device
  from tinygrad.engine.realize import get_runtime, runtime_cache
  dev = Device[device]
  records: list[dict[str, Any]] = []
  attempts: list[dict[str, Any]] = []
  before = _runtime_preconstruction_device_snapshot(dev)

  def snapshot(status: str, error: BaseException | None = None) -> dict[str, Any]:
    after = _runtime_preconstruction_device_snapshot(dev)
    result = {
      "enabled": True, "status": status, "device": device, "count": count,
      "ordered_program_keys": list(program_keys),
      "timeline_before": dict(before), "timeline_after": after,
      "prof_exec_counter_before": before["prof_exec_counter"],
      "prof_exec_counter_after": after["prof_exec_counter"],
      "no_compute_dispatch_during_preconstruction":
        before["prof_exec_counter"] == after["prof_exec_counter"],
      "runtime_cache_retains_code_allocations": all(
        runtime_cache.get((program.key, device)) is record["_runtime"]
        for program, record in zip(programs[:len(records)], records)),
      "attempts": [dict(attempt) for attempt in attempts],
      "runtimes": [{key: value for key, value in record.items() if key != "_runtime"}
                   for record in records],
    }
    result["all_checks_pass"] = (
      len(records) == count and
      result["no_compute_dispatch_during_preconstruction"] and
      result["runtime_cache_retains_code_allocations"] and
      all(record["all_checks_pass"] for record in records))
    if error is not None:
      result.update({"exception": type(error).__name__, "error": str(error)})
    return result

  preexisting = [key for program, key in zip(programs, program_keys)
                 if (program.key, device) in runtime_cache]
  if preexisting:
    evidence = snapshot("REJECTED_PREEXISTING_CACHE")
    evidence["preexisting_program_keys"] = preexisting
    raise FrozenRuntimePreconstructionError(
      "selected runtime cache entries existed before explicit preconstruction", evidence)

  prior_ranges: list[tuple[int, int]] = []
  prior_runtime_ids: list[int] = []
  previous_timeline = before
  for epoch, (program, program_key, expected_identity) in enumerate(
      zip(programs, program_keys, identities)):
    attempt = {
      "epoch": epoch, "program_key": program_key,
      "expected_program_identity": dict(expected_identity),
      "get_runtime_begin": _runtime_preconstruction_device_snapshot(dev),
      "get_runtime_returned": False,
    }
    attempts.append(attempt)
    try:
      runtime = get_runtime(device, program)
    except BaseException as exc:
      attempt.update({
        "get_runtime_exception": type(exc).__name__,
        "get_runtime_error": str(exc),
      })
      evidence = snapshot("GET_RUNTIME_ERROR", exc)
      evidence.update({
        "failure_boundary": "get_runtime_call_raised_before_return",
        "failed_attempt": dict(attempt),
      })
      raise FrozenRuntimePreconstructionError(str(exc), evidence) from exc
    attempt["get_runtime_returned"] = True
    attempt["get_runtime_end"] = _runtime_preconstruction_device_snapshot(dev)
    try:
      lib = getattr(runtime, "lib_gpu", None)
      lib_va, lib_nbytes = int(getattr(lib, "va_addr", 0)), int(getattr(lib, "size", 0))
      lib_end = lib_va + lib_nbytes
      entry_va = int(getattr(runtime, "prog_addr", 0))
      descriptor_va = int(getattr(runtime, "aql_prog_addr", 0))
      runtime_identity = _aql_target_program_identity(runtime)
      cache_bindings = [{
        "program_key": key[0].hex(), "device": key[1],
      } for key, cached in runtime_cache.items() if cached is runtime]
      after_runtime = _runtime_preconstruction_device_snapshot(dev)
      checks = {
        "program_key_matches_retained_sequence": program.key.hex() == program_key,
        "runtime_identity_matches_retained_binary": runtime_identity == expected_identity,
        "runtime_cache_exact_program_binding":
          cache_bindings == [{"program_key": program_key, "device": device}],
        "runtime_object_distinct_from_prior":
          id(runtime) not in prior_runtime_ids,
        "program_library_range_nonempty": lib_va > 0 and lib_nbytes > 0,
        "program_library_disjoint_from_prior":
          all(lib_end <= start or lib_va >= end for start, end in prior_ranges),
        "program_entry_in_library_range": lib_va <= entry_va < lib_end,
        "program_descriptor_in_library_range": lib_va <= descriptor_va < lib_end,
        "timeline_did_not_regress":
          previous_timeline["timeline_value"] is not None and
          after_runtime["timeline_value"] is not None and
          after_runtime["timeline_value"] >= previous_timeline["timeline_value"],
        "device_timeline_drained_after_runtime_init":
          after_runtime["timeline_value"] is not None and
          after_runtime["signal_value"] == after_runtime["timeline_value"] - 1,
        "no_compute_dispatch_during_runtime_init":
          after_runtime["prof_exec_counter"] == before["prof_exec_counter"],
      }
      record = {
        "epoch": epoch, "program_key": program_key,
        "program_identity": runtime_identity,
        "expected_program_identity": expected_identity,
        "runtime_object_id": id(runtime),
        "program_library_va": lib_va, "program_library_nbytes": lib_nbytes,
        "program_entry_va": entry_va, "program_entry_offset": entry_va-lib_va,
        "program_descriptor_va": descriptor_va,
        "program_descriptor_offset": descriptor_va-lib_va,
        "runtime_cache_bindings": cache_bindings,
        "timeline_after_runtime_init": after_runtime,
        "checks": checks, "all_checks_pass": all(checks.values()),
        "_runtime": runtime,
      }
      records.append(record)
      if not record["all_checks_pass"]:
        raise RuntimeError("exact runtime preconstruction lifecycle audit failed")
      prior_ranges.append((lib_va, lib_end))
      prior_runtime_ids.append(id(runtime))
      previous_timeline = after_runtime
    except BaseException as exc:
      evidence = snapshot("POST_GET_RUNTIME_AUDIT_ERROR", exc)
      evidence.update({
        "failure_boundary": "lifecycle_audit_after_get_runtime_return",
        "failed_attempt": dict(attempt),
      })
      raise FrozenRuntimePreconstructionError(str(exc), evidence) from exc

  evidence = snapshot("PASS")
  if not evidence["all_checks_pass"]:
    raise FrozenRuntimePreconstructionError(
      "runtime preconstruction did not retain the exact selected runtime family", evidence)
  return evidence


def _crosscheck_preconstructed_dispatch_runtimes(
    runtime_preconstruction: Mapping[str, Any],
    dispatch_census: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
  """Prove scheduler dispatch reused preconstructed runtime objects in order."""
  if runtime_preconstruction.get("enabled") is not True:
    return {
      "enabled": False, "status": "NOT_REQUESTED",
      "all_checks_pass": True,
    }
  expected_rows = runtime_preconstruction.get("runtimes")
  calls = dispatch_census.get("calls") if isinstance(dispatch_census, Mapping) else None
  if not isinstance(expected_rows, list) or not isinstance(calls, list):
    return {
      "enabled": True, "status": "INCOMPLETE",
      "expected_runtime_object_ids": [],
      "dispatch_runtime_object_ids": [],
      "checks": {
        "preconstruction_runtime_rows_available": isinstance(expected_rows, list),
        "dispatch_call_rows_available": isinstance(calls, list),
        "dispatch_count_matches_preconstruction": False,
        "ordered_runtime_object_ids_match": False,
      },
      "all_checks_pass": False,
    }
  expected_ids = [row.get("runtime_object_id") for row in expected_rows
                  if isinstance(row, Mapping)]
  expected_keys = [row.get("program_key") for row in expected_rows
                   if isinstance(row, Mapping)]
  dispatch_ids, dispatch_keys = [], []
  complete_dispatch_rows = True
  for call in calls:
    lifecycle = call.get("runtime_lifecycle") if isinstance(call, Mapping) else None
    runtime_id = lifecycle.get("runtime_object_id") if isinstance(lifecycle, Mapping) else None
    program_key = call.get("program_key") if isinstance(call, Mapping) else None
    dispatch_ids.append(runtime_id)
    dispatch_keys.append(program_key)
    complete_dispatch_rows &= runtime_id is not None and program_key is not None
  checks = {
    "preconstruction_runtime_rows_available":
      len(expected_ids) == len(expected_rows) and
      all(runtime_id is not None for runtime_id in expected_ids),
    "dispatch_call_rows_available": complete_dispatch_rows,
    "dispatch_count_matches_preconstruction": len(dispatch_ids) == len(expected_ids),
    "ordered_program_keys_match": dispatch_keys == expected_keys,
    "ordered_runtime_object_ids_match": dispatch_ids == expected_ids,
    "observed_dispatch_prefix_reuses_preconstructed_runtimes":
      dispatch_ids == expected_ids[:len(dispatch_ids)] and
      dispatch_keys == expected_keys[:len(dispatch_keys)],
  }
  passed = all(checks.values())
  return {
    "enabled": True,
    "status": "PASS" if passed else (
      "INCOMPLETE" if checks["observed_dispatch_prefix_reuses_preconstructed_runtimes"] else "MISMATCH"),
    "expected_runtime_object_ids": expected_ids,
    "dispatch_runtime_object_ids": dispatch_ids,
    "expected_program_keys": expected_keys,
    "dispatch_program_keys": dispatch_keys,
    "checks": checks, "all_checks_pass": passed,
  }


def _dispatch_error_runtime_reuse_evidence(
    runtime_preconstruction: Mapping[str, Any], exc: BaseException,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
  """Recover partial queue census and runtime reuse evidence from a failed realization."""
  packet_census = _amd_dispatch_census_from_exception(exc)
  return packet_census, _crosscheck_preconstructed_dispatch_runtimes(
    runtime_preconstruction, packet_census)


def _frozen_program_set_ordinal_target_identity(binding: Any, epoch: int) -> dict[str, str]:
  """Derive one exact runtime identity without changing the family order."""
  programs, binaries = binding.artifact.programs, binding.artifact.binaries
  if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < len(programs) or \
     len(programs) != len(binaries):
    raise ValueError("frozen v2 ordinal is outside the complete retained PROGRAM family")
  program, binary = programs[epoch], binaries[epoch]
  identity = {
    "function_name": program.arg.function_name,
    "binary_sha256": hashlib.sha256(binary).hexdigest(),
  }
  if not identity["function_name"] or not identity["binary_sha256"]:
    raise ValueError("frozen v2 ordinal contains an incomplete runtime identity")
  return identity


def _frozen_program_set_ordinal_sequence_target_identities(
    binding: Any, epochs: tuple[int, int]) -> tuple[dict[str, str], dict[str, str]]:
  """Derive exact ordered identities for two selected family ordinals."""
  identities = tuple(_frozen_program_set_ordinal_target_identity(binding, epoch) for epoch in epochs)
  if len({(row["function_name"], row["binary_sha256"]) for row in identities}) != 2:
    raise ValueError("frozen v2 ordinal sequence target identities are not distinct")
  return identities[0], identities[1]


def _fixed_base_prefix_reference_operands(q4_blocks: np.ndarray, values: np.ndarray,
                                          scales: np.ndarray, sums: np.ndarray,
                                          prefix_epochs: int
                                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Slice the static-offset prefix from retained full-role fixed-base buffers."""
  if not isinstance(q4_blocks, np.ndarray) or q4_blocks.ndim != 3 or \
     q4_blocks.dtype != np.uint8 or q4_blocks.shape[2] != 144:
    raise ValueError("fixed-base prefix requires uint8 Q4 blocks shaped [N,epochs,144]")
  if not isinstance(prefix_epochs, int) or isinstance(prefix_epochs, bool) or \
     not 1 <= prefix_epochs <= q4_blocks.shape[1]:
    raise ValueError("fixed-base reference prefix is outside the Q4 epoch extent")
  if any(not isinstance(value, np.ndarray) or value.ndim != 3
         for value in (values, scales, sums)):
    raise ValueError("fixed-base prefix requires rank-three retained DS4 arrays")
  if values.shape[0] != q4_blocks.shape[1] * 2 or \
     scales.shape[0] != values.shape[0] or sums.shape != scales.shape:
    raise ValueError("retained full-role DS4 arrays differ from the Q4 epoch extent")
  records = prefix_epochs * 2
  return (
    np.ascontiguousarray(q4_blocks[:, :prefix_epochs, :]).reshape(-1),
    np.ascontiguousarray(values[:records]),
    np.ascontiguousarray(scales[:records]),
    np.ascontiguousarray(sums[:records]),
  )


def _fixed_base_ordinal_reference_operands(q4_blocks: np.ndarray, values: np.ndarray,
                                           scales: np.ndarray, sums: np.ndarray,
                                           epoch: int
                                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Slice one static-offset ordinal from retained full-role fixed-base buffers."""
  if not isinstance(q4_blocks, np.ndarray) or q4_blocks.ndim != 3 or \
     q4_blocks.dtype != np.uint8 or q4_blocks.shape[2] != 144:
    raise ValueError("fixed-base ordinal requires uint8 Q4 blocks shaped [N,epochs,144]")
  if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < q4_blocks.shape[1]:
    raise ValueError("fixed-base reference ordinal is outside the Q4 epoch extent")
  if any(not isinstance(value, np.ndarray) or value.ndim != 3
         for value in (values, scales, sums)):
    raise ValueError("fixed-base ordinal requires rank-three retained DS4 arrays")
  if values.shape[0] != q4_blocks.shape[1] * 2 or \
     scales.shape[0] != values.shape[0] or sums.shape != scales.shape:
    raise ValueError("retained full-role DS4 arrays differ from the Q4 epoch extent")
  record = epoch * 2
  return (
    np.ascontiguousarray(q4_blocks[:, epoch:epoch+1, :]).reshape(-1),
    np.ascontiguousarray(values[record:record+2]),
    np.ascontiguousarray(scales[record:record+2]),
    np.ascontiguousarray(sums[record:record+2]),
  )


def _fixed_base_ordinal_sequence_reference_operands(
    q4_blocks: np.ndarray, values: np.ndarray, scales: np.ndarray, sums: np.ndarray,
    epochs: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Concatenate exactly two selected static-offset ordinals for the reference."""
  if not isinstance(q4_blocks, np.ndarray) or q4_blocks.ndim != 3 or \
     q4_blocks.dtype != np.uint8 or q4_blocks.shape[2] != 144:
    raise ValueError("fixed-base ordinal sequence requires uint8 Q4 blocks shaped [N,epochs,144]")
  if not isinstance(epochs, tuple) or len(epochs) != 2 or \
     any(not isinstance(epoch, int) or isinstance(epoch, bool) for epoch in epochs) or \
     not 0 <= epochs[0] < epochs[1] < q4_blocks.shape[1]:
    raise ValueError("fixed-base reference requires exactly two strictly increasing ordinals")
  if any(not isinstance(value, np.ndarray) or value.ndim != 3
         for value in (values, scales, sums)):
    raise ValueError("fixed-base ordinal sequence requires rank-three retained DS4 arrays")
  if values.shape[0] != q4_blocks.shape[1] * 2 or \
     scales.shape[0] != values.shape[0] or sums.shape != scales.shape:
    raise ValueError("retained full-role DS4 arrays differ from the Q4 epoch extent")
  records = tuple(record for epoch in epochs for record in (epoch*2, epoch*2+1))
  return (
    np.ascontiguousarray(np.concatenate(
      [q4_blocks[:, epoch:epoch+1, :] for epoch in epochs], axis=1)).reshape(-1),
    np.ascontiguousarray(values[list(records)]),
    np.ascontiguousarray(scales[list(records)]),
    np.ascontiguousarray(sums[list(records)]),
  )


def _validate_v2_fixed_base_prefix_epochs(role_spec: ExactRoleSpec, prefix_epochs: int) -> int:
  """Admit bounded diagnostic prefixes plus the role's complete epoch count."""
  allowed = tuple(sorted({1, 2, 3, role_spec.epochs}))
  if not isinstance(prefix_epochs, int) or isinstance(prefix_epochs, bool) or prefix_epochs not in allowed:
    raise ValueError(f"frozen v2 GPU probe prefix_epochs must be one of {allowed} for role {role_spec.role!r}")
  return prefix_epochs


def _validate_v2_fixed_base_ordinal(role_spec: ExactRoleSpec, epoch: int) -> int:
  """Admit one diagnostic ordinal without weakening contiguous-prefix admission."""
  if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < role_spec.epochs:
    raise ValueError(
      f"frozen v2 GPU probe epoch must be in [0,{role_spec.epochs}) for role {role_spec.role!r}")
  return epoch


def _validate_v2_fixed_base_ordinal_sequence(
    role_spec: ExactRoleSpec, epochs: tuple[int, int] | list[int]) -> tuple[int, int]:
  """Admit exactly two increasing research ordinals, independent of prefix admission."""
  if not isinstance(epochs, (tuple, list)) or len(epochs) != 2 or \
     any(not isinstance(epoch, int) or isinstance(epoch, bool) for epoch in epochs):
    raise ValueError("frozen v2 ordinal sequence requires exactly two integer ordinals")
  selected = (epochs[0], epochs[1])
  if not 0 <= selected[0] < selected[1] < role_spec.epochs:
    raise ValueError(
      f"frozen v2 ordinal sequence must be strictly increasing within [0,{role_spec.epochs}) "
      f"for role {role_spec.role!r}")
  return selected


def run_frozen_epoch_program_set_ordinal_probe(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    epoch: int) -> dict[str, Any]:
  """Run one frozen v2 ordinal against full-role fixed-base producer buffers."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_ordinal_probe.v2"
  role_spec = admit_exact_role_spec(role_spec)
  epoch = _validate_v2_fixed_base_ordinal(role_spec, epoch)

  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_frozen_epoch_program_set import load_frozen_epoch_program_set_binding
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  from extra.qk.q4k_q8_activation_producer import (
    PhysicalDS4Q8ActivationSpec, produce_physical_ds4_q8_1_tensor,
  )

  binding = load_frozen_epoch_program_set_binding(role_spec, frozen_bundle)
  if len(binding.program_keys) != role_spec.epochs:
    raise RuntimeError("frozen v2 ordinal probe requires the complete admitted PROGRAM family")
  program = binding.artifact.programs[epoch]
  if program.key.hex() != binding.program_keys[epoch]:
    raise RuntimeError("frozen v2 ordinal PROGRAM key differs from its admitted family position")
  variant_row = binding.artifact.manifest["variants"][epoch]
  if variant_row["epoch"] != epoch or variant_row["program_key"] != binding.program_keys[epoch]:
    raise RuntimeError("frozen v2 ordinal manifest differs from its admitted family position")
  target_identity = _frozen_program_set_ordinal_target_identity(binding, epoch)
  m, n, k = role_spec.shape
  words_np = _random_q4_words(n, k, 20260721)
  activation_np = np.random.default_rng(20260722).standard_normal(
    (m, k), dtype=np.float32).astype(np.float16)
  packed_weight = Tensor(words_np, dtype=dtypes.uint32, device="AMD")
  activation = Tensor(activation_np, dtype=dtypes.float16, device="AMD")
  tile = produce_physical_ds4_q8_1_tensor(
    activation.cast(dtypes.float32).contiguous(),
    PhysicalDS4Q8ActivationSpec(m, k))

  output_seed = activation.flatten()[:1].cast(dtypes.float32)
  zero = output_seed._apply_uop(lambda u: u.mul(0)).expand(m*n)
  zeroed_output = Tensor.empty(m*n, dtype=dtypes.float32, device="AMD")
  zeroed_output.assign(zero)
  fixed_inputs = (packed_weight, tile.values, tile.scales, tile.sums)
  output = zeroed_output.custom_kernel(
    *fixed_inputs, fxn=lambda *_buffers, program=program: program)[0]

  family_calls = [node for node in output.uop.toposort()
                  if node.op is Ops.CALL and node.src[0] in binding.artifact.programs]
  if len(family_calls) != 1 or family_calls[0].src[0] is not program:
    raise RuntimeError("frozen v2 ordinal graph did not retain exactly its selected PROGRAM")
  arguments = get_call_arg_uops(family_calls[0])
  if len(arguments) != 5:
    raise RuntimeError("frozen v2 ordinal graph lost the five-buffer ABI")
  if arguments[0].buf_uop is not zeroed_output.uop.buf_uop:
    raise RuntimeError("frozen v2 ordinal graph lost its explicitly zeroed output allocation")
  graph_evidence = {
    "program_calls": 1, "expected_program_calls": 1,
    "selected_epoch": epoch, "selected_program_key": binding.program_keys[epoch],
    "selected_sink_key": variant_row["sink_key"], "selected_offsets": dict(variant_row["offsets"]),
    "single_program_ordinal": True, "five_buffer_abi": True,
    "distinct_concrete_buffer_vas_deferred_to_dispatch_census": True,
    "initial_output_zeroed": True,
    "buffer_uop_keys": [value.buf_uop.key.hex() for value in arguments],
    "full_role_producer_calls": 1,
  }
  retained_outputs = _retained_producer_tensors([tile])
  graph_evidence.update({
    "retained_producer_tensor_count": len(retained_outputs),
    "retained_producer_tensor_names": ["values", "scales", "sums"],
    "retained_as_companion_realization_outputs": True,
  })

  dispatch_census_key = "aql_packet_census" if bool(getattr(Device["AMD"], "is_aql", False)) \
    else "pm4_dispatch_census"
  try:
    packet_census = _realize_with_amd_dispatch_census(
      output, target_program_identities=(target_identity,),
      target_program_keys=(binding.program_keys[epoch],),
      target_launch_dims=((tuple(program.arg.global_size), tuple(program.arg.local_size)),),
      require_all_five_vas_fixed=True, require_all_five_vas_distinct=True,
      retained_outputs=retained_outputs)
  except BaseException as exc:
    packet_census = _amd_dispatch_census_from_exception(exc)
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal target dispatch failed",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False,
      "production_scheduler_used": False, "scheduler_prefix_semantics_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "epoch": epoch,
      "graph": graph_evidence, "target_program_identity": target_identity,
      **({"dispatch": {dispatch_census_key: packet_census}} if packet_census is not None else {}),
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  actual_values, actual_scales, actual_sums = \
    tile.values.numpy(), tile.scales.numpy(), tile.sums.numpy()
  q4_blocks = words_np.view(np.uint8).reshape(n, role_spec.epochs, 144)
  q4_epoch, values_epoch, scales_epoch, sums_epoch = _fixed_base_ordinal_reference_operands(
    q4_blocks, actual_values, actual_scales, actual_sums, epoch)
  ref_spec = Q4KQ81MMQTileSpec(
    role="frozen_epoch_program_set_ordinal", m=m, n=n, k=256,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  actual_ds4 = Q81MMQDS4Activation(
    values_epoch, scales_epoch.astype(np.float16).astype(np.float32),
    sums_epoch.astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=256, m_tile=m))
  consumer_reference = q4k_q8_1_mmq_ds4_tile_reference(q4_epoch, actual_ds4, ref_spec)
  consumer_comparison = _numeric_comparison(
    output.numpy().reshape(m, n), consumer_reference)

  oracle_values, oracle_scales, oracle_sums = q8_1_mmq_ds4_quantize_reference(
    activation_np.astype(np.float32))
  producer_diagnostic = _producer_oracle_diagnostic(
    actual_values, actual_scales, actual_sums,
    oracle_values, oracle_scales, oracle_sums)
  passed = consumer_comparison["status"] == "pass"
  return {
    "schema": schema, "status": "PASS" if passed else "CONSUMER_MISMATCH",
    "exact_blocker": None if passed else
      "ordinal output differs from reference built from retained full-role producer bytes",
    "research_only": True, "production_dispatch_changed": False,
    "production_scheduler_used": False, "scheduler_prefix_semantics_changed": False,
    "default_route": "direct_packed", "role": role_spec.role,
    "shape": list(role_spec.shape), "epoch": epoch,
    "producer": "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
    "graph": graph_evidence,
    "dispatch": {
      "launcher": "tinygrad_scheduler", "mode": "single_static_offset_program_ordinal",
      "count": 1, "epoch": epoch, "program_key": binding.program_keys[epoch],
      "target_program_identity": target_identity,
      dispatch_census_key: packet_census,
    },
    "correctness": {
      "status": "PASS" if passed else "CONSUMER_MISMATCH",
      "comparison": consumer_comparison,
      "authority": "same_session_retained_full_role_producer_bytes_with_exact_static_offset_ordinal_and_fp16_metadata_roundtrip",
    },
    "producer_diagnostic": {
      **producer_diagnostic,
      "source_oracle": "extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
    },
    "family_identity": binding.family_identity,
    "frozen_bundle": str(Path(frozen_bundle).resolve()),
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def run_frozen_epoch_program_set_ordinal_sequence_probe(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    epochs: tuple[int, int] | list[int]) -> dict[str, Any]:
  """Run exactly two selected frozen v2 ordinals over one fixed five-buffer ABI."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_ordinal_sequence_probe.v2"
  role_spec = admit_exact_role_spec(role_spec)
  epochs = _validate_v2_fixed_base_ordinal_sequence(role_spec, epochs)

  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_frozen_epoch_program_set import load_frozen_epoch_program_set_binding
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  from extra.qk.q4k_q8_activation_producer import (
    PhysicalDS4Q8ActivationSpec, produce_physical_ds4_q8_1_tensor,
  )

  binding = load_frozen_epoch_program_set_binding(role_spec, frozen_bundle)
  if len(binding.program_keys) != role_spec.epochs:
    raise RuntimeError("frozen v2 ordinal sequence requires the complete admitted PROGRAM family")
  programs = tuple(binding.artifact.programs[epoch] for epoch in epochs)
  variant_rows = tuple(binding.artifact.manifest["variants"][epoch] for epoch in epochs)
  for epoch, program, row in zip(epochs, programs, variant_rows):
    if program.key.hex() != binding.program_keys[epoch] or \
       row["epoch"] != epoch or row["program_key"] != binding.program_keys[epoch]:
      raise RuntimeError("frozen v2 ordinal sequence differs from its admitted family positions")
  target_identities = _frozen_program_set_ordinal_sequence_target_identities(binding, epochs)
  m, n, k = role_spec.shape
  words_np = _random_q4_words(n, k, 20260721)
  activation_np = np.random.default_rng(20260722).standard_normal(
    (m, k), dtype=np.float32).astype(np.float16)
  packed_weight = Tensor(words_np, dtype=dtypes.uint32, device="AMD")
  activation = Tensor(activation_np, dtype=dtypes.float16, device="AMD")
  tile = produce_physical_ds4_q8_1_tensor(
    activation.cast(dtypes.float32).contiguous(),
    PhysicalDS4Q8ActivationSpec(m, k))

  output_seed = activation.flatten()[:1].cast(dtypes.float32)
  zero = output_seed._apply_uop(lambda u: u.mul(0)).expand(m*n)
  zeroed_output = Tensor.empty(m*n, dtype=dtypes.float32, device="AMD")
  zeroed_output.assign(zero)
  fixed_inputs = (packed_weight, tile.values, tile.scales, tile.sums)
  output = zeroed_output
  for program in programs:
    output = output.custom_kernel(
      *fixed_inputs, fxn=lambda *_buffers, program=program: program)[0]

  family_calls = [node for node in output.uop.toposort()
                  if node.op is Ops.CALL and node.src[0] in binding.artifact.programs]
  if [call.src[0] for call in family_calls] != list(programs):
    raise RuntimeError("frozen v2 ordinal sequence did not retain its exact ordered PROGRAMs")
  arguments = [get_call_arg_uops(call) for call in family_calls]
  if len(arguments) != 2 or any(len(row) != 5 for row in arguments):
    raise RuntimeError("frozen v2 ordinal sequence lost its two-call five-buffer ABI")
  if arguments[0][0].buf_uop is not zeroed_output.uop.buf_uop:
    raise RuntimeError("frozen v2 ordinal sequence lost its explicitly zeroed output allocation")
  if any(arguments[0][slot].buf_uop is not row[slot].buf_uop
         for row in arguments for slot in range(5)):
    raise RuntimeError("frozen v2 ordinal sequence changed a buffer identity between calls")
  if family_calls[0] not in arguments[1][0].toposort():
    raise RuntimeError("frozen v2 ordinal sequence lost its slot-zero ordering chain")
  graph_evidence = {
    "program_calls": 2, "expected_program_calls": 2,
    "selected_epochs": list(epochs),
    "selected_program_keys": [binding.program_keys[epoch] for epoch in epochs],
    "selected_sink_keys": [row["sink_key"] for row in variant_rows],
    "selected_offsets": [dict(row["offsets"]) for row in variant_rows],
    "exact_ordered_ordinal_sequence": True, "five_buffer_abi": True,
    "all_calls_share_buffer_identity": True,
    "distinct_concrete_buffer_vas_deferred_to_dispatch_census": True,
    "slot0_ordered": True, "initial_output_zeroed": True,
    "buffer_uop_keys": [value.buf_uop.key.hex() for value in arguments[0]],
    "full_role_producer_calls": 1,
  }
  retained_outputs = _retained_producer_tensors([tile])
  graph_evidence.update({
    "retained_producer_tensor_count": len(retained_outputs),
    "retained_producer_tensor_names": ["values", "scales", "sums"],
    "retained_as_companion_realization_outputs": True,
  })

  dispatch_census_key = "aql_packet_census" if bool(getattr(Device["AMD"], "is_aql", False)) \
    else "pm4_dispatch_census"
  try:
    packet_census = _realize_with_amd_dispatch_census(
      output, target_program_identities=target_identities,
      target_launch_dims=tuple(
        (tuple(program.arg.global_size), tuple(program.arg.local_size)) for program in programs),
      require_all_five_vas_fixed=True, require_all_five_vas_distinct=True,
      target_program_keys=tuple(binding.program_keys[epoch] for epoch in epochs),
      retained_outputs=retained_outputs)
  except BaseException as exc:
    packet_census = _amd_dispatch_census_from_exception(exc)
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal sequence target dispatch failed",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False,
      "production_scheduler_used": False, "scheduler_prefix_semantics_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "epochs": list(epochs),
      "graph": graph_evidence, "target_program_identities": target_identities,
      **({"dispatch": {dispatch_census_key: packet_census}} if packet_census is not None else {}),
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  actual_values, actual_scales, actual_sums = \
    tile.values.numpy(), tile.scales.numpy(), tile.sums.numpy()
  q4_blocks = words_np.view(np.uint8).reshape(n, role_spec.epochs, 144)
  q4_selected, values_selected, scales_selected, sums_selected = \
    _fixed_base_ordinal_sequence_reference_operands(
      q4_blocks, actual_values, actual_scales, actual_sums, epochs)
  ref_spec = Q4KQ81MMQTileSpec(
    role="frozen_epoch_program_set_ordinal_sequence", m=m, n=n, k=512,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  actual_ds4 = Q81MMQDS4Activation(
    values_selected, scales_selected.astype(np.float16).astype(np.float32),
    sums_selected.astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=512, m_tile=m))
  consumer_reference = q4k_q8_1_mmq_ds4_tile_reference(
    q4_selected, actual_ds4, ref_spec)
  consumer_comparison = _numeric_comparison(
    output.numpy().reshape(m, n), consumer_reference)

  oracle_values, oracle_scales, oracle_sums = q8_1_mmq_ds4_quantize_reference(
    activation_np.astype(np.float32))
  producer_diagnostic = _producer_oracle_diagnostic(
    actual_values, actual_scales, actual_sums,
    oracle_values, oracle_scales, oracle_sums)
  passed = consumer_comparison["status"] == "pass"
  return {
    "schema": schema, "status": "PASS" if passed else "CONSUMER_MISMATCH",
    "exact_blocker": None if passed else
      "ordinal sequence output differs from the exact selected retained-byte reference",
    "research_only": True, "production_dispatch_changed": False,
    "production_scheduler_used": False, "scheduler_prefix_semantics_changed": False,
    "default_route": "direct_packed", "role": role_spec.role,
    "shape": list(role_spec.shape), "epochs": list(epochs),
    "producer": "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
    "graph": graph_evidence,
    "dispatch": {
      "launcher": "tinygrad_scheduler", "mode": "two_static_offset_program_ordinals",
      "count": 2, "epochs": list(epochs),
      "program_keys": [binding.program_keys[epoch] for epoch in epochs],
      "target_program_identities": target_identities,
      "all_five_vas_fixed": True, "all_five_vas_distinct": True,
      dispatch_census_key: packet_census,
    },
    "correctness": {
      "status": "PASS" if passed else "CONSUMER_MISMATCH",
      "comparison": consumer_comparison,
      "authority": "same_session_retained_full_role_producer_bytes_with_exact_two_ordinal_selection_and_fp16_metadata_roundtrip",
    },
    "producer_diagnostic": {
      **producer_diagnostic,
      "source_oracle": "extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
    },
    "family_identity": binding.family_identity,
    "frozen_bundle": str(Path(frozen_bundle).resolve()),
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def run_frozen_epoch_program_set_prefix_probe(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    prefix_epochs: int, preconstruct_runtimes: bool = False,
    diagnostic_global_grid: tuple[int, int, int] | list[int] | None = None) -> dict[str, Any]:
  """Run a frozen v2 fixed-base scheduler prefix with one full-role producer."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_prefix_probe.v2"
  if not isinstance(preconstruct_runtimes, bool):
    raise ValueError("preconstruct_runtimes must be a bool")
  role_spec = admit_exact_role_spec(role_spec)
  prefix_epochs = _validate_v2_fixed_base_prefix_epochs(role_spec, prefix_epochs)
  diagnostic_global_grid = _validate_attn_qo_diagnostic_global_grid(
    role_spec, diagnostic_global_grid)
  if diagnostic_global_grid is not None and not preconstruct_runtimes:
    raise ValueError("bounded diagnostic global grid requires exact runtime preconstruction")
  diagnostic_request = None if diagnostic_global_grid is None else {
    "schema": "tinygrad.mmq_attn_qo_bounded_global_grid.v1",
    "enabled": True, "research_only": True, "diagnostic_only": True,
    "production_promotion": False, "promotion_eligible": False,
    "c1_certification_claimed": False, "c1_certification_eligible": False,
    "requested_global_grid": list(diagnostic_global_grid),
    "allowed_global_grid_ladder": [
      list(grid) for grid in ATTN_QO_DIAGNOSTIC_GLOBAL_GRIDS],
    "phase": "requested_before_runtime_preconstruction_or_gpu_dispatch",
  }

  from types import SimpleNamespace
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Device
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.uop.ops import Ops
  from extra.qk.mmq_frozen_epoch_program_set import load_frozen_epoch_program_set_binding
  from extra.qk.mmq_q4k_q8_reference import (
    Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q4KQ81MMQTileSpec,
    Q8_1_MMQ_DS4_LAYOUT, q4k_q8_1_mmq_ds4_tile_reference,
    q8_1_mmq_ds4_quantize_reference,
  )
  from extra.qk.prefill.frozen_epoch_program_set_scheduler import (
    attach_frozen_epoch_program_set_schedule, prepare_frozen_epoch_program_set_schedule,
  )
  from extra.qk.q4k_q8_activation_producer import produce_physical_ds4_q8_1_tensor

  binding = load_frozen_epoch_program_set_binding(role_spec, frozen_bundle)
  target_identities = _frozen_program_set_target_identities(binding, prefix_epochs)
  m, n, k = role_spec.shape
  words_np = _random_q4_words(n, k, 20260721)
  activation_np = np.random.default_rng(20260722).standard_normal(
    (m, k), dtype=np.float32).astype(np.float16)
  packed_weight = Tensor(words_np, dtype=dtypes.uint32, device="AMD")
  activation = Tensor(activation_np, dtype=dtypes.float16, device="AMD")
  linear = SimpleNamespace(
    bias=None, out_features=n, in_features=k, q4k_storage=object(),
    prefill_packed_weight=lambda: packed_weight)
  produced_tiles = []

  def capture_producer(source: Any, spec: Any) -> Any:
    tile = produce_physical_ds4_q8_1_tensor(source, spec)
    produced_tiles.append(tile)
    return tile

  preparation = prepare_frozen_epoch_program_set_schedule(
    linear, activation, role_spec=role_spec, frozen_bundle=frozen_bundle,
    enabled=True, prefix_epochs=prefix_epochs, binding=binding,
    activation_producer=capture_producer)
  if preparation is None:
    raise RuntimeError("frozen v2 fixed-base scheduler preparation unexpectedly remained disabled")
  if len(produced_tiles) != 1:
    raise RuntimeError("frozen v2 fixed-base scheduler did not expose exactly one full-role producer tile")

  selected_programs = binding.artifact.programs[:prefix_epochs]
  if any(node in selected_programs
         for tensor in preparation.operands for node in tensor.uop.toposort()):
    raise RuntimeError("frozen v2 preparation attached a selected target PROGRAM too early")
  graph_evidence = {
    "program_calls": 0, "expected_program_calls": prefix_epochs,
    "program_keys": list(binding.program_keys[:prefix_epochs]),
    "ordered_program_prefix": False, "five_buffer_abi": True,
    "all_calls_share_buffer_identity": False, "slot0_ordered": False,
    "target_programs_attached_during_preparation": False,
    "attachment_stage": "awaiting_synchronized_preparation",
    "full_role_producer_calls": len(produced_tiles),
  }
  retained_outputs = _retained_producer_tensors(produced_tiles)
  graph_evidence.update({
    "retained_producer_tensor_count": len(retained_outputs),
    "retained_producer_tensor_names": ["values", "scales", "sums"],
    "retained_as_companion_realization_outputs": False,
    "producer_and_output_initialization_phase_separated": True,
  })

  runtime_preconstruction: dict[str, Any] = {
    "enabled": False, "status": "NOT_REQUESTED", "count": 0,
  }
  if preconstruct_runtimes:
    try:
      runtime_preconstruction = _preconstruct_frozen_program_runtimes(
        tuple(selected_programs), tuple(binding.program_keys[:prefix_epochs]),
        target_identities)
    except BaseException as exc:
      partial = getattr(exc, "runtime_preconstruction", None)
      return {
        "schema": schema, "status": "BLOCKED",
        "exact_blocker": "frozen v2 exact runtime preconstruction failed before realization",
        "exception": type(exc).__name__, "error": str(exc),
        "research_only": True, "production_dispatch_changed": False,
        "default_route": "direct_packed", "role": role_spec.role,
        "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
        "graph": graph_evidence, "target_program_identities": target_identities,
        "runtime_preconstruction": dict(partial) if isinstance(partial, Mapping) else {
          "enabled": True, "status": "PRECONSTRUCTION_ERROR",
          "exception": type(exc).__name__, "error": str(exc),
        },
        **({"bounded_global_grid_diagnostic": diagnostic_request}
           if diagnostic_request is not None else {}),
        "family_identity": binding.family_identity,
        "frozen_bundle": str(Path(frozen_bundle).resolve()),
        "compile_performed": False, "requires_recompile": False,
        "hip_used": False, "no_fallback": True,
      }

  try:
    preparation_phase = _realize_and_synchronize_five_buffer_preparation(
      preparation.operands)
  except BaseException as exc:
    partial = _preparation_phase_from_exception(exc)
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "producer/output initialization failed before target dispatch",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
      "graph": graph_evidence, "target_program_identities": target_identities,
      "runtime_preconstruction": runtime_preconstruction,
      "preparation_phase": partial,
      **({"bounded_global_grid_diagnostic": diagnostic_request}
         if diagnostic_request is not None else {}),
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  try:
    schedule = attach_frozen_epoch_program_set_schedule(
      preparation, preparation_receipt=preparation_phase)
  except BaseException as exc:
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen target attachment rejected the synchronized preparation",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
      "graph": graph_evidence, "target_program_identities": target_identities,
      "runtime_preconstruction": runtime_preconstruction,
      "preparation_phase": preparation_phase,
      **({"bounded_global_grid_diagnostic": diagnostic_request}
         if diagnostic_request is not None else {}),
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  output = schedule.output
  calls = [node for node in output.uop.toposort()
           if node.op is Ops.CALL and node.src[0] in selected_programs]
  if [call.src[0] for call in calls] != list(selected_programs):
    raise RuntimeError("frozen v2 fixed-base graph lost its exact ordered PROGRAM prefix")
  arguments = [get_call_arg_uops(call) for call in calls]
  if any(len(row) != 5 for row in arguments):
    raise RuntimeError("frozen v2 fixed-base graph lost the five-buffer ABI")
  if any(arguments[0][slot].buf_uop is not row[slot].buf_uop
         for row in arguments for slot in range(5)):
    raise RuntimeError("frozen v2 fixed-base graph changed a buffer identity within the prefix")
  if any(previous not in current[0].toposort()
         for previous, current in zip(calls, arguments[1:])):
    raise RuntimeError("frozen v2 fixed-base graph lost its slot-zero ordering chain")
  attached_keys = [value.buf_uop.key.hex() for value in arguments[0]]
  prepared_keys = [row["buffer_uop_key"] for row in preparation_phase["allocations"]]
  if attached_keys != prepared_keys:
    raise RuntimeError("frozen v2 target attachment differs from synchronized preparation buffers")
  graph_evidence.update({
    "program_calls": len(calls), "ordered_program_prefix": True,
    "all_calls_share_buffer_identity": True, "slot0_ordered": True,
    "buffer_uop_keys": attached_keys,
    "prepared_buffer_uop_keys": prepared_keys,
    "attachment_matches_synchronized_preparation": True,
    "attachment_stage": "attached_after_synchronized_preparation",
  })
  bounded_grid_evidence = None
  if diagnostic_global_grid is not None:
    output, bounded_grid_evidence = _apply_diagnostic_global_grid_to_target_calls(
      output, selected_programs, diagnostic_global_grid)
    calls = [node for node in output.uop.toposort()
             if node.op is Ops.CALL and node.src[0] in selected_programs]
    graph_evidence.update({
      "bounded_global_grid_diagnostic": True,
      "call_launch_only_override": True,
      "program_nodes_replaced": False,
      "program_keys_preserved": bounded_grid_evidence["program_keys_preserved"],
      "binary_identities_preserved": bounded_grid_evidence["binary_identities_preserved"],
      "buffer_abi_preserved": bounded_grid_evidence["buffer_abi_preserved"],
      "local_sizes_preserved": bounded_grid_evidence["local_sizes_preserved"],
      "full_grid_correctness_claimed": False,
      "c1_certification_claimed": False,
      "production_promotion": False,
    })

  dispatch_census_key = "aql_packet_census" if bool(getattr(Device["AMD"], "is_aql", False)) \
    else "pm4_dispatch_census"
  target_launch_dims = tuple(
    (diagnostic_global_grid or tuple(program.arg.global_size), tuple(program.arg.local_size))
    for program in selected_programs)
  try:
    packet_census = _realize_with_amd_dispatch_census(
      output, target_program_identities=target_identities,
      target_program_keys=tuple(binding.program_keys[:prefix_epochs]),
      target_launch_dims=target_launch_dims,
      require_all_five_vas_fixed=True, require_all_five_vas_distinct=True,
      retained_outputs=())
  except BaseException as exc:
    packet_census, runtime_reuse_crosscheck = \
      _dispatch_error_runtime_reuse_evidence(runtime_preconstruction, exc)
    prepared_vas = [row["va"] for row in preparation_phase["allocations"]]
    observed_calls = packet_census.get("calls", []) if isinstance(packet_census, Mapping) else []
    preparation_dispatch_crosscheck = {
      "expected_vas": prepared_vas,
      "observed_call_count": len(observed_calls),
      "observed_calls_match_prepared_allocations":
        all(row.get("kernarg_qwords") == prepared_vas for row in observed_calls),
    }
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 fixed-base realization failed during or after audited target dispatch",
      "exception": type(exc).__name__, "error": str(exc),
      "research_only": True, "production_dispatch_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
      "graph": graph_evidence, "target_program_identities": target_identities,
      "runtime_preconstruction": runtime_preconstruction,
      "runtime_reuse_crosscheck": runtime_reuse_crosscheck,
      "preparation_phase": preparation_phase,
      "preparation_dispatch_crosscheck": preparation_dispatch_crosscheck,
      **({"bounded_global_grid_diagnostic": bounded_grid_evidence}
         if bounded_grid_evidence is not None else {}),
      **({"dispatch": {dispatch_census_key: packet_census}} if packet_census is not None else {}),
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  runtime_reuse_crosscheck = _crosscheck_preconstructed_dispatch_runtimes(
    runtime_preconstruction, packet_census)
  prepared_vas = [row["va"] for row in preparation_phase["allocations"]]
  observed_calls = packet_census.get("calls", [])
  preparation_dispatch_crosscheck = {
    "expected_vas": prepared_vas,
    "observed_call_count": len(observed_calls),
    "expected_call_count": prefix_epochs,
    "observed_calls_match_prepared_allocations":
      all(row.get("kernarg_qwords") == prepared_vas for row in observed_calls),
  }
  preparation_dispatch_crosscheck["all_checks_pass"] = bool(
    len(observed_calls) == prefix_epochs and
    preparation_dispatch_crosscheck["observed_calls_match_prepared_allocations"])
  if not preparation_dispatch_crosscheck["all_checks_pass"]:
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "target kernargs differ from synchronized five-buffer preparation allocations",
      "research_only": True, "production_dispatch_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
      "graph": graph_evidence, "target_program_identities": target_identities,
      "runtime_preconstruction": runtime_preconstruction,
      "runtime_reuse_crosscheck": runtime_reuse_crosscheck,
      "preparation_phase": preparation_phase,
      "preparation_dispatch_crosscheck": preparation_dispatch_crosscheck,
      **({"bounded_global_grid_diagnostic": bounded_grid_evidence}
         if bounded_grid_evidence is not None else {}),
      "dispatch": {dispatch_census_key: packet_census},
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }
  if not runtime_reuse_crosscheck["all_checks_pass"]:
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 dispatch did not reuse preconstructed runtimes in exact order",
      "research_only": True, "production_dispatch_changed": False,
      "default_route": "direct_packed", "role": role_spec.role,
      "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
      "graph": graph_evidence, "target_program_identities": target_identities,
      "runtime_preconstruction": runtime_preconstruction,
      "runtime_reuse_crosscheck": runtime_reuse_crosscheck,
      "preparation_phase": preparation_phase,
      "preparation_dispatch_crosscheck": preparation_dispatch_crosscheck,
      **({"bounded_global_grid_diagnostic": bounded_grid_evidence}
         if bounded_grid_evidence is not None else {}),
      "dispatch": {dispatch_census_key: packet_census},
      "family_identity": binding.family_identity,
      "frozen_bundle": str(Path(frozen_bundle).resolve()),
      "compile_performed": False, "requires_recompile": False,
      "hip_used": False, "no_fallback": True,
    }

  tile = produced_tiles[0]
  actual_values, actual_scales, actual_sums = \
    tile.values.numpy(), tile.scales.numpy(), tile.sums.numpy()
  q4_blocks = words_np.view(np.uint8).reshape(n, role_spec.epochs, 144)
  q4_prefix, values_prefix, scales_prefix, sums_prefix = _fixed_base_prefix_reference_operands(
    q4_blocks, actual_values, actual_scales, actual_sums, prefix_epochs)
  ref_spec = Q4KQ81MMQTileSpec(
    role="frozen_epoch_program_set_prefix", m=m, n=n, k=prefix_epochs*256,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  actual_ds4 = Q81MMQDS4Activation(
    values_prefix, scales_prefix.astype(np.float16).astype(np.float32),
    sums_prefix.astype(np.float16).astype(np.float32),
    Q81MMQDS4ActivationSpec(m=m, k=prefix_epochs*256, m_tile=m))
  consumer_reference = q4k_q8_1_mmq_ds4_tile_reference(q4_prefix, actual_ds4, ref_spec)
  got_output = output.numpy().reshape(m, n)
  if diagnostic_global_grid is None:
    consumer_comparison = _numeric_comparison(got_output, consumer_reference)
    untouched_comparison = None
  else:
    bounded_columns = diagnostic_global_grid[0] * 128
    consumer_comparison = _numeric_comparison(
      got_output[:, :bounded_columns], consumer_reference[:, :bounded_columns])
    untouched_comparison = _exact_zero_comparison(got_output[:, bounded_columns:])

  oracle_values, oracle_scales, oracle_sums = q8_1_mmq_ds4_quantize_reference(
    activation_np.astype(np.float32))
  producer_diagnostic = _producer_oracle_diagnostic(
    actual_values, actual_scales, actual_sums,
    oracle_values, oracle_scales, oracle_sums)
  passed = consumer_comparison["status"] == "pass" and (
    untouched_comparison is None or untouched_comparison["status"] == "pass")
  return {
    "schema": schema, "status": "PASS" if passed else "CONSUMER_MISMATCH",
    "exact_blocker": None if passed else
      "target output differs from reference built from retained full-role producer bytes",
    "research_only": True, "production_dispatch_changed": False,
    "default_route": "direct_packed", "role": role_spec.role,
    "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
    "producer": "extra.qk.q4k_q8_activation_producer.produce_physical_ds4_q8_1_tensor",
    "graph": graph_evidence, "runtime_preconstruction": runtime_preconstruction,
    **({"bounded_global_grid_diagnostic": {
      **bounded_grid_evidence,
      "phase": "audited_target_dispatch_completed",
      "effective_target_launch_dims": [
        [list(global_size), list(local_size)] for global_size, local_size in target_launch_dims],
      "full_grid_correctness_claimed": False,
      "preconstructed_runtime_reuse_attested": runtime_reuse_crosscheck["all_checks_pass"],
    }} if bounded_grid_evidence is not None else {}),
    "runtime_reuse_crosscheck": runtime_reuse_crosscheck,
    "preparation_phase": preparation_phase,
    "preparation_dispatch_crosscheck": preparation_dispatch_crosscheck,
    "dispatch": {
      "launcher": "tinygrad_scheduler", "mode": "static_offset_program_chain",
      "count": prefix_epochs, "program_keys": list(binding.program_keys[:prefix_epochs]),
      "target_program_identities": target_identities,
      dispatch_census_key: packet_census,
    },
    "correctness": {
      "status": "PASS" if passed else "CONSUMER_MISMATCH",
      "comparison": consumer_comparison,
      **({"untouched_output_comparison": untouched_comparison,
          "bounded_columns": diagnostic_global_grid[0] * 128,
          "full_grid_correctness_claimed": False,
          "authority": "diagnostic_launched_column_prefix_and_zero_untouched_suffix_only"}
         if diagnostic_global_grid is not None else {
           "authority": "same_session_retained_full_role_producer_bytes_with_static_offset_prefix_and_fp16_metadata_roundtrip"}),
    },
    "producer_diagnostic": {
      **producer_diagnostic,
      "source_oracle": "extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
    },
    "scheduler_evidence": schedule.evidence,
    "family_identity": binding.family_identity,
    "frozen_bundle": str(Path(frozen_bundle).resolve()),
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def run_frozen_scheduler_prefix_two_probe_isolated(*, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                                   frozen_bundle: str | Path,
                                                   address_mode: str,
                                                   change_slot: str = "all",
                                                   timeout_seconds: float = 180.0,
                                                   child_env_overrides: Mapping[str, str] | None = None
                                                   ) -> dict[str, Any]:
  """Run the scheduler prefix-two diagnostic in a fresh, health-guarded child."""
  try:
    role_spec = admit_exact_role_spec(role_spec)
    marker0, marker1 = tuple(object() for _ in range(4)), tuple(object() for _ in range(4))
    _scheduler_prefix_two_launches(address_mode, (marker0, marker1), change_slot)
    env_overrides = _validated_child_env_overrides(child_env_overrides)
  except (TypeError, ValueError) as exc:
    return {"schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1", "status": "BLOCKED",
            "exact_blocker": str(exc)}
  if timeout_seconds <= 0:
    return {"schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1", "status": "BLOCKED",
            "exact_blocker": "timeout_seconds must be positive"}
  from extra.qk.mmq_target_epoch_orchestrator import parse_kernel_faults, read_kernel_log_since, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(env_overrides or None))
  except BaseException: health_before = False
  if not health_before:
    return {"schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1", "status": "BLOCKED",
            "exact_blocker": "pre-run GPU health probe failed", "health_before": False,
            "child_env_overrides": env_overrides}
  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  child_env.update(env_overrides)
  role_expr = f"exact_role_spec({role_spec.role!r}, shape={role_spec.shape!r})"
  bundle_arg = repr(str(Path(frozen_bundle).resolve()))
  code = (
    "import json; from extra.qk.mmq_exact_role_spec import exact_role_spec; "
    "from extra.qk.mmq_llama_five_buffer_gpu_harness import run_frozen_scheduler_prefix_two_probe; "
    f"print(json.dumps(run_frozen_scheduler_prefix_two_probe(role_spec={role_expr}, "
    f"frozen_bundle={bundle_arg}, address_mode={address_mode!r}, change_slot={change_slot!r})), flush=True)")
  started = time.time()
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    proc = None
  try: kernel_faults = parse_kernel_faults(read_kernel_log_since(started))
  except BaseException as exc: kernel_faults = [f"kernel-log scan failed: {type(exc).__name__}: {exc}"]
  try: health_after = bool(spawned_tiny_health_probe(env_overrides or None))
  except BaseException: health_after = False
  if proc is None:
    return {"schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1", "status": "BLOCKED",
            "exact_blocker": "scheduler prefix-two child timed out", "timeout": True,
            "timeout_seconds": timeout_seconds, "health_before": health_before, "health_after": health_after,
            "kernel_faults": kernel_faults, "child_env_overrides": env_overrides}
  result = None
  for line in reversed(proc.stdout.strip().splitlines()):
    try: candidate = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(candidate, dict):
      result = candidate
      break
  if result is None:
    result = {"schema": "tinygrad.mmq_frozen_scheduler_prefix_two_probe.v1", "status": "BLOCKED",
              "exact_blocker": "scheduler prefix-two child returned no structured result",
              "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
  result.update({"health_before": health_before, "health_after": health_after,
                 "kernel_faults": kernel_faults, "child_env_overrides": env_overrides})
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
  elif proc.returncode != 0 and result.get("status") == "PASS":
    result.update({"status": "BLOCKED", "exact_blocker": "scheduler prefix-two child exited non-zero",
                   "returncode": proc.returncode})
  return result


def run_frozen_scheduler_producer_prefix_probe_isolated(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    epoch_limit: int, timeout_seconds: float = 180.0,
    child_env_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
  """Run the real-producer scheduler prefix in a fresh, health-guarded child."""
  schema = "tinygrad.mmq_frozen_scheduler_producer_prefix_probe.v1"
  try:
    role_spec = admit_exact_role_spec(role_spec)
    if not isinstance(epoch_limit, int) or isinstance(epoch_limit, bool) or epoch_limit not in (1, 2):
      raise ValueError("producer-backed scheduler prefix epoch_limit must be 1 or 2")
    env_overrides = _validated_child_env_overrides(child_env_overrides)
  except (TypeError, ValueError) as exc:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": str(exc)}
  if timeout_seconds <= 0:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  from extra.qk.mmq_target_epoch_orchestrator import parse_kernel_faults, read_kernel_log_since, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(env_overrides or None))
  except BaseException: health_before = False
  if not health_before:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": "pre-run GPU health probe failed",
            "health_before": False, "child_env_overrides": env_overrides}

  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  child_env.update(env_overrides)
  role_expr = f"exact_role_spec({role_spec.role!r}, shape={role_spec.shape!r})"
  bundle_arg = repr(str(Path(frozen_bundle).resolve()))
  code = (
    "import json; from extra.qk.mmq_exact_role_spec import exact_role_spec; "
    "from extra.qk.mmq_llama_five_buffer_gpu_harness import run_frozen_scheduler_producer_prefix_probe; "
    f"print(json.dumps(run_frozen_scheduler_producer_prefix_probe(role_spec={role_expr}, "
    f"frozen_bundle={bundle_arg}, epoch_limit={epoch_limit})), flush=True)")
  started = time.time()
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    proc = None
  try: kernel_faults = parse_kernel_faults(read_kernel_log_since(started))
  except BaseException as exc: kernel_faults = [f"kernel-log scan failed: {type(exc).__name__}: {exc}"]
  try: health_after = bool(spawned_tiny_health_probe(env_overrides or None))
  except BaseException: health_after = False
  if proc is None:
    return {"schema": schema, "status": "BLOCKED",
            "exact_blocker": "producer-backed scheduler prefix child timed out",
            "timeout": True, "timeout_seconds": timeout_seconds,
            "health_before": health_before, "health_after": health_after,
            "kernel_faults": kernel_faults, "child_env_overrides": env_overrides}
  result = None
  for line in reversed(proc.stdout.strip().splitlines()):
    try: candidate = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(candidate, dict):
      result = candidate
      break
  if result is None:
    result = {"schema": schema, "status": "BLOCKED",
              "exact_blocker": "producer-backed scheduler prefix child returned no structured result",
              "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
  result.update({"health_before": health_before, "health_after": health_after,
                 "kernel_faults": kernel_faults, "child_env_overrides": env_overrides})
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
  elif proc.returncode != 0 and result.get("status") == "PASS":
    result.update({"status": "BLOCKED", "exact_blocker": "producer-backed scheduler prefix child exited non-zero",
                   "returncode": proc.returncode})
  return result


def _run_frozen_epoch_program_set_prefix_probe_worker(
    role_spec: ExactRoleSpec, frozen_bundle: str, prefix_epochs: int,
    preconstruct_runtimes: bool, child_env_overrides: Mapping[str, str],
    diagnostic_global_grid: tuple[int, int, int] | None = None) -> dict[str, Any]:
  """Spawn-safe worker; apply the narrow queue env before device creation."""
  os.environ.update(dict(child_env_overrides))
  os.environ["DEV"] = "AMD"
  return run_frozen_epoch_program_set_prefix_probe(
    role_spec=role_spec, frozen_bundle=frozen_bundle, prefix_epochs=prefix_epochs,
    preconstruct_runtimes=preconstruct_runtimes,
    diagnostic_global_grid=diagnostic_global_grid)


def run_frozen_epoch_program_set_prefix_probe_isolated(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    prefix_epochs: int, preconstruct_runtimes: bool = False,
    diagnostic_global_grid: tuple[int, int, int] | list[int] | None = None,
    timeout_seconds: float = 180.0,
    child_env_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
  """Run the v2 fixed-base prefix in the existing health-guarded child flow."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_prefix_probe.v2"
  try:
    if not isinstance(preconstruct_runtimes, bool):
      raise ValueError("preconstruct_runtimes must be a bool")
    role_spec = admit_exact_role_spec(role_spec)
    prefix_epochs = _validate_v2_fixed_base_prefix_epochs(role_spec, prefix_epochs)
    diagnostic_global_grid = _validate_attn_qo_diagnostic_global_grid(
      role_spec, diagnostic_global_grid)
    if diagnostic_global_grid is not None and not preconstruct_runtimes:
      raise ValueError("bounded diagnostic global grid requires exact runtime preconstruction")
    env_overrides = _validated_child_env_overrides(child_env_overrides)
    env_overrides.setdefault("AMD_AQL", "1")
  except (TypeError, ValueError) as exc:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": str(exc)}
  if timeout_seconds <= 0:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_before = False
  if not health_before:
    return {
      "schema": schema, "status": "BLOCKED", "exact_blocker": "pre-run GPU health probe failed",
      "health_before": False, "child_env_overrides": env_overrides,
    }

  from tinygrad.runtime.process_isolated import run_isolated
  started = time.time()
  isolated = run_isolated(
    _run_frozen_epoch_program_set_prefix_probe_worker,
    args=(role_spec, str(Path(frozen_bundle).resolve()), prefix_epochs,
          preconstruct_runtimes, env_overrides, diagnostic_global_grid),
    timeout_seconds=timeout_seconds, start_method="spawn")
  kernel_faults, kernel_fault_evidence = collect_kernel_fault_evidence(started)
  try: health_after = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_after = False
  if isolated.timed_out:
    result = dict(isolated.result) if isinstance(isolated.result, dict) else {
      "schema": schema, "status": "BLOCKED",
    }
    result.update({
      "status": "BLOCKED",
      "exact_blocker": "frozen v2 fixed-base prefix child timed out",
      "timeout": True, "timeout_seconds": timeout_seconds,
      "health_before": health_before, "health_after": health_after,
      "kernel_faults": kernel_faults, "kernel_fault_evidence": kernel_fault_evidence,
      "child_env_overrides": env_overrides,
    })
    if isinstance(isolated.evidence, dict):
      result["isolated_failure_evidence"] = dict(isolated.evidence)
    return result
  if isolated.status == "passed" and isinstance(isolated.result, dict):
    result = dict(isolated.result)
  else:
    result = dict(isolated.result) if isinstance(isolated.result, dict) else {
      "schema": schema, "status": "BLOCKED",
    }
    result.update({
      "status": "BLOCKED",
      "exact_blocker": isolated.error or
        "frozen v2 fixed-base prefix child returned no structured result",
      "child_status": isolated.status,
      "stdout_tail": isolated.stdout[-2000:], "stderr_tail": isolated.stderr[-2000:],
    })
    if isinstance(isolated.evidence, dict):
      result["isolated_failure_evidence"] = dict(isolated.evidence)
      if "preparation_phase" in isolated.evidence:
        result["preparation_phase"] = isolated.evidence["preparation_phase"]
      if "runtime_preconstruction" in isolated.evidence:
        result["runtime_preconstruction"] = isolated.evidence["runtime_preconstruction"]
      census_key = "aql_packet_census" if "aql_packet_census" in isolated.evidence \
        else "pm4_dispatch_census" if "pm4_dispatch_census" in isolated.evidence else None
      if census_key is not None:
        result["dispatch"] = {census_key: isolated.evidence[census_key]}
  result.update({
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": kernel_faults, "kernel_fault_evidence": kernel_fault_evidence,
    "child_env_overrides": env_overrides,
  })
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
  return result


def run_frozen_epoch_program_set_ordinal_probe_isolated(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    epoch: int, timeout_seconds: float = 180.0,
    child_env_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
  """Run one v2 fixed-base ordinal in a fresh, health-guarded child."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_ordinal_probe.v2"
  try:
    role_spec = admit_exact_role_spec(role_spec)
    epoch = _validate_v2_fixed_base_ordinal(role_spec, epoch)
    env_overrides = _validated_child_env_overrides(child_env_overrides)
    env_overrides.setdefault("AMD_AQL", "1")
  except (TypeError, ValueError) as exc:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": str(exc)}
  if timeout_seconds <= 0:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  from extra.qk.mmq_target_epoch_orchestrator import parse_kernel_faults, read_kernel_log_since, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_before = False
  if not health_before:
    return {
      "schema": schema, "status": "BLOCKED", "exact_blocker": "pre-run GPU health probe failed",
      "health_before": False, "child_env_overrides": env_overrides,
    }

  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  child_env.update(env_overrides)
  role_expr = f"exact_role_spec({role_spec.role!r}, shape={role_spec.shape!r})"
  bundle_arg = repr(str(Path(frozen_bundle).resolve()))
  code = (
    "import json; from extra.qk.mmq_exact_role_spec import exact_role_spec; "
    "from extra.qk.mmq_llama_five_buffer_gpu_harness import run_frozen_epoch_program_set_ordinal_probe; "
    f"print(json.dumps(run_frozen_epoch_program_set_ordinal_probe(role_spec={role_expr}, "
    f"frozen_bundle={bundle_arg}, epoch={epoch})), flush=True)")
  started = time.time()
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    proc = None
  try: kernel_faults = parse_kernel_faults(read_kernel_log_since(started))
  except BaseException as exc: kernel_faults = [f"kernel-log scan failed: {type(exc).__name__}: {exc}"]
  try: health_after = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_after = False
  if proc is None:
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal child timed out",
      "timeout": True, "timeout_seconds": timeout_seconds,
      "health_before": health_before, "health_after": health_after,
      "kernel_faults": kernel_faults, "child_env_overrides": env_overrides,
    }
  result = None
  for line in reversed(proc.stdout.strip().splitlines()):
    try: candidate = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(candidate, dict):
      result = candidate
      break
  if result is None:
    result = {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal child returned no structured result",
      "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:],
      "stderr_tail": proc.stderr[-2000:],
    }
  result.update({
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": kernel_faults, "child_env_overrides": env_overrides,
  })
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
  elif proc.returncode != 0 and result.get("status") == "PASS":
    result.update({
      "status": "BLOCKED", "exact_blocker": "frozen v2 ordinal child exited non-zero",
      "returncode": proc.returncode,
    })
  return result


def run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
    *, role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC, frozen_bundle: str | Path,
    epochs: tuple[int, int] | list[int], timeout_seconds: float = 180.0,
    child_env_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
  """Run two selected v2 ordinals in a fresh, health-guarded child."""
  schema = "tinygrad.mmq_frozen_epoch_program_set_ordinal_sequence_probe.v2"
  try:
    role_spec = admit_exact_role_spec(role_spec)
    epochs = _validate_v2_fixed_base_ordinal_sequence(role_spec, epochs)
    env_overrides = _validated_child_env_overrides(child_env_overrides)
    env_overrides.setdefault("AMD_AQL", "1")
  except (TypeError, ValueError) as exc:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": str(exc)}
  if timeout_seconds <= 0:
    return {"schema": schema, "status": "BLOCKED", "exact_blocker": "timeout_seconds must be positive"}
  from extra.qk.mmq_target_epoch_orchestrator import parse_kernel_faults, read_kernel_log_since, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_before = False
  if not health_before:
    return {
      "schema": schema, "status": "BLOCKED", "exact_blocker": "pre-run GPU health probe failed",
      "health_before": False, "child_env_overrides": env_overrides,
    }

  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  child_env.update(env_overrides)
  role_expr = f"exact_role_spec({role_spec.role!r}, shape={role_spec.shape!r})"
  bundle_arg = repr(str(Path(frozen_bundle).resolve()))
  code = (
    "import json; from extra.qk.mmq_exact_role_spec import exact_role_spec; "
    "from extra.qk.mmq_llama_five_buffer_gpu_harness import "
    "run_frozen_epoch_program_set_ordinal_sequence_probe; "
    f"print(json.dumps(run_frozen_epoch_program_set_ordinal_sequence_probe(role_spec={role_expr}, "
    f"frozen_bundle={bundle_arg}, epochs={epochs!r})), flush=True)")
  started = time.time()
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    proc = None
  try: kernel_faults = parse_kernel_faults(read_kernel_log_since(started))
  except BaseException as exc: kernel_faults = [f"kernel-log scan failed: {type(exc).__name__}: {exc}"]
  try: health_after = bool(spawned_tiny_health_probe(env_overrides))
  except BaseException: health_after = False
  if proc is None:
    return {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal sequence child timed out",
      "timeout": True, "timeout_seconds": timeout_seconds,
      "health_before": health_before, "health_after": health_after,
      "kernel_faults": kernel_faults, "child_env_overrides": env_overrides,
    }
  result = None
  for line in reversed(proc.stdout.strip().splitlines()):
    try: candidate = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(candidate, dict):
      result = candidate
      break
  if result is None:
    result = {
      "schema": schema, "status": "BLOCKED",
      "exact_blocker": "frozen v2 ordinal sequence child returned no structured result",
      "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:],
      "stderr_tail": proc.stderr[-2000:],
    }
  result.update({
    "health_before": health_before, "health_after": health_after,
    "kernel_faults": kernel_faults, "child_env_overrides": env_overrides,
  })
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
  elif proc.returncode != 0 and result.get("status") == "PASS":
    result.update({
      "status": "BLOCKED", "exact_blocker": "frozen v2 ordinal sequence child exited non-zero",
      "returncode": proc.returncode,
    })
  return result


def run_full_grid_target_role_probe_isolated(*, timeout_seconds: float = 900.0,
                                              role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC,
                                              warmups: int = 0, rounds: int = 1,
                                              epoch_limit: int | None = None,
                                              n_chunk_tiles: int | None = None,
                                              epoch_start: int = 0,
                                              host_accumulate: bool = False,
                                              in_kernel_accumulate: bool = False,
                                              per_epoch_check: bool = False,
                                              persistent_buffers: bool = False,
                                              preloaded_epochs: bool = False,
                                              sync_each_epoch: bool = False,
                                              stable_metadata_staging: bool = False,
                                              stable_epoch_staging: bool = False,
                                              wait_each_dispatch: bool = True,
                                              frozen_bundle: str | Path | None = None,
                                              child_env_overrides: Mapping[str, str] | None = None) -> dict[str, Any]:
  try: role_spec = admit_exact_role_spec(role_spec)
  except (TypeError, ValueError) as exc:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "exact_blocker": f"exact role admission failed: {exc}"}
  role_identity = {"role": role_spec.role, "shape": list(role_spec.shape)}
  if timeout_seconds <= 0: return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1",
                                  "status": "BLOCKED", **role_identity,
                                  "exact_blocker": "timeout_seconds must be positive"}
  if in_kernel_accumulate and host_accumulate:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "in_kernel_accumulate and host_accumulate are mutually exclusive"}
  if in_kernel_accumulate and per_epoch_check:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "per_epoch_check is unsafe with in_kernel_accumulate because it performs intermediate readback"}
  if in_kernel_accumulate and not (persistent_buffers or preloaded_epochs):
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "in_kernel_accumulate requires persistent_buffers"}
  if stable_epoch_staging and not stable_metadata_staging:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "stable_epoch_staging requires stable_metadata_staging"}
  if not wait_each_dispatch and not (
      in_kernel_accumulate and persistent_buffers and preloaded_epochs and
      stable_metadata_staging and stable_epoch_staging and not per_epoch_check):
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "asynchronous epoch dispatch requires in-place accumulation, "
            "persistent preloaded buffers, all-input fixed-VA staging, and no intermediate readback"}
  if frozen_bundle is not None and not in_kernel_accumulate:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "frozen target bundle requires in_kernel_accumulate",
            "compile_performed": False, "requires_recompile": False}
  try: env_overrides = _validated_child_env_overrides(child_env_overrides)
  except ValueError as exc:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": str(exc)}
  health_overrides = dict(env_overrides)
  from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence, spawned_tiny_health_probe
  try: health_before = bool(spawned_tiny_health_probe(health_overrides or None))
  except BaseException: health_before = False
  mode_health_before = health_before
  health_mode = {"amd_aql_env": health_overrides.get("AMD_AQL"),
                 "before": mode_health_before, "after": None}
  if not health_before or not mode_health_before:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            **role_identity, "exact_blocker": "pre-run GPU health probe failed", "health_before": health_before,
            "mode_health_before": mode_health_before, "health_mode": health_mode,
            "child_env_overrides": env_overrides}
  child_env = dict(os.environ)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  child_env.update(env_overrides)
  frozen_arg = repr(str(Path(frozen_bundle).resolve())) if frozen_bundle is not None else "None"
  role_expr = f"exact_role_spec({role_spec.role!r}, shape={role_spec.shape!r})"
  code = ("import json; from extra.qk.mmq_exact_role_spec import exact_role_spec; "
          "from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_target_role_probe; "
          f"print(json.dumps(run_full_grid_target_role_probe(warmups={int(warmups)}, rounds={int(rounds)}, "
          f"role_spec={role_expr}, "
          f"epoch_limit={repr(epoch_limit)}, n_chunk_tiles={repr(n_chunk_tiles)}, epoch_start={int(epoch_start)}, "
          f"host_accumulate={bool(host_accumulate)}, in_kernel_accumulate={bool(in_kernel_accumulate)}, "
          f"per_epoch_check={bool(per_epoch_check)}, "
          f"persistent_buffers={bool(persistent_buffers)}, preloaded_epochs={bool(preloaded_epochs)}, "
          f"sync_each_epoch={bool(sync_each_epoch)}, stable_metadata_staging={bool(stable_metadata_staging)}, "
          f"stable_epoch_staging={bool(stable_epoch_staging)}, "
          f"wait_each_dispatch={bool(wait_each_dispatch)}, "
          f"frozen_bundle={frozen_arg})), flush=True)")
  started = time.time()
  try:
    proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    kernel_faults, kernel_fault_evidence = collect_kernel_fault_evidence(started)
    try: health_after = bool(spawned_tiny_health_probe(health_overrides or None))
    except BaseException: health_after = False
    mode_health_after = health_after
    health_mode["after"] = mode_health_after
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "exact_blocker": f"target-role compile/{epoch_limit if epoch_limit is not None else 'full'}-epoch dispatch timed out",
            "timeout_seconds": timeout_seconds, "timeout": True, "health_before": health_before,
            "mode_health_before": mode_health_before, "kernel_faults": kernel_faults,
            "kernel_fault_evidence": kernel_fault_evidence,
            "health_after": health_after, "mode_health_after": mode_health_after,
            "health_mode": health_mode, "child_env_overrides": env_overrides,
            "compile_performed": False if frozen_bundle is not None else None,
            "requires_recompile": False if frozen_bundle is not None else None, **role_identity}
  result = None
  for line in reversed(proc.stdout.strip().splitlines()):
    try: candidate = json.loads(line)
    except json.JSONDecodeError: continue
    if isinstance(candidate, dict):
      result = candidate
      break
  kernel_faults, kernel_fault_evidence = collect_kernel_fault_evidence(started)
  try: health_after = bool(spawned_tiny_health_probe(health_overrides or None))
  except BaseException: health_after = False
  mode_health_after = health_after
  health_mode["after"] = mode_health_after
  if result is None:
    return {"schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1", "status": "BLOCKED",
            "exact_blocker": "target-role child returned no structured result", "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:],
            "kernel_faults": kernel_faults, "kernel_fault_evidence": kernel_fault_evidence,
            "health_before": health_before, "health_after": health_after,
            "mode_health_before": mode_health_before, "mode_health_after": mode_health_after,
            "health_mode": health_mode, "child_env_overrides": env_overrides,
            "diagnostic": {"epoch_limit": epoch_limit, "n_chunk_tiles": n_chunk_tiles,
                           "epoch_start": epoch_start, "host_accumulate": host_accumulate,
                           "in_kernel_accumulate": in_kernel_accumulate,
                           "per_epoch_check": per_epoch_check, "persistent_buffers": persistent_buffers,
                           "preloaded_epochs": preloaded_epochs, "sync_each_epoch": sync_each_epoch,
                           "stable_metadata_staging": stable_metadata_staging,
                           "stable_epoch_staging": stable_epoch_staging,
                           "role": role_spec.role, "shape": list(role_spec.shape),
                           "frozen_bundle": str(Path(frozen_bundle).resolve()) if frozen_bundle is not None else None}}
  result.update({"kernel_faults": kernel_faults, "kernel_fault_evidence": kernel_fault_evidence,
                 "health_before": health_before, "health_after": health_after,
                 "mode_health_before": mode_health_before, "mode_health_after": mode_health_after,
                 "health_mode": health_mode, "child_env_overrides": env_overrides})
  if kernel_faults:
    result.update({"status": "BLOCKED", "exact_blocker": "AMD kernel fault/reset marker observed"})
  elif not health_after or not mode_health_after:
    result.update({"status": "BLOCKED", "exact_blocker": "post-run GPU health probe failed"})
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
  parser.add_argument("--target-role", action="store_true",
                      help="run the isolated exact Qwen3 target-role probe")
  parser.add_argument("--target-role-inventory", type=Path, default=DEFAULT_INVENTORY)
  parser.add_argument("--target-role-name", default=DEFAULT_EXACT_ROLE_SPEC.role)
  parser.add_argument("--target-role-epochs", type=int, default=None)
  parser.add_argument("--target-role-output", type=Path)
  parser.add_argument("--target-role-timeout", type=float, default=900.0)
  parser.add_argument("--target-role-per-epoch-check", action="store_true")
  parser.add_argument("--target-role-persistent", action="store_true")
  parser.add_argument("--target-role-preloaded", action="store_true")
  parser.add_argument("--target-role-stable-metadata", action="store_true")
  parser.add_argument("--target-role-stable-epochs", action="store_true")
  parser.add_argument("--target-role-no-wait-each-dispatch", action="store_true")
  parser.add_argument("--target-role-host-accumulate", action="store_true")
  parser.add_argument("--target-role-in-kernel-accumulate", action="store_true")
  parser.add_argument("--target-role-frozen-bundle", type=Path)
  parser.add_argument("--target-role-amd-aql", choices=("0", "1"))
  parser.add_argument("--scheduler-prefix-two", choices=("same", "changed"),
                      help="run two scheduler-owned frozen PROGRAM calls with same or changed input VAs")
  parser.add_argument("--scheduler-prefix-two-change-slot", choices=_SCHEDULER_PREFIX_CHANGE_SLOTS, default="all",
                      help="with changed mode, change only this input ABI slot (default: all)")
  parser.add_argument("--scheduler-producer-prefix-epochs", type=int, choices=(1, 2),
                      help="run a 1/2-epoch frozen scheduler prefix with the real physical Q8 producer")
  parser.add_argument("--scheduler-v2-fixed-base-prefix-epochs", type=int,
                      help="run a 1/2/3-epoch or admitted-full-role frozen v2 static-offset prefix")
  parser.add_argument("--scheduler-v2-fixed-base-preconstruct-runtimes", action="store_true",
                      help="preconstruct/cache the selected exact runtimes before the v2 prefix realization")
  parser.add_argument("--scheduler-v2-fixed-base-diagnostic-global-grid", type=int, nargs=3,
                      metavar=("X", "Y", "Z"),
                      help="research-only attn_qo CALL launch grid; never C1/promotion evidence")
  parser.add_argument("--scheduler-v2-fixed-base-ordinal", type=int,
                      help="run one research-only frozen v2 static-offset ordinal")
  parser.add_argument("--scheduler-v2-fixed-base-ordinal-sequence", type=int, nargs=2,
                      metavar=("FIRST", "SECOND"),
                      help="run exactly two increasing research-only frozen v2 static-offset ordinals")
  args = parser.parse_args()
  if args.scheduler_v2_fixed_base_preconstruct_runtimes and \
     args.scheduler_v2_fixed_base_prefix_epochs is None:
    parser.error("--scheduler-v2-fixed-base-preconstruct-runtimes requires "
                 "--scheduler-v2-fixed-base-prefix-epochs")
  if args.scheduler_v2_fixed_base_diagnostic_global_grid is not None and \
     args.scheduler_v2_fixed_base_prefix_epochs is None:
    parser.error("--scheduler-v2-fixed-base-diagnostic-global-grid requires "
                 "--scheduler-v2-fixed-base-prefix-epochs")
  if args.scheduler_v2_fixed_base_diagnostic_global_grid is not None and \
     not args.scheduler_v2_fixed_base_preconstruct_runtimes:
    parser.error("--scheduler-v2-fixed-base-diagnostic-global-grid requires "
                 "--scheduler-v2-fixed-base-preconstruct-runtimes")
  if args.scheduler_v2_fixed_base_ordinal_sequence is not None:
    if args.target_role_frozen_bundle is None:
      parser.error("--scheduler-v2-fixed-base-ordinal-sequence requires --target-role-frozen-bundle")
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_frozen_epoch_program_set_ordinal_sequence_probe_isolated(
      role_spec=role_spec, frozen_bundle=args.target_role_frozen_bundle,
      epochs=args.scheduler_v2_fixed_base_ordinal_sequence,
      timeout_seconds=args.target_role_timeout,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql}
        if args.target_role_amd_aql is not None else None)
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
  if args.scheduler_v2_fixed_base_ordinal is not None:
    if args.target_role_frozen_bundle is None:
      parser.error("--scheduler-v2-fixed-base-ordinal requires --target-role-frozen-bundle")
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_frozen_epoch_program_set_ordinal_probe_isolated(
      role_spec=role_spec, frozen_bundle=args.target_role_frozen_bundle,
      epoch=args.scheduler_v2_fixed_base_ordinal,
      timeout_seconds=args.target_role_timeout,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql}
        if args.target_role_amd_aql is not None else None)
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
  if args.scheduler_v2_fixed_base_prefix_epochs is not None:
    if args.target_role_frozen_bundle is None:
      parser.error("--scheduler-v2-fixed-base-prefix-epochs requires --target-role-frozen-bundle")
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_frozen_epoch_program_set_prefix_probe_isolated(
      role_spec=role_spec, frozen_bundle=args.target_role_frozen_bundle,
      prefix_epochs=args.scheduler_v2_fixed_base_prefix_epochs,
      preconstruct_runtimes=args.scheduler_v2_fixed_base_preconstruct_runtimes,
      diagnostic_global_grid=args.scheduler_v2_fixed_base_diagnostic_global_grid,
      timeout_seconds=args.target_role_timeout,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql}
        if args.target_role_amd_aql is not None else None)
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
  if args.scheduler_producer_prefix_epochs is not None:
    if args.target_role_frozen_bundle is None:
      parser.error("--scheduler-producer-prefix-epochs requires --target-role-frozen-bundle")
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_frozen_scheduler_producer_prefix_probe_isolated(
      role_spec=role_spec, frozen_bundle=args.target_role_frozen_bundle,
      epoch_limit=args.scheduler_producer_prefix_epochs, timeout_seconds=args.target_role_timeout,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql} if args.target_role_amd_aql is not None else None)
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
  if args.scheduler_prefix_two:
    if args.target_role_frozen_bundle is None:
      parser.error("--scheduler-prefix-two requires --target-role-frozen-bundle")
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_frozen_scheduler_prefix_two_probe_isolated(
      role_spec=role_spec, frozen_bundle=args.target_role_frozen_bundle,
      address_mode=args.scheduler_prefix_two, change_slot=args.scheduler_prefix_two_change_slot,
      timeout_seconds=args.target_role_timeout,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql} if args.target_role_amd_aql is not None else None)
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
  if args.target_role:
    role_spec = exact_role_spec(args.target_role_name, inventory=args.target_role_inventory)
    row = run_full_grid_target_role_probe_isolated(
      role_spec=role_spec,
      timeout_seconds=args.target_role_timeout, warmups=0, rounds=1,
      epoch_limit=args.target_role_epochs, n_chunk_tiles=role_spec.program.grid[0],
      host_accumulate=args.target_role_host_accumulate,
      in_kernel_accumulate=args.target_role_in_kernel_accumulate,
      per_epoch_check=args.target_role_per_epoch_check,
      persistent_buffers=args.target_role_persistent, preloaded_epochs=args.target_role_preloaded,
      stable_metadata_staging=args.target_role_stable_metadata,
      stable_epoch_staging=args.target_role_stable_epochs,
      wait_each_dispatch=not args.target_role_no_wait_each_dispatch,
      frozen_bundle=args.target_role_frozen_bundle,
      child_env_overrides={"AMD_AQL": args.target_role_amd_aql} if args.target_role_amd_aql is not None else None,
    )
    encoded = json.dumps(row, indent=2, sort_keys=True)
    if args.target_role_output is not None:
      args.target_role_output.parent.mkdir(parents=True, exist_ok=True)
      args.target_role_output.write_text(encoded + "\n")
    print(encoded)
    return 0 if row.get("status") == "PASS" else 1
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
