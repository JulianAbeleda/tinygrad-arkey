#!/usr/bin/env python3
"""Executable truth for the exact BoltBeam single-buffer prefill candidate.

This is a diagnostic authority around the existing graph-GEMM generated
transport.  It neither defines a route nor duplicates a kernel builder.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, pathlib, re, subprocess
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any

import numpy as np

from extra.qk.prefill.anchor_isa_resource_capture import _program_surface
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-single-buffer-execution-authority.v1"
SELECTED_SURFACE = "route_pf16_graph_gemm.generated_lds_matmul_transport"
M, N, K = 512, 12288, 4096

@dataclass
class PreparedCandidateExecution:
  compiled: UOp
  program: UOp
  call: UOp
  output: Any
  reference: np.ndarray
  identity: str
  def kernel_call(self, *, wait:bool=True) -> float:
    from tinygrad.engine.realize import ExecContext, exec_kernel
    elapsed = exec_kernel(ExecContext(jit=True, wait=wait, update_stats=False), self.call, self.program)
    if elapsed is None: raise RuntimeError("prepared candidate kernel returned no device timing")
    return float(elapsed)

def _prepared_candidate(compiled:UOp, program:UOp, output:Any, reference:np.ndarray, identity:str) -> PreparedCandidateExecution:
  calls = [u for u in compiled.toposort() if u.op is Ops.CALL and u.src and u.src[0] is program]
  if len(calls) != 1: raise RuntimeError(f"expected one exact candidate CALL, found {len(calls)}")
  return PreparedCandidateExecution(compiled, program, calls[0], output, reference, identity)


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _git_state() -> dict[str, Any]:
  try:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
    return {"revision": revision, "dirty": dirty}
  except Exception: return {"revision": None, "dirty": True}


@contextmanager
def _temporary_env(values: dict[str, str]):
  old = {key: os.environ.get(key) for key in values}
  os.environ.update(values)
  try: yield
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value


def _program_identity(program: UOp) -> dict[str, Any]:
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE), None)
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY), None)
  if not isinstance(source, str) or not isinstance(binary, bytes):
    raise RuntimeError("candidate PROGRAM has no compiler source/binary")
  return {"program_key": program.key.hex(), "source_sha256": _sha256(source.encode()),
          "binary_sha256": _sha256(binary), "binary_bytes": len(binary)}


def _compiler_lds_truth(program: UOp) -> dict[str, Any]:
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE), "")
  lowered = program.src[0].toposort()
  local_defs = [u for u in lowered if u.op is Ops.DEFINE_LOCAL]
  # AMD:ISA compiler rendering uses the AMDOps names below.  Keep the source
  # check broad enough for the HIP renderer as well, without accepting a local
  # allocation alone as proof that transport actually happened.
  lower_source = source.lower()
  shared = re.findall(r"__attribute__\(\(shared.*?\)\)(half|float|int|short|char)\s+\w+\[(\d+)\]", lower_source)
  itemsize = {"half": 2, "short": 2, "float": 4, "int": 4, "char": 1}
  rendered_lds_bytes = sum(itemsize[kind] * int(count) for kind, count in shared)
  shared_accesses = len(re.findall(r"\(\*\(buf\d+\+", lower_source))
  shared_stores = len(re.findall(r"\(\*\(buf\d+\+[^)]*\)\)\s*=", lower_source))
  shared_loads = shared_accesses - shared_stores
  local_stores = [u for u in lowered if u.op is Ops.STORE and any(x.op is Ops.DEFINE_LOCAL for x in u.backward_slice)]
  local_loads = [u for u in lowered if u.op is Ops.LOAD and any(x.op is Ops.DEFINE_LOCAL for x in u.backward_slice)]
  stores = sum(lower_source.count(x) for x in ("ds_store", "ds_write")) or len(local_stores) or shared_stores
  loads = sum(lower_source.count(x) for x in ("ds_load", "ds_read")) or len(local_loads) or shared_loads
  barriers = lower_source.count("s_barrier") + lower_source.count("__syncthreads")
  lds_bytes = max(max((getattr(u.dtype, "size", 0) for u in local_defs), default=0), rendered_lds_bytes)
  return {"define_local_count": len(local_defs), "shared_declaration_count": len(shared), "lds_bytes": lds_bytes,
          "lds_store_markers": stores, "lds_load_markers": loads, "barrier_markers": barriers,
          "actual_compiler_lds_staging": bool((local_defs or shared) and stores and loads and barriers)}


def _candidate_programs(compiled_linear: UOp, identity: str) -> list[UOp]:
  found = []
  for u in compiled_linear.toposort():
    if u.op is not Ops.PROGRAM: continue
    context = getattr(u.src[0].arg, "candidate_context", None)
    if context is not None and context.canonical_identity == identity: found.append(u)
  return list(dict.fromkeys(found))


def _expected_structure(payload: dict[str, Any]) -> dict[str, Any]:
  schedule = payload["schedule"]
  windows = schedule["lds"]["windows"]
  return {"threads": schedule["threads"], "tile": schedule["tile"], "waves": schedule["waves"],
          "wave_threads": schedule["waves"]["m"] * schedule["waves"]["n"] * payload["workload"]["target"]["wave_size"],
          "lds_windows": windows, "lds_strides": schedule["lds"]["strides"],
          "lds_bytes": max(end for _start, end in windows.values())}


def _emitted_tile_lds_proof(program: UOp) -> dict[str, Any]:
  """Recompute the exact candidate structure from optimized emitted UOps."""
  sink = program.src[0]
  buffers = [u for u in sink.toposort() if u.op is Ops.BUFFER and isinstance(u.tag, tuple) and u.tag[0] == "kernel_tile_lds"]
  errors = []
  if len(buffers) != 1: return {"passed": False, "errors": [f"expected one tagged tile LDS buffer, found {len(buffers)}"]}
  buf = buffers[0]
  if not buf.src or buf.src[0].arg * buf.dtype.itemsize != 20480: errors.append("tagged tile LDS allocation is not 20480 bytes")
  def _address(u):
    address = u.src[0]
    if address.op is Ops.INDEX: return address.src[0], address.src[1], 1
    if address.op is Ops.SHRINK and len(address.src) == 3 and address.src[2].op is Ops.CONST:
      return address.src[0], address.src[1], address.src[2].arg
    return None
  stores = [u for u in sink.toposort() if u.op is Ops.STORE and u.src[0].op is Ops.INDEX and u.src[0].src[0] is buf]
  loads = [u for u in sink.toposort() if u.op is Ops.LOAD and (address := _address(u)) is not None and
           buf in address[0].backward_slice_with_self]
  if len(stores) != 32: errors.append(f"expected 32 scalar cooperative store formulas, found {len(stores)}")
  if len(loads) != 48: errors.append(f"expected 48 packed fragment load formulas, found {len(loads)}")
  allowed = re.compile(r"^[0-9lidx ()+*<>&|/%-]+$")
  def _addresses(rows):
    addresses = [address for u in rows if (address := _address(u)) is not None]
    expressions = [(address[1].render(), address[2]) for address in addresses]
    if any(allowed.fullmatch(expr) is None for expr,_width in expressions): raise ValueError("emitted LDS index is not restricted affine integer syntax")
    return {int(eval(compile(expr, "<candidate-lds-index>", "eval"), {"__builtins__": {}},  # pylint: disable=eval-used
                     {"lidx0": lane, "lidx1": wave_m, "lidx2": wave_n}))+element
            for expr,width in expressions for lane in range(32) for wave_m in range(4) for wave_n in range(2) for element in range(width)}
  expected_a = {row*40+k for row in range(128) for k in range(32)}
  expected_b = {5120+row*40+k for row in range(128) for k in range(32)}
  try:
    store_addresses, load_addresses = _addresses(stores), _addresses(loads)
    if store_addresses != expected_a | expected_b: errors.append("cooperative stores do not exactly cover stride-80 A/B data intervals")
    if load_addresses != expected_a | expected_b: errors.append("fragment loads do not exactly consume staged A/B data intervals")
  except ValueError as exc: errors.append(str(exc))
  barriers = [u for u in sink.toposort() if u.op is Ops.BARRIER]
  if len(barriers) != 1: errors.append(f"expected one shared barrier, found {len(barriers)}")
  elif any(barriers[0] not in u.backward_slice for u in loads): errors.append("a fragment load is not ordered after the shared barrier")
  wmmas = [u for u in sink.toposort() if u.op is Ops.WMMA]
  if len(wmmas) != 16: errors.append(f"expected 16 emitted WMMA atoms, found {len(wmmas)}")
  for wmma in wmmas:
    if wmma.arg[1:6] != ((16, 16, 16), dtypes.half, dtypes.float, "AMD", 32):
      errors.append("emitted WMMA descriptor differs from validated RDNA3 atom"); break
    if any(not any(load in src.backward_slice for load in loads) for src in wmma.src[:2]):
      errors.append("emitted WMMA operand is not bound to proven LDS fragment loads"); break
  return {"passed": not errors, "errors": errors, "allocation_count": len(buffers), "store_formula_count": len(stores),
          "load_formula_count": len(loads), "wmma_count": len(wmmas), "tile": {"m": 128, "n": 128, "k": 32},
          "waves": {"m": 4, "n": 2}, "lds_windows": {"a": [0, 10240], "b": [10240, 20480]},
          "lds_strides": {"a": 80, "b": 80}, "producer_data_elements": 8192, "consumer_data_elements": 8192}


def _structural_binding(payload: dict[str, Any], program: UOp, lds: dict[str, Any]) -> dict[str, Any]:
  expected = _expected_structure(payload)
  emitted_proof = _emitted_tile_lds_proof(program)
  local_size = tuple(program.arg.local_size or ())
  actual_threads = math.prod(local_size) if local_size else None
  wave_size = payload["workload"]["target"]["wave_size"]
  wave_count = actual_threads // wave_size if actual_threads is not None and actual_threads % wave_size == 0 else None
  emitted_waves = ({"m": local_size[1], "n": local_size[2]}
                   if local_size == (wave_size, expected["waves"]["m"], expected["waves"]["n"]) else None)
  actual = {"threads": actual_threads, "local_size": list(local_size), "lds_bytes": lds["lds_bytes"],
            # Rendered source does not retain enough semantic metadata to prove
            # candidate windows, strides, or wave ownership. Unknown is a hard
            # failure, not permission to infer them from the attached context.
            "wave_count": wave_count, "tile": emitted_proof.get("tile") if emitted_proof["passed"] else None,
            "waves": emitted_proof.get("waves") if emitted_proof["passed"] else emitted_waves,
            "lds_windows": emitted_proof.get("lds_windows") if emitted_proof["passed"] else None,
            "lds_strides": emitted_proof.get("lds_strides") if emitted_proof["passed"] else None}
  evidence = {
    "threads": "PROGRAM launch local_size", "lds_bytes": "lowered DEFINE_LOCAL/rendered shared declaration",
    "wave_count": "launch threads divided by target wave size" if wave_count is not None else None,
    "tile": "optimized UOp affine producer/consumer proof" if emitted_proof["passed"] else None,
    "waves": "optimized UOp proof + ordered PROGRAM local axes" if emitted_proof["passed"] else
             "ordered PROGRAM local axes (wave_size, waves_m, waves_n)" if emitted_waves is not None else None,
    "lds_windows": "optimized UOp exhaustive address-set proof" if emitted_proof["passed"] else None,
    "lds_strides": "optimized UOp exhaustive address-set proof" if emitted_proof["passed"] else None,
  }
  errors = []
  if actual_threads != expected["threads"]: errors.append(f"threads: expected {expected['threads']}, emitted {actual_threads}")
  if expected["wave_threads"] != expected["threads"]: errors.append("payload waves do not account for payload threads")
  if actual["lds_bytes"] != expected["lds_bytes"]:
    errors.append(f"LDS bytes: expected {expected['lds_bytes']}, emitted {actual['lds_bytes']}")
  expected_wave_count = expected["waves"]["m"] * expected["waves"]["n"]
  if wave_count != expected_wave_count: errors.append(f"wave count: expected {expected_wave_count}, emitted {wave_count}")
  for field in ("tile", "waves", "lds_windows", "lds_strides"):
    if actual[field] is None: errors.append(f"{field}: emitted structure is unproven")
    elif actual[field] != expected[field]: errors.append(f"{field}: emitted structure differs from payload")
  return {"expected": expected, "actual": actual, "evidence": evidence, "emitted_proof": emitted_proof,
          "matches_payload": not errors, "pre_gpu_eligible": not errors, "errors": errors}


def _require_pre_gpu_structure(binding: dict[str, Any]) -> None:
  if not binding["pre_gpu_eligible"]:
    raise RuntimeError("candidate emitted-program structure is not proven; refusing GPU execution: " + "; ".join(binding["errors"]))


def _case_arrays(case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  # References are independently defined and cheap despite covering every
  # output element of the exact anchor.
  if case == "constant":
    av, bv = np.float16(0.03125), np.float16(-0.0625)
    a, b = np.full((M, K), av, np.float16), np.full((N, K), bv, np.float16)
    ref = np.full((M, N), np.float32(K) * np.float32(av) * np.float32(bv), np.float32)
  elif case == "row_col":
    av = ((np.arange(M) % 7) - 3).astype(np.float16)[:, None] * np.float16(1/64)
    bv = ((np.arange(N) % 11) - 5).astype(np.float16)[:, None] * np.float16(1/64)
    a, b = np.broadcast_to(av, (M, K)).copy(), np.broadcast_to(bv, (N, K)).copy()
    ref = np.float32(K) * av.astype(np.float32) @ bv.astype(np.float32).T
  else: raise ValueError(f"unknown case {case!r}")
  return a, b, ref


def run(payload: dict[str, Any], candidate_hash: str, *, case: str = "constant",
        atol: float = 0.125, rtol: float = 0.002,
        prepared_out:list[PreparedCandidateExecution]|None=None) -> dict[str, Any]:
  """Compile and execute the exact candidate through its production route."""
  identity = canonical_candidate_hash(payload)
  if candidate_hash != identity: raise ValueError("candidate hash does not match exact payload")
  if (payload["workload"]["shape"] != {"m": M, "n": N, "k": K} or
      payload["workload"]["role"] != "ffn_gate_up"):
    raise ValueError("execution authority only accepts the exact ffn_gate_up anchor")

  from tinygrad import Device, Tensor
  from tinygrad.codegen import to_program_cache
  from tinygrad.codegen.opt import postrange
  from tinygrad.engine.realize import compile_linear, run_linear, runtime_cache
  from tinygrad.helpers import getenv
  from extra.qk.prefill_graph_gemm_route import route_pf16_graph_gemm
  from extra.qk.pure_search_guard import effective_routes

  class Linear:
    bias = None
    _prefill_graph_role = "ffn_gate_up"

  a_np, b_np, ref = _case_arrays(case)
  env = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON": json.dumps(payload, separators=(",", ":")),
         "BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH": identity,
         "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1",
         "PREFILL_WMMA_PIPE_PRIMITIVE": "0",
         "PREFILL_DBUF": "0", "PURE_MACHINE_SEARCH_ALLOW_ROLLBACK": "0"}
  old_opts, old_contexts = postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS
  old_local = getattr(postrange, "_WARMSTART_LOCAL_STAGE_KEYS", None)
  try:
    with _temporary_env(env):
      getenv.cache_clear(); to_program_cache.clear()
      x, w = Tensor(a_np), Tensor(b_np)
      out = route_pf16_graph_gemm(Linear(), x, w=w)
      if out is None: raise RuntimeError("exact route declined the anchor")
      linear = out.schedule_linear()
      compiled = compile_linear(linear)
      programs = _candidate_programs(compiled, identity)
      if len(programs) != 1:
        raise RuntimeError(f"expected one exact candidate PROGRAM, found {len(programs)}")
      program = programs[0]
      context = program.src[0].arg.candidate_context
      surface, lds, hashes = _program_surface(program), _compiler_lds_truth(program), _program_identity(program)
      if dump := os.environ.get("BOLTBEAM_AUTHORITY_SOURCE_DUMP"):
        pathlib.Path(dump).write_text(next(u.arg for u in program.src if u.op is Ops.SOURCE))
      if not surface["strict_pure"]: raise RuntimeError(f"candidate selected forbidden executable surface: {surface}")
      if not lds["actual_compiler_lds_staging"]: raise RuntimeError(f"candidate did not emit compiler LDS staging: {lds}")
      structural = _structural_binding(payload, program, lds)
      effective = next(row for row in effective_routes(os.environ) if row["family"] == "prefill_gemm")
      if not structural["pre_gpu_eligible"]:
        route_id = effective["effective_route"]
        return {"schema": SCHEMA, "canonical_identity": identity, "route_id": route_id,
                "selected_route_id": route_id, "environment": {"device": str(Device.DEFAULT), "git": _git_state()},
                "route_binding_complete": False, "route_authority": effective,
                "structural_binding": structural, "binding_errors": list(structural["errors"]), "program": hashes,
                "runtime": {"status": "not_run", "executed_binary_sha256": None, "binary_equal": None},
                "surface": surface, "executable_truth": {
                  "selected_route_id": route_id, "selected_surface": SELECTED_SURFACE,
                  "manifest_route_backed": True, "fallback_used": False, "strict_pure": False,
                  "compiler_surface_forbidden_markers_absent": surface["strict_pure"],
                  "ops_ins_absent": surface["ops_ins_count"] == 0,
                  "asm_source_absent": surface["source_kind"] != "native_isa",
                  "candidate_context_equal": (context.schema_version == payload["schema_version"] and
                                                context.canonical_identity == identity),
                  "runtime_binary_matches_candidate": None, **lds},
                "correctness": {"status": "not_run", "reason": "emitted_program_structure_unproven",
                                "case": case, "comparison": "not_run", "elements": 0, "passed": False},
                "fallback_used": bool(effective["rolled_back_to_oracle"]), "strict_pure": False,
                "runtime_binary_matches_candidate": None, "passed": False}
      prepared = _prepared_candidate(compiled, program, out, ref, identity)
      run_linear(compiled, jit=True, wait=True)
      output = out.float().numpy()
      runtime = runtime_cache.get((program.key, str(Device.DEFAULT)))
      if runtime is None: runtime = runtime_cache.get((program.key, Device.DEFAULT))
      loaded = getattr(runtime, "lib", None)
      if not isinstance(loaded, bytes): raise RuntimeError("exact executed binary absent from runtime cache")
      loaded_hash = _sha256(loaded)
  finally:
    postrange._WARMSTART_OPTS, postrange._WARMSTART_CANDIDATE_CONTEXTS = old_opts, old_contexts
    postrange._WARMSTART_LOCAL_STAGE_KEYS = old_local
    getenv.cache_clear(); to_program_cache.clear()

  abs_err = np.abs(output - ref)
  correct = bool(np.all(np.isfinite(output)) and np.allclose(output, ref, atol=atol, rtol=rtol))
  binary_equal = loaded_hash == hashes["binary_sha256"]
  context_equal = (context.schema_version == payload["schema_version"] and context.canonical_identity == identity)
  route_id = effective["effective_route"]
  route_surface_agrees = route_id != "prefill_pipe_role_selective_generated"
  route_strict_pure = bool(effective["strict_pure"] and route_surface_agrees and structural["matches_payload"])
  executable_truth = {"selected_route_id": route_id, "selected_surface": SELECTED_SURFACE,
    "manifest_route_backed": True, "fallback_used": False, "strict_pure": route_strict_pure,
    "compiler_surface_forbidden_markers_absent": surface["strict_pure"],
    "ops_ins_absent": surface["ops_ins_count"] == 0, "asm_source_absent": surface["source_kind"] != "native_isa",
    "candidate_context_equal": context_equal, "runtime_binary_matches_candidate": binary_equal, **lds}
  binding_errors = list(structural["errors"])
  if not route_surface_agrees: binding_errors.append(f"manifest effective route {route_id!r} does not describe selected generated surface")
  passed = bool(correct and binary_equal and context_equal and route_strict_pure and not binding_errors)
  if prepared_out is not None: prepared_out.append(prepared)
  return {"schema": SCHEMA, "canonical_identity": identity, "route_id": route_id,
          "selected_route_id": route_id, "environment": {"device": str(Device.DEFAULT), "git": _git_state()},
          "route_binding_complete": False if binding_errors else passed, "route_authority": effective,
          "structural_binding": structural, "binding_errors": binding_errors, "program": hashes,
          "runtime": {"executed_binary_sha256": loaded_hash, "binary_equal": binary_equal},
          "surface": surface, "executable_truth": executable_truth,
          "correctness": {"case": case, "comparison": "full_output", "elements": int(output.size),
                          "atol": atol, "rtol": rtol, "max_abs_error": float(abs_err.max()),
                          "mean_abs_error": float(abs_err.mean()), "passed": correct},
          "fallback_used": bool(effective["rolled_back_to_oracle"]), "strict_pure": route_strict_pure,
          "runtime_binary_matches_candidate": binary_equal, "passed": passed}


def _load_payload(path: pathlib.Path) -> dict[str, Any]:
  row = json.loads(path.read_text())
  if row.get("schema_version"): return row
  return row["rows"][0]["search_row"]["full_kernel_candidate"]


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--candidate", type=pathlib.Path, required=True)
  ap.add_argument("--candidate-hash"); ap.add_argument("--case", default="constant", choices=("constant", "row_col"))
  ap.add_argument("--output", type=pathlib.Path)
  args = ap.parse_args()
  payload = _load_payload(args.candidate); identity = args.candidate_hash or canonical_candidate_hash(payload)
  report = run(payload, identity, case=args.case)
  text = json.dumps(report, indent=2) + "\n"
  if args.output: args.output.write_text(text)
  print(text, end="")
  return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
