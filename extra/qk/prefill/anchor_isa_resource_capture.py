#!/usr/bin/env python3
"""Bind exact-shape pure-scheduler and S9-oracle compiler evidence.

This module deliberately composes existing kernel builders and AMD evidence
parsers.  It is diagnostic only and never participates in runtime selection.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, subprocess
from typing import Any

from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.engine.realize import Estimates
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu, parse_amdgpu_metadata
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-pure-anchor-isa-resource-capture.v1"
SHAPE = {"M": 512, "N": 12288, "K": 4096}
ROLE = "ffn_gate_up"
PURE_ROUTE = "prefill_v2_scheduler_matmul_default"
ORACLE_ROUTE = "prefill_pipe_role_selective_generated"
ANCHOR_LDS_BYTES = 20480


def _git_state() -> dict[str, Any]:
  try:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip())
    return {"revision": revision, "dirty": dirty}
  except Exception:
    return {"revision": None, "dirty": True}


def _sha256(data: bytes) -> str: return hashlib.sha256(data).hexdigest()


def _program_surface(program: UOp) -> dict[str, Any]:
  if program.op is not Ops.PROGRAM: raise ValueError("capture requires a lowered PROGRAM")
  linear = next((u for u in program.src if u.op is Ops.LINEAR), None)
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE), None)
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY), None)
  if linear is None or not isinstance(source, str) or not isinstance(binary, bytes):
    raise ValueError("PROGRAM must bind LINEAR, SOURCE, and BINARY")
  ops_ins_count = sum(u.op is Ops.INS for u in linear.toposort())
  asm_source = source.lstrip().startswith(".text") or bool(ops_ins_count)
  forbidden = []
  if ops_ins_count: forbidden.append("Ops.INS")
  if asm_source: forbidden.append("assembly_source")
  return {
    "ops_ins_count": ops_ins_count,
    "source_kind": "native_isa" if asm_source else "compiler_rendered",
    "forbidden_markers": forbidden,
    "strict_pure": not forbidden,
  }


def capture_program(program: UOp, *, candidate_id: str, route_id: str,
                    expected_pure: bool) -> dict[str, Any]:
  """Capture identity/resources/ISA for one already-lowered exact program."""
  surface = _program_surface(program)
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  try:
    metadata = parse_amdgpu_metadata(binary)
    disassembly, disassembly_tool = disassemble_amdgpu(binary)
    metadata_kind = "amdgpu_code_object_notes"
  except (subprocess.CalledProcessError, ValueError):
    # AMD:ISA uses Tinygrad's deliberately minimal native ELF.  Its kernel
    # descriptor and rendered instruction source are the existing authorities.
    from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
    from tinygrad.runtime.autogen import amdgpu_kd
    desc = kernel_descriptor_from_elf(binary)
    vgpr_gran = ((desc.compute_pgm_rsrc1 & amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) >>
                 amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT_SHIFT)
    sgpr_gran = ((desc.compute_pgm_rsrc1 & amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) >>
                 amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT_SHIFT)
    metadata = {"vgpr_allocated": (vgpr_gran + 1) * 8, "sgpr_allocated": (sgpr_gran + 1) * 8,
                "lds_bytes": desc.group_segment_fixed_size, "scratch_bytes": desc.private_segment_fixed_size,
                "compute_pgm_rsrc1": desc.compute_pgm_rsrc1, "compute_pgm_rsrc2": desc.compute_pgm_rsrc2}
    disassembly, disassembly_tool = source, "tinygrad AMD:ISA rendered source"
    metadata_kind = "tinygrad_native_elf_kernel_descriptor"
  hashes = {"program_key": program.key.hex(), "source_sha256": _sha256(source.encode()),
            "binary_sha256": _sha256(binary), "isa_sha256": _sha256(disassembly.encode())}
  if "symbol" in metadata and metadata["symbol"] != program.arg.function_name + ".kd":
    raise RuntimeError("metadata symbol does not match bound program")
  purity_matches = surface["strict_pure"] is expected_pure
  return {
    "candidate_id": candidate_id, "route_id": route_id, "expected_pure": expected_pure,
    "purity_matches_expectation": purity_matches, "program": {
      "function_name": program.arg.function_name, "target": str(program.src[1].arg),
      "launch": {"global_size": list(getattr(program.arg, "global_size", ())),
                 "local_size": list(getattr(program.arg, "local_size", ()))},
      **hashes,
    }, "surface": surface, "resources": {"authority": metadata_kind, **metadata},
    "isa": {"bytes": len(disassembly.encode()), "line_count": len(disassembly.splitlines()),
            "disassembly_tool": disassembly_tool},
  }


def capture_candidate_program(program: UOp, payload: dict[str, Any], candidate_hash: str, *,
                              route_id: str="pure.single_buffer.anchor") -> dict[str, Any]:
  """Compiled-resource authority for an exact route-bound typed pipeline candidate.

  Candidate fields describe intent.  This authority accepts them only after the
  lowered PROGRAM context, code-object descriptor, and final ISA independently
  agree that the requested compiler-owned single-buffer kernel was emitted.
  """
  identity = canonical_candidate_hash(payload)
  if candidate_hash != identity: raise ValueError("candidate_hash does not match canonical candidate payload")
  if program.op is not Ops.PROGRAM: raise ValueError("candidate authority requires a lowered PROGRAM")
  context = program.src[0].arg.candidate_context
  if context is None: raise RuntimeError("compiled PROGRAM has no full-kernel candidate context")
  if context.schema_version != payload["schema_version"] or context.canonical_identity != identity:
    raise RuntimeError("compiled PROGRAM candidate context does not match canonical payload")
  pipeline = getattr(context, "pipeline", None)
  expected_lds_bytes = ANCHOR_LDS_BYTES if pipeline is None else pipeline.active_lds_bytes
  if expected_lds_bytes not in (ANCHOR_LDS_BYTES, 2 * ANCHOR_LDS_BYTES):
    raise RuntimeError("typed pipeline active LDS is outside the buffer1/buffer2 authority")

  row = capture_program(program, candidate_id=identity, route_id=route_id, expected_pure=True)
  resources, surface = row["resources"], row["surface"]
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  disassembly, _ = disassemble_amdgpu(binary)
  try: isa = analyze_final_isa(disassembly, wavefront_size=resources.get("wavefront_size"))
  except ValueError as exc:
    # The shared MMQ analyzer's epoch classifier is intentionally stricter
    # than dense prefill.  Instruction presence below remains authoritative.
    isa = {"status": "dense_prefill_epoch_not_applicable", "reason": str(exc)}
  counts = isa.get("instruction_counts", isa.get("counts", {}))
  # analyze_final_isa has evolved across evidence schemas.  Direct mnemonic
  # evidence is retained as the fail-closed authority for generated transport.
  lowered = disassembly.lower()
  ds_store_count = sum(1 for line in lowered.splitlines() if "ds_store" in line)
  ds_load_count = sum(1 for line in lowered.splitlines() if "ds_load" in line)
  local_defs = [u for u in program.src[0].toposort() if u.op is Ops.DEFINE_LOCAL]
  tagged_local_buffers = [u for u in program.src[0].toposort() if u.op is Ops.BUFFER and
                          isinstance(u.tag, tuple) and u.tag[0] == "kernel_tile_lds"]
  local_sizes = [u.dtype.nbytes() for u in local_defs]
  local_sizes += [u.src[0].arg * u.dtype.itemsize for u in tagged_local_buffers if u.src and u.src[0].op is Ops.CONST]
  local_size = row["program"]["launch"]["local_size"]
  workgroup_threads = 0 if not local_size else __import__("math").prod(local_size)
  required = ("vgpr", "sgpr", "vgpr_spills", "sgpr_spills", "lds_bytes", "scratch_bytes",
              "max_workgroup_threads", "wavefront_size", "target")
  missing = [name for name in required if name not in resources]
  errors = []
  if missing: errors.append(f"compiled AMD metadata missing authority fields: {missing}")
  if not surface["strict_pure"]: errors.append("compiled surface contains handwritten assembly")
  if resources.get("lds_bytes") != expected_lds_bytes:
    errors.append(f"compiled LDS allocation is not the typed {expected_lds_bytes}-byte pipeline allocation")
  if local_sizes != [expected_lds_bytes]:
    errors.append(f"compiler IR does not contain exactly one typed {expected_lds_bytes}-byte LDS allocation")
  if not ds_store_count or not ds_load_count:
    errors.append("final ISA does not prove compiler-emitted LDS stores and loads")
  if resources.get("scratch_bytes") != 0 or resources.get("vgpr_spills") != 0 or resources.get("sgpr_spills") != 0:
    errors.append("compiled candidate uses scratch or register spills")
  if not workgroup_threads or resources.get("max_workgroup_threads") != workgroup_threads:
    errors.append("program launch workgroup does not equal compiled metadata")
  if resources.get("wavefront_size") != payload["workload"]["target"]["wave_size"]:
    errors.append("compiled wavefront size does not match candidate target")
  if not str(resources.get("target", "")).endswith(payload["workload"]["target"]["arch"]):
    errors.append("compiled AMD target does not match candidate target")
  if row["program"]["binary_sha256"] != _sha256(binary):
    errors.append("captured binary identity changed during authority evaluation")
  return {"schema": "prefill-pure-anchor-compiled-resource-authority.v1",
          "canonical_identity": identity, "candidate_hash": identity,
          "status": "pass" if not errors else "fail", "passed": not errors, "errors": errors,
          "git": _git_state(),
          "route_id": route_id, "candidate_context": {"schema_version": context.schema_version,
            "canonical_identity": context.canonical_identity},
          "program": row["program"], "surface": surface,
          "resources": {name: resources.get(name) for name in required} | {"authority": resources["authority"]},
          "isa": {**row["isa"], "analysis": isa, "instruction_counts": counts,
                  "compiler_ir_define_local_sizes": local_sizes,
                  "ds_store_count": ds_store_count, "ds_load_count": ds_load_count,
                  "compiler_emitted_pipeline_lds": not errors and ds_store_count > 0 and ds_load_count > 0,
                  "compiler_emitted_single_buffer_lds": (pipeline is None or pipeline.buffer_count == 1) and not errors and ds_store_count > 0 and ds_load_count > 0},
          "pipeline": {"buffer_count": 1 if pipeline is None else pipeline.buffer_count,
                       "active_lds_bytes": expected_lds_bytes},
          "binding": {"context_matches_payload": context.canonical_identity == identity,
                      "binary_sha256": row["program"]["binary_sha256"],
                      "isa_sha256": row["program"]["isa_sha256"]}}


def build_pure_program() -> UOp:
  # This is the exact default `_prefill_v2_opts`: TC, UPCAST(2,4), no LOCAL, UNROLL(8).
  from extra.qk.prefill_v2_schedule_search import _compile_native_program
  return _compile_native_program(SHAPE["M"], SHAPE["N"], SHAPE["K"], 2, 4, 0, 8)


def build_s9_oracle_program() -> UOp:
  from extra.qk.prefill_schedule_spec import describe_prefill_schedule, emit_prefill_gemm_from_spec
  spec = describe_prefill_schedule(SHAPE["N"], SHAPE["K"], role=ROLE)
  built = emit_prefill_gemm_from_spec(spec)
  if built is None: raise RuntimeError("S9 oracle schedule is not tile legal")
  insts, lds_bytes, bm, bn, threads, name = built
  grid = (SHAPE["N"] // bn, SHAPE["M"] // bm, 1)
  lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
  args = (UOp.placeholder((SHAPE["M"], SHAPE["K"]), dtypes.half, 0).base,
          UOp.placeholder((SHAPE["N"], SHAPE["K"]), dtypes.half, 1).base,
          UOp.placeholder((SHAPE["M"], SHAPE["N"]), dtypes.half, 2).base)
  sink = UOp.sink(*args, lds, UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1"),
                  UOp.special(threads, "lidx0"), arg=KernelInfo(name=name,
                    estimates=Estimates(ops=SHAPE["M"]*SHAPE["N"]*SHAPE["K"]*2)))
  raw = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                              UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=i) for i in insts))))
  return to_program(raw, Device[Device.DEFAULT].renderer)


def capture_anchor() -> dict[str, Any]:
  rows = [capture_program(build_pure_program(), candidate_id="pure.default.m512n12288k4096",
                          route_id=PURE_ROUTE, expected_pure=True),
          capture_program(build_s9_oracle_program(), candidate_id="oracle.s9.m512n12288k4096",
                          route_id=ORACLE_ROUTE, expected_pure=False)]
  return {"schema": SCHEMA, "anchor": {"role": ROLE, "shape": SHAPE}, "git": _git_state(), "captures": rows,
          "binding_complete": all(row["purity_matches_expectation"] for row in rows),
          "oracle_policy": "evidence_only_never_candidate_substrate"}


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, required=True)
  ap.add_argument("--allow-dirty", action="store_true")
  args = ap.parse_args(argv)
  if _git_state()["dirty"] and not args.allow_dirty:
    ap.error("refusing authority capture from a dirty worktree; use --allow-dirty for diagnostics")
  report = capture_anchor()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(args.output)


if __name__ == "__main__": main()
