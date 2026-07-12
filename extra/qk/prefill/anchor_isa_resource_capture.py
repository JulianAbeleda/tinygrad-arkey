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

from extra.qk.mmq_compile_evidence import disassemble_amdgpu, parse_amdgpu_metadata

ROOT = pathlib.Path(__file__).resolve().parents[3]
SCHEMA = "prefill-pure-anchor-isa-resource-capture.v1"
SHAPE = {"M": 512, "N": 12288, "K": 4096}
ROLE = "ffn_gate_up"
PURE_ROUTE = "prefill_v2_scheduler_matmul_default"
ORACLE_ROUTE = "prefill_pipe_role_selective_generated"


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
  return {"schema": SCHEMA, "anchor": {"role": ROLE, "shape": SHAPE}, "captures": rows,
          "binding_complete": all(row["purity_matches_expectation"] for row in rows),
          "oracle_policy": "evidence_only_never_candidate_substrate"}


def main(argv: list[str] | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--output", type=pathlib.Path, required=True)
  args = ap.parse_args(argv)
  report = capture_anchor()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n")
  print(args.output)


if __name__ == "__main__": main()
