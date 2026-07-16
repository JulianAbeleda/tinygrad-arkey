#!/usr/bin/env python3
"""Final AMD code-object/resource evidence for ``emit_q4k_q8_mmq_wmma``.

This is a compile-evidence gate, not an occupancy gate.  Occupancy cannot be
derived from an AMD code object and is therefore reported as unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_compile_evidence import analyze_final_isa, disassemble_amdgpu, parse_amdgpu_metadata
from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_wmma, emit_q4k_q8_mmq_wmma

SCHEMA = "tinygrad.q4k_q8_mmq_uop_resource_gate.v1"
SHAPES = ((16, 16, 256), (32, 32, 512))
SIGNED_WMMA_MNEMONIC = "v_wmma_i32_16x16x16_iu8"
_METADATA_FIELDS = ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills",
                    "wavefront_size", "max_workgroup_threads", "symbol")


def _sha256(value: bytes) -> str: return hashlib.sha256(value).hexdigest()


def build_sink(m: int, n: int, k: int) -> UOp:
  spec = describe_q4k_q8_mmq_wmma(m=m, n=n, k=k)
  return emit_q4k_q8_mmq_wmma(spec)(
    UOp.placeholder((m, n), dtypes.float32, 10),
    UOp.placeholder((n * (k // 256) * 36,), dtypes.uint32, 11),
    UOp.placeholder((m * k,), dtypes.int8, 12),
    UOp.placeholder((m * (k // 32),), dtypes.float32, 13))


def _require_final_program(program: UOp) -> tuple[str, bytes]:
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM:
    raise ValueError("final lowering result must be Ops.PROGRAM")
  if len(program.src) != 5 or program.src[3].op is not Ops.SOURCE or program.src[4].op is not Ops.BINARY:
    raise ValueError("final PROGRAM must contain SOURCE and BINARY artifacts")
  source, binary = program.src[3].arg, program.src[4].arg
  if not isinstance(source, str) or not source: raise ValueError("final PROGRAM source is missing")
  if not isinstance(binary, bytes) or not binary: raise ValueError("final PROGRAM binary is missing")
  return source, binary


def capture_program_evidence(program: UOp, authored_sink: UOp) -> dict[str, Any]:
  """Validate and summarize one already-final ``Ops.PROGRAM`` (mock-friendly)."""
  source, binary = _require_final_program(program)
  if not isinstance(authored_sink, UOp) or authored_sink.op is not Ops.SINK:
    raise ValueError("authored emitter result must be Ops.SINK")
  metadata = parse_amdgpu_metadata(binary)
  missing = [field for field in _METADATA_FIELDS if field not in metadata]
  if missing: raise ValueError("AMDGPU metadata missing " + ", ".join(missing))
  for field in _METADATA_FIELDS[:-1]:
    if not isinstance(metadata[field], int) or isinstance(metadata[field], bool) or metadata[field] < 0:
      raise ValueError(f"invalid AMDGPU metadata {field}")
  if not isinstance(metadata["symbol"], str): raise ValueError("invalid AMDGPU metadata symbol")

  function_name = program.arg.function_name
  if getattr(authored_sink.arg, "name", None) != function_name:
    raise ValueError("authored sink name does not match PROGRAM function")
  if metadata["symbol"] != function_name + ".kd": raise ValueError("metadata symbol does not match PROGRAM function")
  global_size, local_size = tuple(program.arg.global_size), tuple(program.arg.local_size)
  if not global_size or not local_size or any(not isinstance(x, int) or x <= 0 for x in (*global_size, *local_size)):
    raise ValueError("PROGRAM launch sizes must be positive integers")
  if metadata["max_workgroup_threads"] != math.prod(local_size):
    raise ValueError("metadata max workgroup does not match PROGRAM local size")

  disassembly, disassembly_tool = disassemble_amdgpu(binary)
  isa = analyze_final_isa(disassembly, wavefront_size=metadata["wavefront_size"])
  wmma = [row["mnemonic"] for row in isa.get("instructions", ()) if row.get("mnemonic", "").startswith("v_wmma_")]
  if not wmma: raise ValueError("final ISA contains no WMMA instruction")
  if set(wmma) != {SIGNED_WMMA_MNEMONIC}:
    raise ValueError(f"final ISA WMMA mnemonic is not exactly {SIGNED_WMMA_MNEMONIC}")
  if len(wmma) != 4: raise ValueError("final ISA must contain exactly four signed WMMA instructions")

  authored_nodes = authored_sink.toposort()
  authored_wmma = sum(node.op in (Ops.WMMA, Ops.SHAPED_WMMA) for node in authored_nodes)
  if authored_wmma != 0: raise ValueError("emitter unexpectedly authors WMMA UOps")
  linear = next((src for src in program.src if src.op is Ops.LINEAR), None)
  if linear is None: raise ValueError("final PROGRAM has no LINEAR UOps")
  # HIP PROGRAMs retain Ops.WMMA here; ISA-renderer PROGRAMs may already carry
  # their textual instruction argument.  Both are compiler-authored, neither
  # is an authored emitter node.
  lowered_wmma = sum(node.op in (Ops.WMMA, Ops.SHAPED_WMMA) or str(node.arg).startswith(SIGNED_WMMA_MNEMONIC)
                     for node in linear.src)
  if lowered_wmma < 1: raise ValueError("final PROGRAM has no lowered signed WMMA UOp")

  return {
    "schema": SCHEMA, "status": "admitted", "function_name": function_name,
    "program_name": function_name, "program_key": program.key.hex(),
    "launch": {"global_size": list(global_size), "local_size": list(local_size)},
    "identity": {"rendered_source_sha256": _sha256(source.encode()), "binary_sha256": _sha256(binary),
                 "binary_bytes": len(binary)},
    "resources": {field: metadata[field] for field in _METADATA_FIELDS[:-1]},
    "metadata_symbol": metadata["symbol"],
    "uops": {"authored_total": len(authored_nodes), "authored_wmma": authored_wmma,
             "final_program_linear_total": len(linear.src), "final_program_lowered_wmma": lowered_wmma},
    "final_isa": {"instruction_count": isa["instruction_count"], "wmma_count": len(wmma),
                  "wmma_mnemonic": SIGNED_WMMA_MNEMONIC, "scratch_sites": isa["scratch_sites"],
                  "disassembly_sha256": _sha256(disassembly.encode()), "disassembly_tool": disassembly_tool},
    "occupancy": {"status": "unavailable", "reason": "not measured; code-object metadata and ISA do not provide occupancy"},
    "occupancy_gate_called": False,
  }


def capture_shape(m: int, n: int, k: int, *, device: str = "AMD") -> dict[str, Any]:
  if (m, n, k) not in SHAPES: raise ValueError(f"unsupported evidence shape {(m, n, k)}")
  sink = build_sink(m, n, k)
  return capture_program_evidence(to_program(sink, Device[device].renderer), sink)


def capture_all(*, device: str = "AMD") -> dict[str, Any]:
  return {"schema": SCHEMA, "device": device, "captures": [capture_shape(*shape, device=device) for shape in SHAPES]}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--output", type=Path, help="write JSON atomically enough for a one-shot CLI capture")
  args = parser.parse_args()
  text = json.dumps(capture_all(device=args.device), indent=2, sort_keys=True) + "\n"
  if args.output is None: print(text, end="")
  else: args.output.write_text(text)


if __name__ == "__main__": main()
