#!/usr/bin/env python3
"""Exact compiler, code-object, resource, and final-ISA evidence for the bounded MMQ atom."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any, Mapping

from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_q4k_q8_atom import Q4K_WORDS_PER_BLOCK, _q4k_q8_1_bounded_ds4_coop_tile_kernel
from extra.qk.mmq_q4k_q8_reference import Q4_K_BLOCK_ELEMS

SHAPE = {"M": 16, "N": 16, "K": 256}
COMPILER_ENV = ("AMD", "AMD_CC", "AMD_COMGR_SAVE_TEMPS", "AMD_ISA_REG_ACCUM", "COALESCED_LOAD_LOWERING",
                "DEV", "HIPCC", "NOOPT", "REG_STORE_DEVEC", "ROCM_PATH", "SCHED_LIST", "SCHED_UNROLL",
                "WARP_REDUCE_LOWERING")


def _sha256(data:bytes) -> str: return hashlib.sha256(data).hexdigest()


def build_mmq_sink(spec:Any) -> UOp:
  """Construct the sole canonical source sink used for compile evidence."""
  if getattr(spec, "writeback_mode", None) not in ("gated_matrix_v0", "direct_owner_v0"):
    raise ValueError("spec has no supported writeback_mode")
  m, n, k = SHAPE.values()
  return _q4k_q8_1_bounded_ds4_coop_tile_kernel(m, n, k, "ffn_gate_up", spec.writeback_mode)(
    UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
    UOp.placeholder(((k // 128) * m * 128,), dtypes.int8, 2),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 3),
    UOp.placeholder(((k // 128) * m * 4,), dtypes.float32, 4))


def compile_mmq_program(spec:Any, device:str="AMD") -> UOp:
  program = to_program(build_mmq_sink(spec), Device[device].renderer)
  if program.op is not Ops.PROGRAM or len(program.src) != 5 or program.src[3].op is not Ops.SOURCE or program.src[4].op is not Ops.BINARY:
    raise RuntimeError("MMQ lowering did not produce a source-bound binary PROGRAM")
  return program


def _tool(names:tuple[str, ...]) -> str:
  for name in names:
    if Path(name).is_file() or (found := shutil.which(name)): return str(Path(name) if Path(name).is_file() else found)
  raise FileNotFoundError(f"none of {names!r} is available")


def _run_binary_tool(binary:bytes, names:tuple[str, ...], args:tuple[str, ...]) -> tuple[str, str]:
  tool = _tool(names)
  with tempfile.NamedTemporaryFile(suffix=".hsaco") as f:
    f.write(binary); f.flush()
    proc = subprocess.run((tool, *args, f.name), check=True, capture_output=True, text=True)
  version = subprocess.run((tool, "--version"), check=True, capture_output=True, text=True).stdout.splitlines()[0]
  return proc.stdout, f"{tool}: {version}"


def _int_field(text:str, field:str) -> int:
  match = re.search(rf"^\s*\.{re.escape(field)}:\s*(\d+)\s*$", text, re.MULTILINE)
  if not match: raise ValueError(f"AMDGPU metadata missing .{field}")
  return int(match.group(1))


def parse_amdgpu_metadata(binary:bytes) -> dict[str, Any]:
  text, tool = _run_binary_tool(binary,
    ("/opt/rocm/llvm/bin/llvm-readelf", "llvm-readelf-21", "llvm-readelf-20", "llvm-readelf"), ("--notes",))
  def string_field(field:str) -> str:
    match = re.search(rf"^\s*\.{re.escape(field)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if not match: raise ValueError(f"AMDGPU metadata missing .{field}")
    return match.group(1).strip()
  target_match = re.search(r"^amdhsa\.target:\s*(\S+)\s*$", text, re.MULTILINE)
  if not target_match: raise ValueError("AMDGPU metadata missing amdhsa.target")
  return {"vgpr": _int_field(text, "vgpr_count"), "sgpr": _int_field(text, "sgpr_count"),
          "vgpr_spills": _int_field(text, "vgpr_spill_count"), "sgpr_spills": _int_field(text, "sgpr_spill_count"),
          "lds_bytes": _int_field(text, "group_segment_fixed_size"),
          "scratch_bytes": _int_field(text, "private_segment_fixed_size"),
          "max_workgroup_threads": _int_field(text, "max_flat_workgroup_size"),
          "wavefront_size": _int_field(text, "wavefront_size"),
          "dynamic_stack": string_field("uses_dynamic_stack").lower() == "true",
          "symbol": string_field("symbol"), "target": target_match.group(1), "metadata_tool": tool}


def disassemble_amdgpu(binary:bytes) -> tuple[str, str]:
  text, tool = _run_binary_tool(binary,
    ("/opt/rocm/llvm/bin/llvm-objdump", "llvm-objdump-21", "llvm-objdump-20", "llvm-objdump"), ("-d",))
  # llvm-objdump prints the temporary input path; it is storage provenance, not ISA identity.
  text = re.sub(r"^.*:\s+file format (.+)$", r"<code-object>:\tfile format \1", text, count=1, flags=re.MULTILINE)
  return text, tool


_INST = re.compile(r"^\s*([a-z][a-z0-9_]*)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):\s*([0-9A-Fa-f ]+)\s*$")
_REG = re.compile(r"\b([vs])(?:\[(\d+):(\d+)\]|(\d+)\b)|\b(vcc(?:_lo|_hi)?|scc|exec(?:_lo|_hi)?)\b")


def _registers(text:str) -> list[str]:
  out: list[str] = []
  for match in _REG.finditer(text):
    if special := match.group(5): out.append(special); continue
    kind, lo, hi, scalar = match.group(1), match.group(2), match.group(3), match.group(4)
    if scalar is not None: out.append(kind + scalar)
    else: out.extend(f"{kind}{idx}" for idx in range(int(lo), int(hi) + 1))
  return out


def _instruction_class(mnemonic:str) -> tuple[str, str]:
  if mnemonic.startswith(("global_load", "flat_load", "buffer_load")): return "global_load", "vmem"
  if mnemonic.startswith(("global_store", "flat_store", "buffer_store")): return "global_store", "vmem"
  if mnemonic.startswith("ds_load"): return "lds_load", "lds"
  if mnemonic.startswith("ds_store"): return "lds_store", "lds"
  if mnemonic.startswith("scratch_load"): return "global_load", "vmem"
  if mnemonic.startswith("scratch_store"): return "global_store", "vmem"
  if mnemonic.startswith("s_barrier"): return "barrier", "salu"
  if mnemonic.startswith("s_waitcnt"): return "waitcnt", "salu"
  if mnemonic.startswith(("s_branch", "s_cbranch", "s_cmp", "v_cmp")): return "branch_predicate", "salu"
  if any(token in mnemonic for token in ("wmma", "mfma", "dot")): return "dot_mfma", "valu"
  if mnemonic.startswith("v_"):
    return ("valu_float" if re.search(r"(?:^|_)(?:f16|f32|f64|bf16)(?:_|$)", mnemonic) else "valu_int"), "valu"
  return "salu", "salu"


def _read_write_registers(mnemonic:str, operands:str, instruction_class:str) -> tuple[list[str], list[str]]:
  pieces = [piece.strip() for piece in operands.split(",")]
  all_regs = _registers(operands)
  if not pieces or instruction_class in ("global_store", "lds_store", "barrier", "waitcnt") or mnemonic.startswith(("s_branch", "s_cbranch", "s_endpgm")):
    return all_regs, []
  first = _registers(pieces[0])
  writes = list(first)
  reads = _registers(",".join(pieces[1:]))
  if mnemonic.startswith(("v_cmp", "s_cmp")):
    writes = first if first and first[0] in ("vcc", "vcc_lo", "vcc_hi", "scc") else (["vcc"] if mnemonic.startswith("v_cmp") else ["scc"])
    reads = [reg for reg in all_regs if reg not in writes]
  if "_co_" in mnemonic and len(pieces) > 1:
    carry = _registers(pieces[1])
    writes.extend(reg for reg in carry if reg not in writes)
    reads = _registers(",".join(pieces[2:]))
  if "saveexec" in mnemonic and "exec" not in writes: writes.append("exec")
  return list(dict.fromkeys(reads)), list(dict.fromkeys(writes))


def analyze_final_isa(disassembly:str) -> dict[str, Any]:
  rows, max_vgpr, max_sgpr = [], -1, -1
  for line in disassembly.splitlines():
    match = _INST.match(line)
    if not match: continue
    mnemonic, operands, pc, encoding = match.groups()
    for reg in _registers(operands):
      if reg.startswith("v") and reg[1:].isdigit(): max_vgpr = max(max_vgpr, int(reg[1:]))
      if reg.startswith("s") and reg[1:].isdigit(): max_sgpr = max(max_sgpr, int(reg[1:]))
    instruction_class, issue_domain = _instruction_class(mnemonic)
    reads, writes = _read_write_registers(mnemonic, operands, instruction_class)
    epoch = ({"global_load": "load_decode", "global_store": "writeback", "lds_load": "dot_k_loop",
              "lds_store": "stage", "barrier": "visibility_sync", "waitcnt": "visibility_sync"}.get(instruction_class, "unknown"))
    rows.append({"index": len(rows), "pc": int(pc, 16), "mnemonic": mnemonic, "operands": operands.strip(),
                 "encoding": encoding.strip().split(), "instruction_class": instruction_class,
                 "issue_domain": issue_domain, "reads": reads, "writes": writes, "dependencies": [], "epoch": epoch,
                 "active_lanes": None, "transactions": None,
                 "operand_role_source": "amd_isa_destination_first_convention"})
  if not rows: raise ValueError("no AMD instructions parsed from disassembly")
  def selected(prefixes:tuple[str, ...]) -> list[dict[str, Any]]:
    return [row for row in rows if row["mnemonic"].startswith(prefixes)]
  stores = selected(("global_store",))
  return {"instruction_count": len(rows), "encoded_dwords": sum(len(r["encoding"]) for r in rows),
          "global_load_sites": len(selected(("global_load",))), "global_store_sites": len(stores),
          "ds_load_sites": len(selected(("ds_load",))), "ds_store_sites": len(selected(("ds_store",))),
          "barrier_sites": len(selected(("s_barrier",))), "waitcnt_sites": len(selected(("s_waitcnt",))),
          "scratch_sites": len(selected(("scratch_",))),
          "branch_sites": len(selected(("s_branch", "s_cbranch"))),
          "predicate_sites": len(selected(("v_cmp", "s_cmp", "s_cbranch"))),
          "max_referenced_vgpr": max_vgpr, "max_referenced_sgpr": max_sgpr,
          "store_instructions": [{"index": r["index"], "pc": r["pc"], "mnemonic": r["mnemonic"],
                                  "operands": r["operands"]} for r in stores], "instructions": rows}


@dataclass(frozen=True)
class MMQCompileEvidence:
  program: UOp
  sink_text: str
  source: str
  binary: bytes
  disassembly: str
  metadata: Mapping[str, Any]
  isa: Mapping[str, Any]
  compiler: Mapping[str, Any]

  @property
  def hashes(self) -> dict[str, str]:
    return {"uop_sha256": _sha256(self.sink_text.encode()), "lowered_sink_sha256": _sha256(repr(self.program.src[0]).encode()),
            "linear_sha256": _sha256(repr(self.program.src[2]).encode()), "rendered_source_sha256": _sha256(self.source.encode()),
            "binary_sha256": _sha256(self.binary), "isa_sha256": _sha256(self.disassembly.encode())}

  def manifest(self) -> dict[str, Any]:
    return {"schema": "tinygrad.mmq_compile_manifest.v1", "function_name": self.program.arg.function_name,
            "target": self.program.src[1].arg, "program_key": self.program.key.hex(),
            "launch": {"global_size": list(self.program.arg.global_size), "local_size": list(self.program.arg.local_size)},
            "binary_bytes": len(self.binary), "hashes": self.hashes, "resources": dict(self.metadata),
            "isa_summary": {k: v for k, v in self.isa.items() if k != "instructions"}, "compiler": dict(self.compiler),
            "compiler_environment": {key: os.environ[key] for key in COMPILER_ENV if key in os.environ}}


def capture_loaded_mmq_program(spec:Any, device:str="AMD") -> MMQCompileEvidence:
  program = compile_mmq_program(spec, device)
  from tinygrad.engine.realize import runtime_cache
  runtime = runtime_cache.get((program.key, device))
  if runtime is None: raise RuntimeError("executed MMQ program is absent from the runtime cache")
  loaded = getattr(runtime, "lib", None)
  if not isinstance(loaded, bytes): raise RuntimeError("loaded MMQ runtime does not expose code-object bytes")
  compiled = program.src[4].arg
  if loaded != compiled: raise RuntimeError("loaded MMQ code object differs from compiled program binary")
  source = program.src[3].arg
  metadata = parse_amdgpu_metadata(loaded)
  disassembly, disasm_tool = disassemble_amdgpu(loaded)
  isa = analyze_final_isa(disassembly)
  renderer, compiler = Device[device].renderer, Device[device].compiler
  compiler_info = {"renderer": type(renderer).__name__, "compiler": type(compiler).__name__,
                   "compiler_cache_key": compiler.cachekey, "disassembly_tool": disasm_tool}
  evidence = MMQCompileEvidence(program, repr(build_mmq_sink(spec)), source, loaded, disassembly, metadata, isa, compiler_info)
  if metadata["symbol"] != program.arg.function_name + ".kd": raise RuntimeError("metadata symbol does not match program")
  if not metadata["target"].endswith(str(renderer.target.arch)): raise RuntimeError("metadata target does not match renderer")
  if metadata["max_workgroup_threads"] != __import__("math").prod(program.arg.local_size):
    raise RuntimeError("metadata workgroup size does not match program launch")
  return evidence
