#!/usr/bin/env python3
from __future__ import annotations

import ctypes, pathlib, re
from typing import Any

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _relocs = elf_loader(lib)
  rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1 = desc.compute_pgm_rsrc1
  gran_vgpr = rsrc1 & 0x3f
  gran_sgpr = (rsrc1 >> 6) & 0xf
  return {"vgpr": (gran_vgpr + 1) * 8, "sgpr": (gran_sgpr + 1) * 8, "lds": desc.group_segment_fixed_size,
          "scratch": desc.private_segment_fixed_size, "kernarg": desc.kernarg_size, "rsrc1": hex(rsrc1)}


def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)


# canonical cross-lane opcode set: the ONE source for "does this ISA use a cross-lane primitive?".
# both the histogram (startswith) and the gate probes (regex) derive from this tuple.
CROSS_LANE_OPS = ("ds_bpermute", "ds_permute", "ds_swizzle", "v_permlane")
CROSS_LANE_RE = r"\b(" + "|".join(CROSS_LANE_OPS) + ")"
def has_cross_lane(asm: str) -> bool: return re.search(CROSS_LANE_RE, asm) is not None

def _hist(asm: str) -> dict[str, int]:
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "cross_lane": 0, "barrier_wait": 0, "scratch": 0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+|s_barrier)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if op.startswith(CROSS_LANE_OPS): h["cross_lane"] += 1
    if op == "s_barrier" or "s_waitcnt" in op: h["barrier_wait"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
  return h


def _parse_debug(lines: list[str]) -> dict[str, Any]:
  program_ms: dict[str, float] = {}
  for raw in lines:
    line = ANSI.sub("", raw)
    if "***" not in line: continue
    name_match = re.search(r"\*\*\*\s+\S+\s+\d+\s+(\S+)", line)
    if not name_match: continue
    vals = [float(x) for x in re.findall(r"(?<![A-Za-z])(\d+(?:\.\d+)?)\s*(?:ms|us)", line)]
    if not vals: continue
    ms = vals[-1] / (1000.0 if " us" in line and " ms" not in line else 1.0)
    program_ms[name_match.group(1)] = program_ms.get(name_match.group(1), 0.0) + ms
  return {"program_ms": program_ms}
