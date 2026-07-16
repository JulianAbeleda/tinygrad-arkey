"""Fail-closed resources for native AMDISARenderer minimal-ELF PROGRAMs."""
from __future__ import annotations

import ctypes
from math import prod
from typing import Any

from tinygrad.renderer.amd.dsl import FixedBitField, Reg
from tinygrad.renderer.isa import CompilerCaptureProof
from tinygrad.uop.ops import Ops, ProgramInfo, UOp


def _reject(message: str): raise ValueError(f"native AMD resource evidence rejected: {message}")


def _target_arch(target: str) -> str:
  if not isinstance(target, str) or not target: _reject("explicit target is required")
  arch = target.rsplit(":", 1)[-1]
  if arch != "gfx1100": _reject(f"unsupported native target {target!r}")
  return arch


def _native_instruction_resources(linear: UOp, *, allow_symbolic_control_flow: bool = False) -> tuple[int, int]:
  """Return exact used VGPR/SGPR ends, rejecting any scratch/spill or foreign construct."""
  if linear.op is not Ops.LINEAR or not linear.src: _reject("one non-empty final LINEAR stream is required")
  vgpr_end = sgpr_end = 0
  for row in linear.src:
    if row.op is not Ops.INS: _reject("final LINEAR contains a non-instruction row")
    inst = row.arg
    if allow_symbolic_control_flow and isinstance(inst, tuple) and inst[:1] in (("label",), ("branch",)):
      if "scratch" in str(inst).lower() or "spill" in str(inst).lower():
        _reject("LINEAR symbolic control flow contains scratch/spill constructs")
      continue
    text = f"{type(inst).__name__} {inst}".lower()
    if "scratch" in text or "spill" in text: _reject("final LINEAR contains scratch/spill instructions or constructs")
    if not type(inst).__module__.startswith("tinygrad.runtime.autogen.amd.") or not callable(getattr(inst, "to_bytes", None)):
      _reject("final LINEAR contains a non-native AMD instruction")
    fields = getattr(inst, "_fields", None)
    if not isinstance(fields, list): _reject("native AMD instruction has no typed fields")
    for name, field in fields:
      if isinstance(field, FixedBitField): continue
      value = getattr(inst, name, None)
      if not isinstance(value, Reg): continue
      if not isinstance(value.offset, int) or not isinstance(value.sz, int) or value.sz <= 0:
        _reject("native AMD instruction contains an invalid register operand")
      if 256 <= value.offset < 512: vgpr_end = max(vgpr_end, value.offset - 256 + value.sz)
      elif 0 <= value.offset < 106: sgpr_end = max(sgpr_end, value.offset + value.sz)
  return vgpr_end, sgpr_end


def amd_native_program_resources(program: UOp, *, target: str) -> dict[str, Any]:
  """Inspect one final native AMD PROGRAM without external ELF tools.

  Zero scratch/spills is admitted only after the exact renderer-final stream
  contains no scratch instruction or non-native renderer construct and
  byte-for-byte reassembles to the supplied final minimal ELF.
  """
  arch = _target_arch(target)
  if not isinstance(program, UOp) or program.op is not Ops.PROGRAM or not isinstance(program.arg, ProgramInfo):
    _reject("a final ProgramInfo PROGRAM is required")
  if len(program.src) != 5 or tuple(row.op for row in program.src[1:]) != \
     (Ops.DEVICE, Ops.LINEAR, Ops.SOURCE, Ops.BINARY):
    _reject("PROGRAM must contain sink/device/LINEAR/source/binary exactly")
  if program.src[1].arg != "AMD": _reject("PROGRAM is not native AMD")
  linear, source, binary = program.src[2], program.src[3].arg, program.src[4].arg
  if not isinstance(source, str) or source != "\n".join(str(row.arg) for row in linear.src):
    _reject("PROGRAM source differs from its LINEAR stream")
  if not isinstance(binary, bytes) or not binary: _reject("final ELF binary is missing")

  local_size, global_size = program.arg.local_size, program.arg.global_size
  if not isinstance(local_size, tuple) or not local_size or any(type(x) is not int or x <= 0 for x in local_size):
    _reject("ProgramInfo local launch is not fully static and positive")
  if not isinstance(global_size, tuple) or not global_size or any(type(x) is not int or x <= 0 for x in global_size):
    _reject("ProgramInfo global launch is not fully static and positive")
  workgroup_threads = prod(local_size)

  try:
    from tinygrad.renderer.isa.amd import AMDISARenderer
    from tinygrad.renderer.amd.elf import (assemble_linear, descriptor_register_counts,
      kernel_descriptor_from_elf)
    from tinygrad.runtime.support.elf import elf_loader
    from tinygrad.helpers import Target
    renderer = AMDISARenderer(Target.parse(f"AMD:ISA:{arch}"))
    # This backend has no spill ABI: all three construction hooks must remain hard errors.
    for method, args in ((renderer.stack_pointer, ()), (renderer.spill, (None, None)),
                         (renderer.fill, (None, None, None))):
      try: method(*args)
      except NotImplementedError as exc:
        if "no spills" not in str(exc): _reject("AMDISARenderer spill hard-error contract drift")
      else: _reject("AMDISARenderer unexpectedly permits spill/stack construction")
    # Reject scratch/spill evidence before final scheduling can inspect or transform it.
    _native_instruction_resources(linear, allow_symbolic_control_flow=True)
    final_linear = renderer._final_linear(linear)
    used_vgpr, used_sgpr = _native_instruction_resources(final_linear)
    _, sections, relocs = elf_loader(binary)
    if len(sections) != 3 or [section.name for section in sections] != [".text", ".rodata", ".strtab"] or relocs:
      _reject("binary is not the native minimal AMD ELF")
    rodata = sections[1].content
    descriptor = kernel_descriptor_from_elf(binary)
    if len(rodata) != ctypes.sizeof(descriptor): _reject("ELF has a malformed kernel descriptor")
    allocated_vgpr, allocated_sgpr = descriptor_register_counts(descriptor, is_cdna=False)
    from tinygrad.runtime.autogen import amdgpu_kd
    private_properties = int(descriptor.kernel_code_properties) & (
      amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER |
      amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT |
      amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE |
      amdgpu_kd.KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK)
    private_rsrc = int(descriptor.compute_pgm_rsrc2) & amdgpu_kd.COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT
    if int(descriptor.private_segment_fixed_size) != 0 or private_properties or private_rsrc:
      _reject("ELF descriptor enables a private/scratch segment")
    # Match AMDISARenderer.asm exactly.  Proof-bearing streams project compiler-owned register storage out of the
    # assembly PROGRAM metadata; reassembling them with proof=None can produce a different descriptor/ELF even though
    # the final instruction stream is identical.
    proof = linear.arg if isinstance(linear.arg, CompilerCaptureProof) else None
    if assemble_linear(renderer._assembly_program(program, proof), final_linear, arch) != binary:
      _reject("final LINEAR does not reassemble to the supplied ELF")
  except ValueError:
    raise
  except Exception as exc:
    raise ValueError(f"native AMD resource evidence rejected: malformed native program: {type(exc).__name__}: {exc}") from exc

  if used_vgpr > allocated_vgpr: _reject("final LINEAR VGPR use exceeds the ELF descriptor allocation")
  wave32_bit = 1 << amdgpu_kd.KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32_SHIFT
  wavefront_size = 32 if int(descriptor.kernel_code_properties) & wave32_bit else 64
  if wavefront_size != 32: _reject("native target requires a wave32 descriptor")
  return {"schema": "tinygrad.amd.native_program_resources.v1",
    "authority": {"vgpr_lds_wave": "final_elf_descriptor", "sgpr_scratch_spills": "renderer_final_linear",
                  "workgroup": "program_info_launch"},
    "target": arch, "vgpr": allocated_vgpr, "sgpr": used_sgpr,
    "allocated_vgpr": allocated_vgpr, "allocated_sgpr": allocated_sgpr,
    "used_vgpr": used_vgpr, "lds_bytes": int(descriptor.group_segment_fixed_size),
    "scratch_bytes": 0, "vgpr_spills": 0, "sgpr_spills": 0,
    "scratch_spill_proof": {"amdisarenderer_spill_construction": "hard_error",
      "pre_and_final_linear_scratch_spill_instructions": "absent", "private_segment_fixed_size": 0,
      "private_segment_properties": "disabled", "byte_identical_reassembly": True},
    "workgroup_threads": workgroup_threads, "max_workgroup_threads": workgroup_threads,
    "wavefront_size": wavefront_size, "global_size": list(global_size), "local_size": list(local_size)}


__all__ = ["amd_native_program_resources"]
