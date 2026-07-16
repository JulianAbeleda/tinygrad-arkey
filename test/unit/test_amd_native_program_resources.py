from functools import lru_cache
import ctypes
import subprocess

import pytest

from extra.qk.prefill.amd_native_program_resources import amd_native_program_resources
from extra.qk.prefill.q4k_q8_five_buffer_pipeline import compile_q4k_q8_five_buffer_pipeline
from test.unit.test_q4k_q8_five_buffer_execution_adapter import _entry
from tinygrad.renderer.amd.elf import descriptor_register_counts, kernel_descriptor_from_elf
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.uop.ops import Ops, UOp


@lru_cache(maxsize=1)
def _programs():
  entry = _entry((64, 16, 256), role="native_resource_test")
  pipeline = compile_q4k_q8_five_buffer_pipeline(entry.payload, entry.canonical_identity)
  return pipeline.producer, pipeline.mmq


def test_both_pipeline_programs_use_native_final_authorities_without_external_tools(monkeypatch):
  programs = _programs()
  monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("external metadata tool is forbidden"))
  for program in programs:
    resources = amd_native_program_resources(program, target="AMD:ISA:gfx1100")
    descriptor = kernel_descriptor_from_elf(program.src[4].arg)
    allocated_vgpr, allocated_sgpr = descriptor_register_counts(descriptor, is_cdna=False)
    assert resources["vgpr"] == resources["allocated_vgpr"] == allocated_vgpr
    assert resources["allocated_sgpr"] == allocated_sgpr is None and resources["sgpr"] > 0
    assert resources["lds_bytes"] == descriptor.group_segment_fixed_size
    assert resources["scratch_bytes"] == resources["vgpr_spills"] == resources["sgpr_spills"] == 0
    assert resources["wavefront_size"] == 32 and resources["workgroup_threads"] == 32
    assert resources["target"] == "gfx1100"
    assert resources["authority"] == {"vgpr_lds_wave": "final_elf_descriptor",
      "sgpr_scratch_spills": "renderer_final_linear", "workgroup": "program_info_launch"}
    assert resources["scratch_spill_proof"] == {"amdisarenderer_spill_construction": "hard_error",
      "pre_and_final_linear_scratch_spill_instructions": "absent", "private_segment_fixed_size": 0,
      "private_segment_properties": "disabled", "byte_identical_reassembly": True}


@pytest.mark.parametrize("mutation, match", (
  ("binary", "malformed native program"), ("device", "not native AMD"),
  ("source", "source differs"), ("instruction", "scratch/spill"),
))
def test_malformed_or_non_native_program_rejected(mutation, match):
  program = _programs()[0]
  if mutation == "binary": changed = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=b"not-an-elf"),))
  elif mutation == "device": changed = program.replace(src=(program.src[0], UOp(Ops.DEVICE, arg="CPU"), *program.src[2:]))
  elif mutation == "source": changed = program.replace(src=program.src[:3] + (UOp(Ops.SOURCE, arg="drift"), program.src[4]))
  else:
    linear = program.src[2].replace(src=(UOp(Ops.INS, arg="scratch_load_dword spill"), *program.src[2].src))
    source = UOp(Ops.SOURCE, arg="\n".join(str(row.arg) for row in linear.src))
    changed = program.replace(src=(program.src[0], program.src[1], linear, source, program.src[4]))
  with pytest.raises(ValueError, match=match): amd_native_program_resources(changed, target="AMD:ISA:gfx1100")


@pytest.mark.parametrize("target", ("", "AMD", "AMD:ISA:gfx1200"))
def test_missing_or_non_gfx1100_target_rejected(target):
  with pytest.raises(ValueError, match="target"):
    amd_native_program_resources(_programs()[0], target=target)


def test_private_segment_descriptor_mutation_rejected():
  program = _programs()[0]
  binary = bytearray(program.src[4].arg)
  _, sections, _ = elf_loader(bytes(binary))
  rodata = next(section for section in sections if section.name == ".rodata")
  descriptor = kernel_descriptor_from_elf(bytes(binary))
  descriptor.private_segment_fixed_size = 4
  offset = int(rodata.header.sh_offset)
  binary[offset:offset + ctypes.sizeof(descriptor)] = bytes(descriptor)
  changed = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=bytes(binary)),))
  with pytest.raises(ValueError, match="private/scratch segment"):
    amd_native_program_resources(changed, target="AMD:ISA:gfx1100")


def test_binary_identity_mismatch_rejected():
  program = _programs()[0]
  binary = bytearray(program.src[4].arg); binary[-1] ^= 1
  changed = program.replace(src=program.src[:4] + (UOp(Ops.BINARY, arg=bytes(binary)),))
  with pytest.raises(ValueError, match="reassemble"):
    amd_native_program_resources(changed, target="AMD:ISA:gfx1100")
