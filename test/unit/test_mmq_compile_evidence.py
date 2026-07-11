from types import SimpleNamespace

import pytest

from extra.qk import mmq_compile_evidence as ce
from extra.qk.mmq_experiment import canonical_candidate


DISASM = """
0000000000001800 <kernel>:
  global_load_b32 v1, v[2:3], off // 000000001800: DC500000 017C0002
  ds_store_b32 v4, v1 offset:0 // 000000001808: D8340000 00000104
  s_waitcnt lgkmcnt(0) // 000000001810: BF89FC07
  v_cmp_eq_u32_e32 vcc_lo, 0, v0 // 000000001814: 7D520080
  global_store_b32 v[2:3], v1, off // 000000001818: DC700000 007C0102
  s_endpgm // 000000001820: BFB00000
"""

NOTES = """
    .group_segment_fixed_size: 256
    .max_flat_workgroup_size: 32
    .name: kernel
    .private_segment_fixed_size: 0
    .sgpr_count: 29
    .sgpr_spill_count: 0
    .symbol: kernel.kd
    .uses_dynamic_stack: false
    .vgpr_count: 27
    .vgpr_spill_count: 0
    .wavefront_size: 32
amdhsa.target: amdgcn-amd-amdhsa--gfx1100
"""


def test_build_and_compile_exact_candidate_program():
  gated = ce.compile_mmq_program(canonical_candidate("gated_matrix_v0"))
  direct = ce.compile_mmq_program(canonical_candidate("direct_owner_v0"))
  assert gated.arg.local_size == direct.arg.local_size == (32, 1, 1)
  assert gated.arg.global_size == direct.arg.global_size == (16, 16, 1)
  assert gated.src[3].arg != direct.src[3].arg
  assert gated.src[4].arg.startswith(b"\x7fELF") and direct.src[4].arg.startswith(b"\x7fELF")


def test_final_isa_analysis_derives_store_and_register_facts():
  result = ce.analyze_final_isa(DISASM)
  assert result["instruction_count"] == 6
  assert result["global_load_sites"] == 1
  assert result["global_store_sites"] == 1
  assert result["ds_store_sites"] == 1
  assert result["waitcnt_sites"] == 1
  assert result["predicate_sites"] == 1
  assert result["max_referenced_vgpr"] == 4
  assert result["store_instructions"][0]["pc"] == 0x1818
  load, store = result["instructions"][0], result["instructions"][4]
  assert load["instruction_class"] == "global_load" and load["issue_domain"] == "vmem"
  assert load["reads"] == ["v2", "v3"] and load["writes"] == ["v1"]
  assert store["instruction_class"] == "global_store" and store["epoch"] == "writeback"
  assert store["reads"] == ["v2", "v3", "v1"] and store["writes"] == []
  assert store["active_lanes"] is None and store["transactions"] is None
  assert result["instructions"][3]["writes"] == ["vcc_lo"]


def test_metadata_parser_requires_and_returns_exact_fields(monkeypatch):
  monkeypatch.setattr(ce, "_run_binary_tool", lambda binary, names, args: (NOTES, "fixture readelf"))
  result = ce.parse_amdgpu_metadata(b"ELF")
  assert result == {"vgpr": 27, "sgpr": 29, "vgpr_spills": 0, "sgpr_spills": 0, "lds_bytes": 256,
                    "scratch_bytes": 0, "max_workgroup_threads": 32, "wavefront_size": 32,
                    "dynamic_stack": False, "symbol": "kernel.kd", "target": "amdgcn-amd-amdhsa--gfx1100",
                    "metadata_tool": "fixture readelf"}
  monkeypatch.setattr(ce, "_run_binary_tool", lambda binary, names, args: (NOTES.replace("    .vgpr_count: 27\n", ""), "x"))
  with pytest.raises(ValueError, match="vgpr_count"): ce.parse_amdgpu_metadata(b"ELF")


def test_disassembly_identity_ignores_temporary_input_path(monkeypatch):
  monkeypatch.setattr(ce, "_run_binary_tool", lambda binary, names, args: ("/tmp/random.hsaco: file format elf64-amdgpu\n", "objdump"))
  text, _ = ce.disassemble_amdgpu(b"ELF")
  assert text == "<code-object>:\tfile format elf64-amdgpu\n"


def test_capture_fails_closed_without_loaded_program(monkeypatch):
  import tinygrad.engine.realize as realize
  monkeypatch.setattr(realize, "runtime_cache", {})
  with pytest.raises(RuntimeError, match="absent from the runtime cache"):
    ce.capture_loaded_mmq_program(canonical_candidate("direct_owner_v0"))
