"""Shape pins moved from code into promotion records (measured_shapes / measured_gemv_shapes / measured_topologies /
measured_leases); the emitters validate only their real limits."""
import json
import pytest

from tinygrad.llm import model_route_plan as mrp
from tinygrad.llm.decode_kernels import Q4KGEMVEpilogue, q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel, q6k_spec_for_role
from tinygrad.llm.q6k_v_mmvq import emit_q6k_v_four_warp_fp16_direct, validate_four_warp
from tinygrad.llm.shared_q8_attention import _emit_q8_provider, validate_q8_provider


def test_records_hold_the_measured_shapes():
  assert mrp.shape_measured("decode_ffn_down_resadd", {"dim": 4096, "hidden_dim": 12288, "vocab": 1})
  assert not mrp.shape_measured("decode_ffn_down_resadd", {"dim": 5120, "hidden_dim": 17408})
  assert mrp.shape_measured("decode_q6k_vocab_four_warp_fp16", {"rows": 151936, "k": 4096}, field="measured_gemv_shapes")
  assert not mrp.shape_measured("decode_q6k_vocab_four_warp_fp16", {"rows": 248320, "k": 5120}, field="measured_gemv_shapes")
  assert mrp.shape_measured("decode_flash_llama_vec_wide", {"num_blocks": 36, "n_heads": 32, "n_kv_heads": 8, "head_dim": 128},
                            field="measured_topologies")
  assert not mrp.shape_measured("decode_flash_llama_vec_wide", {"num_blocks": 64, "n_heads": 24, "n_kv_heads": 4, "head_dim": 256},
                                field="measured_topologies")
  lease = mrp.measured_entry("decode_shared_q8_attention", {"num_blocks": 36, "dim": 4096}, field="measured_leases")
  assert lease["blocks"] == list(range(1, 13)) + list(range(14, 19)) + [25]
  assert mrp.measured_entry("decode_shared_q8_attention", {"num_blocks": 64, "dim": 5120}, field="measured_leases") is None
  assert mrp.shape_measured("decode_norm_fusion", {"dim": 1})  # a record without measured shapes keeps validate as its only rule


def test_the_gate_refuses_an_unmeasured_shape_after_validate_passes():
  shapes = mrp.DecodeRouteShapes(dim=5120, hidden_dim=17408, n_heads=40, n_kv_heads=8, head_dim=128, vocab=1, eps=1e-6)
  gate = mrp.DecodeRouteShapeGate(shapes, {"decode_ffn_down_resadd": lambda s: None})
  assert not gate.admit("decode_ffn_down_resadd", True) and "measured" in gate.report_lines()[0]
  ok = mrp.DecodeRouteShapeGate(mrp.DecodeRouteShapes(4096, 12288, 32, 8, 128, 151936, 1e-6), {"decode_ffn_down_resadd": lambda s: None})
  assert ok.admit("decode_ffn_down_resadd", True)


def test_emitters_state_only_their_real_limits():
  Q4KGEMVEpilogue("ffn_down_resadd").validate(5120, 17408)
  q6k_spec_for_role(5120, 17408, role="ffn_down", row_tile=2, reduction="in_kernel", epilogue="ffn_down_resadd").validate()
  q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(17408, 5120)
  emit_q6k_v_four_warp_fp16_direct(rows=5120, k=6144)
  validate_four_warp(248320, 5120, warps_per_row=2)
  with pytest.raises(ValueError): validate_four_warp(10, 5120, warps_per_row=4)   # 20 blocks over 8 slices
  with pytest.raises(ValueError): validate_four_warp(10, 300)
  _emit_q8_provider(5120); validate_q8_provider(32)
  with pytest.raises(ValueError): validate_q8_provider(48)


def test_registry_and_search_family():
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  from extra.llm_research import decode_emitter_space as space
  assert callable(build_registered_llm_emitter("q6_vocab_four_warp.v1", {"rows": 248320, "k": 6144}))
  assert callable(build_registered_llm_emitter("q6_four_warp_gemv.v1", {"rows": 5120, "k": 6144, "warps_per_row": 4}))
  assert callable(build_registered_llm_emitter("shared_q8_provider.v1", {"k": 5120, "source_dtype": "fp16"}))
  assert callable(build_registered_llm_emitter("q6_v.v1", {}))
  fam = space.BY_FAMILY[space.Q6_FOUR_WARP_FAMILY]
  j = fam.to_json()
  assert j["quants"] == ["Q6_K"] and j["lanes_per_block_slice"] == 16 and j["installed_rule"] == {"warps_per_row": 4}
  sched = {"launch": {"threads": 256}, "tile": {"m": 1, "n": 1, "k": 4096}, "memory": {"a": {"space": "global"}, "b": {"vector_width": 2}}}
  assert fam.parameters(sched, 5120, 6144, "Q6_K", 32) == {"rows": 5120, "k": 6144, "warps_per_row": 8}
