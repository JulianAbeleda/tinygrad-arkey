import pytest

from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import (AMD_ISA_TARGET,
  admit_q4k_q8_five_buffer_compile, build_q4k_q8_five_buffer_sink, compile_q4k_q8_five_buffer_program)
from extra.qk.runtime_specs import FULL_KERNEL_CANDIDATE_SCHEMA, derive_packed_weight_candidate, derive_q4k_q8_1_five_buffer_candidate
from tinygrad import dtypes
from tinygrad.uop.ops import Ops


def _payload(shape=(256,256,4096), role="ffn_gate_up"):
  m,n,k = shape
  return {"schema_version":FULL_KERNEL_CANDIDATE_SCHEMA,
    "workload":{"profile":"compile_adapter_test","role":role,"shape":{"m":m,"n":n,"k":k},
      "dtypes":{"a":"fp16","b":"fp16","c":"fp16","accumulator":"fp32"},
      "layout":{"a":"row_major","b":"transposed_row_major","c":"row_major"},
      "target":{"backend":"AMD","arch":"gfx1100","wave_size":32}},
    "schedule":{"tile":{"m":128,"n":128,"k":32},"waves":{"m":4,"n":2},"threads":256,
      "lane_ownership":"rdna3_wmma_f32_16x16x16_f16_lds2_static",
      "cooperative_load":{x:{"lane_mapping":"cooperative_row_stride_64_b128","vector_width":8,"alignment":16} for x in ("a","b")},
      "lds":{"windows":{"a":[0,10240],"b":[10240,20480]},"strides":{"a":80,"b":80},
        "padding":16,"banks":32,"store_vector_width":8,"load_vector_width":8},
      "pipeline":{"buffer_count":1,"stage_count":1,"epoch_graph":[]},
      "wmma":{"instruction_family":"wmma_f32_16x16x16_f16","fragment_layout":"rdna3_wmma_f32_16x16x16_f16_lds2_static",
        "accumulator_ownership":"wmma_accum_wm_x_wn_8_vgprs"},
      "dependency_policy":{"waitcnt":{"vm":0,"lgkm":0},"barriers":[]},
      "residency":{"preload":["a","b"],"resident":["accumulator"],"reuse":{"a":4,"b":2}},
      "epilogue":{"lane_mapping":"wmma_accumulator_scalar_b16","vector_width":1},"numerical_mode":"ieee_fp16_acc_fp32"},
    "static_constraints":{"max_lds_bytes":65536,"max_vgpr_per_thread":256,"allow_spill":False},
    "applicability":{"exact_shape":True,"profiles":["compile_adapter_test"],"roles":[role],"targets":["AMD:gfx1100:wave32"]}}


def _entry(shape=(256, 256, 4096), role="ffn_gate_up"):
  return derive_q4k_q8_1_five_buffer_candidate(_payload(shape, role))


def test_cpu_structural_bridge_binds_exact_identity_and_five_buffer_abi():
  entry = _entry()
  sink, admission = build_q4k_q8_five_buffer_sink(entry.payload, entry.canonical_identity)
  assert sink.arg.candidate_context is admission.context
  assert sink.arg.candidate_context.canonical_identity == entry.canonical_identity
  assert sink.arg.candidate_context.geometry is None
  assert sink.arg.candidate_context.pipeline.transport == "direct_global"
  assert admission.active_lds_bytes == 0 and admission.capability.transport == "direct_global"
  assert admission.operand_plan is not entry.payload["kernel_abi"]
  with pytest.raises(TypeError): admission.operand_plan["family"] = "legacy"
  assert len([u for u in sink.toposort() if u.op is Ops.STORE]) == 1
  assert {u.arg.slot:u.dtype.base for u in sink.toposort() if u.op is Ops.PARAM} == {
    0:dtypes.float32, 1:dtypes.uint32, 2:dtypes.int8, 3:dtypes.float32, 4:dtypes.float32}


def test_bridge_fails_closed_on_legacy_tail_identity_and_target_drift():
  entry = _entry()
  legacy_entry = derive_packed_weight_candidate(_payload(), "Q4_K")
  with pytest.raises(ValueError, match="five-buffer operand_plan"):
    admit_q4k_q8_five_buffer_compile(legacy_entry.payload, legacy_entry.canonical_identity)
  with pytest.raises(ValueError, match="identity"):
    admit_q4k_q8_five_buffer_compile(entry.payload, "0" * 64)
  tail = _entry((256, 256, 4224))
  with pytest.raises(ValueError, match="no tails"):
    admit_q4k_q8_five_buffer_compile(tail.payload, tail.canonical_identity)
  with pytest.raises(ValueError, match="target drift"):
    compile_q4k_q8_five_buffer_program(entry.payload, entry.canonical_identity, target="AMD:ISA:gfx1200")


def _assert_final_program(program, admission):
  assert program.op is Ops.PROGRAM
  assert [u for u in program.toposort() if u.op is Ops.PROGRAM] == [program]
  assert program.src[0].arg.candidate_context is admission.context
  assert program.src[0].arg.candidate_context.canonical_identity == admission.canonical_identity
  assert program.src[0].arg.candidate_context.geometry is None
  assert program.src[0].arg.candidate_context.pipeline is admission.plan
  assert {u.arg.slot:u.dtype.base for u in program.src[0].toposort() if u.op is Ops.PARAM} == {
    0:dtypes.float32, 1:dtypes.uint32, 2:dtypes.int8, 3:dtypes.float32, 4:dtypes.float32}
  assert (tuple(program.arg.globals), tuple(program.arg.outs), tuple(program.arg.ins)) == \
         ((0, 1, 2, 3, 4), (0,), (1, 2, 3, 4))
  assert program.src[3].op is Ops.SOURCE and program.src[4].op is Ops.BINARY
  assert program.src[3].arg.count("v_wmma_i32_16x16x16_iu8") == 1
  assert "ds_read" not in program.src[3].arg and "ds_write" not in program.src[3].arg


def test_amd_isa_compiles_small_multi_grid_to_one_identity_bound_program():
  entry = _entry()
  program, admission = compile_q4k_q8_five_buffer_program(entry.payload, entry.canonical_identity, target=AMD_ISA_TARGET)
  _assert_final_program(program, admission)
  assert program.arg.global_size[0] > 1 and program.arg.global_size[1] > 1


def test_amd_isa_compiles_real_14b_attn_kv_role_shape():
  entry = _entry((512, 1024, 5120), role="attn_kv")
  program, admission = compile_q4k_q8_five_buffer_program(entry.payload, entry.canonical_identity)
  _assert_final_program(program, admission)
