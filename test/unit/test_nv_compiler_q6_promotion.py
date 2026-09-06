from types import SimpleNamespace

import tinygrad.llm.model as model
import pytest
from extra.llm_research.prefill.nv_compiler_q6k_model_arm import _ordinary_prefill_jit, _captured_program_calls
from extra.llm_research.prefill.nv_compiler_streamk_codegen import q4_down_candidate_context, q4_down_fixup_map
from extra.llm_research.prefill.nv_compiler_q4k_streamk_transform import transform_compiler_q4k_to_streamk, active_fixup_source
from extra.llm_research.prefill.nv_compiler_q4k_down_asset import validate_streamk_inputs, WORDS_U32, RECORD_U32
from tinygrad import Tensor, dtypes


def _env(values):
  return lambda key, default=0: values.get(key, default)


@pytest.mark.parametrize("override", ["NV_LLAMA_FATTN_MMA_PP512", "NV_LLAMA_Q6_VOCAB_PP512"])
def test_implicit_llama_consumers_follow_projection_stack(monkeypatch, override):
  config=SimpleNamespace(prefill_ubatch=512, num_blocks=36, dim=4096, hidden_dim=12288,
    n_heads=32, n_kv_heads=8, head_dim=128, num_experts=0)
  monkeypatch.setattr(model, "Device", SimpleNamespace(DEFAULT="NV"))
  monkeypatch.setattr(model, "getenv", _env({}))
  assert model._nv_llama_prefill_role_enabled(config, override)
  monkeypatch.setattr(model, "getenv", _env({"NV_COMPILER_Q4_IMMA_K_PP512":0}))
  assert model._nv_llama_prefill_role_enabled(config, override)
  monkeypatch.setattr(model, "getenv", _env({override:1}))
  assert model._nv_llama_prefill_role_enabled(config, override)
  monkeypatch.setattr(model, "Device", SimpleNamespace(DEFAULT="CPU"))
  assert not model._nv_llama_prefill_role_enabled(config, override)


def test_generated_q6_down_is_the_compiler_stack_default(monkeypatch):
  config=SimpleNamespace()
  monkeypatch.setattr(model,"_nv_compiler_q4_imma_k_pp512_enabled",lambda _:True)
  monkeypatch.setattr(model,"getenv",_env({}))
  assert model._nv_compiler_q6_imma_pp512_enabled(config)
  assert model._nv_compiler_q6_imma_role_enabled(config,"ffn_down")
  assert not model._nv_compiler_q6_imma_role_enabled(config,"attn_v")


def test_generated_q6_down_has_explicit_rollback_and_preempts_llama(monkeypatch):
  config=SimpleNamespace()
  monkeypatch.setattr(model,"_nv_compiler_q4_imma_k_pp512_enabled",lambda _:True)
  monkeypatch.setattr(model,"_nv_llama_full_packed_pp512_enabled",lambda _:True)
  monkeypatch.setattr(model,"getenv",_env({"NV_COMPILER_Q6_IMMA_PP512":0}))
  assert not model._nv_compiler_q6_imma_role_enabled(config,"ffn_down")
  assert model._nv_llama_packed_q6k_down_enabled(config)
  monkeypatch.setattr(model,"getenv",_env({"NV_COMPILER_Q6_IMMA_PP512":1}))
  assert model._nv_compiler_q6_imma_role_enabled(config,"ffn_down")
  assert not model._nv_llama_packed_q6k_down_enabled(config)

def test_ordinary_qualified_mode_selects_generated_stack(monkeypatch):
  config=SimpleNamespace(prefill_ubatch=512, num_blocks=36, dim=4096, hidden_dim=12288,
    n_heads=32, n_kv_heads=8, head_dim=128, num_experts=0)
  monkeypatch.setattr(model, "getenv", _env({}))
  monkeypatch.setattr(model, "Device", SimpleNamespace(DEFAULT="NV"))
  assert model._nv_q4_production_mode(config) == "llama"
  monkeypatch.setattr(model, "getenv", _env({"NV_COMPILER_Q4_IMMA_PP512":1,"NV_COMPILER_Q4_IMMA_K_PP512":1}))
  assert model._nv_q4_production_mode(config) == "compiler"

def test_compiler_gate_streamk_defaults_on_with_explicit_rollback(monkeypatch):
  config=SimpleNamespace()
  monkeypatch.setattr(model,"_nv_q4_imma_pp512_mode",lambda:"compiler")
  monkeypatch.setattr(model,"_nv_compiler_q4_imma_pp512_qualified",lambda _:True)
  monkeypatch.setattr(model,"getenv",_env({}))
  assert model._nv_compiler_q4_gate_streamk_enabled(config)
  monkeypatch.setattr(model,"getenv",_env({"NV_COMPILER_Q4_GATE_STREAMK":0}))
  assert not model._nv_compiler_q4_gate_streamk_enabled(config)
  monkeypatch.setattr(model,"_nv_q4_imma_pp512_mode",lambda:"llama")
  monkeypatch.setattr(model,"getenv",_env({}))
  assert not model._nv_compiler_q4_gate_streamk_enabled(config)

def test_q6_rollback_keeps_llama_down_lease(monkeypatch):
  config=SimpleNamespace()
  monkeypatch.setattr(model,"_nv_compiler_q4_imma_k_pp512_enabled",lambda _:True)
  monkeypatch.setattr(model,"_nv_llama_full_packed_pp512_enabled",lambda _:True)
  monkeypatch.setattr(model,"getenv",_env({"NV_COMPILER_Q6_IMMA_PP512":0}))
  assert model._nv_llama_packed_q6k_down_enabled(config)

def test_compiler_k_lease_is_closed_off_nv_and_on_llama_override(monkeypatch):
  config=SimpleNamespace(prefill_ubatch=512, num_blocks=36, dim=4096, hidden_dim=12288,
    n_heads=32, n_kv_heads=8, head_dim=128, num_experts=0)
  monkeypatch.setattr(model, "getenv", _env({}))
  monkeypatch.setattr(model, "Device", SimpleNamespace(DEFAULT="CPU"))
  assert not model._nv_compiler_q4_imma_k_pp512_enabled(config)
  monkeypatch.setattr(model, "getenv", _env({"NV_COMPILER_Q4_IMMA_PP512":1}))
  assert not model._nv_compiler_q4_imma_k_pp512_enabled(config)
  monkeypatch.setattr(model, "getenv", _env({"NV_LLAMA_PACKED_Q4K_PP512":1}))
  monkeypatch.setattr(model, "Device", SimpleNamespace(DEFAULT="NV"))
  assert not model._nv_compiler_q4_imma_k_pp512_enabled(config)

def test_ordinary_census_selects_concrete_capture():
  class J: pass
  m=J(); m.config=SimpleNamespace(prefill_v2=True); m.prefill_v2_jits={(0,True):"concrete"}; m.prefill_v2_greedy_jit="fallback"
  assert _ordinary_prefill_jit(m,0,True) == "concrete"
  assert _ordinary_prefill_jit(m,1,True) == "fallback"

def test_q4_down_streamk_context_uses_wide_down_geometry():
  ctx=q4_down_candidate_context(); g=ctx.schedule
  assert (g.m,g.n,g.k,g.tile_m,g.tile_n,g.tile_k,g.owners)==(512,4096,12288,128,128,64,170)
  assert g.output_tiles==128 and g.work_units==24576
  assert ctx.validate().partial_slots==340
  rows,active=q4_down_fixup_map()
  assert len(rows)==128 and len(active)==128 and {len(x) for x in rows}=={2,3}
  assert sum(len(x) for x in rows)==290 and max(max(x) for x in rows)<340

def test_q4_streamk_transform_accepts_down_emitted_abi_and_rejects_wrong_k():
  src='''extern "C" __global__ void __launch_bounds__(256) r(float* data0_2097152, unsigned int* data1_1966080, unsigned int* data2_7077888) {
  int gidx0 = blockIdx.x; /* 32 */
  int gidx1 = blockIdx.y; /* 4 */
  float buf0[64];
  (*(buf0+0)) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 192; Ridx0++) { (*(buf0+1)) = 0.0f; }
  int alu242 = 0;
  *((float2*)((data0_2097152+alu242))) = make_float2((*(buf0+0)),(*(buf0+32)));
}'''
  out=transform_compiler_q4k_to_streamk(src,tiles_n=32,k_blocks=192,output_stride=4096,kernel_name="q4_down_streamk")
  assert "q4_down_streamk" in out and "Ridx0 = k_begin; Ridx0 < k_end" in out and "partials+(slot*16384)" in out
  with pytest.raises(ValueError,match="K loop"):
    transform_compiler_q4k_to_streamk(src,tiles_n=32,k_blocks=64,output_stride=4096)
  fix=active_fixup_source(max_contributors=3)
  assert "map[3*tile+2]" in fix and "s2>=0" in fix

def test_q4_down_streamk_input_abi_orders_words_before_record():
  words=Tensor.empty(WORDS_U32,dtype=dtypes.uint32,device="CPU")
  record=Tensor.empty(RECORD_U32,dtype=dtypes.uint32,device="CPU")
  validate_streamk_inputs(words,record)
  with pytest.raises(ValueError,match="sizes mismatch"):
    validate_streamk_inputs(record,words)

def test_streamk_launch_preserves_canonical_argument_order():
  from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import _streamk_main
  from tinygrad.codegen.opt.packed_weight import PackedWeightTransform, Q8ActivationRecordTransform
  words=Tensor.empty(PackedWeightTransform("Q4_K",4096,12288).packed_bytes//4,dtype=dtypes.uint32,device="CPU")
  record=Tensor.empty(Q8ActivationRecordTransform(512,12288).storage_units,dtype=dtypes.uint32,device="CPU")
  partial, ids, main = object(), object(), object()
  class Output:
    def uop_program(self, *args, fxn):
      assert args == (partial, ids, words, record)
      assert fxn() is main
      return (self, *args)
  out=Output()
  assert _streamk_main(SimpleNamespace(main_program=main),out,partial,ids,words,record)[0] is out
  with pytest.raises(ValueError,match="sizes mismatch"):
    _streamk_main(SimpleNamespace(main_program=main),out,partial,ids,record,words)
