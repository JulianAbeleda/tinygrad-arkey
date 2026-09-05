from types import SimpleNamespace

import tinygrad.llm.model as model
import pytest
from extra.llm_research.prefill.nv_compiler_q6k_model_arm import _ordinary_prefill_jit, _captured_program_calls


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
