from types import SimpleNamespace

import tinygrad.llm.model as model


def _env(values):
  return lambda key, default=0: values.get(key, default)


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
