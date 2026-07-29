import json

import pytest

from tinygrad.llm import runtime_state


class _Model:
  max_context = 32
  config = type("Config", (), {"prefill_v2": False, "prefill_concrete_kv": False})()
  _cached_tokens = [1, 2]

  def reset_generation_state(self): self.reset = True


@pytest.fixture(autouse=True)
def no_device_probe(monkeypatch):
  monkeypatch.setattr(runtime_state, "_device_target", lambda: "test-target")


def test_cli_compatibility_reexports_runtime_owner():
  from tinygrad.llm import cli
  for name in ("SimpleTokenizer", "models", "_quant_from_name", "DEFAULT_REGISTRY_PATH",
               "build_registry", "RuntimeFault", "RuntimeState"):
    assert getattr(cli, name) is getattr(runtime_state, name)
  # Other tests may have imported cli before the autouse device-probe monkeypatch.
  assert callable(cli._device_target)


def test_registry_merges_file_and_derives_quant(tmp_path):
  path = tmp_path / "models.json"
  path.write_text(json.dumps({"models": [{"id": "builtin", "path": "custom-Q4_K_M.gguf"},
                                           {"id": "local", "path": "local-F16.gguf"}]}))
  registry = runtime_state.build_registry({"builtin": "builtin-Q6_K.gguf"}, path)
  assert registry["builtin"]["path"] == "custom-Q4_K_M.gguf"
  assert registry["builtin"]["quant"] == "Q6_K"
  assert registry["local"]["quant"] == "F16"
  assert registry["local"]["target"] == "test-target"


def test_tokenizer_builds_from_gguf_metadata_after_extraction():
  tok = runtime_state.SimpleTokenizer.from_gguf_kv({
    "tokenizer.ggml.tokens": ["a", "<eos>"], "tokenizer.ggml.token_type": [1, 2],
    "tokenizer.ggml.pre": "llama3", "tokenizer.ggml.eos_token_id": 1})
  assert tok.encode("a") == [0]
  assert tok.is_end(1)


def test_adopt_status_cache_and_unload(monkeypatch):
  state = runtime_state.RuntimeState({"model": {"id": "model", "status": "available"}})
  model = _Model()
  state.adopt(model, {"general.architecture": "llama"}, object(), "model", "Model", "model-Q4_K_M.gguf")
  assert state.status()["loaded"] is True
  assert state.status()["quant"] == "Q4_K_M"
  assert state.clear_prefix_cache() == {"prefix_cache_cleared": True, "cleared_tokens": 2}
  monkeypatch.setattr(runtime_state.gc, "collect", lambda: 0)
  assert state.unload() == {"loaded": False, "unloaded": "model"}
  assert state.registry["model"]["status"] == "available"


def test_missing_model_is_a_runtime_fault():
  state = runtime_state.RuntimeState({})
  with pytest.raises(runtime_state.RuntimeFault, match="not in the registry") as exc:
    state.load("missing", warmup=False)
  assert exc.value.err_type == "unknown_model"
