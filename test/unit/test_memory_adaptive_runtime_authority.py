import inspect, json
import pytest

import tinygrad.llm.memory_adaptive_authority as authority
from tinygrad.llm.memory_adaptive_authority import (_resolve_for_test, refresh_memory_adaptive_policy,
  resolve_memory_adaptive_policy)


def selected(policy, candidate="overlay"):
  result = {**policy, "selected_candidate_id": candidate}
  return {"decision": "SELECTED", "selected_candidate_id": candidate, "interrupted": False,
          "from_cache": False, "policy": result, "cache_record": {"result": result}}


def test_exact_hit_returns_policy():
  policy = {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "direct-packed-baseline"}
  assert _resolve_for_test("model.gguf", runner=lambda **_: selected(policy)) == selected(policy)


def test_stale_or_malformed_result_is_a_miss():
  assert _resolve_for_test("model.gguf", runner=lambda **_: {"decision": "SELECTED", "policy": None}) is None
  assert _resolve_for_test("model.gguf", runner=lambda **_: {"decision": "SELECTED", "policy": {}, "interrupted": False}) is None
  assert _resolve_for_test("model.gguf", runner=lambda **_: {"decision": "REFUSE"}) is None


def test_interruption_is_not_swallowed():
  def interrupted(**_): raise KeyboardInterrupt
  with pytest.raises(KeyboardInterrupt): _resolve_for_test("model.gguf", runner=interrupted)


def test_public_signature_has_only_model_path():
  assert list(inspect.signature(resolve_memory_adaptive_policy).parameters) == ["selected_model_source"]


def test_selected_envelope_is_not_unwrapped():
  result = selected({"strategy": "DIRECT_PACKED_FALLBACK"})
  resolved = _resolve_for_test("model.gguf", runner=lambda **_: result)
  assert resolved is not None
  assert resolved["decision"] == "SELECTED"
  assert resolved["policy"]["strategy"] == "DIRECT_PACKED_FALLBACK"


def test_interrupted_or_nonpersistent_selection_is_not_runtime_authority():
  result = selected({"strategy": "DIRECT_PACKED_FALLBACK"})
  assert _resolve_for_test("model.gguf", runner=lambda **_: {**result, "interrupted": True}) is None
  assert _resolve_for_test("model.gguf", runner=lambda **_: {**result, "cache_record": None}) is None


def test_normal_resolution_is_read_only_and_does_not_import_controller(tmp_path, monkeypatch):
  cache = tmp_path / "policy.json"
  record = selected({"strategy": "DIRECT_PACKED_FALLBACK"})["cache_record"]
  cache.write_text(json.dumps(record))
  monkeypatch.setattr(authority, "_cache_path", lambda _: cache)
  monkeypatch.setattr(authority, "_write", lambda *args: pytest.fail("normal resolution wrote cache"))
  resolved = resolve_memory_adaptive_policy("model.gguf")
  assert resolved is not None and resolved["from_cache"] is True
  assert cache.read_text() == json.dumps(record)


def test_explicit_refresh_runs_controller_and_persists(tmp_path, monkeypatch):
  cache = tmp_path / "policy.json"
  monkeypatch.setattr(authority, "_cache_path", lambda _: cache)
  result = selected({"strategy": "FULL_RESIDENT_OVERLAY"})
  monkeypatch.setattr("extra.qk.memory_adaptive_search_controller.run_controller", lambda **kwargs: result)
  assert refresh_memory_adaptive_policy("model.gguf") == result
  assert json.loads(cache.read_text()) == result["cache_record"]


def test_refresh_does_not_swallow_process_control(monkeypatch):
  def stop(**kwargs): raise SystemExit(7)
  monkeypatch.setattr("extra.qk.memory_adaptive_search_controller.run_controller", stop)
  with pytest.raises(SystemExit): refresh_memory_adaptive_policy("model.gguf")
