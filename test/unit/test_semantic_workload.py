import hashlib
import pytest
from extra.llm_research.semantic_workload import bind_exact_gguf_workload, load_exact_gguf_model, semantic_identity_to_workload

TARGET = {"backend":"METAL", "target_id":"apple_m4_10c", "resolved_target_hash":"b"*64}

def _identity(): return {"phase":"decode", "tensor_name":"blk.0.ffn_down.weight", "module_path":"blk.0.ffn_down", "role":"ffn_down",
  "logical_m":1, "logical_n":4096, "logical_k":12288, "source_quant_storage":"Q6_K", "source_layout":"gguf_packed_row_major",
  "module_representation":"nn_linear", "input_dtype":"float16", "output_dtype":"float16", "accumulator_dtype":"float32"}

def test_semantic_identity_serializes_exact_workload_without_fixture_substitution():
  out = semantic_identity_to_workload(_identity(), model_hash="a"*64, target=TARGET, tolerance={"atol":1e-3,"rtol":1e-3})
  assert out["shape"] == {"m":1,"n":4096,"k":12288} and out["fixture_shape_substitution"] == "forbidden"
  assert out["semantic_identity"]["module_path"] == "blk.0.ffn_down"
  assert out["semantic_identity"]["module_representation"] == "nn_linear"

@pytest.mark.parametrize("change", [{"metadata_status":"unavailable"}, {"logical_k":0}, {"source_quant_storage":"F32"}])
def test_semantic_workload_rejects_unavailable_or_ambiguous_identity(change):
  identity = _identity(); identity.update(change)
  with pytest.raises(ValueError): semantic_identity_to_workload(identity, model_hash="a"*64, target=TARGET, tolerance={"atol":1e-3,"rtol":1e-3})

def test_exact_binding_streams_hash_validates_k_and_reports_packed_bytes(monkeypatch, tmp_path):
  path = tmp_path / "tiny.gguf"; path.write_bytes(b"abc" * 100)
  workload = semantic_identity_to_workload(_identity(), model_hash=hashlib.sha256(path.read_bytes()).hexdigest(), target=TARGET, tolerance={"atol":1e-3,"rtol":1e-3})
  class Weight: shape = (4096, 12288)
  import tinygrad.llm.gguf as gguf
  monkeypatch.setattr(gguf, "gguf_load_with_metadata", lambda _path: ({}, {"blk.0.ffn_down.weight":Weight()},
    {"tensor_infos":[("blk.0.ffn_down.weight", (12288,4096), 14, 0)]}))
  binding = bind_exact_gguf_workload(workload, path)
  assert binding.packed_source_bytes == 4096 * 12288 // 256 * 210
  workload["shape"]["k"] = 7
  with pytest.raises(ValueError, match="divisible"): bind_exact_gguf_workload(workload, path)

def test_verified_model_can_bind_repeated_tensors_without_reloading(monkeypatch, tmp_path):
  path = tmp_path / "tiny.gguf"; path.write_bytes(b"abc")
  workload = semantic_identity_to_workload(_identity(), model_hash=hashlib.sha256(b"abc").hexdigest(), target=TARGET, tolerance={"atol":1e-3,"rtol":1e-3})
  class Weight: shape = (4096, 12288)
  import tinygrad.llm.gguf as gguf
  calls = []
  monkeypatch.setattr(gguf, "gguf_load_with_metadata", lambda _path: (calls.append(_path) or ({}, {"blk.0.ffn_down.weight":Weight()}, {"tensor_infos":[("blk.0.ffn_down.weight", (), 14, 0)]})))
  model = load_exact_gguf_model(workload, path)
  bind_exact_gguf_workload(workload, path, model=model); bind_exact_gguf_workload(workload, path, model=model)
  assert len(calls) == 1

@pytest.mark.parametrize("model_hash,target", [
  ("A"*64, TARGET), ("a"*63, TARGET), ("a"*64, {"backend":"METAL","target_id":"apple_m4_10c"}),
  ("a"*64, {"backend":"METAL","target_id":"apple_m4_10c","resolved_target_hash":"Z"*64})])
def test_semantic_workload_rejects_noncanonical_model_or_target_identity(model_hash, target):
  with pytest.raises(ValueError):
    semantic_identity_to_workload(_identity(), model_hash=model_hash, target=target, tolerance={"atol":1e-3,"rtol":1e-3})
