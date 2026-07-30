"""Pure bridge from a recorded semantic identity to an exact provider workload."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import hashlib, pathlib
from tinygrad.engine.metadata import PROGRAM_IDENTITY_FIELDS

@dataclass(frozen=True)
class ExactGGUFBinding:
  tensor_name: str; weight: Any; activation: Any; model_hash: str; packed_source_bytes: int

@dataclass(frozen=True)
class ExactGGUFModel:
  """Verified GGUF backing that can serve several exact tensor bindings."""
  model_hash: str; state: Mapping[str, Any]; metadata: Mapping[str, Any]

def load_exact_gguf_model(workload:Mapping[str, Any], gguf_path:str|pathlib.Path) -> ExactGGUFModel:
  """Stream-verify and load a GGUF once; caching belongs to the long-lived adapter."""
  path = pathlib.Path(gguf_path)
  if workload.get("fixture_shape_substitution") != "forbidden": raise ValueError("exact binding forbids fixture substitution")
  digest_state = hashlib.sha256()
  with path.open("rb") as source:
    for block in iter(lambda: source.read(1 << 20), b""): digest_state.update(block)
  digest = digest_state.hexdigest()
  if digest != workload.get("model_hash"): raise ValueError("GGUF content hash mismatch")
  from tinygrad.llm.gguf import gguf_load_with_metadata
  _kv, state, meta = gguf_load_with_metadata(path)
  return ExactGGUFModel(digest, state, meta)

def bind_exact_gguf_workload(workload:Mapping[str, Any], gguf_path:str|pathlib.Path,
                             *, model:ExactGGUFModel|None=None) -> ExactGGUFBinding:
  """Bind the recorded tensor without fixture resizing, reusing a verified model when supplied."""
  path = pathlib.Path(gguf_path)
  if workload.get("fixture_shape_substitution") != "forbidden": raise ValueError("exact binding forbids fixture substitution")
  loaded = load_exact_gguf_model(workload, path) if model is None else model
  if loaded.model_hash != workload.get("model_hash"): raise ValueError("cached GGUF content hash mismatch")
  identity, shape = workload["semantic_identity"], workload["shape"]
  name = identity["tensor_name"]
  if name not in loaded.state: raise ValueError("recorded GGUF tensor is absent")
  row = next((row for row in loaded.metadata["tensor_infos"] if row[0] == name), None)
  if row is None or int(row[2]) not in (12, 14): raise ValueError("recorded tensor is not Q4_K/Q6_K")
  if shape["k"] % 256: raise ValueError("exact packed K must be divisible by 256")
  if tuple(loaded.state[name].shape) != (shape["n"], shape["k"]): raise ValueError("recorded GGUF tensor shape mismatch")
  from tinygrad import Tensor, dtypes
  activation = Tensor.arange(shape["m"]*shape["k"], dtype=dtypes.float16).reshape(shape["m"], shape["k"])
  block_bytes = 144 if int(row[2]) == 12 else 210
  return ExactGGUFBinding(name, loaded.state[name], activation, loaded.model_hash, shape["n"]*shape["k"]//256*block_bytes)

def semantic_identity_to_workload(identity:Mapping[str, Any], *, model_hash:str, target:Mapping[str, Any], tolerance:Mapping[str, float]) -> dict[str, Any]:
  if not isinstance(identity, Mapping) or identity.get("metadata_status", "semantic") != "semantic":
    raise ValueError("semantic identity is unavailable")
  if any(key not in identity for key in PROGRAM_IDENTITY_FIELDS): raise ValueError("semantic identity is incomplete")
  if any(not isinstance(identity[key], int) or isinstance(identity[key], bool) or identity[key] < 1 for key in ("logical_m", "logical_n", "logical_k")):
    raise ValueError("semantic identity has invalid MNK")
  if not isinstance(model_hash, str) or len(model_hash) != 64 or any(char not in "0123456789abcdef" for char in model_hash):
    raise ValueError("model_hash must be a lowercase SHA-256")
  if not isinstance(target, Mapping) or not all(isinstance(target.get(key), str) and target[key] for key in ("backend", "target_id")) or \
     not isinstance(target.get("resolved_target_hash"), str) or len(target["resolved_target_hash"]) != 64 or \
     any(char not in "0123456789abcdef" for char in target["resolved_target_hash"]):
    raise ValueError("exact resolved target is required")
  if not isinstance(tolerance, Mapping) or not all(isinstance(tolerance.get(key), (int, float)) and not isinstance(tolerance[key], bool) and tolerance[key] >= 0 for key in ("atol", "rtol")):
    raise ValueError("exact workload requires non-negative atol/rtol")
  if identity["source_quant_storage"] not in ("Q4_K", "Q6_K"): raise ValueError("unsupported exact packed source")
  if not all(isinstance(identity[key], str) and identity[key] for key in
             ("tensor_name", "module_path", "role", "source_layout", "module_representation", "input_dtype", "output_dtype")):
    raise ValueError("semantic identity has ambiguous strings")
  semantic = {key:identity[key] for key in PROGRAM_IDENTITY_FIELDS}
  return {"schema":"tinygrad.semantic_provider_workload.v1", "model_hash":model_hash, "target":dict(target),
    "semantic_identity":semantic, "operation":"matmul", "shape":{"m":identity["logical_m"], "n":identity["logical_n"], "k":identity["logical_k"]},
    "operands":{"a":{"dtype":identity["input_dtype"]}, "b":{"quantization":identity["source_quant_storage"], "layout":identity["source_layout"]},
                "c":{"dtype":identity["output_dtype"]}}, "tolerance":{"atol":tolerance["atol"], "rtol":tolerance["rtol"]},
    "fixture_shape_substitution":"forbidden"}
