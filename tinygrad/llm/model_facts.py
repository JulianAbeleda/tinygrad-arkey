from __future__ import annotations

from dataclasses import dataclass
from tinygrad.helpers import Metadata
from tinygrad.dtype import DType, least_upper_dtype, sum_acc_dtype
from typing import Any


QK_ROUTE_ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head")
_ROUTE_ROLE_ALIASES = {
  "ffn_gate": "ffn_gate_up", "ffn_up": "ffn_gate_up", "ffn_gate_up": "ffn_gate_up",
  "ffn_gate_shexp": "ffn_gate_up", "ffn_up_shexp": "ffn_gate_up",
  "ffn_down": "ffn_down", "ffn_down_shexp": "ffn_down",
  "attn_q": "attn_qo", "attn_output": "attn_qo", "attn_qo": "attn_qo",
  "attn_k": "attn_kv", "attn_v": "attn_kv", "attn_kv": "attn_kv",
  "output": "lm_head", "lm_head": "lm_head",
}
GGML_QUANT_LABELS = {
  12: "Q4_K",
  14: "Q6_K",
}

@dataclass(frozen=True)
class ProgramIdentityMetadata(Metadata):
  phase: str = ""; tensor_name: str = ""; module_path: str = ""; role: str = ""
  logical_m: int = 0; logical_n: int = 0; logical_k: int = 0
  source_quant_storage: str = ""; source_layout: str = ""; module_representation: str = ""
  input_dtype: str = ""; output_dtype: str = ""; accumulator_dtype: str|None = None
  def __post_init__(self):
    if self.phase not in ("decode", "prefill") or self.role != normalize_route_role(self.role) or min(self.logical_m, self.logical_n, self.logical_k) < 1:
      raise ValueError("invalid program semantic identity")

def program_identity_factory(fact: "TensorFact", *, module_representation:str, source_layout:str|None=None):
  if fact.role is None: return None
  source_layout = source_layout or ("gguf_packed_row_major" if fact.quant_label in ("Q4_K", "Q6_K") else "dense_row_major")
  def factory(_module, x):
    # QK adapters can dispatch either their generated float32 route or the normal Tensor fallback.
    # Until that choice is observed at the KernelProgram boundary, their result dtype is not factual.
    if module_representation != "nn_linear": return None
    if len(x.shape) < 2 or not isinstance(x.shape[-2], int) or x.shape[-2] < 1: return None
    weight_dtype, input_dtype = getattr(getattr(_module, "weight", None), "dtype", None), getattr(x, "dtype", None)
    if not isinstance(weight_dtype, DType) or not isinstance(input_dtype, DType): return None
    phase = getattr(_module, "_program_phase", None)
    if phase not in ("decode", "prefill"): return None
    product_dtype = least_upper_dtype(input_dtype, weight_dtype)
    bias_dtype = getattr(getattr(_module, "bias", None), "dtype", None)
    output_dtype = least_upper_dtype(product_dtype, bias_dtype) if isinstance(bias_dtype, DType) else product_dtype
    return ProgramIdentityMetadata(name=fact.role, caller="", phase=phase, tensor_name=fact.name,
      module_path=fact.module_path, role=fact.role, logical_m=x.shape[-2], logical_n=fact.rows, logical_k=fact.cols,
      source_quant_storage=fact.quant_label, source_layout=source_layout, module_representation=module_representation,
      input_dtype=str(input_dtype), output_dtype=str(output_dtype), accumulator_dtype=str(sum_acc_dtype(product_dtype)))
  return factory


def attach_program_identity_metadata(root: Any, facts: tuple["TensorFact", ...], *, primitive_linears: list[Any], module_at) -> tuple[tuple[str, Any], ...]:
  """Attach source-fact metadata to runtime modules without touching model state tensors."""
  primitive_ids = {id(linear) for linear in primitive_linears}
  attached = []
  for fact in facts:
    if fact.role is None: continue
    try: linear = module_at(root, fact.module_path)
    except (AttributeError, IndexError, ValueError): continue
    if not hasattr(linear, "__call__"): continue
    linear._program_tensor_fact = fact
    representation = "qk_primitive_adapter" if id(linear) in primitive_ids else "nn_linear"
    linear.call_metadata_factory = program_identity_factory(fact, module_representation=representation)
    attached.append((fact.module_path, linear))
  return tuple(attached)


def normalize_route_role(role_or_name: str) -> str:
  """Normalize a route role or tensor/module name to the production role vocabulary."""
  value = str(role_or_name or "")
  leaf = value[:-len(".weight")] if value.endswith(".weight") else value
  leaf = leaf.rsplit(".", 1)[-1]
  return _ROUTE_ROLE_ALIASES.get(leaf, value)


def packed_linear_quant(linear: Any) -> str:
  """Return the packed quant family carried by a runtime linear, if any."""
  if not hasattr(linear, "prefill_packed_weight"): return ""
  if hasattr(linear, "q4k_storage"): return "Q4_K"
  if hasattr(linear, "q6k_storage"): return "Q6_K"
  return ""


def route_role_for_linear(linear: Any) -> str:
  """Resolve a runtime linear's attached role with name fallback for compatibility."""
  for attr in ("_prefill_graph_role", "route_role", "role"):
    value = str(getattr(linear, attr, "") or "")
    if value: return normalize_route_role(value)
  role = normalize_route_role(str(getattr(linear, "name", "") or ""))
  return role if role in QK_ROUTE_ROLES else ""


@dataclass(frozen=True)
class TensorFact:
  name:str; module_path:str; ggml_type:int; rows:int; cols:int; quant_label:str; role:str|None

  @property
  def shape(self) -> tuple[int, int]: return (self.rows, self.cols)

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "module_path": self.module_path, "ggml_type": self.ggml_type,
            "rows": self.rows, "cols": self.cols, "quant_label": self.quant_label, "role": self.role}


@dataclass(frozen=True)
class ModelFacts:
  architecture:str; hidden_size:int|None; intermediate_size:int|None; n_heads:int|None
  n_kv_heads:int|None; head_dim:int|None; tensors:tuple[TensorFact, ...]

  def tensors_for_role(self, role:str) -> tuple[TensorFact, ...]: return tuple(t for t in self.tensors if t.role == role)

  def to_json(self) -> dict[str, Any]:
    return {"architecture": self.architecture, "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size, "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
            "tensors": [t.to_json() for t in self.tensors]}


@dataclass(frozen=True)
class QwenDenseRoleResolver:
  architecture:str; hidden_size:int|None; intermediate_size:int|None; n_heads:int|None; n_kv_heads:int|None; head_dim:int|None

  @classmethod
  def from_kv(cls, kv: dict[str, Any]) -> "QwenDenseRoleResolver":
    arch = str(kv.get("general.architecture", ""))
    hidden_size = _int_or_none(kv.get(f"{arch}.embedding_length"))
    n_heads = _int_or_none(kv.get(f"{arch}.attention.head_count"))
    n_kv_heads = _int_or_none(kv.get(f"{arch}.attention.head_count_kv", n_heads))
    head_dim = _int_or_none(kv.get(f"{arch}.attention.key_length"))
    if head_dim is None and hidden_size is not None and n_heads: head_dim = hidden_size // n_heads
    intermediate_size = _int_or_none(kv.get(f"{arch}.feed_forward_length"))
    return cls(arch, hidden_size, intermediate_size, n_heads, n_kv_heads, head_dim)

  @property
  def kv_size(self) -> int|None: return None if self.n_kv_heads is None or self.head_dim is None else self.n_kv_heads * self.head_dim

  @property
  def q_size(self) -> int|None: return None if self.n_heads is None or self.head_dim is None else self.n_heads * self.head_dim

  def resolve(self, name: str, rows: int, cols: int) -> str | None:
    if not self.architecture.startswith("qwen"): return None
    leaf = name.rsplit(".", 2)[-2:] if "." in name else [name]
    suffix = ".".join(leaf)
    if suffix in ("ffn_gate.weight", "ffn_up.weight"): return self._if_shape(rows, cols, self.intermediate_size, self.hidden_size, "ffn_gate_up")
    if suffix == "ffn_down.weight": return self._if_shape(rows, cols, self.hidden_size, self.intermediate_size, "ffn_down")
    if suffix == "attn_q.weight": return self._if_shape(rows, cols, self.q_size, self.hidden_size, "attn_qo")
    if suffix == "attn_output.weight": return self._if_shape(rows, cols, self.hidden_size, self.q_size, "attn_qo")
    if suffix in ("attn_k.weight", "attn_v.weight"): return self._if_shape(rows, cols, self.kv_size, self.hidden_size, "attn_kv")
    if suffix == "output.weight" or name.endswith("lm_head.weight"): return self._if_shape(rows, cols, None, self.hidden_size, "lm_head")
    return None

  @staticmethod
  def _if_shape(rows: int, cols: int, expected_rows: int | None, expected_cols: int | None, role: str) -> str | None:
    if expected_rows is not None and rows != expected_rows: return None
    if expected_cols is not None and cols != expected_cols: return None
    return role


def model_facts_from_gguf_metadata(kv: dict[str, Any], meta: dict[str, Any]) -> ModelFacts:
  resolver = QwenDenseRoleResolver.from_kv(kv)
  tensors = tuple(tensor_fact_from_gguf_row(row, resolver) for row in meta.get("tensor_infos", ())
                  if _is_route_weight_row(row))
  return ModelFacts(resolver.architecture, resolver.hidden_size, resolver.intermediate_size,
                    resolver.n_heads, resolver.n_kv_heads, resolver.head_dim, tensors)


def tensor_fact_from_gguf_row(row: Any, resolver: QwenDenseRoleResolver | None = None) -> TensorFact:
  name, dims, ggml_type = _normalize_tensor_info(row)
  rows, cols = _matrix_rows_cols(dims)
  module_path = _module_path(name)
  quant_label = GGML_QUANT_LABELS.get(ggml_type, f"GGML_TYPE_{ggml_type}")
  role = resolver.resolve(name, rows, cols) if resolver is not None else None
  return TensorFact(name, module_path, ggml_type, rows, cols, quant_label, role)


def _normalize_tensor_info(row: Any) -> tuple[str, tuple[int, ...], int]:
  if isinstance(row, dict):
    name, dims, ggml_type = row["name"], row.get("dims", row.get("shape")), row.get("type", row.get("ggml_type"))
  else:
    name, dims, ggml_type = row[:3]
  return str(name), tuple(int(x) for x in dims), int(ggml_type)


def _matrix_rows_cols(dims: tuple[int, ...]) -> tuple[int, int]:
  if len(dims) == 0: return (1, 1)
  if len(dims) == 1: return (int(dims[0]), 1)
  return (int(dims[1]), int(dims[0]))


def _module_path(name: str) -> str:
  return name[:-7] if name.endswith(".weight") else name.rsplit(".", 1)[0] if "." in name else name

def _is_route_weight_row(row: Any) -> bool:
  name, dims, _ggml_type = _normalize_tensor_info(row)
  return name.endswith(".weight") and len(dims) == 2


def _int_or_none(value: Any) -> int | None:
  return None if value is None else int(value)
