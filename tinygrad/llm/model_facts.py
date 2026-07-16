from __future__ import annotations

from dataclasses import dataclass
from typing import Any


QK_ROUTE_ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head")
GGML_QUANT_LABELS = {
  12: "Q4_K",
  14: "Q6_K",
}


@dataclass(frozen=True)
class TensorFact:
  name: str
  module_path: str
  ggml_type: int
  rows: int
  cols: int
  quant_label: str
  role: str | None

  @property
  def shape(self) -> tuple[int, int]:
    return (self.rows, self.cols)

  def to_json(self) -> dict[str, Any]:
    return {"name": self.name, "module_path": self.module_path, "ggml_type": self.ggml_type,
            "rows": self.rows, "cols": self.cols, "quant_label": self.quant_label, "role": self.role}


@dataclass(frozen=True)
class ModelFacts:
  architecture: str
  hidden_size: int | None
  intermediate_size: int | None
  n_heads: int | None
  n_kv_heads: int | None
  head_dim: int | None
  tensors: tuple[TensorFact, ...]

  def tensors_for_role(self, role: str) -> tuple[TensorFact, ...]:
    return tuple(t for t in self.tensors if t.role == role)

  def to_json(self) -> dict[str, Any]:
    return {"architecture": self.architecture, "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size, "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
            "tensors": [t.to_json() for t in self.tensors]}


@dataclass(frozen=True)
class QwenDenseRoleResolver:
  architecture: str
  hidden_size: int | None
  intermediate_size: int | None
  n_heads: int | None
  n_kv_heads: int | None
  head_dim: int | None

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
  def kv_size(self) -> int | None:
    if self.n_kv_heads is None or self.head_dim is None: return None
    return self.n_kv_heads * self.head_dim

  @property
  def q_size(self) -> int | None:
    if self.n_heads is None or self.head_dim is None: return None
    return self.n_heads * self.head_dim

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
