from __future__ import annotations

from dataclasses import dataclass
from tinygrad.helpers import Metadata
from tinygrad.dtype import dtypes
from typing import Any
from tinygrad.engine.metadata import bind_buffer_metadata, bind_buffer_metadata_region, buffer_byte_length, buffer_metadata, register_call_metadata_resolver
from tinygrad.llm.qk_layout import Q4_K, Q6_K, QUANT_FORMATS, QuantFormat
from tinygrad.llm.roles import DENSE_PROJECTION_ROLES, normalize_program_role


QK_ROUTE_ROLES = DENSE_PROJECTION_ROLES
GGML_QUANT_LABELS = {
  12: Q4_K.name,
  14: Q6_K.name,
}
PROGRAM_SOURCE_LAYOUTS = ("gguf_packed_row_major",)
PROGRAM_MODULE_REPRESENTATIONS = ("nn_linear", "qk_primitive_adapter")
PROGRAM_DTYPES = ("float16", "float32", "bfloat16", str(dtypes.half), str(dtypes.float), str(dtypes.bfloat16))

@dataclass(frozen=True)
class ProgramIdentityMetadata(Metadata):
  phase: str = ""; tensor_name: str = ""; module_path: str = ""; role: str = ""
  logical_m: int = 0; logical_n: int = 0; logical_k: int = 0
  source_quant_storage: str = ""; source_layout: str = ""; module_representation: str = ""
  input_dtype: str = ""; output_dtype: str = ""; accumulator_dtype: str|None = None
  def __post_init__(self):
    if self.phase not in ("decode", "prefill") or self.role not in QK_ROUTE_ROLES or \
       self.role != normalize_route_role(self.role) or min(self.logical_m, self.logical_n, self.logical_k) < 1 or \
       self.source_quant_storage not in GGML_QUANT_LABELS.values() or self.source_layout not in PROGRAM_SOURCE_LAYOUTS or \
       self.module_representation not in PROGRAM_MODULE_REPRESENTATIONS or self.input_dtype not in PROGRAM_DTYPES or \
       self.output_dtype not in PROGRAM_DTYPES or self.accumulator_dtype is not None and self.accumulator_dtype not in PROGRAM_DTYPES:
      raise ValueError("invalid program semantic identity")

def attach_program_identity_metadata(root: Any, facts: tuple["TensorFact", ...], *, primitive_linears: list[Any], module_at) -> tuple[tuple[str, Any], ...]:
  """Attach source-fact metadata to runtime modules without touching model state tensors."""
  attached = []
  for fact in facts:
    if fact.role is None: continue
    try: linear = module_at(root, fact.module_path)
    except (AttributeError, IndexError, ValueError): continue
    if not hasattr(linear, "__call__"): continue
    linear._program_tensor_fact = fact
    linear.call_metadata_binding = bind_module_program_tensor_facts
    bind_module_program_tensor_facts(linear)
    attached.append((fact.module_path, linear))
  return tuple(attached)


def normalize_route_role(role_or_name: str) -> str:
  """Normalize a route role or tensor/module name to the production role vocabulary."""
  return normalize_program_role(role_or_name)


# One authority for which linears carry a resident-fp16 overlay weight: the raw module attribute names the
# model walk covers. The canonical role set is derived from the name->role alias table, so the walk and the
# inventory byte estimate cannot drift apart (scope 5.1 / R4).
PREFILL_OVERLAY_LINEAR_NAMES = ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp",
                                "ffn_down_shexp", "attn_q", "attn_k", "attn_v", "attn_output")
PREFILL_OVERLAY_ROLES = frozenset(normalize_route_role(name) for name in PREFILL_OVERLAY_LINEAR_NAMES)


def is_prefill_overlay_role(name_or_role: str) -> bool:
  """True when a linear name or canonical role is covered by the resident-fp16 overlay."""
  return normalize_route_role(name_or_role) in PREFILL_OVERLAY_ROLES


def estimate_prefill_overlay_bytes(names_and_numels) -> int:
  """fp16 bytes (2/elt) for overlay-covered weights from (name, numel) pairs."""
  return sum(numel * 2 for name, numel in names_and_numels if is_prefill_overlay_role(name))


def packed_linear_quant(linear: Any) -> QuantFormat | None:
  """Return the typed packed quant format carried by a runtime linear, if any."""
  if not hasattr(linear, "prefill_packed_weight"): return None
  if hasattr(linear, "q4k_storage"): return Q4_K
  if hasattr(linear, "q6k_storage"): return Q6_K
  return None


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
class ProgramTensorFact:
  """Immutable source fact attached to a concrete weight allocation out-of-band."""
  fact: TensorFact
  alias: str
  def __post_init__(self):
    if self.alias not in ("weight", "packed"): raise ValueError("program tensor alias must be weight or packed")


def bind_program_tensor_fact(value:Any, fact:TensorFact, *, alias:str) -> None:
  """Associate an immutable GGUF fact with an existing weight/packed buffer."""
  bind_buffer_metadata(value, ProgramTensorFact(fact, alias))

def program_tensor_facts(value:Any) -> tuple[ProgramTensorFact, ...]:
  """Resolve exact facts through stable buffer aliases and offset views."""
  return tuple(item for item in buffer_metadata(value) if isinstance(item, ProgramTensorFact))

def bind_module_program_tensor_facts(module:Any) -> None:
  """Refresh bindings from a module's current post-load/post-realize tensors."""
  fact = getattr(module, "_program_tensor_fact", None)
  if not isinstance(fact, TensorFact): return
  if (weight := getattr(module, "weight", None)) is not None: bind_program_tensor_fact(weight, fact, alias="weight")
  for storage_name, tensor_name in (("q4k_storage", "words"), ("q6k_storage", "halfs")):
    storage = getattr(module, storage_name, None)
    if storage is not None and (packed := getattr(storage, tensor_name, None)) is not None:
      bind_program_tensor_fact(packed, fact, alias="packed")
  for name in ("_prefill_q4k_words", "_prefill_q6k_halfs"):
    if (packed := getattr(module, name, None)) is not None: bind_program_tensor_fact(packed, fact, alias="packed")

def bind_gguf_program_tensor_facts(meta:dict, facts:tuple[TensorFact, ...]) -> tuple[str, ...]:
  """Bind generic dequant routes to their exact source payload intervals."""
  from tinygrad.llm.gguf_memory_scan import gguf_tensor_spans
  raw = meta.get("raw_tensor")
  if raw is None: return ()
  try: file_size = int(raw.uop.nbytes())
  except (AttributeError, TypeError, ValueError): return ()
  by_name = {fact.name: fact for fact in facts if fact.role is not None}
  bound = []
  for span in gguf_tensor_spans(meta, file_size):
    fact = by_name.get(span.name)
    if fact is None or span.payload_bytes is None: continue
    if bind_buffer_metadata_region(raw, span.absolute_offset, span.payload_bytes, ProgramTensorFact(fact, "weight")): bound.append(span.name)
  return tuple(bound)

def program_identities_from_call(call:Any) -> tuple[ProgramIdentityMetadata, ...]:
  """Derive identities only from registered inputs of the concrete selected program."""
  from tinygrad.uop.ops import Ops
  if getattr(call, "op", None) is not Ops.CALL or not getattr(call, "src", ()): return ()
  program = call.src[0]
  out_slots = tuple(getattr(getattr(program, "arg", None), "outs", ()))
  in_slots = tuple(getattr(getattr(program, "arg", None), "ins", ()))
  if not out_slots: return ()
  try:
    slots = tuple(int(slot) for slot in in_slots)
    output_slots = tuple(int(slot) for slot in out_slots)
  except (TypeError, ValueError): return ()
  if any(slot < 0 or slot+1 >= len(call.src) for slot in (*output_slots, *slots)): return ()
  input_args = tuple(call.src[slot+1] for slot in slots)
  observed = []
  for arg in input_args:
    for binding in program_tensor_facts(arg):
      if binding not in observed: observed.append(binding)
  if not observed: return ()

  def dtype_name(arg):
    dtype = getattr(arg, "dtype", None)
    return str(getattr(dtype, "base", dtype))
  def elements(arg):
    try:
      byte_length, itemsize = buffer_byte_length(arg), arg.dtype.itemsize
      return 0 if byte_length is None or byte_length % itemsize else byte_length // itemsize
    except (AttributeError, RuntimeError, TypeError, ValueError): return 0
  output_dtypes = {dtype_name(call.src[slot+1]) for slot in output_slots}
  if len(output_dtypes) != 1: return ()
  output_dtype = next(iter(output_dtypes))
  value_args = tuple(call.src[slot+1] for slot in slots if slot not in output_slots and not program_tensor_facts(call.src[slot+1]))
  identities = []
  for binding in observed:
    fact = binding.fact
    if fact.role is None or fact.quant_label not in QUANT_FORMATS: continue
    candidates = tuple(arg for arg in value_args if elements(arg) >= fact.cols and elements(arg) % fact.cols == 0)
    logical_ms = {elements(arg) // fact.cols for arg in candidates}
    if len(logical_ms) != 1: continue
    logical_m = next(iter(logical_ms))
    activation = next(arg for arg in candidates if elements(arg) // fact.cols == logical_m)
    phase = "decode" if logical_m == 1 else "prefill"
    identity = ProgramIdentityMetadata(name=fact.role, caller="", phase=phase, tensor_name=fact.name,
      module_path=fact.module_path, role=fact.role, logical_m=logical_m, logical_n=fact.rows, logical_k=fact.cols,
      source_quant_storage=fact.quant_label, source_layout="gguf_packed_row_major",
      module_representation="qk_primitive_adapter" if binding.alias == "packed" else "nn_linear",
      input_dtype=dtype_name(activation), output_dtype=output_dtype, accumulator_dtype=str(dtypes.float32))
    if identity not in identities: identities.append(identity)
  return tuple(identities)

register_call_metadata_resolver(program_identities_from_call)


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
  shared_expert_size:int|None = None

  @classmethod
  def from_kv(cls, kv: dict[str, Any]) -> "QwenDenseRoleResolver":
    arch = str(kv.get("general.architecture", ""))
    hidden_size = _int_or_none(kv.get(f"{arch}.embedding_length"))
    n_heads = _int_or_none(kv.get(f"{arch}.attention.head_count"))
    n_kv_heads = _int_or_none(kv.get(f"{arch}.attention.head_count_kv", n_heads))
    head_dim = _int_or_none(kv.get(f"{arch}.attention.key_length"))
    if head_dim is None and hidden_size is not None and n_heads: head_dim = hidden_size // n_heads
    intermediate_size = _int_or_none(kv.get(f"{arch}.feed_forward_length"))
    shared_expert_size = _int_or_none(kv.get(f"{arch}.expert_shared_feed_forward_length"))
    return cls(arch, hidden_size, intermediate_size, n_heads, n_kv_heads, head_dim, shared_expert_size)

  @property
  def kv_size(self) -> int|None: return None if self.n_kv_heads is None or self.head_dim is None else self.n_kv_heads * self.head_dim

  @property
  def q_size(self) -> int|None: return None if self.n_heads is None or self.head_dim is None else self.n_heads * self.head_dim

  def resolve(self, name: str, rows: int, cols: int) -> str | None:
    if not self.architecture.startswith("qwen"): return None
    leaf = name.rsplit(".", 2)[-2:] if "." in name else [name]
    suffix = ".".join(leaf)
    if suffix in ("ffn_gate.weight", "ffn_up.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
      # The shared-expert gate/up projections carry the shared expert size, not the dense FFN size, on MoE GGUFs.
      return self._if_shape(rows, cols, self.shared_expert_size if suffix.endswith("_shexp.weight") else self.intermediate_size,
                            self.hidden_size, "ffn_gate_up")
    if suffix in ("ffn_down.weight", "ffn_down_shexp.weight"):
      return self._if_shape(rows, cols, self.hidden_size,
                            self.shared_expert_size if suffix.endswith("_shexp.weight") else self.intermediate_size, "ffn_down")
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
