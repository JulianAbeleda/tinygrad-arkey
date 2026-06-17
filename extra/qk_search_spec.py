"""Schema authority for the bounded decode/context machine-search layer.

Single source of truth for the *vocabulary* and *record shapes* of the
table-driven search described in
`structure/Development/machine-search-decode-context-plan-2026-06-16.md`:

    search spec -> candidate generator -> isolated runner -> scorer -> accepted policy

This module is the scaffold (Phase 1): it owns the enums, the validated row /
accepted-policy dataclasses, the `assemble_search_row` SSOT constructor, and a
*read-only* adapter that maps an existing `qk_generated_policy` artifact into the
`AcceptedPolicy` shape (proving the schema models reality). It deliberately does
NOT run hardware, change runtime behaviour, or write into `bench/` artifacts.

Conventions mirror the existing codebase so this stays a thin authority, not a
rebuild:
- enums + `*_choices()` + `validate_*()` follow `extra/qk_modes.py`;
- `@dataclass(frozen=True)` with raise-on-invalid validation follows `QKConfig`
  (`tinygrad/llm/model.py`);
- `assemble_search_row(**validated) -> dict` follows `assemble_row`
  (`extra/qk_flywheel_dataset.py`) and is the "new experiment = new ROW, not a
  new script" entry point the anti-re-sprawl rule requires;
- all IO routes through `extra/llm_eval_common.py` (no new IO code).

Baseline note: the canonical llama.cpp / HBM-peak numbers currently DISAGREE
across the repo (`qk_experiment_matrix.LLAMA_REFS` 8B=101.2 vs docs 105.7;
`qk_bandwidth_roofline.DEFAULT_PEAK_MEM_GBS` 960 vs capstone 859). `baseline()`
imports the existing constants rather than re-typing them; reconciling the
numbers is a separate follow-up so it does not perturb existing matrix goldens.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from extra.llm_eval_common import read_id_jsonl, read_json_object, write_json, write_jsonl


class Phase(str, Enum):
  """Optimization phase — orthogonal search spaces (do not merge; see non-goals)."""
  DECODE = "decode"
  LONG_CONTEXT_DECODE = "long_context_decode"
  PREFILL = "prefill"


class Model(str, Enum):
  """Search target model."""
  QWEN3_8B = "qwen3_8b"
  QWEN3_14B = "qwen3_14b"
  QWEN3_32B = "qwen3_32b"


class OpScope(str, Enum):
  """The op a search row targets."""
  Q4K_GEMV = "q4k_gemv"
  Q6K_GEMV = "q6k_gemv"
  ATTENTION = "attention"
  FFN_DOWN = "ffn_down"
  LM_HEAD = "lm_head"
  SCHEDULER = "scheduler"


class SearchSpace(str, Enum):
  """The lever a search row explores."""
  PRIMITIVE_POLICY = "primitive_policy"
  DEMOTION = "demotion"
  FLASH_THRESHOLD = "flash_threshold"
  FLASH_VARIANT = "flash_variant"   # decode flash-attention primitive family: {v1, hoisted} x KV-split L
  STORAGE = "storage"
  SCHEDULE = "schedule"
  LDS_BLOCKING = "lds_blocking"


class Objective(str, Enum):
  """What a search row maximizes/minimizes."""
  TOK_S = "tok_s"
  HBM_PCT = "hbm_pct"
  SERVING_LATENCY = "serving_latency"


# The fork is AMD-only; backend is a schema field so the invariant is explicit
# (a non-AMD backend on a QK primitive path is rejected at the runtime gate too).
BACKENDS: frozenset[str] = frozenset({"AMD"})

# qwen3_8b -> "8B": bridges Model values to the existing LLAMA_REFS / model-bytes keys.
_MODEL_SIZE_KEY: dict[str, str] = {Model.QWEN3_8B.value: "8B", Model.QWEN3_14B.value: "14B", Model.QWEN3_32B.value: "32B"}


def phase_choices() -> tuple[str, ...]:
  return tuple(p.value for p in Phase)


def model_choices() -> tuple[str, ...]:
  return tuple(m.value for m in Model)


def op_scope_choices() -> tuple[str, ...]:
  return tuple(s.value for s in OpScope)


def search_space_choices() -> tuple[str, ...]:
  return tuple(s.value for s in SearchSpace)


def objective_choices() -> tuple[str, ...]:
  return tuple(o.value for o in Objective)


def backend_choices() -> tuple[str, ...]:
  return tuple(sorted(BACKENDS))


def _validate(enum_cls:type[Enum], value:str, label:str) -> str:
  """Return `value` if it is a valid member of `enum_cls`, else raise ValueError."""
  try:
    return enum_cls(value).value
  except ValueError:
    raise ValueError(f"unknown {label} {value!r}; expected one of {sorted(m.value for m in enum_cls)}")


def validate_phase(value:str) -> str: return _validate(Phase, value, "phase")
def validate_model(value:str) -> str: return _validate(Model, value, "model")
def validate_op_scope(value:str) -> str: return _validate(OpScope, value, "op_scope")
def validate_search_space(value:str) -> str: return _validate(SearchSpace, value, "search_space")
def validate_objective(value:str) -> str: return _validate(Objective, value, "objective")


def validate_backend(value:str) -> str:
  if value not in BACKENDS: raise ValueError(f"unknown backend {value!r}; expected one of {sorted(BACKENDS)}")
  return value


def model_size_key(model:str) -> str:
  """Map a Model value (`qwen3_8b`) to the size key (`8B`) used by LLAMA_REFS / model bytes."""
  if model not in _MODEL_SIZE_KEY: raise ValueError(f"unknown model {model!r}; expected one of {sorted(_MODEL_SIZE_KEY)}")
  return _MODEL_SIZE_KEY[model]


def _validate_ctx_range(ctx_range:tuple[int, int]) -> tuple[int, int]:
  if not (isinstance(ctx_range, (tuple, list)) and len(ctx_range) == 2):
    raise ValueError(f"ctx_range must be a 2-tuple, got {ctx_range!r}")
  lo, hi = ctx_range
  if not (isinstance(lo, int) and isinstance(hi, int)) or isinstance(lo, bool) or isinstance(hi, bool):
    raise ValueError(f"ctx_range bounds must be ints, got {ctx_range!r}")
  if lo < 1: raise ValueError(f"ctx_range lower bound must be >= 1, got {lo}")
  if hi < lo: raise ValueError(f"ctx_range must be non-decreasing, got [{lo}, {hi}]")
  return (lo, hi)


@dataclass(frozen=True)
class Constraints:
  """Bounds a search row must respect — encoded as data, validated on construction."""
  exact_required: bool = True
  dnll_epsilon: float = 0.0
  max_storage_mb: int | None = None
  ctx_range: tuple[int, int] = (1, 4096)
  no_beam_remote: bool = True

  def __post_init__(self):
    if not isinstance(self.exact_required, bool): raise ValueError("exact_required must be bool")
    if not isinstance(self.no_beam_remote, bool): raise ValueError("no_beam_remote must be bool")
    if not isinstance(self.dnll_epsilon, (int, float)) or isinstance(self.dnll_epsilon, bool) or self.dnll_epsilon < 0:
      raise ValueError(f"dnll_epsilon must be a non-negative number, got {self.dnll_epsilon!r}")
    if self.max_storage_mb is not None and (not isinstance(self.max_storage_mb, int) or isinstance(self.max_storage_mb, bool) or self.max_storage_mb <= 0):
      raise ValueError(f"max_storage_mb must be None or a positive int, got {self.max_storage_mb!r}")
    object.__setattr__(self, "ctx_range", _validate_ctx_range(self.ctx_range))

  def to_dict(self) -> dict[str, Any]:
    return {"exact_required": self.exact_required, "dnll_epsilon": self.dnll_epsilon,
            "max_storage_mb": self.max_storage_mb, "ctx_range": list(self.ctx_range),
            "no_beam_remote": self.no_beam_remote}

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "Constraints":
    return Constraints(exact_required=d.get("exact_required", True), dnll_epsilon=d.get("dnll_epsilon", 0.0),
                       max_storage_mb=d.get("max_storage_mb"), ctx_range=tuple(d.get("ctx_range", (1, 4096))),
                       no_beam_remote=d.get("no_beam_remote", True))


@dataclass(frozen=True)
class SearchRow:
  """One bounded search experiment. Constructing it validates every field."""
  row_id: str
  phase: str
  model: str
  op_scope: str
  backend: str
  search_space: str
  objective: str
  constraints: Constraints = field(default_factory=Constraints)

  def __post_init__(self):
    if not isinstance(self.row_id, str) or not self.row_id: raise ValueError("row_id must be a non-empty string")
    object.__setattr__(self, "phase", validate_phase(self.phase))
    object.__setattr__(self, "model", validate_model(self.model))
    object.__setattr__(self, "op_scope", validate_op_scope(self.op_scope))
    object.__setattr__(self, "backend", validate_backend(self.backend))
    object.__setattr__(self, "search_space", validate_search_space(self.search_space))
    object.__setattr__(self, "objective", validate_objective(self.objective))
    if not isinstance(self.constraints, Constraints): raise ValueError("constraints must be a Constraints")

  def to_dict(self) -> dict[str, Any]:
    return {"id": self.row_id, "phase": self.phase, "model": self.model, "op_scope": self.op_scope,
            "backend": self.backend, "search_space": self.search_space, "objective": self.objective,
            "constraints": self.constraints.to_dict()}

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "SearchRow":
    return SearchRow(row_id=d["id"], phase=d["phase"], model=d["model"], op_scope=d["op_scope"],
                     backend=d["backend"], search_space=d["search_space"], objective=d["objective"],
                     constraints=Constraints.from_dict(d.get("constraints", {})))


def assemble_search_row(*, row_id:str, phase:str, model:str, op_scope:str, backend:str, search_space:str,
                        objective:str, constraints:Constraints | None = None) -> dict[str, Any]:
  """Single source of truth for a search-row dict — validates and returns the canonical shape.

  This is the *only* sanctioned way to add an experiment: a new row, not a new
  script (anti-re-sprawl). Delegates validation to `SearchRow.__post_init__`.
  """
  return SearchRow(row_id=row_id, phase=phase, model=model, op_scope=op_scope, backend=backend,
                   search_space=search_space, objective=objective,
                   constraints=constraints if constraints is not None else Constraints()).to_dict()


@dataclass(frozen=True)
class AcceptedPolicy:
  """A durable, runtime-consumable accepted-policy record (the doc's accepted artifact)."""
  model: str
  phase: str
  backend: str
  ctx_range: tuple[int, int]
  objective: str
  baseline_tok_s: float
  accepted_tok_s: float
  quality_gate: str
  exactness: str
  commit: str
  memory_cap_mb: int | None = None
  hardware: str = "required"

  def __post_init__(self):
    object.__setattr__(self, "phase", validate_phase(self.phase))
    object.__setattr__(self, "model", validate_model(self.model))
    object.__setattr__(self, "backend", validate_backend(self.backend))
    object.__setattr__(self, "objective", validate_objective(self.objective))
    object.__setattr__(self, "ctx_range", _validate_ctx_range(self.ctx_range))
    for name in ("baseline_tok_s", "accepted_tok_s"):
      val = getattr(self, name)
      if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
        raise ValueError(f"{name} must be a non-negative number, got {val!r}")
    for name in ("quality_gate", "exactness", "commit", "hardware"):
      val = getattr(self, name)
      if not isinstance(val, str) or not val: raise ValueError(f"{name} must be a non-empty string")
    if self.memory_cap_mb is not None and (not isinstance(self.memory_cap_mb, int) or isinstance(self.memory_cap_mb, bool) or self.memory_cap_mb <= 0):
      raise ValueError(f"memory_cap_mb must be None or a positive int, got {self.memory_cap_mb!r}")

  def to_dict(self) -> dict[str, Any]:
    return {"model": self.model, "phase": self.phase, "backend": self.backend, "ctx_range": list(self.ctx_range),
            "objective": self.objective, "baseline_tok_s": self.baseline_tok_s, "accepted_tok_s": self.accepted_tok_s,
            "quality_gate": self.quality_gate, "exactness": self.exactness, "memory_cap_mb": self.memory_cap_mb,
            "hardware": self.hardware, "commit": self.commit}

  @staticmethod
  def from_dict(d:dict[str, Any]) -> "AcceptedPolicy":
    return AcceptedPolicy(model=d["model"], phase=d["phase"], backend=d["backend"],
                          ctx_range=tuple(d["ctx_range"]), objective=d["objective"],
                          baseline_tok_s=d["baseline_tok_s"], accepted_tok_s=d["accepted_tok_s"],
                          quality_gate=d["quality_gate"], exactness=d["exactness"], commit=d["commit"],
                          memory_cap_mb=d.get("memory_cap_mb"), hardware=d.get("hardware", "required"))


def baseline(model:str) -> dict[str, Any]:
  """Return the canonical baseline numbers for a model, from the existing repo constants.

  Imports `LLAMA_REFS` / `DEFAULT_MODEL_BYTES` / `DEFAULT_PEAK_MEM_GBS` lazily so this
  schema module stays import-light (no transitive heavy deps at import time) and so the
  baseline numbers have exactly one home. See the module docstring re: the known
  cross-module numeric disagreement (not resolved here).
  """
  from extra.qk_bandwidth_roofline import DEFAULT_MODEL_BYTES, DEFAULT_PEAK_MEM_GBS
  from extra.qk_experiment_matrix import LLAMA_REFS
  size = model_size_key(validate_model(model))
  return {"size": size, "llama_tok_s": LLAMA_REFS[size], "model_bytes": DEFAULT_MODEL_BYTES[size],
          "hbm_peak_gbs": DEFAULT_PEAK_MEM_GBS}


def from_generated_policy(policy:dict[str, Any], *, model:str, baseline_tok_s:float, accepted_tok_s:float,
                          ctx_range:tuple[int, int] = (1, 4096), objective:str = Objective.TOK_S.value) -> AcceptedPolicy:
  """Read-only adapter: map an existing `qk_generated_policy` artifact -> AcceptedPolicy.

  Proves the new schema models the real accepted artifacts under
  `bench/qk-shared-storage-20260612/*/policy.json`. Does NOT mutate or write back the
  artifact. tok/s figures come from the experiment matrix (the artifact does not store
  the explicit-vs-generated comparison), so the caller supplies them.
  """
  if policy.get("kind") != "qk_generated_policy":
    raise ValueError(f"not a qk_generated_policy artifact (kind={policy.get('kind')!r})")
  if policy.get("generator_version") not in (0, 1):
    raise ValueError(f"unsupported generator_version {policy.get('generator_version')!r}")
  commit = policy.get("commit")
  if not isinstance(commit, str) or not commit: raise ValueError("artifact missing string commit")
  cap_bytes = (policy.get("storage_policy") or {}).get("cap_bytes")
  memory_cap_mb = None if cap_bytes in (None, 0) else max(1, int(cap_bytes) // (1024 * 1024))
  # These generated policies are exact (Q4_K/Q6_K dequant is lossless) per the decode arc.
  return AcceptedPolicy(model=model, phase=Phase.DECODE.value, backend="AMD", ctx_range=ctx_range,
                        objective=objective, baseline_tok_s=baseline_tok_s, accepted_tok_s=accepted_tok_s,
                        quality_gate="dNLL <= baseline + epsilon", exactness="byte-identical",
                        commit=commit, memory_cap_mb=memory_cap_mb, hardware="required")


# --- IO helpers (route through llm_eval_common; the scaffold adds no new IO code) ---

def load_search_rows(path:pathlib.Path) -> list[SearchRow]:
  """Load + validate a search-spec table (JSONL, unique ids)."""
  return [SearchRow.from_dict(row) for row in read_id_jsonl(path)]


def save_search_rows(path:pathlib.Path, rows:list[SearchRow]) -> None:
  write_jsonl(path, [row.to_dict() for row in rows])


def load_accepted_policy(path:pathlib.Path) -> AcceptedPolicy:
  return AcceptedPolicy.from_dict(read_json_object(path))


def save_accepted_policy(path:pathlib.Path, policy:AcceptedPolicy) -> None:
  write_json(path, policy.to_dict())
