"""Centralized enum definitions for kernel `mode` and `prompt_format` values.

This is the single source of truth for the stringly-typed mode/format values
used across the QK / LLM tooling. It encodes the valid sets as enums and
exposes validators so callers stop relying on ad-hoc string-literal checks.

Two distinct namespaces are modeled because the codebase uses `mode` for two
unrelated things:

- `PolicyMode`  - llm rollout/train/eval policy selection (explicit/generated/baseline).
- `PrimitiveMode` - q4_k_bench kernel primitive variant (serial/partial/...).

`KernelMode` is the union of both, so a single validator can accept any value
that is valid as *some* kernel mode. `PromptFormat` covers prompt rendering.

The enum `.value` is exactly the legacy string, and the `choices()` helpers
preserve the historical argparse ordering, so routing call sites through this
module is behavior-preserving (NFC).
"""
from __future__ import annotations

from enum import Enum


class PolicyMode(str, Enum):
  """LLM rollout / adapter-train / eval policy modes."""
  GENERATED = "generated"
  EXPLICIT = "explicit"
  BASELINE = "baseline"


class PrimitiveMode(str, Enum):
  """q4_k_bench kernel primitive variants."""
  SERIAL = "serial"
  PARTIAL = "partial"
  PACKED_LOAD = "packed_load"
  HOIST_SCALE_MIN = "hoist_scale_min"
  VECTOR_LOAD = "vector_load"
  GROUPED = "grouped"
  TILE_CUSTOM = "tile_custom"


class PromptFormat(str, Enum):
  """Prompt rendering format."""
  CHAT = "chat"
  RAW = "raw"


# Union of every value that is valid as some kernel `mode`.
KERNEL_MODES: frozenset[str] = frozenset(
  m.value for m in (*PolicyMode, *PrimitiveMode))

PROMPT_FORMATS: frozenset[str] = frozenset(f.value for f in PromptFormat)


def policy_mode_choices() -> tuple[str, ...]:
  """Argparse choices for llm policy mode, in historical order."""
  return (PolicyMode.GENERATED.value, PolicyMode.EXPLICIT.value, PolicyMode.BASELINE.value)


def eval_run_mode_choices() -> tuple[str, ...]:
  """Argparse choices for the eval harness --run-mode flag, in historical order."""
  return (PolicyMode.EXPLICIT.value, PolicyMode.GENERATED.value)


def primitive_mode_choices() -> tuple[str, ...]:
  """Argparse choices for q4_k_bench --primitive-mode, in historical order."""
  return (
    PrimitiveMode.SERIAL.value, PrimitiveMode.PARTIAL.value, PrimitiveMode.PACKED_LOAD.value,
    PrimitiveMode.HOIST_SCALE_MIN.value, PrimitiveMode.VECTOR_LOAD.value, PrimitiveMode.GROUPED.value,
    PrimitiveMode.TILE_CUSTOM.value,
  )


def prompt_format_choices() -> tuple[str, ...]:
  """Argparse choices for --prompt-format, in historical order."""
  return (PromptFormat.CHAT.value, PromptFormat.RAW.value)


def validate_policy_mode(value:str) -> PolicyMode:
  """Return the PolicyMode for `value` or raise ValueError on an unknown mode."""
  try:
    return PolicyMode(value)
  except ValueError:
    raise ValueError(f"unknown mode {value!r}; expected one of {sorted(m.value for m in PolicyMode)}")


def validate_primitive_mode(value:str) -> PrimitiveMode:
  """Return the PrimitiveMode for `value` or raise ValueError on an unknown mode."""
  try:
    return PrimitiveMode(value)
  except ValueError:
    raise ValueError(f"unknown primitive mode {value!r}; expected one of {sorted(m.value for m in PrimitiveMode)}")


def validate_prompt_format(value:str) -> PromptFormat:
  """Return the PromptFormat for `value` or raise ValueError on an unknown format."""
  try:
    return PromptFormat(value)
  except ValueError:
    raise ValueError(f"unknown prompt_format {value!r}; expected one of {sorted(f.value for f in PromptFormat)}")
