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


class Verdict(str, Enum):
  """decode_eval per-run verdicts (single source of truth; .value == the legacy string -> NFC).

  The set is exactly what extra/qk_decode_eval.py:classify() emits. The JSON schema enum, the lifecycle
  search_policy map, the evaluator contract, and the bench README are asserted == this enum by
  test/unit/test_verdict_ssot.py, so the four cannot drift from the producer again. (The newer table-driven
  PMS-R2 evaluator has its own producer enum below: TierVerdict, asserted by the same test.)
  """
  PASS_PROMOTE = "PASS_PROMOTE"
  PASS_OPT_IN = "PASS_OPT_IN"
  PASS_ORACLE_LOCAL_AB = "PASS_ORACLE_LOCAL_AB"
  LOCAL_PASS_WD_FAIL = "LOCAL_PASS_WD_FAIL"
  FAIL_CORRECTNESS = "FAIL_CORRECTNESS"
  FAIL_LOCAL_AB = "FAIL_LOCAL_AB"
  FAIL_ORACLE_LOCAL_AB = "FAIL_ORACLE_LOCAL_AB"
  NEEDS_GPU_STATE_TOOLING = "NEEDS_GPU_STATE_TOOLING"
  SELFTEST_PASS = "SELFTEST_PASS"
  REST = "REST"


class TierVerdict(str, Enum):
  """PMS-R2 candidate-evaluator verdicts (single source of truth; .value == the emitted string).

  The set is exactly what extra/qk_candidate_evaluator.py emits: classify() returns the tier strings, evaluate()
  returns the PMS_R2_* outcome. test/unit/test_verdict_ssot.py asserts this enum == the evaluator's producer, so they
  cannot drift. This is a SEPARATE namespace from Verdict (the decode_eval producer) — different harness, different set.
  """
  # classify() tiers (candidate-vs-baseline % delta)
  PROMOTE_TIER_A = "PROMOTE_TIER_A"
  PROMOTE_TIER_B = "PROMOTE_TIER_B"
  SPEED_EQUIVALENT_PASS = "SPEED_EQUIVALENT_PASS"
  INCONCLUSIVE = "INCONCLUSIVE"
  REFUTED_REGRESSION = "REFUTED_REGRESSION"
  REFUTED_CORRECTNESS = "REFUTED_CORRECTNESS"
  BLOCKED_ROUTE_NOT_BOUND = "BLOCKED_ROUTE_NOT_BOUND"
  # evaluate() replay outcomes
  PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS = "PMS_R2_PASS_EVALUATOR_REPLAYS_KNOWN_DECISIONS"
  PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE = "PMS_R2_BLOCKED_AUTHORITY_HARNESS_INCOMPLETE"


TIER_VERDICTS: frozenset[str] = frozenset(v.value for v in TierVerdict)

# Union of every value that is valid as some kernel `mode`.
KERNEL_MODES: frozenset[str] = frozenset(
  m.value for m in (*PolicyMode, *PrimitiveMode))

PROMPT_FORMATS: frozenset[str] = frozenset(f.value for f in PromptFormat)

VERDICTS: frozenset[str] = frozenset(v.value for v in Verdict)
# verdict -> lifecycle decision. Values are copied VERBATIM from the live
# bench/qk-lifecycle-search/search_policy.json:verdict_to_lifecycle_decision (NFC -- do NOT reword them).
VERDICT_LIFECYCLE: dict[str, str] = {
  Verdict.PASS_PROMOTE: "candidate_promotable_owner_decision",
  Verdict.PASS_OPT_IN: "opt_in_candidate_banked",
  Verdict.PASS_ORACLE_LOCAL_AB: "reference_oracle_target_informs_codegen_non_promotable",
  Verdict.LOCAL_PASS_WD_FAIL: "refute_for_promotion_bank_learning",
  Verdict.FAIL_CORRECTNESS: "refute_candidate",
  Verdict.FAIL_LOCAL_AB: "refute_candidate",
  Verdict.FAIL_ORACLE_LOCAL_AB: "reference_oracle_does_not_beat_comparator",
  Verdict.NEEDS_GPU_STATE_TOOLING: "stop_search_needs_gpu_state",
  Verdict.SELFTEST_PASS: "selftest_only_not_perf",
  Verdict.REST: "bank_baseline_or_rest",
}


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
