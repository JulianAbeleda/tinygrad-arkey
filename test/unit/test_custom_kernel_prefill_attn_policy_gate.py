"""TG8 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the second of the
three pure policy gates this package addresses is `model.py:72`'s
`(n_heads, n_kv_heads, 512) in ADMITTED_GRIDS and backend == "AMD" and arch == "gfx1100"`.

The shape half (ADMITTED_GRIDS) is a shape gate and stays. The target-equality half is split out into its own,
separately named and separately testable policy predicate, `_custom_kernel_prefill_attn_promoted`.

Unlike TG3's quant-gate split, this is NOT decomposed into an independent capability question: the route
injects an already-captured, hand-authored AMD gfx1100 machine-code program (extra/llm_research/
generate_shared_attention_captures) via Tensor.uop_program, not a generically renderer-lowered operation, so
there is no separate "can this target express it" fact apart from "was a captured program promoted for it".
Promotion must therefore default CLOSED (not open like ModelRoutePlan.target_promoted's "no record" default):
an 8B/14B Qwen3 model on Metal satisfies the identical ADMITTED_GRIDS shape as on AMD, so a fail-open default
would attempt to inject raw AMD ISA on a non-AMD renderer. See the module-level comment above
`_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS` in tinygrad/llm/model.py for the full reasoning.
"""
from tinygrad.llm.fused_attention import ADMITTED_GRIDS
from tinygrad.llm.model import (
  _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS, _custom_kernel_prefill_attn_promoted, _should_use_custom_kernel_prefill_attn,
)

_ADMITTED_SHAPE = next(iter(ADMITTED_GRIDS))  # (n_heads, n_kv_heads, 512) e.g. (32, 8, 512)


def test_shape_gate_and_policy_gate_are_independently_decidable():
  n_heads, n_kv_heads, _ = _ADMITTED_SHAPE
  # Shape admitted, target promoted -> True.
  assert _should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, "AMD", "gfx1100")
  # Shape admitted, target NOT promoted -> False purely on policy.
  assert not _should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, "METAL", "Apple9")
  # Shape not admitted (arbitrary unmatched grid), target promoted -> False purely on shape.
  assert (999, 3, 512) not in ADMITTED_GRIDS
  assert not _should_use_custom_kernel_prefill_attn(999, 3, "AMD", "gfx1100")


def test_policy_promotion_is_a_named_independently_testable_predicate():
  assert _custom_kernel_prefill_attn_promoted("AMD", "gfx1100")
  assert not _custom_kernel_prefill_attn_promoted("METAL", "Apple9")
  assert not _custom_kernel_prefill_attn_promoted("NV", "sm_80")
  assert not _custom_kernel_prefill_attn_promoted(None, None)


def test_promoted_targets_default_is_closed_not_open():
  """Unlike ModelRoutePlan.target_promoted (TG3's "no record -> admitted" default, correct for the quant
  primitives because a real capability check ALSO fails closed on Metal), this route has no independent
  capability check, so an unpromoted target must read as denied, never as undecided-and-therefore-admitted."""
  assert _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS == frozenset({("AMD", "gfx1100")})
  # A shape that is proven identical on AMD (8B: n_heads=32, n_kv_heads=8) must NOT silently light up on a
  # same-shaped Metal model just because the shape gate alone matched.
  assert (32, 8, 512) in ADMITTED_GRIDS
  assert not _custom_kernel_prefill_attn_promoted("METAL", "Apple9")


def test_amd_admission_is_unchanged_by_the_tg8_split():
  """Structural-only (scope section 8: no AMD hardware here). Every (n_heads, n_kv_heads) pair the pre-TG8
  gate admitted on AMD/gfx1100 is still admitted after splitting shape from policy."""
  for n_heads, n_kv_heads, _ in ADMITTED_GRIDS:
    assert _should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, "AMD", "gfx1100")
