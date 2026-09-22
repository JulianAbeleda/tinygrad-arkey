"""TG8 (docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md): the second of the
three pure policy gates this package addresses is `model.py:72`'s
`(n_heads, n_kv_heads, 512) in ADMITTED_GRIDS and backend == "AMD" and arch == "gfx1100"`.

The shape half (ADMITTED_GRIDS) is a shape gate and stays. The target-equality half is split out into its own,
separately named and separately testable policy predicate, `_custom_kernel_prefill_attn_promoted`.

Unlike TG3's quant-gate split, this is NOT decomposed into an independent capability question: the fused
kernel is built as UOps by FlashPrefillAttentionSpec.emit() and injected via Tensor.uop_program, and the
per-target fragment decomposition is the unproven-until-measured part. Promotion therefore defaults CLOSED
(not open like ModelRoutePlan.target_promoted's "no record" default) and is sourced from the BoltBeam
route-policy record in tinygrad/llm/generated/ (see the module-level comment above
`_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS` in tinygrad/llm/model.py for the full reasoning).
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
  assert _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS == frozenset({("AMD", "gfx1100"), ("NV", "sm_120")})
  # A shape that is proven identical on AMD (8B: n_heads=32, n_kv_heads=8) must NOT silently light up on a
  # same-shaped Metal model just because the shape gate alone matched.
  assert (32, 8, 512) in ADMITTED_GRIDS
  assert not _custom_kernel_prefill_attn_promoted("METAL", "Apple9")


def test_promoted_set_is_sourced_from_the_boltbeam_record():
  """The enforced set must be exactly the checked-in promotion record, not a second copy."""
  import json
  from pathlib import Path
  from tinygrad.llm import model as llm_model
  record_path = Path(llm_model.__file__).with_name("generated") / "custom-kernel-prefill-attention-route-policy.json"
  record = json.loads(record_path.read_text())
  assert record["schema"] == "boltbeam.route_policy.v1"
  assert frozenset((t.get("backend"), t.get("architecture")) for t in record["promoted_targets"]) == \
    _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS
  # NV sm_120 entered the record only after 5090 e2e token parity (P5 of the NV fused-attention port
  # scope): first-token digits identical to the SDPA baseline, max_abs_error ~1e-4, full write coverage.
  assert ("NV", "sm_120") in _CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS


def test_amd_admission_is_unchanged_by_the_tg8_split():
  """Structural-only (scope section 8: no AMD hardware here). Every (n_heads, n_kv_heads) pair the pre-TG8
  gate admitted on AMD/gfx1100 is still admitted after splitting shape from policy."""
  for n_heads, n_kv_heads, _ in ADMITTED_GRIDS:
    assert _should_use_custom_kernel_prefill_attn(n_heads, n_kv_heads, "AMD", "gfx1100")
