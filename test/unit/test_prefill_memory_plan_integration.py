import json
from types import SimpleNamespace

import pytest

from tinygrad.llm.admission import AdmissionInputs, ContextMemoryTerms, admit_selected_model_memory, plan_selected_model_memory
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.llm.model import Transformer
from tinygrad.llm.prefill_memory_plan import Strategy


def _facts(total=32_000_000_000, free=30_000_000_000):
  probe = ProbeRecord("unit-test", "2026-07-15T00:00:00+00:00")
  return DeviceFacts("AMD", "AMD", "gfx-test", total, free,
                     DeviceCapabilities(global_allocation_granularity=64 * 1024), probe, probe)


def _inputs(**updates):
  values = dict(requested=2048, trained_ctx=8192, free_vram=None, q4_bytes=6_000_000_000,
    est_fp16=8_000_000_000, num_blocks=32, n_heads=32, n_kv_heads=8, head_dim=128,
    prefill_ubatch=512, v2_on=True, model_label="selected-model")
  values.update(updates)
  return AdmissionInputs(**values)


def test_metadata_builder_is_the_geometry_owner():
  kv = {"general.architecture": "unit", "unit.context_length": 8192, "unit.block_count": 34,
        "unit.nextn_predict_layers": 2, "unit.attention.head_count": 32, "unit.attention.head_count_kv": 8,
        "unit.embedding_length": 4096, "unit.attention.key_length": 128, "unit.rope.dimension_count": 96}
  inp = AdmissionInputs.from_model_metadata(2048, kv, free_vram=30_000_000_000, q4_bytes=6_000_000_000,
    est_fp16=8_000_000_000, prefill_ubatch=512, v2_on=True)
  assert (inp.trained_ctx, inp.num_blocks, inp.n_heads, inp.n_kv_heads, inp.head_dim, inp.rope_dim) == (8192, 32, 32, 8, 128, 96)
  terms = ContextMemoryTerms.from_inputs(inp, resident_fp16=False)
  assert (terms.weights, terms.kv_per_tok, terms.prefill_per_tok, terms.flash_scratch, terms.kv_scale_per_tok) == \
         (6_000_000_000, 131072, 65536, 798720, 1024)


def test_auto_exposes_multiple_feasible_strategies_and_uses_packed_baseline():
  admission, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True)
  assert plan.decision is None
  assert plan.feasible_strategies == (Strategy.FULL_RESIDENT_OVERLAY, Strategy.DIRECT_PACKED_FALLBACK)
  assert effective is Strategy.DIRECT_PACKED_FALLBACK
  assert admission.report["prefill_memory_selection_deferred"] is True
  assert json.loads(admission.prefill_memory_plan)["decision"] is None


def test_infeasible_overlay_degrades_instead_of_refusing():
  inp = _inputs(q4_bytes=7_000_000_000, est_fp16=7_000_000_000)
  facts = _facts(total=12_000_000_000, free=12_000_000_000)
  admission, plan, effective = plan_selected_model_memory(inp, facts, direct_packed_supported=True,
                                                          policy={"strategy": "FULL_RESIDENT_OVERLAY"})
  # R1: a preferred overlay that cannot fit degrades, never REFUSEs.
  assert effective is Strategy.DIRECT_PACKED_FALLBACK
  assert plan.decision is Strategy.DIRECT_PACKED_FALLBACK
  # Loudness: the degradation reason names the byte shortfall.
  degradation = admission.report["prefill_overlay_degradation"]
  assert "exceeds" in degradation and "budget" in degradation
  # R6: the degraded packed load gets the packed-sized context, not one sized against the phantom overlay.
  packed = admit_selected_model_memory(inp, facts, direct_packed_supported=True, resident_fp16=False)
  assert admission.max_context == packed[0].max_context


def test_explicit_safe_overlay_is_selected_and_serialized():
  admission, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True,
                                                          policy={"strategy": "FULL_RESIDENT_OVERLAY"})
  assert plan.decision is Strategy.FULL_RESIDENT_OVERLAY
  assert effective is Strategy.FULL_RESIDENT_OVERLAY
  payload = json.loads(admission.prefill_memory_plan)
  assert payload["decision"] == Strategy.FULL_RESIDENT_OVERLAY.value
  # Legacy reporting fields stay available, while the nested scanned budget is
  # the single allocation-admission authority used by the planner.
  assert {"total_bytes", "free_bytes", "safety_reserve", "provenance"} <= set(payload["device"])
  assert payload["device"]["scanned_budget"]["admitted_bytes"] == plan.admitted_budget_bytes


def test_fp16_capability_with_promotion_policy_selects_overlay_when_feasible():
  # AMD-shaped: fp16 expressible (v2_on) + promoted-candidate policy -> overlay when feasible.
  _, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True,
                                                  policy={"strategy": "FULL_RESIDENT_OVERLAY"})
  assert effective is Strategy.FULL_RESIDENT_OVERLAY
  assert plan.decision is Strategy.FULL_RESIDENT_OVERLAY


def test_without_fp16_dtype_capability_the_overlay_candidate_is_unsupported():
  _, plan, effective = plan_selected_model_memory(_inputs(v2_on=False), _facts(), direct_packed_supported=True)
  assert effective is Strategy.DIRECT_PACKED_FALLBACK
  assert plan.decision is Strategy.DIRECT_PACKED_FALLBACK
  assert Strategy.FULL_RESIDENT_OVERLAY not in plan.feasible_strategies


def test_nv_shaped_fp16_capability_without_promotion_uses_packed_baseline():
  # NV-shaped: fp16 expressible, no promoted candidate (policy None) -> baseline, deferred like today's auto.
  admission, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True)
  assert effective is Strategy.DIRECT_PACKED_FALLBACK
  assert admission.report["prefill_memory_selection_deferred"] is True
  assert json.loads(admission.prefill_memory_plan)["decision"] is None


def test_fp16_spend_gb_composes_overlay_and_kv_against_scanned_budget():
  inp = _inputs()
  admission, _, effective = plan_selected_model_memory(inp, _facts(), direct_packed_supported=True,
                                                       policy={"strategy": "FULL_RESIDENT_OVERLAY"})
  assert effective is Strategy.FULL_RESIDENT_OVERLAY
  kv_bytes = admission.kv_per_tok * admission.max_context  # fp16 KV elected (kv_quant False)
  assert admission.report["fp16_spend_gb"] == pytest.approx((inp.est_fp16 + kv_bytes) / 1e9)
  assert admission.report["fp16_spend_gb"] <= admission.report["budget_gb"]
  packed_admission, _, packed_effective = plan_selected_model_memory(inp, _facts(), direct_packed_supported=True)
  assert packed_effective is Strategy.DIRECT_PACKED_FALLBACK
  packed_kv = packed_admission.kv_per_tok * packed_admission.max_context
  assert packed_admission.report["fp16_spend_gb"] == pytest.approx(packed_kv / 1e9)


def test_non_overlay_plan_never_walks_or_realizes_pf16_weights():
  admission, _, _ = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True)
  model = Transformer.__new__(Transformer)
  model.config = SimpleNamespace(prefill_memory_plan=admission.prefill_memory_plan)
  model._prefill_v2_covered = lambda: (_ for _ in ()).throw(AssertionError("must not inspect overlay weights"))
  assert model.realize_prefill_v2_weights() == 0


@pytest.mark.parametrize("total,free,granularity", [(None, 30_000_000_000, 64*1024),
                                                      (32_000_000_000, None, 64*1024),
                                                      (32_000_000_000, 30_000_000_000, None)])
def test_selected_model_plan_fails_closed_without_complete_scanned_budget(total, free, granularity):
  probe = ProbeRecord("unit-test", "2026-07-15T00:00:00+00:00")
  facts = DeviceFacts("AMD", "AMD", "gfx-test", total, free,
                      DeviceCapabilities(global_allocation_granularity=granularity), probe, probe)
  with pytest.raises(RuntimeError, match="requires scanned total/free VRAM and allocator granularity"):
    plan_selected_model_memory(_inputs(), facts, direct_packed_supported=True)
