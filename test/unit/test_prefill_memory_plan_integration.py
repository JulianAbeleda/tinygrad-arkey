import json
from types import SimpleNamespace

import pytest

from tinygrad.llm.admission import AdmissionInputs, plan_selected_model_memory
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
    prefill_ubatch=512, v2_on=True, resident_fp16_admit=False, model_label="selected-model")
  values.update(updates)
  return AdmissionInputs(**values)


def test_auto_exposes_multiple_feasible_strategies_and_uses_packed_baseline():
  admission, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True)
  assert plan.decision is None
  assert plan.feasible_strategies == (Strategy.FULL_RESIDENT_OVERLAY, Strategy.DIRECT_PACKED_FALLBACK)
  assert effective is Strategy.DIRECT_PACKED_FALLBACK
  assert admission.report["prefill_memory_selection_deferred"] is True
  assert json.loads(admission.prefill_memory_plan)["decision"] is None


def test_explicit_overlay_cannot_bypass_shared_byte_budget():
  with pytest.raises(RuntimeError, match="memory plan refused load|requested --max_context"):
    plan_selected_model_memory(_inputs(q4_bytes=7_000_000_000, est_fp16=7_000_000_000),
                               _facts(total=12_000_000_000, free=12_000_000_000),
                               direct_packed_supported=True, overlay_requested=True)


def test_explicit_safe_overlay_is_selected_and_serialized():
  admission, plan, effective = plan_selected_model_memory(_inputs(), _facts(), direct_packed_supported=True,
                                                           overlay_requested=True)
  assert plan.decision is Strategy.FULL_RESIDENT_OVERLAY
  assert effective is Strategy.FULL_RESIDENT_OVERLAY
  assert json.loads(admission.prefill_memory_plan)["decision"] == Strategy.FULL_RESIDENT_OVERLAY.value


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
