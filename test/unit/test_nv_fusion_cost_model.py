"""Hermetic CPU tests for the predicted-wall-delta cost model.

The model is the llama-shaped arithmetic contract that every decode fusion
harness must carry BEFORE the wall bracket runs: the norm never enters the
matmul inner loop (llama keeps one fused rms_norm_f32), so folding an epilogue
into a GEMV is predicted as blocks x [(R-1) x M_removed - R x launch_us].  The
bracket then CONFIRMS the range, EXPLAINS a same-side gap with named residual
causes, or CONTRADICTS (fails closed) when the measurement lands on the
opposite side of zero.
"""
import pytest

from extra.llm_research.decode.nv_fusion_cost_model import (
  predict_wall_delta, reconcile_cost_prediction, LAUNCH_US_RANGE)


def test_prediction_arithmetic_m1_shape():
  # M1: 36 blocks fold the 2.30us norm epilogue into the GEMV with R=2.
  pred = predict_wall_delta(36, {"E_32_32_4_f14a5cc0": 2.30},
                            {"E_32_32_4_f14a5cc0": 2}, launch_us=LAUNCH_US_RANGE)
  # per block: (2-1)*2.30 - 2*launch  ->  [2.30-2*2.0, 2.30-2*1.0] = [-1.7, +0.3]
  assert pred["blocks"] == 36
  assert pred["range_us"] == [round(-1.7 * 36, 3), round(0.3 * 36, 3)]
  # point uses the launch midpoint 1.5: (2.30 - 3.0) * 36 = -25.2
  assert pred["predicted_delta_us"] == round(-0.7 * 36, 3)
  # the range straddles zero: the premise is not sign-decisive
  assert pred["decisive"] is False
  assert "E_32_32_4_f14a5cc0" in pred["terms"]
  assert pred["terms"]["E_32_32_4_f14a5cc0"]["redundancy"] == 2


def test_reconcile_confirms_inside_range():
  pred = predict_wall_delta(36, {"E": 2.30}, {"E": 2})
  # measured at the point prediction: confirmed
  out = reconcile_cost_prediction(pred["predicted_delta_us"], pred)
  assert out["result"] == "CONFIRMED" and not out["residual_causes"]


def test_reconcile_explains_same_side_gap():
  pred = predict_wall_delta(36, {"E": 2.30}, {"E": 2})
  # same side of zero as the point prediction (-25.2), far outside the range:
  # a bigger-than-predicted win -> launch-overlap explanation
  out = reconcile_cost_prediction(-120.0, pred)
  assert out["result"] == "EXPLAINED"
  assert [c["cause"] for c in out["residual_causes"]] == ["launch_overlap"]
  # predicted-loss shape (removed median 5.0, R=2 -> point +72, range [36, 108]):
  # a bigger-than-predicted loss -> critical-path/traffic explanations
  loss_pred = predict_wall_delta(36, {"E": 5.0}, {"E": 2})
  out = reconcile_cost_prediction(200.0, loss_pred)
  assert out["result"] == "EXPLAINED"
  assert "in_kernel_critical_path" in [c["cause"] for c in out["residual_causes"]]
  # same premise, measured on the opposite side of zero -> contradiction
  out = reconcile_cost_prediction(-20.0, loss_pred)
  assert out["result"] == "CONTRADICTED"


def test_reconcile_contradicts_sign_flip():
  pred = predict_wall_delta(36, {"E": 2.30}, {"E": 2})
  # The M1 wall bracket measured +84.4us (candidate SLOWER) while the predicted
  # point is -25.2us (candidate faster): opposite sides of zero, outside range.
  out = reconcile_cost_prediction(84.4, pred)
  assert out["result"] == "CONTRADICTED"
  assert "in_kernel_critical_path" in [c["cause"] for c in out["residual_causes"]]
  assert out["measured_delta_us"] == 84.4


def test_prediction_validates_inputs():
  with pytest.raises(ValueError): predict_wall_delta(0, {"E": 1.0}, {"E": 1})
  with pytest.raises(ValueError): predict_wall_delta(1, {"E": 1.0}, {"F": 1})
  with pytest.raises(ValueError): predict_wall_delta(1, {"E": 1.0}, {"E": 0})
  with pytest.raises(ValueError): predict_wall_delta(1, {"E": 1.0}, {"E": 1},
                                                     launch_us=(2.0, 1.0))


def test_prediction_win_shape_is_decisive():
  # A launch-only removal with R=1 (nothing re-executes in-kernel) is a
  # decisive win across the whole launch range.
  pred = predict_wall_delta(36, {"E": 2.30}, {"E": 1})
  assert pred["range_us"][1] < 0 and pred["decisive"] is True
