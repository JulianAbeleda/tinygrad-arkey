import pytest

from extra.llm_research.decode.nv_wait_adjusted_cut_forecast import (
  CALIBRATED_WAIT_COST_US, LEGACY_WAIT_COST_US, P4_WALL_DELTA_US,
  PROMOTION_GATE_US, calibrate_wait_cost, forecast,
)
from extra.llm_research.decode.nv_dependency_closed_cut import analyze
from extra.llm_research.decode.nv_dependency_closed_cut import attention_cuts


Q_PREFIX = {1:"E_2_8_16_4_", 3:"r_8_16_8_", 5:"E_8_2_16_4_", 7:"r_8_8_16_2_4_"}
K_PREFIX = {0:"E_4_2_8_16_4_", 2:"r_2_8_4_4_16_", 4:"E_2_8_16_4_4_"}


def _block_dag(blocks:int=36, d:float=1.0) -> dict:
  """36 self-contained attention blocks with an independent Q/K fork each."""
  nodes, edges = [], []
  for b in range(blocks):
    base = 9*b
    names = {off:f"{p}_{b}" for off,p in Q_PREFIX.items()}
    names.update({off:f"{p}_{b}" for off,p in K_PREFIX.items()})
    names[6] = f"r_8_16_8_x_{b}"
    names[8] = f"flash_block_tiled_{b}"
    for off in range(9):
      nodes.append({"id":base+off, "name":names[off], "duration_us":d, "group_id":b})
    edges += [{"from":base+0,"to":base+1},{"from":base+1,"to":base+3},
              {"from":base+3,"to":base+5},{"from":base+5,"to":base+7},
              {"from":base+0,"to":base+2},{"from":base+2,"to":base+4},
              {"from":base+4,"to":base+6},{"from":base+6,"to":base+8},
              {"from":base+7,"to":base+8}]
    if b+1 < blocks: edges.append({"from":base+8,"to":base+9})
  return {"nodes":nodes, "edges":edges}


def test_calibration_arithmetic_from_p4_record():
  true_wait = (184.992 + 10.474) / 73
  assert round(true_wait, 3) == CALIBRATED_WAIT_COST_US == 2.678
  assert round(CALIBRATED_WAIT_COST_US / LEGACY_WAIT_COST_US, 3) == 7.377


def test_forecast_reproduces_wall_and_closes_both_cuts():
  dag = _block_dag(36, d=2.0)
  calibrated = forecast(dag)
  # The default calibration pins the Q cut to the P4 wall delta.
  assert calibrated["wait_cost_us"] == pytest.approx(3.1455, abs=1e-3)
  q = calibrated["candidates"]["attention_q"]
  assert q["raw_saving_us"] == 216.0 and q["effective_waits"] == 72
  assert q["costed_saving_us_calibrated"] == pytest.approx(P4_WALL_DELTA_US, abs=1e-3)
  assert q["verdict"] == "CPU_NO_GO"
  k = calibrated["candidates"]["attention_k"]
  assert k["costed_saving_us_calibrated"] == pytest.approx(-192.566, abs=1e-3)
  assert k["verdict"] == "CPU_NO_GO"
  assert calibrated["verdict"] == "CPU_NO_GO"
  # The legacy model still thinks both cuts are worth a GPU arm: that is the
  # exact P4 failure mode the calibrated gate exists to prevent.
  legacy = forecast(dag, LEGACY_WAIT_COST_US)
  assert legacy["verdict"] == "GPU_ELIGIBLE"


def test_gate_is_not_vacuously_closed_at_low_wait_cost():
  dag = _block_dag(36, d=3.0)
  row = forecast(dag, 1.0)["candidates"]["attention_q"]
  assert row["costed_saving_us_calibrated"] > 200
  assert row["verdict"] == "GPU_ELIGIBLE"
  assert forecast(dag, 1.0)["verdict"] == "GPU_ELIGIBLE"


def test_calibrate_wait_cost_returns_wall_reproducing_cost():
  dag = _block_dag(36, d=2.0)
  q_cut = attention_cuts(dag)["attention_q"]
  w = calibrate_wait_cost(dag, q_cut)
  assert w == pytest.approx(3.1455, abs=1e-3)
  assert w > CALIBRATED_WAIT_COST_US  # schedule absorbs waits; honest cost is higher


def test_break_even_and_gate_wait_costs():
  dag = _block_dag(36, d=2.0)
  row = forecast(dag)["candidates"]["attention_q"]
  assert row["break_even_wait_cost_us"] == pytest.approx(216.0/72, abs=1e-3)
  assert row["gate_wait_cost_us"] == pytest.approx((216.0 - PROMOTION_GATE_US)/72, abs=1e-3)
  assert row["gate_wait_cost_us"] < CALIBRATED_WAIT_COST_US


def test_per_cut_verdict_matches_top_level():
  dag = _block_dag(36, d=1.0)
  result = forecast(dag)
  top = result["verdict"]
  assert all(c["verdict"] == top for c in result["candidates"].values())
  assert result["baseline_span_us"] == 324.0


def test_malformed_graph_fails_closed():
  dag = _block_dag(36, d=1.0)
  dag["nodes"][0]["duration_us"] = -1
  with pytest.raises(ValueError):
    analyze(dag, CALIBRATED_WAIT_COST_US)
