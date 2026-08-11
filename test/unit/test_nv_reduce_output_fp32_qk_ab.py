"""Hermetic CPU tests for the fp32 q/k reduce-output wall-bracket harness."""
import argparse
import json
import pathlib
import sys

import pytest

from extra.llm_research.decode.nv_reduce_output_fp32_qk_ab import (
  CENSUS_REFERENCE, CENSUS_SCHEMA, LOGITS_SCHEMA, SCHEMA, TIMING_SCHEMA,
  _assert_candidate_configured, _assert_control_closed, _child_command, _child_root,
  _configure, _fused_body_families, _gates, no_go_record, tok_per_s,
  validate_census, validate_logits_gate, validate_timing_bracket,
)


class _FakeBlock:
  pass


class _FakeModel:
  def __init__(self, blocks=3):
    self.blk = [_FakeBlock() for _ in range(blocks)]


def test_candidate_arm_sets_model_and_every_block_flag():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  model = _FakeModel()
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "candidate")
  gates = _gates(model)
  assert gates["reduce_output_rmsnorm_promoted"] is True
  assert gates["block_reduce_output_rmsnorm_promoted"] == [True, True, True]
  assert gates["decode_direct_greedy_promoted"] is True


def test_control_arm_keeps_closed_graph_and_gates_report_it():
  model = _FakeModel()
  _configure(model, "control")
  gates = _gates(model)
  assert gates["reduce_output_rmsnorm_promoted"] is False
  assert gates["block_reduce_output_rmsnorm_promoted"] == [False, False, False]
  _assert_control_closed(gates)  # closed graph is a no-op


def test_candidate_gate_fails_closed_on_missing_block_flag():
  _assert_candidate_configured({"reduce_output_rmsnorm_promoted": True,
                                "block_reduce_output_rmsnorm_promoted": [True, True, True]})
  with pytest.raises(RuntimeError, match=r"block\[1\]"):
    _assert_candidate_configured({"reduce_output_rmsnorm_promoted": True,
                                  "block_reduce_output_rmsnorm_promoted": [True, False, True]})
  with pytest.raises(RuntimeError, match=r"model\._decode_reduce_output_rmsnorm_promoted"):
    _assert_candidate_configured({"reduce_output_rmsnorm_promoted": False,
                                  "block_reduce_output_rmsnorm_promoted": [True, True, True]})


def test_control_gate_fails_closed_if_promoted_route_observed():
  model = _FakeModel()
  model._decode_reduce_output_rmsnorm_promoted = True
  with pytest.raises(RuntimeError, match=r"model\._decode_reduce_output_rmsnorm_promoted"):
    _assert_control_closed(_gates(model))


def _control_census():
  return {"schema": CENSUS_SCHEMA, "kernels": 936,
          "norms_roles": {"q_norm_reduce": 36, "k_norm_reduce": 36, "q_norm_epilogue": 72,
                          "k_norm_epilogue": 72, "rmsnorm_reduce": 56, "rmsnorm_epilogue": 55,
                          "final_rmsnorm_epilogue": 1},
          "population_counts": {"norms": 328, "quant_core": 217, "flash": 20,
                                "residual_cast_contiguous": 9, "other": 6},
          "program_counts": {"r_2_8_4_4_16_q": 36, "r_8_16_8_k": 36,
                             "E_2_8_16_4_4_qe": 36, "E_4_2_8_16_4_qe": 36,
                             "E_2_8_16_4_ke": 36, "E_8_2_16_4_ke": 36,
                             "r_16_256_r": 56, "E_32_32_4_f14a5cc0d0ed4c90e": 55,
                             "E_32_32_4_c6fef3561a9fbeaff": 1,
                             "q4k_g3_lanemap_gemv_x": 217, "flash_block_tiled_y": 20,
                             "E_32_32_4_fab82d40f922cf5fz": 9, "E_2_tok": 6}}


def _candidate_census():
  return {"schema": CENSUS_SCHEMA, "kernels": 751,
          "norms_roles": {"q_norm_reduce": 0, "k_norm_reduce": 0, "q_norm_epilogue": 0,
                          "k_norm_epilogue": 0, "rmsnorm_reduce": 37, "rmsnorm_epilogue": 37,
                          "final_rmsnorm_epilogue": 0},
          "population_counts": {"norms": 74, "quant_core": 217, "flash": 20,
                                "residual_cast_contiguous": 9, "other": 6},
          "program_counts": {"reduce_output_rmsnorm_32_128": 36, "reduce_output_rmsnorm_8_128": 36,
                             "reduce_output_rmsnorm_1_4096": 19,
                             "r_32_32_4_4_b248195950c8b6f47": 17, "r_8_32_4_4_37811d44743bc147": 17,
                             "r_16_256_r": 37, "E_32_32_4_f14a5cc0d0ed4c90e": 37,
                             "q4k_g3_lanemap_gemv_x": 217, "flash_block_tiled_y": 20,
                             "E_32_32_4_fab82d40f922cf5fz": 9, "E_2_tok": 6}}


def test_fused_body_families_counts_by_prefix():
  counts = {"reduce_output_rmsnorm_1_4096": 19, "reduce_output_rmsnorm_32_128": 36,
            "reduce_output_rmsnorm_8_128": 36, "q4k_g3_lanemap_gemv_x": 217}
  bodies = _fused_body_families({"program_counts": counts})
  assert bodies == {"c6": 19, "q": 36, "k": 36, "total": 91}


def test_fused_body_families_handles_hash_suffixed_names():
  counts = {"reduce_output_rmsnorm_32_128_ab12cd": 36, "reduce_output_rmsnorm_8_128_deadbeef": 36}
  bodies = _fused_body_families({"program_counts": counts})
  assert bodies["q"] == 36 and bodies["k"] == 36 and bodies["total"] == 72


def test_validate_census_gate_passes_on_expected_fp32_shape():
  result = validate_census(_control_census(), _candidate_census())
  assert result["gate_pass"] is True
  assert result["fused_bodies_candidate"] == 91
  assert result["fused_bodies_c6_candidate"] == 19
  assert result["fused_bodies_q_candidate"] == 36
  assert result["fused_bodies_k_candidate"] == 36
  assert result["q_norm_reduce_drop"] == 36
  assert result["k_norm_reduce_drop"] == 36
  assert result["q_norm_epilogue_drop"] == 72
  assert result["k_norm_epilogue_drop"] == 72
  assert result["rmsnorm_reduce_drop"] == 19
  assert result["rmsnorm_epilogue_drop"] == 18
  assert result["conditions"]["qk_bodies_present"] is True
  assert result["conditions"]["final_epilogue_fused_consistent"] is True
  assert result["conditions"]["q_reduce_remaining_matches_reference"] is True
  assert result["conditions"]["k_reduce_remaining_matches_reference"] is True
  assert result["honest_net_program_delta"] == -185


def test_validate_census_fails_closed_when_q_bodies_absent():
  candidate = _candidate_census()
  del candidate["program_counts"]["reduce_output_rmsnorm_32_128"]
  candidate["norms_roles"]["q_norm_reduce"] = 36
  candidate["norms_roles"]["q_norm_epilogue"] = 72
  candidate["kernels"] += 36
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["qk_bodies_present"] is False
  assert any("FAIL CLOSED" in note and "q bodies=0" in note for note in result["fail_closed"])


def test_validate_census_fails_closed_when_k_bodies_absent():
  candidate = _candidate_census()
  del candidate["program_counts"]["reduce_output_rmsnorm_8_128"]
  candidate["norms_roles"]["k_norm_reduce"] = 36
  candidate["norms_roles"]["k_norm_epilogue"] = 72
  candidate["kernels"] += 36
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["qk_bodies_present"] is False
  assert any("k bodies=0" in note for note in result["fail_closed"])


def test_validate_census_fails_closed_when_both_qk_bodies_absent():
  candidate = _candidate_census()
  candidate["program_counts"] = {name: count for name, count in candidate["program_counts"].items()
                                 if not name.startswith("reduce_output_rmsnorm_32_128")
                                 and not name.startswith("reduce_output_rmsnorm_8_128")}
  candidate["norms_roles"]["q_norm_reduce"] = 36
  candidate["norms_roles"]["k_norm_reduce"] = 36
  candidate["norms_roles"]["q_norm_epilogue"] = 72
  candidate["norms_roles"]["k_norm_epilogue"] = 72
  candidate["kernels"] += 72
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["qk_bodies_present"] is False
  assert any("selector still rejects the fp32 route" in note for note in result["fail_closed"])


def test_validate_census_fails_on_q_reduce_drop_inconsistent_with_bodies():
  candidate = _candidate_census()
  candidate["norms_roles"]["q_norm_reduce"] = 36  # no q reduce dropped at all
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["q_reduce_drop_consistent"] is False


def test_validate_census_fails_when_warp_coop_reduce_remaining_mismatches_reference():
  candidate = _candidate_census()
  candidate["norms_roles"]["q_norm_reduce"] = 18  # 18 materializing reduces, reference says 0
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["q_reduce_remaining_matches_reference"] is False
  assert result["conditions"]["q_reduce_drop_consistent"] is True  # drop 18 == 36 - 18


def test_validate_census_fails_on_k_epilogue_drop_inconsistent_with_bodies():
  candidate = _candidate_census()
  candidate["norms_roles"]["k_norm_epilogue"] = 54  # drop 18 != 36 k bodies
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["k_epilogue_drop_consistent"] is False


def test_validate_census_fails_on_c6_route_change():
  candidate = _candidate_census()
  candidate["norms_roles"]["rmsnorm_reduce"] = 20  # C6 route must stay unchanged (56 -> 38)
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["rmsnorm_reduce_drop_consistent"] is False


def test_validate_census_fails_if_final_epilogue_moves():
  candidate = _candidate_census()
  candidate["norms_roles"]["final_rmsnorm_epilogue"] = 1  # final norm epilogue did not fuse
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["final_epilogue_fused_consistent"] is False


def test_validate_census_fails_closed_if_control_has_bodies():
  control = _control_census()
  control["program_counts"] = dict(control["program_counts"])
  control["program_counts"]["reduce_output_rmsnorm_1_4096"] = 3
  result = validate_census(control, _candidate_census())
  assert result["gate_pass"] is False
  assert result["conditions"]["control_has_no_bodies"] is False
  assert any("control census unexpectedly" in note for note in result["fail_closed"])


def test_validate_census_reports_non_norms_shifts_with_exact_names():
  candidate = _candidate_census()
  candidate["program_counts"] = dict(candidate["program_counts"])
  candidate["program_counts"]["q4k_g3_lanemap_gemv_x"] = 218
  candidate["population_counts"] = dict(candidate["population_counts"])
  candidate["population_counts"]["quant_core"] = 218
  candidate["kernels"] += 1
  result = validate_census(_control_census(), candidate)
  # Non-norms shifts are reported, not hidden, and do not false-fail the gate.
  assert result["gate_pass"] is True
  assert result["non_norms_population_deltas"]["quant_core"] == 1
  assert result["callify_redirect_side_effects"]["q4k_g3_lanemap_gemv_x"] == 1


def test_validate_census_honest_net_program_delta_and_reference():
  result = validate_census(_control_census(), _candidate_census())
  assert result["honest_net_program_delta"] == -185
  assert result["reference"]["observed_total"] == CENSUS_REFERENCE["fused_bodies_total"] == 91
  assert result["reference"]["observed_q"] == CENSUS_REFERENCE["fused_bodies_q"] == 36


def test_validate_census_rejects_bad_schema():
  control = _control_census()
  control["schema"] = "tinygrad.nv_reduce_output_primitive_ab.census.v1"
  with pytest.raises(ValueError, match="census row requires schema"):
    validate_census(control, _candidate_census())


def test_validate_census_rejects_missing_schema():
  candidate = _candidate_census()
  del candidate["schema"]
  with pytest.raises(ValueError, match="census row requires schema"):
    validate_census(_control_census(), candidate)


def _logits(tokens, digest, shape=(2, 1, 151936)):
  return {"schema": LOGITS_SCHEMA, "tokens": tokens, "logits_sha256": digest, "shape": list(shape)}


def test_validate_logits_gate_exact_output():
  control = _logits([1, 2], "a" * 64)
  assert validate_logits_gate(control, _logits([1, 2], "a" * 64))["gate_pass"] is True
  assert validate_logits_gate(control, _logits([1, 2], "b" * 64))["gate_pass"] is False
  assert validate_logits_gate(control, _logits([1, 3], "a" * 64))["gate_pass"] is False


def test_validate_logits_gate_rejects_bad_schema():
  control = _logits([1], "a" * 64)
  candidate = _logits([1], "a" * 64)
  candidate["schema"] = TIMING_SCHEMA
  with pytest.raises(ValueError, match="logits row requires schema"):
    validate_logits_gate(control, candidate)


def _timing(median, stream="s" * 64):
  return {"schema": TIMING_SCHEMA, "median_ms_per_token": median, "token_stream_hash": stream}


def test_validate_timing_bracket_promotion_requires_both_controls():
  below = validate_timing_bracket([_timing(5.324), _timing(5.274), _timing(5.320)])
  assert below["candidate_minus_control_a_us"] == pytest.approx(50.0)
  assert below["candidate_minus_control_b_us"] == pytest.approx(46.0)
  assert below["promoted"] is False
  above = validate_timing_bracket([_timing(5.324), _timing(5.270), _timing(5.320)])
  assert above["candidate_minus_control_b_us"] == pytest.approx(50.0)
  assert above["promoted"] is True
  divergent = validate_timing_bracket([_timing(5.324), _timing(5.260, stream="x" * 64), _timing(5.320)])
  assert divergent["all_token_hashes_equal"] is False
  assert divergent["promoted"] is False
  with pytest.raises(ValueError):
    validate_timing_bracket([_timing(5.0), _timing(4.9)])


def test_validate_timing_bracket_rejects_bad_schema():
  row = _timing(5.0)
  row["schema"] = CENSUS_SCHEMA
  with pytest.raises(ValueError, match="timing bracket row 0 requires schema"):
    validate_timing_bracket([row, _timing(4.9), _timing(5.0)])


def test_tok_per_s_conversion():
  assert tok_per_s(5.0) == pytest.approx(200.0)
  assert tok_per_s(4.95) == pytest.approx(202.020202, rel=1e-4)


def test_child_command_includes_timeout_flock_lock():
  args = argparse.Namespace(timeout=600, lock_wait=90, lock="/tmp/gpu-bench.lock", model="/m",
                            depth=512, count=32, max_context=1024, reps=5, settled_continuous=True)
  out = pathlib.Path("/tmp/nv-fp32-qk-ab-control-logits.json")
  cmd = _child_command(args, "logits", "control", out, include_reps=False)
  assert cmd[0] == "timeout" and cmd[1] == "600s"
  assert cmd[2] == "flock" and cmd[3] == "-w" and cmd[4] == "90" and cmd[5] == "/tmp/gpu-bench.lock"
  assert sys.executable in cmd
  assert "--mode" in cmd and "logits" in cmd and "--arm" in cmd and "control" in cmd
  assert "--reps" not in cmd and "--settled-continuous" not in cmd
  assert "nv_reduce_output_fp32_qk_ab.py" in cmd[-1] or any("fp32_qk_ab" in part for part in cmd)
  timing = _child_command(args, "timing-child", "candidate", pathlib.Path("/tmp/nv-timing.json"))
  assert "--reps" in timing and "--settled-continuous" in timing and "candidate" in timing


def test_child_root_derivation():
  assert str(_child_root(pathlib.Path("/tmp/fp32-ab-record.json"), ".children")) == "/tmp/fp32-ab-record.children"
  assert str(_child_root(pathlib.Path("/tmp/fp32-ab-record.json"), ".timing")) == "/tmp/fp32-ab-record.timing"


def test_no_go_record_shape():
  record = no_go_record(model="/m", depth=512)
  assert record["schema"] == SCHEMA
  assert record["verdict"] == "NO-GO"
  assert record["target"] == {"model": "/m", "depth": 512, "device": "NV sm_120", "gpu": "RTX 5090"}
  assert record["construction"]["population"] == "norms"
  for phase in ("smoke", "logits_gate", "census", "wall_bracket"):
    assert record[phase]["run"] is False and record[phase]["result"] == "NOT_AUTHORIZED"
  assert record["census_reference"]["fused_bodies_total"] == 91
  assert record["wall_bracket"]["promotion_us"] == 50.0
  assert record["hard_stop_notes"]
  assert record["isolation_notes"]
  assert any("fp32-qk-route-scope-20260810.md" in c for c in record["citations"])


def test_ab_writes_no_go_when_qk_bodies_missing_in_smoke(monkeypatch, tmp_path):
  import extra.llm_research.decode.nv_reduce_output_fp32_qk_ab as ab_module

  def _run_child(cmd, out):
    mode = cmd[cmd.index("--mode") + 1]
    if mode == "smoke":
      result = {"schema": ab_module.SMOKE_SCHEMA, "survive": True, "fused_body_present": True,
                "fused_c6_body_present": True, "fused_q_body_present": False, "fused_k_body_present": False,
                "program_count": 1}
      pathlib.Path(out).write_text(json.dumps(result))
      return result
    raise AssertionError(f"unexpected child {cmd}")

  monkeypatch.setattr(ab_module, "_run_child", _run_child)
  record_path = tmp_path / "record.json"
  args = argparse.Namespace(model="/m", depth=512, count=32, max_context=1024, reps=5,
                            settled_continuous=True, timeout=600, lock_wait=90,
                            lock="/tmp/gpu-bench.lock", out=str(record_path))
  ab_module.ab(args)
  record = json.loads(record_path.read_text())
  assert record["verdict"] == "NO-GO"
  assert record["smoke"]["result"] == "NO-GO"
  assert record["logits_gate"]["result"] == "NOT_AUTHORIZED"
  assert record["census"]["result"] == "NOT_AUTHORIZED"


def test_ab_writes_no_go_when_census_fails_closed(monkeypatch, tmp_path):
  import extra.llm_research.decode.nv_reduce_output_fp32_qk_ab as ab_module

  calls = []

  def _run_child(cmd, out):
    mode = cmd[cmd.index("--mode") + 1]
    arm = cmd[cmd.index("--arm") + 1]
    calls.append((mode, arm))
    if (mode, arm) == ("smoke", "candidate"):
      result = {"schema": ab_module.SMOKE_SCHEMA, "survive": True, "fused_body_present": True,
                "fused_c6_body_present": True, "fused_q_body_present": True, "fused_k_body_present": True,
                "program_count": 1}
    elif mode == "logits":
      result = {"schema": ab_module.LOGITS_SCHEMA, "tokens": [1], "logits_sha256": "a" * 64, "shape": [1, 1, 151936]}
    elif mode == "census":
      # candidate census omits the q/k bodies: the gate must fail closed
      candidate = _candidate_census()
      del candidate["program_counts"]["reduce_output_rmsnorm_32_128"]
      del candidate["program_counts"]["reduce_output_rmsnorm_8_128"]
      candidate["norms_roles"]["q_norm_reduce"] = 36
      candidate["norms_roles"]["k_norm_reduce"] = 36
      candidate["norms_roles"]["q_norm_epilogue"] = 72
      candidate["norms_roles"]["k_norm_epilogue"] = 72
      candidate["kernels"] += 72
      result = candidate if arm == "candidate" else _control_census()
    else:
      raise AssertionError(f"unexpected child {mode}/{arm}")
    pathlib.Path(out).write_text(json.dumps(result))
    return result

  monkeypatch.setattr(ab_module, "_run_child", _run_child)
  record_path = tmp_path / "record.json"
  args = argparse.Namespace(model="/m", depth=512, count=32, max_context=1024, reps=5,
                            settled_continuous=True, timeout=600, lock_wait=90,
                            lock="/tmp/gpu-bench.lock", out=str(record_path))
  ab_module.ab(args)
  record = json.loads(record_path.read_text())
  assert record["verdict"] == "NO-GO"
  assert record["smoke"]["result"] == "PASS"
  assert record["logits_gate"]["result"] == "PASS"
  assert record["census"]["result"] == "FAIL"
  assert record["census"]["conditions"]["qk_bodies_present"] is False
  assert record["wall_bracket"]["result"] == "NOT_AUTHORIZED"
  assert ("smoke", "candidate") in calls and ("logits", "control") in calls and ("census", "candidate") in calls


def test_ab_writes_no_go_when_logits_gate_fails(monkeypatch, tmp_path):
  import extra.llm_research.decode.nv_reduce_output_fp32_qk_ab as ab_module

  def _run_child(cmd, out):
    mode = cmd[cmd.index("--mode") + 1]
    arm = cmd[cmd.index("--arm") + 1]
    if (mode, arm) == ("smoke", "candidate"):
      result = {"schema": ab_module.SMOKE_SCHEMA, "survive": True, "fused_body_present": True,
                "fused_c6_body_present": True, "fused_q_body_present": True, "fused_k_body_present": True,
                "program_count": 1}
    elif mode == "logits":
      digest = "a" * 64 if arm == "control" else "b" * 64
      result = {"schema": ab_module.LOGITS_SCHEMA, "tokens": [1], "logits_sha256": digest, "shape": [1, 1, 151936]}
    else:
      raise AssertionError(f"unexpected child {mode}/{arm}")
    pathlib.Path(out).write_text(json.dumps(result))
    return result

  monkeypatch.setattr(ab_module, "_run_child", _run_child)
  record_path = tmp_path / "record.json"
  args = argparse.Namespace(model="/m", depth=512, count=32, max_context=1024, reps=5,
                            settled_continuous=True, timeout=600, lock_wait=90,
                            lock="/tmp/gpu-bench.lock", out=str(record_path))
  ab_module.ab(args)
  record = json.loads(record_path.read_text())
  assert record["verdict"] == "NO-GO"
  assert record["logits_gate"]["result"] == "FAIL"
  assert record["census"]["result"] == "NOT_AUTHORIZED"


def test_ab_books_only_when_all_gates_pass_and_bracket_promotes(monkeypatch, tmp_path):
  import extra.llm_research.decode.nv_reduce_output_fp32_qk_ab as ab_module

  def _run_child(cmd, out):
    mode = cmd[cmd.index("--mode") + 1]
    arm = cmd[cmd.index("--arm") + 1]
    if (mode, arm) == ("smoke", "candidate"):
      result = {"schema": ab_module.SMOKE_SCHEMA, "survive": True, "fused_body_present": True,
                "fused_c6_body_present": True, "fused_q_body_present": True, "fused_k_body_present": True,
                "program_count": 1}
    elif mode == "logits":
      result = {"schema": ab_module.LOGITS_SCHEMA, "tokens": [1], "logits_sha256": "a" * 64, "shape": [1, 1, 151936]}
    elif mode == "census":
      result = _candidate_census() if arm == "candidate" else _control_census()
    elif mode == "timing-child":
      median = 5.320 if arm == "control" else 5.270
      result = {"schema": ab_module.TIMING_SCHEMA, "arm": arm, "median_ms_per_token": median,
                "token_stream_hash": "s" * 64}
    else:
      raise AssertionError(f"unexpected child {mode}/{arm}")
    pathlib.Path(out).write_text(json.dumps(result))
    return result

  monkeypatch.setattr(ab_module, "_run_child", _run_child)
  record_path = tmp_path / "record.json"
  args = argparse.Namespace(model="/m", depth=512, count=32, max_context=1024, reps=5,
                            settled_continuous=True, timeout=600, lock_wait=90,
                            lock="/tmp/gpu-bench.lock", out=str(record_path))
  ab_module.ab(args)
  record = json.loads(record_path.read_text())
  assert record["smoke"]["result"] == "PASS"
  assert record["logits_gate"]["result"] == "PASS"
  assert record["census"]["result"] == "PASS"
  assert record["wall_bracket"]["result"] == "PROMOTED"
  assert record["tok_per_s"]["candidate_median_ms"] == 5.270
  assert record["tok_per_s"]["candidate_tok_per_s"] == pytest.approx(1000.0 / 5.270)
  assert record["verdict"] == "BOOKED"
