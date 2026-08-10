"""Hermetic CPU tests for the NV reduce-output primitive wall-bracket harness."""
import argparse
import pathlib
import sys

import pytest

from extra.llm_research.decode.nv_reduce_output_primitive_ab import (
  POP_NORMS, SCHEMA, _assert_candidate_configured, _assert_control_closed,
  _child_command, _child_root, _configure, _gates, _guarded_child, no_go_record,
  tok_per_s, validate_census, validate_logits_gate, validate_timing_bracket,
)


class _FakeBlock:
  pass


class _FakeModel:
  def __init__(self, blocks=3):
    self.blk = [_FakeBlock() for _ in range(blocks)]


def test_candidate_arm_fails_closed_without_callify_flags():
  model = _FakeModel()
  with pytest.raises(RuntimeError, match="CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT"):
    _configure(model, "candidate")
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER=1):
    _configure(model, "candidate")
  assert model._decode_reduce_output_rmsnorm_promoted is True
  assert all(block._decode_reduce_output_rmsnorm_promoted is True for block in model.blk)
  # Census-matching decode route is set on the candidate arm.
  assert model._decode_direct_greedy_promoted is True


def test_control_arm_sets_census_direct_greedy_route_only():
  model = _FakeModel()
  _configure(model, "control")
  # The control arm keeps the closed reduce-output graph but runs the same
  # production-qualified direct greedy flash route as the committed census, so
  # the bracket measures only the reduce-output inter-arm delta.
  assert model._decode_direct_greedy_promoted is True
  assert not hasattr(model, "_decode_reduce_output_rmsnorm_promoted")
  assert not any(hasattr(block, "_decode_reduce_output_rmsnorm_promoted") for block in model.blk)
  assert _gates(model)["decode_direct_greedy_promoted"] is True


def test_candidate_arm_requires_both_callify_flags():
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT
  from tinygrad.helpers import Context
  with Context(CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1):
    with pytest.raises(RuntimeError, match="CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER"):
      _configure(_FakeModel(), "candidate")


def test_candidate_requires_promotion_flag_on_model_and_every_block():
  _assert_candidate_configured({"reduce_output_rmsnorm_promoted": True,
                                "block_reduce_output_rmsnorm_promoted": [True, True, True]})
  with pytest.raises(RuntimeError, match=r"block\[1\]"):
    _assert_candidate_configured({"reduce_output_rmsnorm_promoted": True,
                                  "block_reduce_output_rmsnorm_promoted": [True, False, True]})
  with pytest.raises(RuntimeError, match=r"model\._decode_reduce_output_rmsnorm_promoted"):
    _assert_candidate_configured({"reduce_output_rmsnorm_promoted": False,
                                  "block_reduce_output_rmsnorm_promoted": [True, True, True]})


def test_control_arm_fails_closed_if_promoted_route_observed():
  model = _FakeModel()
  _assert_control_closed(_gates(model))  # closed graph is a no-op
  model._decode_reduce_output_rmsnorm_promoted = True
  with pytest.raises(RuntimeError, match=r"model\._decode_reduce_output_rmsnorm_promoted"):
    _assert_control_closed(_gates(model))
  blocky = _FakeModel()
  blocky.blk[1]._decode_reduce_output_rmsnorm_promoted = True
  with pytest.raises(RuntimeError, match=r"block\[1\]"):
    _assert_control_closed(_gates(blocky))


def test_unknown_arm_rejected():
  with pytest.raises(ValueError):
    _configure(_FakeModel(), "not-an-arm")


def _logits(tokens, digest, shape=(2, 1, 151936)):
  return {"tokens": tokens, "logits_sha256": digest, "shape": list(shape)}


def test_validate_logits_gate_exact_output():
  control = _logits([1, 2], "a" * 64)
  assert validate_logits_gate(control, _logits([1, 2], "a" * 64))["gate_pass"] is True
  bad_sha = validate_logits_gate(control, _logits([1, 2], "b" * 64))
  assert bad_sha["gate_pass"] is False and bad_sha["logits_sha256_equal"] is False
  bad_tokens = validate_logits_gate(control, _logits([1, 3], "a" * 64))
  assert bad_tokens["gate_pass"] is False and bad_tokens["tokens_equal"] is False
  bad_shape = validate_logits_gate(control, _logits([1, 2], "a" * 64, shape=(2, 1, 128)))
  assert bad_shape["gate_pass"] is False and bad_shape["shape_equal"] is False


def _census(kernels, norms_roles, pops, program_counts):
  return {"kernels": kernels, "norms_roles": norms_roles,
          "population_counts": pops, "program_counts": program_counts}


def _control_census():
  return _census(
    kernels=306,
    norms_roles={"rmsnorm_reduce": 19, "q_norm_reduce": 8, "k_norm_reduce": 8, "rmsnorm_epilogue": 19},
    pops={POP_NORMS: 54, "quant_core": 217, "flash": 20, "residual_cast_contiguous": 9, "other": 6},
    program_counts={"r_16_256_abc": 19, "E_32_32_4_f14a5cc0d0ed4c90aaa": 19,
                    "r_2_8_4_4_16_h": 8, "r_8_16_8_h": 8, "q4k_g3_lanemap_gemv_x": 217,
                    "flash_block_tiled_y": 20, "E_32_32_4_fab82d40f922cf5fbbb": 9,
                    "E_32_32_4_other1234": 6})


def _candidate_census():
  return _census(
    kernels=307,
    norms_roles={"rmsnorm_reduce": 13, "q_norm_reduce": 8, "k_norm_reduce": 8, "rmsnorm_epilogue": 13},
    pops={POP_NORMS: 42, "quant_core": 217, "flash": 20, "residual_cast_contiguous": 9, "other": 19},
    program_counts={"r_16_256_abc": 13, "E_32_32_4_f14a5cc0d0ed4c90aaa": 13,
                    "r_2_8_4_4_16_h": 8, "r_8_16_8_h": 8, "q4k_g3_lanemap_gemv_x": 217,
                    "flash_block_tiled_y": 20, "E_32_32_4_fab82d40f922cf5fbbb": 9,
                    "E_32_32_4_other1234": 7, "reduce_output_rmsnorm_1_4096": 6,
                    "E_32_32_4_8eeb0be1271d29e7ccc": 6})


def test_validate_census_fused_bodies_reduce_drop_epilogues():
  result = validate_census(_control_census(), _candidate_census())
  assert result["gate_pass"] is True
  assert result["fused_bodies_candidate"] == 6
  assert result["rmsnorm_reduce_drop"] == 6
  assert result["epilogues_removed"] == 6
  assert result["conditions"]["fused_bodies_present"] is True
  assert result["conditions"]["rmsnorm_reduce_drop_consistent"] is True
  assert result["conditions"]["other_reduce_roles_unchanged"] is True
  # The honest net program delta is recorded, not hidden.
  assert result["honest_net_program_delta"] == 1
  # Callify-redirect side effects carry the exact families.
  effects = result["callify_redirect_side_effects"]
  assert effects["reduce_output_rmsnorm_1_4096"] == 6
  assert effects["r_16_256_abc"] == -6
  assert effects["E_32_32_4_8eeb0be1271d29e7ccc"] == 6


def test_validate_census_fails_without_fused_bodies():
  candidate = _candidate_census()
  del candidate["program_counts"]["reduce_output_rmsnorm_1_4096"]
  candidate["population_counts"]["other"] -= 6
  candidate["kernels"] -= 6
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["fused_bodies_present"] is False


def test_validate_census_fails_without_reduce_drop():
  candidate = _candidate_census()
  candidate["norms_roles"]["rmsnorm_reduce"] = 19
  candidate["program_counts"]["r_16_256_abc"] = 19
  candidate["norms_roles"]["rmsnorm_epilogue"] = 13  # epilogues still removed
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["rmsnorm_reduce_drop"] == 0
  assert result["conditions"]["rmsnorm_reduce_drop_consistent"] is False


def test_validate_census_fails_on_qk_reduce_role_change():
  candidate = _candidate_census()
  candidate["norms_roles"]["q_norm_reduce"] = 9
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["conditions"]["other_reduce_roles_unchanged"] is False


def test_validate_census_fails_without_epilogue_removal():
  candidate = _candidate_census()
  candidate["norms_roles"]["rmsnorm_epilogue"] = 19
  candidate["program_counts"]["E_32_32_4_f14a5cc0d0ed4c90aaa"] = 19
  candidate["population_counts"][POP_NORMS] += 6
  candidate["kernels"] += 6
  result = validate_census(_control_census(), candidate)
  assert result["gate_pass"] is False
  assert result["epilogues_removed"] == 0
  assert result["conditions"]["epilogues_removed_positive"] is False


def test_validate_census_reports_non_norms_shifts_without_hiding_them():
  candidate = _candidate_census()
  candidate["population_counts"]["quant_core"] = 218
  candidate["program_counts"]["q4k_g3_lanemap_gemv_x"] = 218
  candidate["kernels"] += 1
  result = validate_census(_control_census(), candidate)
  # A non-norms shift is reported with the exact family; it does not false-fail
  # the gate the way blanket population equality would (the callify redirect
  # legitimately moves E_32_32_4 residual/contiguous families).
  assert result["gate_pass"] is True
  assert result["non_norms_population_deltas"]["quant_core"] == 1
  assert result["callify_redirect_side_effects"]["q4k_g3_lanemap_gemv_x"] == 1


def _timing(median, stream="s" * 64, token="t" * 64):
  return {"median_ms_per_token": median, "token_stream_hash": stream, "token_hashes": [token]}


def test_validate_timing_bracket_promotion_requires_both_controls():
  # control 5.324 ms vs candidate 5.274 ms: exactly +50 us on arm A, but only
  # +46 us vs arm B, so the bracket is not promoted.
  below = validate_timing_bracket([_timing(5.324), _timing(5.274), _timing(5.320)])
  assert below["all_token_hashes_equal"] is True
  assert below["candidate_minus_control_a_us"] == pytest.approx(50.0)
  assert below["candidate_minus_control_b_us"] == pytest.approx(46.0)
  assert below["promoted"] is False
  # candidate 5.270 ms clears +50 us vs both arms.
  above = validate_timing_bracket([_timing(5.324), _timing(5.270), _timing(5.320)])
  assert above["candidate_minus_control_a_us"] == pytest.approx(54.0)
  assert above["candidate_minus_control_b_us"] == pytest.approx(50.0)
  assert above["promoted"] is True
  # A divergent token stream voids the bracket even with a large win.
  divergent = validate_timing_bracket([_timing(5.324), _timing(5.260, stream="x" * 64), _timing(5.320)])
  assert divergent["all_token_hashes_equal"] is False
  assert divergent["promoted"] is False
  with pytest.raises(ValueError):
    validate_timing_bracket([_timing(5.0), _timing(4.9)])


def test_child_command_construction_includes_timeout_flock_lock():
  args = argparse.Namespace(timeout=600, lock_wait=90, lock="/tmp/gpu-bench.lock", model="/m",
                            depth=512, count=32, max_context=1024, reps=5, settled_continuous=True)
  out = pathlib.Path("/tmp/nv-reduce-output-ab-control-logits.json")
  cmd = _child_command(args, "logits", "control", out, include_reps=False)
  assert cmd[0] == "timeout" and cmd[1] == "600s"
  assert cmd[2] == "flock" and cmd[3] == "-w" and cmd[4] == "90" and cmd[5] == "/tmp/gpu-bench.lock"
  assert sys.executable in cmd
  assert "--mode" in cmd and "logits" in cmd and "--arm" in cmd and "control" in cmd
  assert "--out" in cmd and str(out) in cmd
  assert "--reps" not in cmd and "--settled-continuous" not in cmd
  timing = _child_command(args, "timing-child", "candidate", pathlib.Path("/tmp/nv-timing.json"))
  assert "--reps" in timing and "--settled-continuous" in timing and "candidate" in timing
  smoke = _child_command(args, "smoke", "candidate", pathlib.Path("/tmp/nv-smoke.json"), include_reps=False)
  assert "smoke" in smoke and "candidate" in smoke and "--reps" not in smoke


def test_no_go_record_shape():
  record = no_go_record(model="/m", depth=512)
  assert record["schema"] == SCHEMA
  assert record["verdict"] == "NO-GO"
  assert record["target"] == {"model": "/m", "depth": 512, "device": "NV sm_120", "gpu": "RTX 5090"}
  assert record["construction"]["population"] == "norms"
  for phase in ("smoke", "logits_gate", "census", "wall_bracket"):
    assert record[phase]["run"] is False
    assert record[phase]["result"] == "NOT_AUTHORIZED"
  assert record["wall_bracket"]["promotion_us"] == 50.0
  assert record["census_reference"]["fused_bodies"] == 54
  assert record["census_reference"]["net_call_delta_vs_ordinary"] == 1
  assert record["hard_stop_notes"]
  assert record["isolation_notes"]
  assert record["citations"]
  assert any("nv-reduce-output-wall-bracket-scope-20260809.md" in c for c in record["citations"])
  assert any("nv-generic-reduce-output-census-20260809.json" in c for c in record["citations"])


def test_tok_per_s_conversion():
  assert tok_per_s(5.0) == pytest.approx(200.0)
  assert tok_per_s(4.95) == pytest.approx(202.020202, rel=1e-4)


def test_child_root_derivation():
  assert str(_child_root(pathlib.Path("/tmp/ro-ab-record.json"), ".children")) == "/tmp/ro-ab-record.children"
  assert str(_child_root(pathlib.Path("/tmp/ro-ab-record.json"), ".timing")) == "/tmp/ro-ab-record.timing"


def test_guarded_child_records_no_go_with_child_stderr(monkeypatch, tmp_path):
  import extra.llm_research.decode.nv_reduce_output_primitive_ab as ab_module
  from extra.llm_research.decode.nv_reduce_output_primitive_ab import ChildFailure, no_go_record

  def _boom(cmd, out):
    raise ChildFailure("child failed rc=1: kernel boom", "kernel boom stderr")
  monkeypatch.setattr(ab_module, "_run_child", _boom)
  record = no_go_record()
  result = _guarded_child(record, "logits_gate", ["timeout"], pathlib.Path("/tmp/never.json"),
                          "the candidate exact-logits child failed; raw child stderr captured below")
  assert result is None
  assert record["logits_gate"]["run"] is True
  assert record["logits_gate"]["result"] == "NO-GO"
  assert "kernel boom stderr" in record["logits_gate"]["stderr"]
  assert any("HARD STOP at logits_gate" in note for note in record["hard_stop_notes"])


def test_ab_writes_no_go_record_when_logits_child_fails(monkeypatch, tmp_path):
  import argparse
  import extra.llm_research.decode.nv_reduce_output_primitive_ab as ab_module
  from extra.llm_research.decode.nv_reduce_output_primitive_ab import ChildFailure
  calls = []

  def _run_child(cmd, out):
    mode = cmd[cmd.index("--mode") + 1]
    arm = cmd[cmd.index("--arm") + 1]
    calls.append((mode, arm))
    if (mode, arm) == ("smoke", "candidate"):
      pathlib.Path(out).write_text('{"survive": true, "fused_body_present": true, "program_count": 1}')
      return json_parse(pathlib.Path(out).read_text())
    if (mode, arm) == ("logits", "control"):
      digest = "a" * 64
      pathlib.Path(out).write_text(f'{{"tokens": [1], "logits_sha256": "{digest}", "shape": [1, 1, 151936]}}')
      return json_parse(pathlib.Path(out).read_text())
    if (mode, arm) == ("logits", "candidate"):
      raise ChildFailure("child failed rc=1: invalid diagnostic output at row 0", "invalid diagnostic output at row 0")
    raise AssertionError(f"unexpected child {mode}/{arm}")

  monkeypatch.setattr(ab_module, "_run_child", _run_child)
  record_path = tmp_path / "record.json"
  args = argparse.Namespace(model="/m", depth=512, count=32, max_context=1024, reps=5,
                            settled_continuous=True, timeout=600, lock_wait=90,
                            lock="/tmp/gpu-bench.lock", out=str(record_path))
  ab_module.ab(args)
  record = json_parse(record_path.read_text())
  assert record["verdict"] == "NO-GO"
  assert record["smoke"]["result"] == "PASS"
  assert record["logits_gate"]["run"] is True
  assert record["logits_gate"]["result"] == "NO-GO"
  assert "invalid diagnostic output at row 0" in record["logits_gate"]["stderr"]
  assert record["census"]["result"] == "NOT_AUTHORIZED"
  assert record["wall_bracket"]["result"] == "NOT_AUTHORIZED"
  assert calls == [("smoke", "candidate"), ("logits", "control"), ("logits", "candidate")]


def json_parse(text):
  import json
  return json.loads(text)
