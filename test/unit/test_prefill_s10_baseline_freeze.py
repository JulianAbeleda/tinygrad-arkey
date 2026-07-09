import json

from extra.qk.prefill import s10_baseline_freeze as freeze


def _write(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + "\n")


def test_s10_baseline_freeze_summarizes_s9_artifacts(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _write(s9_dir / "final-report.json", {
    "schema": "prefill-lds2-s9-final-report.v1",
    "verdict": "keep_opt_in",
    "whole_prefill_authority": {
      "status": "valid",
      "baseline_pp512_median": 100.0,
      "baseline_pp4096_median": 90.0,
      "best_pp512": 101.0,
      "best_pp4096": 90.5,
      "materiality_vs_baseline": {"is_material": False},
    },
  })
  _write(s9_dir / "roofline-audit.json", {
    "schema": "prefill-lds2-s9-roofline-audit.v1",
    "verdict": "S9_ROOFLINE_KEEP_OPT_IN",
    "shape": {"m": 512, "n": 12288, "k": 4096},
  })
  route_attribution = {
    "prefill_route_family": "prefill_pipe_role_selective_generated",
    "prefill_route_pure": False,
    "prefill_route_rolled_back": True,
    "prefill_route_provenance": "external_handwritten_kernel",
  }
  _write(whole_dir / "raw-hand-s9-combined-default-authority.json", {
    "schema": "prefill-whole-synced-authority.v1",
    "whole_tok_s": {"512": 4388.33, "4096": 3229.36},
    "route_attribution": route_attribution,
  })
  _write(whole_dir / "raw-hand-s9-combined-best-authority.json", {
    "schema": "prefill-whole-synced-authority.v1",
    "whole_tok_s": {"512": 4413.43, "4096": 3236.92},
    "route_attribution": route_attribution,
  })

  out = freeze.summarize_s9_artifacts(s9_dir, whole_dir)

  assert out["s9_complete_state"] == "S9_COMPLETE_KEEP_OPT_IN"
  assert out["default_vs_opt_in_decision"] == "keep_opt_in"
  assert out["roofline_verdict"] == "S9_ROOFLINE_KEEP_OPT_IN"
  assert out["active_shape"] == {"m": 512, "n": 12288, "k": 4096}
  assert out["current_route_id"] == "prefill_pipe_role_selective_generated"
  assert out["current_role_classification"] == "external_handwritten_kernel"
  assert out["whole_prefill_baseline_band"]["default"] == {"512": 4388.33, "4096": 3229.36}
  assert out["whole_prefill_baseline_band"]["best_opt_in"] == {"512": 4413.43, "4096": 3236.92}


def test_s10_baseline_freeze_records_missing_artifacts_and_not_run_smokes(tmp_path):
  out = freeze.build_artifact(tmp_path / "missing-s9", tmp_path / "missing-whole")

  assert out["schema"] == "prefill-s10-lds2-ownership-baseline-freeze.v1"
  assert out["s9_summary"]["s9_complete_state"] == "unknown_or_incomplete"
  assert out["s9_summary"]["artifacts"]["s9_final_report"]["present"] is False
  assert out["s9_summary"]["artifacts"]["s9_final_report"]["error"] == "missing"
  assert out["smokes"]["micro"]["status"] == "not_run"
  assert "hand_vs_generated_shape_matrix.py" in out["smokes"]["micro"]["blocker"]
  assert out["smokes"]["whole_prefill"]["status"] == "not_run"
  assert "prefill_whole_synced.py" in out["smokes"]["whole_prefill"]["blocker"]


def test_s10_baseline_freeze_optional_smokes_reuse_existing_harnesses(monkeypatch, tmp_path):
  calls = []

  class Proc:
    returncode = 0
    stdout = "ok\n"
    stderr = ""

  def fake_run(argv, cwd, env, text, capture_output, timeout):
    calls.append({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout, "text": text,
                  "capture_output": capture_output})
    return Proc()

  monkeypatch.setattr(freeze.subprocess, "run", fake_run)

  micro = freeze.run_micro_smoke(True, cwd=tmp_path, timeout_s=7)
  whole = freeze.run_whole_smoke(True, model="/tmp/model.gguf", cwd=tmp_path, timeout_s=9)

  assert micro["status"] == "ok"
  assert whole["status"] == "ok"
  assert calls[0]["argv"][1] == "extra/qk/prefill/hand_vs_generated_shape_matrix.py"
  assert "--hand-reps" in calls[0]["argv"]
  assert calls[1]["argv"][1] == "extra/qk/prefill_whole_synced.py"
  assert "--mode" in calls[1]["argv"] and "smoke" in calls[1]["argv"]
  assert "--no-artifact" in calls[1]["argv"]
  assert calls[1]["env"]["PREFILL_V2"] == "1"
