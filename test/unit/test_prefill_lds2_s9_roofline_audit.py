import json

from extra.qk.prefill import lds2_s9_roofline_audit as roof


def _write(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + "\n")


def test_roofline_audit_keeps_opt_in_when_whole_gain_is_tiny(tmp_path):
  s9 = tmp_path / "s9"
  whole = tmp_path / "whole"
  _write(s9 / "combined-search.json", {
    "shape": {"m": 512, "n": 12288, "k": 4096},
    "baseline_tflops": 74.68,
    "best_tflops": 75.49,
  })
  _write(s9 / "final-report.json", {"axes": {"combined": {"best_correct_candidate": {"candidate_id": 7}}}})
  _write(whole / "raw-hand-s9-combined-default-authority.json", {"whole_tok_s": {"512": 4388.33, "4096": 3229.36}})
  _write(whole / "raw-hand-s9-combined-best-authority.json", {"whole_tok_s": {"512": 4413.43, "4096": 3236.92}})

  out = roof.build_audit(s9, whole)

  assert out["roofline"]["bound"] == "compute"
  assert out["micro"]["best_roofline_efficiency"] > out["micro"]["baseline_roofline_efficiency"]
  assert out["whole_prefill"]["max_speedup"] < 0.01
  assert out["verdict"] == "S9_ROOFLINE_KEEP_OPT_IN"


def test_roofline_audit_promotes_when_whole_and_efficiency_clear_thresholds(tmp_path):
  s9 = tmp_path / "s9"
  whole = tmp_path / "whole"
  _write(s9 / "combined-search.json", {
    "shape": {"m": 512, "n": 12288, "k": 4096},
    "baseline_tflops": 74.0,
    "best_tflops": 76.0,
  })
  _write(whole / "raw-hand-s9-combined-default-authority.json", {"whole_tok_s": {"512": 4300.0}})
  _write(whole / "raw-hand-s9-combined-best-authority.json", {"whole_tok_s": {"512": 4400.0}})

  out = roof.build_audit(s9, whole, roofline_efficiency_point_threshold=0.005)

  assert out["verdict"] == "S9_ROOFLINE_PROMOTE_DEFAULT"
