import json

from extra.qk.prefill import lds2_s9_final_report as report


def _write(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + "\n")


def _search(rows, baseline=100.0, best=100.0):
  return {
    "schema": "prefill-lds2-s9-test.v1",
    "baseline_candidate_id": 0,
    "baseline_tflops": baseline,
    "best_tflops": best,
    "rows": rows,
  }


def _row(candidate_id, tflops, status="ok", name=None):
  row = {"candidate_id": candidate_id, "status": status, "tflops": tflops, "rel_rmse": 1e-3}
  if name is not None: row["name"] = name
  return row


def _required_axes(s9_dir, best=101.0):
  rows = [_row(0, 100.0, name="baseline"), _row(1, best), _row(2, 0.0, "WRONG rr=nan")]
  _write(s9_dir / "wait-search.json", _search(rows, best=best))
  _write(s9_dir / "layout-search.json", _search(rows, best=best))
  _write(s9_dir / "lifecycle-search.json", _search(rows, best=best))


def _whole(path, pp512, pp4096=100.0):
  _write(path, {"schema": "prefill-whole-synced-authority.v1", "whole_tok_s": {"512": pp512, "4096": pp4096}})


def test_final_report_blocks_micro_combined_win_without_authority(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _required_axes(s9_dir)
  _write(s9_dir / "combined-search.json", _search([_row(0, 100.0, name="baseline"), _row(3, 106.0)], best=106.0))

  out = report.build_report(s9_dir, whole_dir, material_threshold=0.03)

  assert out["verdict"] == "blocked"
  assert out["axis_status"]["combined"] == "material_win"
  assert "combined micro win needs whole-prefill authority before default promotion" in out["reasons"]


def test_final_report_promotes_with_authority_material_candidate(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _required_axes(s9_dir)
  _write(s9_dir / "combined-search.json", _search([_row(0, 100.0, name="baseline"), _row(3, 106.0)], best=106.0))
  _whole(whole_dir / "raw-hand-s9-default-authority.json", 100.0, 100.0)
  _whole(whole_dir / "raw-hand-s9-best-authority.json", 104.0, 103.1)

  out = report.build_report(s9_dir, whole_dir, material_threshold=0.03)

  assert out["verdict"] == "promote_default"
  assert out["whole_prefill_authority"]["best_pp512"] == 104.0
  assert out["axes"]["wait"]["rejected_or_wrong_candidates"][0]["status"] == "WRONG rr=nan"


def test_final_report_keeps_opt_in_when_authority_refutes_combined_micro_win(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _required_axes(s9_dir)
  _write(s9_dir / "combined-search.json", _search([_row(0, 100.0, name="baseline"), _row(3, 106.0)], best=106.0))
  _whole(whole_dir / "raw-hand-s9-default-authority.json", 100.0, 100.0)
  _whole(whole_dir / "raw-hand-s9-best-authority.json", 101.0, 100.5)

  out = report.build_report(s9_dir, whole_dir, material_threshold=0.03)

  assert out["verdict"] == "keep_opt_in"
  assert "combined micro win did not clear whole-prefill authority threshold" in out["reasons"]


def test_final_report_keeps_opt_in_for_axis_local_win_only(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _required_axes(s9_dir, best=106.0)
  _write(s9_dir / "combined-search.json", _search([_row(0, 100.0, name="baseline"), _row(1, 101.0)], best=101.0))

  out = report.build_report(s9_dir, whole_dir, material_threshold=0.03)

  assert out["verdict"] == "keep_opt_in"
  assert "only axis-local material wins are present" in out["reasons"]


def test_final_report_blocks_when_required_axis_is_missing(tmp_path):
  s9_dir = tmp_path / "s9"
  _write(s9_dir / "wait-search.json", _search([_row(0, 100.0, name="baseline")]))
  _write(s9_dir / "layout-search.json", _search([_row(0, 100.0, name="baseline")]))

  out = report.build_report(s9_dir, tmp_path / "whole")

  assert out["verdict"] == "blocked"
  assert out["axis_status"]["lifecycle"] == "missing"


def test_final_report_rejects_when_correct_candidates_do_not_beat_baseline(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _required_axes(s9_dir, best=101.0)
  _write(s9_dir / "combined-search.json", _search([_row(0, 100.0, name="baseline"), _row(1, 101.0)], best=101.0))

  out = report.build_report(s9_dir, whole_dir, material_threshold=0.03)

  assert out["verdict"] == "reject"
  assert out["axes"]["combined"]["best_correct_candidate"]["candidate_id"] == 1
