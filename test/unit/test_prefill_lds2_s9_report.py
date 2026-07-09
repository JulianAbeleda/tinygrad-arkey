import json

from extra.qk.prefill import lds2_s9_report as report


def _write(path, payload):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + "\n")


def _wait(rows):
  return {"schema": "prefill-lds2-s9-wait-search.v1", "rows": rows}


def _row(candidate_id, tflops, status="ok", wait_policy=None):
  return {
    "candidate_id": candidate_id,
    "status": status,
    "tflops": tflops,
    "wait_policy": wait_policy or {
      "vm_after_coop_load": candidate_id,
      "lgkm_after_coop_store": 0,
      "lgkm_after_frag_load": 0,
    },
  }


def _whole(pp512, pp4096, gate=None):
  payload = {"schema": "prefill-whole-synced-authority.v1", "whole_tok_s": {"512": pp512, "4096": pp4096}}
  if gate is not None:
    payload["prefill_route_binding_gate"] = gate
  return payload


def test_s9_report_promotes_when_micro_and_whole_prefill_are_clean(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  zero = report.ZERO_WAIT_POLICY
  _write(s9_dir / "wait-search.json", _wait([_row(0, 100.0, wait_policy=zero), _row(1, 104.0)]))
  _write(s9_dir / "wait-search-repeat.json", _wait([_row(0, 102.0, wait_policy=zero), _row(1, 106.0)]))
  _write(whole_dir / "s9-repeat-default-a.json", _whole(4400.0, 3200.0))
  _write(whole_dir / "s9-repeat-candidate-a.json", _whole(4420.0, 3220.0))

  out = report.build_report(s9_dir, whole_dir, min_micro_speedup=0.01)

  assert out["verdict"] == "promote"
  assert out["micro_tflops"]["baseline_median"] == 101.0
  assert out["micro_tflops"]["best_median"] == 105.0
  assert out["whole_prefill"]["pp512_median"] == 4410.0
  assert out["whole_prefill"]["pp4096_median"] == 3210.0


def test_s9_report_keeps_opt_in_when_route_binding_fails(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _write(s9_dir / "wait-search.json", _wait([
    _row(0, 100.0, wait_policy=report.ZERO_WAIT_POLICY),
    _row(2, 104.0, wait_policy={"vm_after_coop_load": 0, "lgkm_after_coop_store": 2, "lgkm_after_frag_load": 0}),
  ]))
  _write(whole_dir / "raw-hand-s9-wait-store2-authority.json", _whole(4400.0, 3200.0, {
    "verdict": "PREFILL_ROUTE_BINDING_FAIL",
    "failures": ["prefill_route_pure is not true"],
  }))

  out = report.build_report(s9_dir, whole_dir, min_micro_speedup=0.01)

  assert out["verdict"] == "keep_opt_in"
  assert out["whole_prefill"]["binding_failure_count"] == 1


def test_s9_report_rejects_correctness_when_no_candidate_is_ok(tmp_path):
  s9_dir = tmp_path / "s9"
  whole_dir = tmp_path / "whole"
  _write(s9_dir / "wait-search.json", _wait([_row(0, 0.0, status="WRONG rr=nan", wait_policy=report.ZERO_WAIT_POLICY)]))

  out = report.build_report(s9_dir, whole_dir)

  assert out["verdict"] == "reject_correctness"


def test_s9_report_is_inconclusive_without_s9_artifacts(tmp_path):
  out = report.build_report(tmp_path / "missing-s9", tmp_path / "missing-whole")

  assert out["verdict"] == "inconclusive"
  assert "no S9 wait-search artifacts found" in out["failures"]
