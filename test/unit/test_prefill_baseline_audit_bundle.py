import argparse, json
from pathlib import Path

from extra.qk.prefill import baseline_audit_bundle as audit


def _write(path: Path, payload: dict):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(payload) + "\n")


def _args(tmp_path, **kw):
  base = {
    "baseline": tmp_path / "baseline.json",
    "candidate": tmp_path / "candidate.json",
    "schedule_gate": tmp_path / "schedule.json",
    "route_census": tmp_path / "census.json",
    "shape_matrix": tmp_path / "matrix.json",
    "s9_report": tmp_path / "s9.json",
    "output": tmp_path / "out.json",
    "no_write": True,
    "json": False,
  }
  base.update(kw)
  return argparse.Namespace(**base)


def test_baseline_audit_bundle_compares_authority_artifacts(tmp_path):
  args = _args(tmp_path)
  _write(args.baseline, {
    "schema": "prefill-whole-synced-authority.v1",
    "mode": "authority",
    "graph_gemm": False,
    "whole_tok_s": {"512": 100.0, "4096": 80.0},
    "chunk_ms": {"0": 1.25},
    "route_attribution": {"prefill_route_family": "baseline-route"},
  })
  _write(args.candidate, {
    "schema": "prefill-whole-synced-authority.v1",
    "mode": "authority",
    "graph_gemm": True,
    "whole_tok_s": {"512": 90.0, "4096": 84.0},
    "route_attribution": {"prefill_route_family": "candidate-route"},
  })
  _write(args.schedule_gate, {"schema": "prefill-v2-schedule-table-gate.v1", "verdict": "PASS", "rows": []})
  _write(args.route_census, {
    "m": 512, "n": 5120, "k": 5120,
    "rows": [
      {"route": "generated", "shape": "2x2", "family": "generated", "structure_status": "ok",
       "wmma_count": 16, "inst_per_wmma": 31.0, "waitcnt_per_wmma": 2.5},
      {"route": "bad", "shape": "4x4", "structure_status": "AttributeError", "structure_error": "boom"},
    ],
  })
  _write(args.shape_matrix, {
    "m": 512, "n": 5120, "k": 5120, "generated_env": "dbuf-safe",
    "rows": [{"shape": "2x2", "generated": {"status": "ok", "tflops": 30.0},
              "hand_lds2": {"status": "ok", "tflops": 60.0}}],
  })
  _write(args.s9_report, {
    "schema": "prefill-lds2-s9-report.v1",
    "verdict": "keep_opt_in",
    "whole_prefill": {"pp512_median": 100.0, "pp4096_median": 90.0},
    "whole_prefill_detail": {"pp512_samples": [99.0, 100.0], "pp4096_samples": [89.0, 90.0]},
  })

  out = audit.build_bundle(args)

  assert out["verdict"] == "BASELINE_AUDIT_READY"
  assert out["comparison"]["whole_tok_s"]["512"]["delta_pct"] == -10.0
  assert out["comparison"]["whole_tok_s"]["4096"]["delta_pct"] == 5.0
  assert out["route_census"]["ok_count"] == 1
  assert out["route_census"]["failure_count"] == 1
  assert out["shape_matrix"]["rows"][0]["generated_vs_hand_pct"] == -50.0
  assert out["s9_authority"]["pp512_median"] == 100.0
  assert out["promotion_gates"]["s9_98pct"]["pp512"]["pass"] is False
  assert out["promotion_gates"]["pure_baseline"]["pp4096"]["pass"] is True
  assert out["missing_layers"] == []


def test_baseline_audit_bundle_reports_missing_optional_layers(tmp_path):
  args = _args(tmp_path)
  _write(args.baseline, {"schema": "prefill-whole-synced-authority.v1", "whole_tok_s": {"512": 123.0}})

  out = audit.build_bundle(args)

  assert out["verdict"] == "BASELINE_AUDIT_READY_WITH_GAPS"
  assert out["artifacts"]["baseline"]["present"] is True
  assert out["artifacts"]["route_census"]["error"] == "missing"
  assert set(out["missing_layers"]) == {"candidate", "schedule_gate", "route_census", "shape_matrix", "s9_report"}
  assert "prefill_route_census.py" in out["next_commands"]["route_census"]
  assert "hand_vs_generated_shape_matrix.py" in out["next_commands"]["shape_matrix"]


def test_baseline_audit_bundle_blocks_without_baseline(tmp_path):
  out = audit.build_bundle(_args(tmp_path))

  assert out["verdict"] == "BASELINE_AUDIT_BLOCKED_MISSING_BASELINE"
  assert "baseline_authority" in out["next_commands"]


def test_baseline_audit_bundle_keeps_valid_json_with_trailing_diagnostic(tmp_path):
  p = tmp_path / "artifact.json"
  p.write_text('{"rows": []}\nAMD synchronization failed before finalizing: timeout\n')

  data, err = audit._load_json(p)

  assert data == {"rows": []}
  assert err.startswith("trailing non-json output:")
