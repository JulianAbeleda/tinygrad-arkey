#!/usr/bin/env python3
"""Bundle existing prefill baseline audit artifacts into one diagnosis report."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any


SCHEMA = "prefill-baseline-audit-bundle.v1"
DEFAULT_BASELINE = Path("bench/prefill-whole-synced/baseline-after-s10-warmstart-policy.json")
DEFAULT_CANDIDATE = Path("bench/prefill-whole-synced/s10-composed-warmstart-policy-authority.json")
DEFAULT_SCHEDULE_GATE = Path("bench/prefill-v2-schedule-table/latest.json")
DEFAULT_ROUTE_CENSUS = Path("bench/prefill-baseline-audit/route-census-structural.json")
DEFAULT_SHAPE_MATRIX = Path("bench/prefill-baseline-audit/hand-vs-generated-shape-matrix.json")
DEFAULT_OUTPUT = Path("bench/prefill-baseline-audit/latest.json")


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
  if not path.exists(): return None, "missing"
  raw = path.read_text()
  try: return json.loads(raw), None
  except json.JSONDecodeError as e:
    try:
      data, end = json.JSONDecoder().raw_decode(raw)
      trailing = raw[end:].strip()
      if isinstance(data, dict) and trailing:
        return data, "trailing non-json output: " + trailing.splitlines()[0][:240]
    except Exception:
      pass
    return None, f"{type(e).__name__}: {e}"
  except Exception as e: return None, f"{type(e).__name__}: {e}"


def _round(v: float | None, ndigits: int = 3) -> float | None:
  return None if v is None else round(v, ndigits)


def _num(v: Any) -> float | None:
  if v is None: return None
  try: return float(v)
  except (TypeError, ValueError): return None


def _pct_delta(candidate: Any, baseline: Any) -> float | None:
  c, b = _num(candidate), _num(baseline)
  if c is None or b is None or b == 0: return None
  return round((c - b) / b * 100.0, 2)


def _artifact_status(path: Path, data: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
  out = {"path": str(path), "present": data is not None, "error": error}
  if data is not None:
    out["schema"] = data.get("schema")
    if "verdict" in data: out["verdict"] = data.get("verdict")
  return out


def authority_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
  if report is None: return None
  whole = {str(k): _round(_num(v), 2) for k, v in report.get("whole_tok_s", {}).items()}
  chunk = {str(k): _round(_num(v), 3) for k, v in report.get("chunk_ms", {}).items()}
  return {
    "schema": report.get("schema"),
    "mode": report.get("mode"),
    "graph_gemm": report.get("graph_gemm"),
    "prefill_v2": report.get("prefill_v2"),
    "timing_authority": report.get("timing_authority"),
    "git_short": report.get("git_short"),
    "git_dirty": report.get("git_dirty"),
    "route_attribution": report.get("route_attribution", {}),
    "prefill_role_routes": report.get("prefill_role_routes", {}),
    "whole_tok_s": whole,
    "chunk_ms": chunk,
  }


def authority_comparison(baseline: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any] | None:
  if baseline is None or candidate is None: return None
  keys = sorted(set(str(k) for k in baseline.get("whole_tok_s", {})) | set(str(k) for k in candidate.get("whole_tok_s", {})),
                key=lambda x: int(x) if x.isdigit() else x)
  whole = {}
  for k in keys:
    b = baseline.get("whole_tok_s", {}).get(k, baseline.get("whole_tok_s", {}).get(int(k) if k.isdigit() else k))
    c = candidate.get("whole_tok_s", {}).get(k, candidate.get("whole_tok_s", {}).get(int(k) if k.isdigit() else k))
    whole[k] = {"baseline": _round(_num(b), 2), "candidate": _round(_num(c), 2), "delta_pct": _pct_delta(c, b)}
  return {
    "baseline_family": baseline.get("route_attribution", {}).get("prefill_route_family"),
    "candidate_family": candidate.get("route_attribution", {}).get("prefill_route_family"),
    "whole_tok_s": whole,
  }


def route_census_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
  if report is None: return None
  rows = report.get("rows", [])
  ok = [r for r in rows if r.get("structure_status") == "ok"]
  failures = [r for r in rows if r.get("structure_status") != "ok"]

  def worst(field: str) -> dict[str, Any] | None:
    pool = [r for r in ok if _num(r.get(field)) is not None]
    if not pool: return None
    r = max(pool, key=lambda x: float(x.get(field, 0.0)))
    return {"route": r.get("route"), "shape": r.get("shape"), field: r.get(field)}

  by_route = {}
  for r in ok:
    by_route.setdefault(r.get("route", "?"), {
      "family": r.get("family"), "shape": r.get("shape"), "wmma_count": r.get("wmma_count"),
      "inst_per_wmma": r.get("inst_per_wmma"), "waitcnt_per_wmma": r.get("waitcnt_per_wmma"),
      "global_b128_per_wmma": r.get("global_b128_per_wmma"),
      "ds_store_b128_per_wmma": r.get("ds_store_b128_per_wmma"),
      "ds_load_b128_per_wmma": r.get("ds_load_b128_per_wmma"),
      "future_slot_before_compute": r.get("future_slot_before_compute"),
      "dbuf_d7_ok": r.get("dbuf_d7_ok"),
    })
  return {
    "m": report.get("m"), "n": report.get("n"), "k": report.get("k"),
    "row_count": len(rows), "ok_count": len(ok), "failure_count": len(failures),
    "worst_inst_per_wmma": worst("inst_per_wmma"),
    "worst_waitcnt_per_wmma": worst("waitcnt_per_wmma"),
    "routes": by_route,
    "failures": [{"route": r.get("route"), "shape": r.get("shape"), "status": r.get("structure_status"),
                  "error": r.get("structure_error") or r.get("message")} for r in failures[:12]],
  }


def shape_matrix_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
  if report is None: return None
  rows = []
  for r in report.get("rows", []):
    g, h = r.get("generated", {}), r.get("hand_lds2", {})
    rows.append({
      "shape": r.get("shape"),
      "generated_status": g.get("status"),
      "generated_tflops": _round(_num(g.get("tflops")), 2),
      "hand_status": h.get("status"),
      "hand_tflops": _round(_num(h.get("tflops")), 2),
      "generated_vs_hand_pct": _pct_delta(g.get("tflops"), h.get("tflops")),
      "generated_inst_per_wmma": g.get("inst_per_wmma"),
      "hand_inst_per_wmma": h.get("inst_per_wmma"),
      "generated_waitcnt_per_wmma": g.get("waitcnt_per_wmma"),
      "hand_waitcnt_per_wmma": h.get("waitcnt_per_wmma"),
    })
  return {
    "m": report.get("m"), "n": report.get("n"), "k": report.get("k"),
    "generated_env": report.get("generated_env"),
    "rows": rows,
  }


def schedule_gate_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
  if report is None: return None
  return {
    "schema": report.get("schema"),
    "verdict": report.get("verdict"),
    "route_id": report.get("route_id"),
    "shape_count": report.get("shape_count"),
    "shapes": report.get("shapes"),
    "evidence": report.get("evidence", {}),
    "rows": [{"shape": r.get("shape"), "params": r.get("params"), "measured": r.get("measured")}
             for r in report.get("rows", [])],
    "remaining_blocker": report.get("remaining_blocker"),
  }


def next_commands(args: argparse.Namespace) -> dict[str, str]:
  py = "PYTHONPATH=."
  return {
    "baseline_authority": f"{py} DEV=AMD python3 extra/qk/prefill_whole_synced.py --mode authority --artifact {args.baseline} --json",
    "candidate_authority": f"{py} DEV=AMD PREFILL_GRAPH_GEMM=1 PREFILL_WMMA_PIPE_PRIMITIVE=1 PREFILL_WMMA_LDS_PRIMITIVE=1 PREFILL_DBUF=1 PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE=1 python3 extra/qk/prefill_whole_synced.py --mode authority --require-route prefill_wmma_pipe_lds_dbuf_primitive_generated --artifact {args.candidate} --json",
    "schedule_gate": f"{py} DEV=AMD:ISA python3 extra/qk/prefill_v2_schedule_table_gate.py --shapes 5120x5120 --run-amd",
    "route_census": f"mkdir -p {args.route_census.parent} && {py} DEV=AMD:ISA python3 extra/qk/prefill/prefill_route_census.py --structural-only --routes generated-direct,generated-kmajor,hand-lds2 --shapes '2,2;4,2;2,4' --json > {args.route_census}",
    "shape_matrix": f"mkdir -p {args.shape_matrix.parent} && {py} DEV=AMD:ISA python3 extra/qk/prefill/hand_vs_generated_shape_matrix.py --shapes '2,2;4,2;2,4' --generated-env dbuf-safe --json > {args.shape_matrix}",
  }


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
  loaded = {}
  for name, path in (("baseline", args.baseline), ("candidate", args.candidate), ("schedule_gate", args.schedule_gate),
                     ("route_census", args.route_census), ("shape_matrix", args.shape_matrix)):
    loaded[name] = (*_load_json(path), path)
  baseline, base_err, _ = loaded["baseline"]
  candidate, cand_err, _ = loaded["candidate"]
  schedule, sched_err, _ = loaded["schedule_gate"]
  census, census_err, _ = loaded["route_census"]
  matrix, matrix_err, _ = loaded["shape_matrix"]
  missing_required = [k for k in ("baseline",) if loaded[k][0] is None]
  missing_optional = [k for k in ("candidate", "schedule_gate", "route_census", "shape_matrix") if loaded[k][0] is None]
  verdict = "BASELINE_AUDIT_BLOCKED_MISSING_BASELINE" if missing_required else \
            ("BASELINE_AUDIT_READY_WITH_GAPS" if missing_optional else "BASELINE_AUDIT_READY")
  command_key_for_layer = {
    "baseline": "baseline_authority", "candidate": "candidate_authority", "schedule_gate": "schedule_gate",
    "route_census": "route_census", "shape_matrix": "shape_matrix",
  }
  all_next_commands = next_commands(args)
  missing_layers = missing_required + missing_optional
  bundle = {
    "schema": SCHEMA,
    "verdict": verdict,
    "artifacts": {
      "baseline": _artifact_status(args.baseline, baseline, base_err),
      "candidate": _artifact_status(args.candidate, candidate, cand_err),
      "schedule_gate": _artifact_status(args.schedule_gate, schedule, sched_err),
      "route_census": _artifact_status(args.route_census, census, census_err),
      "shape_matrix": _artifact_status(args.shape_matrix, matrix, matrix_err),
    },
    "baseline": authority_summary(baseline),
    "candidate": authority_summary(candidate),
    "comparison": authority_comparison(baseline, candidate),
    "schedule_gate": schedule_gate_summary(schedule),
    "route_census": route_census_summary(census),
    "shape_matrix": shape_matrix_summary(matrix),
    "missing_layers": missing_layers,
    "next_commands": {command_key_for_layer[layer]: all_next_commands[command_key_for_layer[layer]]
                      for layer in missing_layers},
  }
  return bundle


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
  ap.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
  ap.add_argument("--schedule-gate", type=Path, default=DEFAULT_SCHEDULE_GATE)
  ap.add_argument("--route-census", type=Path, default=DEFAULT_ROUTE_CENSUS)
  ap.add_argument("--shape-matrix", type=Path, default=DEFAULT_SHAPE_MATRIX)
  ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--no-write", action="store_true")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)
  bundle = build_bundle(args)
  if not args.no_write:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
  if args.json:
    print(json.dumps(bundle, indent=2))
  else:
    print(f"{bundle['verdict']} -> {args.output if not args.no_write else '(not written)'}")
    if bundle["missing_layers"]: print("missing layers: " + ", ".join(bundle["missing_layers"]))
  return 0 if not bundle["verdict"].endswith("MISSING_BASELINE") else 2


if __name__ == "__main__":
  raise SystemExit(main())
