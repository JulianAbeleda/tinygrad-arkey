#!/usr/bin/env python3
"""S10 LDS2 ownership baseline-freeze artifact.

This is an artifact runner only. It reuses the existing micro/whole harness
entrypoints when explicitly requested, and otherwise records those smokes as
not_run with a blocker instead of inventing timing logic here.
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from extra.qk.prefill_harness import DEFAULT_MODEL, prefill_authority_argv, prefill_run_profile, prefill_subprocess_env

DEFAULT_S9_DIR = ROOT / "bench/prefill-lds2-s9"
DEFAULT_WHOLE_DIR = ROOT / "bench/prefill-whole-synced"
DEFAULT_OUTPUT = ROOT / "bench/prefill-s10-lds2-ownership/baseline-freeze.json"
S9_FINAL = "final-report.json"
S9_ROOFLINE = "roofline-audit.json"
S9_DEFAULT_WHOLE = "raw-hand-s9-combined-default-authority.json"
S9_BEST_WHOLE = "raw-hand-s9-combined-best-authority.json"
S10_CLASSIFICATION = "compiler_primitive_spec_owned__asm_backend_atom"
S10_ROUTE_ENV = {
  "PREFILL_GRAPH_GEMM": "1",
  "PREFILL_WMMA_PIPE_PRIMITIVE": "1",
  "PREFILL_WMMA_LDS_PRIMITIVE": "1",
  "PREFILL_DBUF": "1",
}


def _load_json(path: pathlib.Path) -> tuple[dict[str, Any] | None, str | None]:
  try:
    data = json.loads(path.read_text())
  except FileNotFoundError:
    return None, "missing"
  except Exception as e:
    return None, f"invalid_json: {e}"
  return (data, None) if isinstance(data, dict) else (None, "json_root_not_object")


def _artifact_ref(path: pathlib.Path, data: dict[str, Any] | None, error: str | None) -> dict[str, Any]:
  out = {"path": str(path), "present": error is None}
  if error is not None: out["error"] = error
  if data is not None and data.get("schema") is not None: out["schema"] = data.get("schema")
  return out


def _whole_band(data: dict[str, Any] | None) -> dict[str, Any] | None:
  if not data: return None
  whole = data.get("whole_tok_s")
  if not isinstance(whole, dict): return None
  return {k: whole.get(k) for k in ("512", "1024", "2048", "4096") if k in whole}


def summarize_s9_artifacts(s9_dir: pathlib.Path = DEFAULT_S9_DIR,
                           whole_dir: pathlib.Path = DEFAULT_WHOLE_DIR) -> dict[str, Any]:
  final_path = s9_dir / S9_FINAL
  roofline_path = s9_dir / S9_ROOFLINE
  default_path = whole_dir / S9_DEFAULT_WHOLE
  best_path = whole_dir / S9_BEST_WHOLE
  final, final_err = _load_json(final_path)
  roofline, roofline_err = _load_json(roofline_path)
  default_whole, default_err = _load_json(default_path)
  best_whole, best_err = _load_json(best_path)

  route_attr = default_whole.get("route_attribution", {}) if default_whole else {}
  if not isinstance(route_attr, dict): route_attr = {}
  final_authority = final.get("whole_prefill_authority", {}) if final else {}
  if not isinstance(final_authority, dict): final_authority = {}
  roofline_shape = roofline.get("shape") if roofline else None

  return {
    "s9_complete_state": "S9_COMPLETE_KEEP_OPT_IN" if final and final.get("verdict") == "keep_opt_in" else "unknown_or_incomplete",
    "default_vs_opt_in_decision": final.get("verdict") if final else None,
    "roofline_verdict": roofline.get("verdict") if roofline else None,
    "active_shape": roofline_shape,
    "current_route_id": route_attr.get("prefill_route_family"),
    "current_route_provenance": route_attr.get("prefill_route_provenance"),
    "current_route_pure": route_attr.get("prefill_route_pure"),
    "current_route_rolled_back": route_attr.get("prefill_route_rolled_back"),
    "current_role_classification": "external_handwritten_kernel" if route_attr.get("prefill_route_rolled_back") else "unknown",
    "s10_target_role_classification": S10_CLASSIFICATION,
    "whole_prefill_baseline_band": {
      "default": _whole_band(default_whole),
      "best_opt_in": _whole_band(best_whole),
      "final_report_summary": {
        "status": final_authority.get("status"),
        "baseline_pp512_median": final_authority.get("baseline_pp512_median"),
        "baseline_pp4096_median": final_authority.get("baseline_pp4096_median"),
        "best_pp512": final_authority.get("best_pp512"),
        "best_pp4096": final_authority.get("best_pp4096"),
        "materiality_vs_baseline": final_authority.get("materiality_vs_baseline"),
      },
    },
    "artifacts": {
      "s9_final_report": _artifact_ref(final_path, final, final_err),
      "s9_roofline_audit": _artifact_ref(roofline_path, roofline, roofline_err),
      "s9_default_whole": _artifact_ref(default_path, default_whole, default_err),
      "s9_best_whole": _artifact_ref(best_path, best_whole, best_err),
    },
  }


def _not_run(blocker: str) -> dict[str, Any]:
  return {"status": "not_run", "blocker": blocker}


def _run_command(name: str, argv: list[str], env: dict[str, str], cwd: pathlib.Path, timeout_s: int) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    proc = subprocess.run([sys.executable, *argv], cwd=cwd, env={**os.environ, **env}, text=True,
                          capture_output=True, timeout=timeout_s)
  except Exception as e:
    return {"status": "blocker", "name": name, "argv": argv, "blocker": f"{type(e).__name__}: {e}",
            "elapsed_s": round(time.perf_counter() - t0, 3)}
  status = "ok" if proc.returncode == 0 else "blocker"
  return {"status": status, "name": name, "argv": argv, "returncode": proc.returncode,
          "elapsed_s": round(time.perf_counter() - t0, 3),
          "stdout_tail": proc.stdout.splitlines()[-20:], "stderr_tail": proc.stderr.splitlines()[-20:]}


def run_micro_smoke(run: bool, cwd: pathlib.Path = ROOT, timeout_s: int = 600) -> dict[str, Any]:
  argv = ["extra/qk/prefill/hand_vs_generated_shape_matrix.py", "--shapes", "2,2", "--hand-reps", "1",
          "--hand-iters", "1", "--json"]
  if not run: return _not_run("pass --run-micro-smoke to invoke existing hand_vs_generated_shape_matrix.py")
  return _run_command("micro_smoke", argv, {"PYTHONPATH": str(cwd), **S10_ROUTE_ENV}, cwd, timeout_s)


def run_whole_smoke(run: bool, model: str = DEFAULT_MODEL, cwd: pathlib.Path = ROOT, timeout_s: int = 1800) -> dict[str, Any]:
  prof = prefill_run_profile("smoke")
  argv = prefill_authority_argv(model, prof, artifact=False)
  if not run: return _not_run("pass --run-whole-smoke to invoke existing prefill_whole_synced.py smoke authority")
  return _run_command("whole_smoke", argv, prefill_subprocess_env(S10_ROUTE_ENV), cwd, timeout_s)


def build_artifact(s9_dir: pathlib.Path = DEFAULT_S9_DIR, whole_dir: pathlib.Path = DEFAULT_WHOLE_DIR,
                   *, run_micro: bool = False, run_whole: bool = False, model: str = DEFAULT_MODEL,
                   cwd: pathlib.Path = ROOT) -> dict[str, Any]:
  return {
    "schema": "prefill-s10-lds2-ownership-baseline-freeze.v1",
    "phase": "S10.0 baseline freeze",
    "s9_summary": summarize_s9_artifacts(s9_dir, whole_dir),
    "smokes": {
      "micro": run_micro_smoke(run_micro, cwd=cwd),
      "whole_prefill": run_whole_smoke(run_whole, model=model, cwd=cwd),
    },
  }


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--s9-dir", type=pathlib.Path, default=DEFAULT_S9_DIR)
  ap.add_argument("--whole-dir", type=pathlib.Path, default=DEFAULT_WHOLE_DIR)
  ap.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
  ap.add_argument("--model", default=os.environ.get("QK_MODEL", DEFAULT_MODEL))
  ap.add_argument("--run-micro-smoke", action="store_true")
  ap.add_argument("--run-whole-smoke", action="store_true")
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args(argv)
  payload = build_artifact(args.s9_dir, args.whole_dir, run_micro=args.run_micro_smoke,
                           run_whole=args.run_whole_smoke, model=args.model)
  out = args.output if args.output.is_absolute() else ROOT / args.output
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
  if args.json:
    print(json.dumps(payload, indent=2, allow_nan=False))
  else:
    s9 = payload["s9_summary"]
    print(f"{s9['s9_complete_state']} decision={s9['default_vs_opt_in_decision']} output={out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
