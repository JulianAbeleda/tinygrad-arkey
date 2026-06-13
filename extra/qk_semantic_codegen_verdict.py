#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from typing import Any

DEFAULT_MODELS = ("8b", "14b")


def _read(path:pathlib.Path) -> dict[str, Any]:
  return json.loads(path.read_text())


def _write(path:pathlib.Path, data:dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(data, indent=2, sort_keys=True))


def _portable(path:pathlib.Path, repo:pathlib.Path) -> str:
  resolved = str(repo.resolve())
  text = str(path)
  if text == resolved: return "."
  return text.replace(resolved + "/", "")


def _decision_row(path:pathlib.Path, repo:pathlib.Path) -> dict[str, Any]:
  data = _read(path)
  return {
    "candidate": path.parent.name,
    "path": _portable(path, repo),
    "status": data.get("status"),
    "gain": data.get("gain"),
    "explicit_tok_s": (data.get("explicit") or {}).get("avg_tok_s"),
    "generated_tok_s": (data.get("generated") or {}).get("avg_tok_s"),
    "ab_match": data.get("ab_match"),
    "reasons": data.get("reasons") or [],
  }


def _model_report(base:pathlib.Path, model:str, repo:pathlib.Path) -> dict[str, Any]:
  root = base / model
  candidates = _read(root / "candidates.json")
  static_gate = _read(root / "static-gate.json")
  microbench = _read(root / "microbench.json")
  decisions = [_decision_row(path, repo) for path in sorted((root / "full-benchmark").glob("*/decision.json"))]
  micro_rows = microbench.get("rows") or []
  micro_accepts = [row for row in micro_rows if row.get("status") == "accept"]
  full_ready = [row for row in micro_accepts if row.get("full_decode_supported")]
  promoted = [row for row in decisions if row.get("status") == "accept"]
  return {
    "model": model.upper(),
    "candidate_summary": candidates.get("summary") or {},
    "static_gate_summary": static_gate.get("summary") or {},
    "microbench_summary": microbench.get("summary") or {},
    "microbench_accepts": [
      {
        "id": row["id"],
        "role": row.get("role"),
        "format": row.get("format"),
        "gain": row.get("gain"),
        "full_decode_supported": row.get("full_decode_supported"),
      }
      for row in micro_accepts
    ],
    "full_decode_ready": [row["id"] for row in full_ready],
    "full_decode_decisions": decisions,
    "promoted": [row["candidate"] for row in promoted],
    "status": "accept" if promoted else "reject",
  }


def build_verdict(base:pathlib.Path, *, repo:pathlib.Path=pathlib.Path.cwd(), models:tuple[str, ...]=DEFAULT_MODELS) -> dict[str, Any]:
  reports = [_model_report(base, model, repo) for model in models]
  full_decisions = [row for report in reports for row in report["full_decode_decisions"]]
  accepts = [row for row in full_decisions if row.get("status") == "accept"]
  rejects = [row for row in full_decisions if row.get("status") == "reject"]
  ties = [row for row in full_decisions if row.get("status") == "tie"]
  needs_rerun = [row for row in full_decisions if row.get("status") == "needs-rerun"]
  invalid = [row for row in full_decisions if row.get("status") == "invalid"]
  accepted_models = [report["model"] for report in reports if report["promoted"]]
  if invalid:
    overall = "semantic_codegen_v1_invalid"
  elif needs_rerun:
    overall = "semantic_codegen_v1_needs_rerun"
  elif len(accepted_models) == len(reports):
    overall = "semantic_codegen_v1_accept"
  elif accepts:
    overall = "semantic_codegen_v1_partial_accept"
  else:
    overall = "semantic_codegen_v1_rejected"
  reasons = []
  for report in reports:
    if report["promoted"]:
      reasons.append(f"{report['model']} promoted {', '.join(report['promoted'])}")
    elif report["full_decode_decisions"]:
      row = report["full_decode_decisions"][0]
      reasons.append(f"{report['model']} full decode {row['status']} {row['candidate']}: {row['gain'] * 100:.2f}%")
    else:
      reasons.append(f"{report['model']} had no full-decode candidate")
  run_32b = overall == "semantic_codegen_v1_accept"
  if not run_32b:
    reasons.append("32B skipped by default because the 8B/14B semantic codegen gate did not both accept")
  return {
    "kind": "qk_semantic_codegen_verdict",
    "schema_version": 1,
    "base": _portable(base, repo),
    "gate_models": [model.upper() for model in models],
    "models": reports,
    "summary": {
      "overall_decision": overall,
      "models": len(reports),
      "accepted_models": accepted_models,
      "microbench_accepts": sum(len(report["microbench_accepts"]) for report in reports),
      "full_decode_candidates": len(full_decisions),
      "full_decode_accepts": len(accepts),
      "full_decode_rejects": len(rejects),
      "full_decode_ties": len(ties),
      "full_decode_needs_rerun": len(needs_rerun),
      "full_decode_invalid": len(invalid),
      "run_32b": run_32b,
      "reasons": reasons,
    },
  }


def verdict_markdown(verdict:dict[str, Any]) -> str:
  lines = [
    "# QK Semantic Codegen v1 Verdict",
    "",
    "This is the 8B/14B gate for the first runtime-supported semantic codegen",
    "surface: exact-tensor Q4_K direct-output GEMV. 32B is intentionally",
    "excluded unless both target models show promise.",
    "",
    "## Summary",
    "",
    f"- overall decision: `{verdict['summary']['overall_decision']}`",
    f"- microbench accepts: `{verdict['summary']['microbench_accepts']}`",
    f"- full-decode candidates: `{verdict['summary']['full_decode_candidates']}`",
    f"- full-decode accepts: `{verdict['summary']['full_decode_accepts']}`",
    f"- run 32B: `{verdict['summary']['run_32b']}`",
    "",
    "Reasons:",
    "",
  ]
  lines += [f"- {reason}" for reason in verdict["summary"]["reasons"]]
  lines += [
    "",
    "## Models",
    "",
    "| model | microbench accepts | full-decode ready | full-decode status | gain % | reference tok/s | generated tok/s |",
    "|---|---:|---|---|---:|---:|---:|",
  ]
  for report in verdict["models"]:
    decision = (report["full_decode_decisions"] or [{}])[0]
    gain = decision.get("gain")
    explicit = decision.get("explicit_tok_s")
    generated = decision.get("generated_tok_s")
    lines.append(
      f"| {report['model']} | {len(report['microbench_accepts'])} | "
      f"`{', '.join(report['full_decode_ready']) or 'none'}` | `{decision.get('status', 'none')}` | "
      f"{'n/a' if gain is None else f'{gain * 100:.2f}'} | "
      f"{'n/a' if explicit is None else f'{explicit:.2f}'} | "
      f"{'n/a' if generated is None else f'{generated:.2f}'} |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
    "A microbench win is not promoted unless the exact tensor-scoped policy also",
    "wins a full autoregressive decode with greedy output A/B passing. This keeps",
    "the codegen surface pointed toward model-level throughput rather than",
    "standalone kernel scores.",
    "",
  ]
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Summarize QK semantic codegen gate results")
  parser.add_argument("--base", type=pathlib.Path, default=pathlib.Path("bench/qk-ansor-transition-20260612/semantic-codegen-v1"))
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=("8b", "14b"))
  args = parser.parse_args()
  verdict = build_verdict(args.base, repo=args.repo.resolve(), models=tuple(args.models))
  _write(args.json, verdict)
  args.md.parent.mkdir(parents=True, exist_ok=True)
  args.md.write_text(verdict_markdown(verdict))
  print(verdict_markdown(verdict))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
