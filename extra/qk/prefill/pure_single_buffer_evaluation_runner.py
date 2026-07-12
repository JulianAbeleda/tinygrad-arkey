#!/usr/bin/env python3
"""Artifact adapter for the pure single-buffer evaluation gate.

This runner does not compile, execute, check numerics, or time kernels.  It
adapts artifacts emitted by those existing authorities and proves that all of
them describe the same candidate, executable binary, and source commit.
"""
from __future__ import annotations

import argparse, json, pathlib
from dataclasses import dataclass
from typing import Any

from extra.qk.prefill.pure_single_buffer_evaluation_gate import EvaluationAuthorities, evaluate

SCHEMA = "prefill-pure-single-buffer-evaluation-runner.v1"
STAGES = ("static_legality", "compile_resources", "route_binding",
          "full_output_correctness", "kernel_timing")


def _at(row: dict[str, Any], *paths: tuple[str, ...]) -> Any:
  for path in paths:
    value: Any = row
    for key in path:
      if not isinstance(value, dict) or key not in value: break
      value = value[key]
    else:
      if value is not None: return value
  return None


def _candidate_identity(row: dict[str, Any]) -> Any:
  return _at(row, ("canonical_identity",), ("candidate_hash",),
             ("full_kernel_candidate_hash",), ("binding", "full_kernel_candidate_hash"),
             ("candidate", "canonical_identity"))


def _binary_identity(row: dict[str, Any]) -> Any:
  return _at(row, ("binary_sha256",), ("program", "binary_sha256"),
             ("binding", "binary_sha256"), ("binding", "executed_binary_sha256"),
             ("execution", "binary_sha256"))


def _commit_identity(row: dict[str, Any]) -> Any:
  return _at(row, ("commit",), ("git_revision",), ("git", "revision"),
             ("environment", "git_revision"), ("environment", "git", "revision"))


def _status(row: dict[str, Any]) -> str:
  if row.get("passed") is True: return "pass"
  status = row.get("status")
  if status in ("pass", "passed", "ok"): return "pass"
  return str(status or "blocked")


@dataclass
class ArtifactAuthorities:
  paths: dict[str, pathlib.Path | None]
  expected_commit: str | None = None
  _binary: str | None = None
  _commit: str | None = None

  def _collector(self, stage: str):
    path = self.paths.get(stage)
    if path is None: return None

    def collect(_payload: dict[str, Any], identity: str) -> dict[str, Any]:
      try: raw = json.loads(path.read_text())
      except FileNotFoundError: raise ValueError(f"artifact does not exist: {path}") from None
      except json.JSONDecodeError as exc: raise ValueError(f"invalid JSON artifact {path}: {exc}") from None
      if not isinstance(raw, dict): raise ValueError(f"artifact is not a JSON object: {path}")

      found_candidate = _candidate_identity(raw)
      if found_candidate != identity:
        raise ValueError(f"artifact candidate join mismatch: expected {identity}, found {found_candidate!r}")
      commit = _commit_identity(raw)
      if not isinstance(commit, str) or not commit:
        raise ValueError("artifact has no source commit identity")
      expected_commit = self.expected_commit or self._commit
      if expected_commit is not None and commit != expected_commit:
        raise ValueError(f"artifact commit join mismatch: expected {expected_commit}, found {commit}")
      if self._commit is None: self._commit = commit

      binary = _binary_identity(raw)
      if stage != "static_legality":
        if not isinstance(binary, str) or len(binary) != 64:
          raise ValueError("artifact has no valid executable binary SHA-256")
        if self._binary is None:
          if stage != "compile_resources": raise ValueError("compile_resources must establish binary identity")
          self._binary = binary
        elif binary != self._binary:
          raise ValueError(f"artifact binary join mismatch: expected {self._binary}, found {binary}")

      evidence = {"canonical_identity": identity, "status": _status(raw),
                  "artifact": str(path), "commit": commit, "authority": raw}
      if binary is not None: evidence["binary_sha256"] = binary
      if stage == "route_binding":
        evidence.update({
          "route_binding_complete": _at(raw, ("route_binding_complete",), ("binding_complete",),
                                        ("binding", "route_binding_complete")),
          "route_id": _at(raw, ("route_id",), ("binding", "route_id")),
          "selected_route_id": _at(raw, ("selected_route_id",), ("binding", "selected_route_id"),
                                   ("executable_truth", "selected_route_id")),
          "runtime_binary_matches_candidate": _at(raw, ("runtime_binary_matches_candidate",),
                                                   ("binding", "runtime_binary_matches_candidate"),
                                                   ("executable_truth", "runtime_binary_matches_candidate")),
          "strict_pure": _at(raw, ("strict_pure",), ("surface", "strict_pure"),
                             ("executable_truth", "strict_pure")),
          "fallback_used": _at(raw, ("fallback_used",), ("binding", "fallback_used"),
                               ("executable_truth", "fallback_used")),
        })
      return evidence
    return collect

  def gate_authorities(self) -> EvaluationAuthorities:
    return EvaluationAuthorities(*(self._collector(stage) for stage in STAGES))


def run(payload: dict[str, Any], candidate_hash: str, paths: dict[str, pathlib.Path | None],
        *, expected_commit: str | None = None, allowed_buffer_counts: tuple[int, ...]=(1,)) -> dict[str, Any]:
  adapter = ArtifactAuthorities(paths, expected_commit)
  report = evaluate(payload, candidate_hash, adapter.gate_authorities(), allowed_buffer_counts=allowed_buffer_counts)
  return {"schema": SCHEMA, "verdict": "PASS" if report["passed"] else "BLOCKED",
          "joins": {"candidate_hash": candidate_hash, "binary_sha256": adapter._binary,
                    "commit": adapter._commit}, "evaluation": report}


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--payload", type=pathlib.Path, required=True)
  ap.add_argument("--candidate-hash", required=True)
  ap.add_argument("--expected-commit")
  ap.add_argument("--allowed-buffer-count", type=int, action="append", dest="allowed_buffer_counts")
  for stage in STAGES: ap.add_argument(f"--{stage.replace('_', '-')}-artifact", type=pathlib.Path)
  ap.add_argument("--output", type=pathlib.Path)
  args = ap.parse_args(argv)
  payload = json.loads(args.payload.read_text())
  paths = {stage: getattr(args, f"{stage}_artifact") for stage in STAGES}
  report = run(payload, args.candidate_hash, paths, expected_commit=args.expected_commit,
               allowed_buffer_counts=tuple(args.allowed_buffer_counts or (1,)))
  text = json.dumps(report, indent=2) + "\n"
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(text, end="")
  return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
