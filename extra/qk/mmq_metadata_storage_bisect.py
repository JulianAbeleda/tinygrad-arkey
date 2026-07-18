"""Fail-closed parent orchestrator for the repeated-target metadata storage bisect.

The target K=256 PROGRAM is compiled and serialized exactly once in the CPU
parent. Three fresh spawned workers then execute the same q4/value-fixed,
metadata-changing sequence through the storage modes implemented by
``mmq_single_epoch_canary``:

* ``preloaded_views``
* ``dedicated_preloaded``
* ``fixed_refreshed``

This is diagnostic evidence only. A numerical PASS does not opt a production
route in, and any missing worker result, contract drift, or artifact identity
drift blocks the aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Callable

from tinygrad.runtime.process_isolated import IsolatedResult, run_isolated
from extra.qk.mmq_single_epoch_canary import METADATA_STORAGE_MODES, _run_epoch_worker


SCHEMA = "tinygrad.mmq_q4k_q8_1.metadata_storage_bisect.v1"
Q4_EPOCH_SEQUENCE = (0, 0, 0)
Q8_EPOCH_SEQUENCE = (0, 0, 0)
Q8_VALUES_EPOCH_SEQUENCE = (0, 0, 0)
Q8_METADATA_EPOCH_SEQUENCE = (0, 1, 2)
REQUIRED_PROGRAM_HASHES = ("binary_sha256", "source_sha256", "serialized_program_sha256")


def _sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    while chunk := handle.read(1 << 20): digest.update(chunk)
  return digest.hexdigest()


def _run_metadata_mode_worker(program_path: str, expected_serialized_sha256: str,
                              metadata_storage_mode: str, device: str,
                              runtime_timeout_ms: int) -> dict[str, Any]:
  """Fresh-child adapter which proves the serialized artifact used by the worker."""
  actual_sha = _sha256_file(program_path)
  if actual_sha != expected_serialized_sha256:
    raise RuntimeError("serialized target PROGRAM identity changed before worker dispatch")
  child = _run_epoch_worker(
    program_path, 0, 3, device, runtime_timeout_ms, False,
    Q4_EPOCH_SEQUENCE, Q4_EPOCH_SEQUENCE, Q8_EPOCH_SEQUENCE,
    Q8_VALUES_EPOCH_SEQUENCE, Q8_METADATA_EPOCH_SEQUENCE, metadata_storage_mode,
  )
  return {
    "metadata_storage_mode": metadata_storage_mode,
    "serialized_program_sha256": actual_sha,
    "child": child,
  }


def _base_result() -> dict[str, Any]:
  return {
    "schema": SCHEMA,
    "status": "BLOCKED",
    "passed": False,
    "diagnostic_complete": False,
    "diagnostic_only": True,
    "promotion_eligible": False,
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "no_fallback": True,
    "sequence": {
      "q4": list(Q4_EPOCH_SEQUENCE),
      "q8": list(Q8_EPOCH_SEQUENCE),
      "q8_values": list(Q8_VALUES_EPOCH_SEQUENCE),
      "q8_metadata": list(Q8_METADATA_EPOCH_SEQUENCE),
    },
    "program_identity": {},
    "same_serialized_artifact": False,
    "modes": [],
    "exact_blocker": None,
  }


def _blocked_mode(mode: str, blocker: str, isolated: IsolatedResult | None = None,
                  *, worker_serialized_sha256: str | None = None,
                  child: dict[str, Any] | None = None) -> dict[str, Any]:
  return {
    "metadata_storage_mode": mode,
    "status": "BLOCKED",
    "passed": False,
    "contract_valid": False,
    "same_serialized_artifact": False,
    "worker_serialized_program_sha256": worker_serialized_sha256,
    "child": child,
    "isolated_status": None if isolated is None else isolated.status,
    "isolated_error": None if isolated is None else isolated.error,
    "stdout_tail": "" if isolated is None else isolated.stdout[-1000:],
    "stderr_tail": "" if isolated is None else isolated.stderr[-1000:],
    "elapsed_seconds": None if isolated is None else isolated.elapsed_seconds,
    "exact_blocker": blocker,
  }


def _validate_mode_result(mode: str, isolated: IsolatedResult,
                          expected_serialized_sha256: str) -> dict[str, Any]:
  if isolated.status != "passed" or not isinstance(isolated.result, dict):
    return _blocked_mode(mode, "spawned metadata worker did not return a structured result", isolated)
  payload = isolated.result
  worker_sha = payload.get("serialized_program_sha256")
  child = payload.get("child")
  if worker_sha != expected_serialized_sha256:
    return _blocked_mode(mode, "spawned worker used a different serialized PROGRAM", isolated,
                         worker_serialized_sha256=worker_sha, child=child if isinstance(child, dict) else None)
  if payload.get("metadata_storage_mode") != mode or not isinstance(child, dict):
    return _blocked_mode(mode, "spawned worker metadata mode/result contract drifted", isolated,
                         worker_serialized_sha256=worker_sha, child=child if isinstance(child, dict) else None)

  expected_sequences = {
    "q4_epoch_sequence": list(Q4_EPOCH_SEQUENCE),
    "q8_epoch_sequence": list(Q8_EPOCH_SEQUENCE),
    "q8_values_epoch_sequence": list(Q8_VALUES_EPOCH_SEQUENCE),
    "q8_metadata_epoch_sequence": list(Q8_METADATA_EPOCH_SEQUENCE),
  }
  contract_valid = (
    child.get("metadata_storage_mode") == mode and
    child.get("target_dispatches") == 3 and
    child.get("no_fallback") is True and
    all(child.get(key) == value for key, value in expected_sequences.items())
  )
  if not contract_valid:
    return _blocked_mode(mode, "spawned worker sequence/dispatch/no-fallback contract failed", isolated,
                         worker_serialized_sha256=worker_sha, child=child)

  numerical_pass = (
    child.get("passed") is True and child.get("status") == "PASS" and
    isinstance(child.get("comparison"), dict) and child["comparison"].get("status") == "pass" and
    child["comparison"].get("mismatch_count") == 0
  )
  return {
    "metadata_storage_mode": mode,
    "status": "PASS" if numerical_pass else "BLOCKED",
    "passed": numerical_pass,
    "contract_valid": True,
    "same_serialized_artifact": True,
    "worker_serialized_program_sha256": worker_sha,
    "child": child,
    "isolated_status": isolated.status,
    "isolated_error": isolated.error,
    "stdout_tail": isolated.stdout[-1000:],
    "stderr_tail": isolated.stderr[-1000:],
    "elapsed_seconds": isolated.elapsed_seconds,
    "exact_blocker": None if numerical_pass else "metadata storage mode numerical comparison failed",
  }


def run_metadata_storage_bisect(
  *,
  compile_fn: Callable[[str | Path], tuple[str, dict[str, Any]]] | None = None,
  runner: Callable[..., IsolatedResult] = run_isolated,
  device: str = "AMD",
  timeout_seconds: float = 120.0,
  runtime_timeout_ms: int = 30_000,
) -> dict[str, Any]:
  """Compile once and execute all metadata storage modes in fresh spawned workers."""
  if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
    raise ValueError("timeout_seconds must be positive")
  if not isinstance(runtime_timeout_ms, int) or runtime_timeout_ms <= 0:
    raise ValueError("runtime_timeout_ms must be positive")
  if compile_fn is None:
    from extra.qk.mmq_target_epoch_orchestrator import compile_target_program_artifact
    compile_fn = compile_target_program_artifact

  out = _base_result()
  with tempfile.TemporaryDirectory(prefix="tinygrad-mmq-metadata-bisect-") as temp_dir:
    try: program_path, compile_evidence = compile_fn(temp_dir)
    except BaseException as exc:
      out["exact_blocker"] = f"target PROGRAM compile/serialization failed: {type(exc).__name__}: {exc}"
      return out
    if not isinstance(compile_evidence, dict):
      out["exact_blocker"] = "target PROGRAM compile evidence is not a mapping"
      return out

    identity = {key: compile_evidence.get(key) for key in REQUIRED_PROGRAM_HASHES}
    if not all(isinstance(value, str) and len(value) == 64 for value in identity.values()):
      out["program_identity"] = identity
      out["exact_blocker"] = "target PROGRAM compile evidence is missing source/binary/serialized identity"
      return out
    try: actual_serialized_sha = _sha256_file(program_path)
    except BaseException as exc:
      out["program_identity"] = identity
      out["exact_blocker"] = f"serialized target PROGRAM could not be hashed: {type(exc).__name__}: {exc}"
      return out
    out["program_identity"] = identity
    if actual_serialized_sha != identity["serialized_program_sha256"]:
      out["exact_blocker"] = "parent serialized target PROGRAM hash does not match compile evidence"
      return out

    for mode in METADATA_STORAGE_MODES:
      try:
        isolated = runner(
          _run_metadata_mode_worker,
          args=(program_path, actual_serialized_sha, mode, device, runtime_timeout_ms),
          timeout_seconds=timeout_seconds,
          start_method="spawn",
        )
      except BaseException as exc:
        out["modes"].append(_blocked_mode(
          mode, f"spawned metadata worker could not be started: {type(exc).__name__}: {exc}"))
        continue
      if not isinstance(isolated, IsolatedResult):
        out["modes"].append(_blocked_mode(mode, "spawned metadata runner returned an invalid result"))
        continue
      out["modes"].append(_validate_mode_result(mode, isolated, actual_serialized_sha))

  out["same_serialized_artifact"] = (
    len(out["modes"]) == len(METADATA_STORAGE_MODES) and
    all(row.get("same_serialized_artifact") is True for row in out["modes"])
  )
  out["diagnostic_complete"] = (
    out["same_serialized_artifact"] and
    all(row.get("contract_valid") is True for row in out["modes"])
  )
  out["passed"] = out["diagnostic_complete"] and all(row.get("passed") is True for row in out["modes"])
  out["status"] = "PASS" if out["passed"] else "BLOCKED"
  blockers = [f"{row['metadata_storage_mode']}: {row['exact_blocker']}"
              for row in out["modes"] if row.get("exact_blocker")]
  out["exact_blocker"] = None if out["passed"] else (
    "; ".join(blockers) if blockers else "metadata storage bisect did not complete"
  )
  return out


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, help="optional JSON evidence path")
  parser.add_argument("--timeout-seconds", type=float, default=120.0)
  parser.add_argument("--runtime-timeout-ms", type=int, default=30_000)
  args = parser.parse_args()
  result = run_metadata_storage_bisect(
    timeout_seconds=args.timeout_seconds, runtime_timeout_ms=args.runtime_timeout_ms,
  )
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n")
  print(encoded)
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = [
  "SCHEMA", "run_metadata_storage_bisect", "_run_metadata_mode_worker",
]
