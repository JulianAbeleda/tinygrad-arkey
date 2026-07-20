"""One-shot forensic runner for the guarded PM4 ``ffn_gate_up`` snapshot.

This diagnostic constructs the exact production prefix-1 runtime and captures
its concrete PM4 command immediately before submission, but the guarded child
must never copy the command into the ring or ring the doorbell.  Both PASS and
BLOCKED envelopes are retained as non-admissible forensic JSON.  A diagnostic
PASS is not correctness evidence and is never published through the frozen
correctness-evidence API.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, TextIO

from extra.qk.mmq_ffn_gate_up_guarded_correctness import (
  ENVELOPE_SCHEMA, PM4_NO_DOORBELL_SCHEMA, SCHEMA,
  build_production_candidate_prefix_runtime,
  run_guarded_pm4_no_doorbell,
  validate_guarded_envelope,
  validate_pm4_no_doorbell_evidence,
)
from extra.qk.mmq_ffn_gate_up_pm4_prefix1_runner import (
  _content_identity, _fsync_directory, _fsync_file, _identity,
  _publish_forensic_envelope, _resolve_input, _resolve_output,
  validate_pm4_prefix1_semantic_preflight,
)


PM4_QUEUE_MODE = "PM4"
PREFIX_EPOCHS = 1
SUBMIT_POLICY = "snapshot_only"
DEFAULT_TIMEOUT_SECONDS = 900.0
CLAIM_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_pm4_no_doorbell_claim.v1"
RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_pm4_no_doorbell_runner_receipt.v1"
_ENVELOPE_KEYS = {
  "schema", "status", "exact_blocker", "queue_mode", "operation_schema",
  "health_before", "health_after", "kernel_faults",
  "kernel_fault_evidence", "launched", "spawn_count", "child_status",
  "timed_out", "error", "elapsed_seconds", "result", "no_retry",
  "retry_count", "no_queue_fallback", "promotion_evidence_eligible",
  "request_identity", "config_identity", "evidence_identity",
}
_CHILD_BLOCKED_REQUIRED = {
  "schema": PM4_NO_DOORBELL_SCHEMA,
  "status": "BLOCKED",
  "queue_mode": PM4_QUEUE_MODE,
  "prefix_epochs": PREFIX_EPOCHS,
  "submit_policy": SUBMIT_POLICY,
  "no_retry": True,
  "retry_count": 0,
  "no_fallback": True,
  "compile_performed": False,
  "requires_recompile": False,
  "promotion_evidence_eligible": False,
  "environment": {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"},
}


def _request_identity(config: Mapping[str, Any]) -> str:
  return _identity({
    "schema": f"{SCHEMA}.pm4_no_doorbell_request",
    "queue_mode": PM4_QUEUE_MODE,
    "prefix_epochs": PREFIX_EPOCHS,
    "submit_policy": SUBMIT_POLICY,
    "config_identity": _identity(dict(config)),
  })


def _claim_path(output: Path) -> Path:
  return output.with_name(f"{output.name}.claim")


def _acquire_claim(output: Path) -> Path:
  claim = _claim_path(output)
  payload = {
    "schema": CLAIM_SCHEMA, "pid": os.getpid(), "output": str(output),
    "queue_mode": PM4_QUEUE_MODE, "prefix_epochs": PREFIX_EPOCHS,
    "submit_policy": SUBMIT_POLICY,
    "promotion_evidence_eligible": False,
  }
  descriptor = os.open(
    claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(descriptor, "wb") as handle:
      handle.write(
        json.dumps(
          payload, sort_keys=True, separators=(",", ":"),
          allow_nan=False).encode() + b"\n")
      handle.flush()
      os.fsync(handle.fileno())
    _fsync_directory(claim.parent)
  except BaseException:
    try: claim.unlink()
    except FileNotFoundError: pass
    raise
  return claim


def _release_claim(claim: Path) -> None:
  claim.unlink()
  _fsync_directory(claim.parent)


def _validate_blocked_child(
    value: Mapping[str, Any], *, config: Mapping[str, Any],
    ) -> dict[str, Any]:
  child = dict(value)
  if child.get("evidence_identity") != _identity({
      key: item for key, item in child.items()
      if key != "evidence_identity"}):
    raise ValueError("blocked no-doorbell child identity differs")
  if any(child.get(key) != expected for key, expected in
         _CHILD_BLOCKED_REQUIRED.items()):
    raise ValueError("blocked no-doorbell child safety contract differs")
  if not isinstance(child.get("exact_blocker"), str) or \
     not child["exact_blocker"] or \
     not isinstance(child.get("exception"), str) or \
     not isinstance(child.get("failed_attempt"), Mapping):
    raise ValueError("blocked no-doorbell child failure facts differ")
  if child.get("config_identity") != _identity(dict(config)) or \
     child.get("request_identity") != _request_identity(config):
    raise ValueError("blocked no-doorbell child binding differs")
  return child


def validate_pm4_no_doorbell_forensic_envelope(
    value: Any, *, config: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Validate PASS or BLOCKED no-doorbell evidence as forensics only."""
  if not isinstance(value, Mapping):
    raise ValueError("PM4 no-doorbell forensic envelope must be a mapping")
  row = dict(value)
  if set(row) != _ENVELOPE_KEYS or \
     row.get("evidence_identity") != _identity({
       key: item for key, item in row.items()
       if key != "evidence_identity"}):
    raise ValueError("PM4 no-doorbell forensic envelope fields/identity differ")
  if row.get("schema") != ENVELOPE_SCHEMA or \
     row.get("queue_mode") != PM4_QUEUE_MODE or \
     row.get("operation_schema") != PM4_NO_DOORBELL_SCHEMA:
    raise ValueError("PM4 no-doorbell forensic operation differs")
  if row.get("no_retry") is not True or row.get("retry_count") != 0 or \
     row.get("no_queue_fallback") is not True or \
     row.get("promotion_evidence_eligible") is not False:
    raise ValueError("PM4 no-doorbell forensic safety contract differs")
  if type(row.get("health_before")) is not bool or \
     type(row.get("health_after")) is not bool or \
     type(row.get("launched")) is not bool or \
     type(row.get("timed_out")) is not bool:
    raise ValueError("PM4 no-doorbell forensic boolean facts differ")
  if row.get("spawn_count") not in (0, 1) or \
     row["launched"] is not (row["spawn_count"] == 1):
    raise ValueError("PM4 no-doorbell forensic spawn facts differ")
  if not isinstance(row.get("kernel_faults"), list) or \
     any(not isinstance(item, str) for item in row["kernel_faults"]) or \
     not isinstance(row.get("kernel_fault_evidence"), Mapping):
    raise ValueError("PM4 no-doorbell forensic fault facts differ")
  elapsed = row.get("elapsed_seconds")
  if elapsed is not None and (
      not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or
      not math.isfinite(elapsed) or elapsed < 0):
    raise ValueError("PM4 no-doorbell forensic elapsed time differs")
  config_identity = _identity(dict(config))
  if row.get("config_identity") != config_identity or \
     row.get("request_identity") != _request_identity(config):
    raise ValueError("PM4 no-doorbell forensic request/config binding differs")
  _content_identity(config_identity, "forensic config identity")
  _content_identity(row["request_identity"], "forensic request identity")

  status, result = row.get("status"), row.get("result")
  if status == "PASS":
    if row.get("exact_blocker") is not None:
      raise ValueError("passing no-doorbell diagnostic has a blocker")
    validated = validate_guarded_envelope(row)
    diagnostic = validate_pm4_no_doorbell_evidence(validated["result"])
    if diagnostic["target_dispatch_submitted"] is not False or \
       diagnostic["native_submit_call_count"] != 0 or \
       diagnostic["promotion_evidence_eligible"] is not False:
      raise ValueError("passing no-doorbell diagnostic submission facts differ")
    return row
  if status != "BLOCKED" or \
     not isinstance(row.get("exact_blocker"), str) or \
     not row["exact_blocker"]:
    raise ValueError("PM4 no-doorbell forensic state differs")
  if result is not None:
    if not isinstance(result, Mapping):
      raise ValueError("blocked no-doorbell child result differs")
    if result.get("status") == "PASS":
      child = validate_pm4_no_doorbell_evidence(result)
      if child["config_identity"] != config_identity or \
         child["request_identity"] != row["request_identity"]:
        raise ValueError("blocked outer envelope child binding differs")
    else:
      _validate_blocked_child(result, config=config)
  return row


def _set_fixed_parent_gpu_environment() -> dict[str, str | None]:
  previous = {
    key: os.environ.get(key) for key in ("DEV", "AMD_AQL", "PROFILE")}
  os.environ.update({"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"})
  return previous


def _restore_parent_gpu_environment(
    previous: Mapping[str, str | None],
    ) -> None:
  for key in ("DEV", "AMD_AQL", "PROFILE"):
    value = previous[key]
    if value is None:
      os.environ.pop(key, None)
    else:
      os.environ[key] = value


def _emit_receipt(
    envelope: Mapping[str, Any], output: Path, stream: TextIO,
    ) -> None:
  nested = envelope.get("result")
  nested_identity = \
    nested.get("evidence_identity") if isinstance(nested, Mapping) else None
  target_submitted = \
    nested.get("target_dispatch_submitted") \
    if isinstance(nested, Mapping) else None
  native_submit_count = \
    nested.get("native_submit_call_count") \
    if isinstance(nested, Mapping) else None
  receipt = {
    "schema": RECEIPT_SCHEMA,
    "diagnostic": "PM4_NO_DOORBELL",
    "status": envelope["status"],
    "output": str(output),
    "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "outer_evidence_identity": envelope["evidence_identity"],
    "nested_diagnostic_evidence_identity": nested_identity,
    "launched": envelope["launched"],
    "spawn_count": envelope["spawn_count"],
    "blocker": envelope["exact_blocker"],
    "target_dispatch_submitted": target_submitted,
    "native_submit_call_count": native_submit_count,
    "promotion_evidence_eligible": False,
  }
  stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
  stream.flush()


def run_pm4_no_doorbell(
    *, frozen_bundle: str | Path, staged_family_manifest: str | Path,
    execution_fixture_v2: str | Path, pm4_c4: str | Path,
    output: str | Path, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    semantic_preflight: Callable[
      [str | Path, str | Path, str | Path, str | Path], str] =
      validate_pm4_prefix1_semantic_preflight,
    claim_acquirer: Callable[[Path], Path] = _acquire_claim,
    guarded_stage: Callable[..., Mapping[str, Any]] =
      run_guarded_pm4_no_doorbell,
    forensic_validator: Callable[..., dict[str, Any]] =
      validate_pm4_no_doorbell_forensic_envelope,
    forensic_publisher: Callable[[Path, Mapping[str, Any]], None] =
      _publish_forensic_envelope,
    error_stream: TextIO | None = None,
    receipt_stream: TextIO | None = None,
    ) -> int:
  """Run and retain exactly one terminal PM4 no-doorbell diagnostic.

  Exit codes: 0 retained diagnostic PASS, 1 retained BLOCKED forensics,
  2 prelaunch failure, and 3 post-invocation validation/persistence failure.
  Exit 3 deliberately leaves the exclusive claim in place.
  """
  errors = sys.stderr if error_stream is None else error_stream
  receipts = sys.stdout if receipt_stream is None else receipt_stream
  claim: Path | None = None
  invoked = False
  try:
    if not isinstance(timeout_seconds, (int, float)) or \
       isinstance(timeout_seconds, bool):
      raise ValueError("timeout_seconds must be numeric")
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
      raise ValueError("timeout_seconds must be finite and positive")
    bundle = _resolve_input(frozen_bundle, "frozen bundle", directory=True)
    family_manifest = _resolve_input(
      staged_family_manifest, "staged family manifest", directory=False)
    fixture = _resolve_input(
      execution_fixture_v2, "execution fixture v2", directory=False)
    c4 = _resolve_input(pm4_c4, "PM4 C4", directory=False)
    output_path = _resolve_output(output)
    candidate_identity = _content_identity(
      semantic_preflight(bundle, family_manifest, fixture, c4),
      "candidate executable identity")
    claim = claim_acquirer(output_path)
    if os.path.lexists(output_path):
      raise FileExistsError(
        f"output became occupied while acquiring claim {output_path}")
    config = {
      "frozen_bundle": str(bundle),
      "staged_family_manifest": str(family_manifest),
      "execution_fixture_v2": str(fixture),
      "runtime_canary_isolation": str(c4),
      "candidate_executable_identity": candidate_identity,
    }
  except BaseException as exc:
    if claim is not None:
      try: _release_claim(claim)
      except BaseException: pass
    print(
      f"PM4 no-doorbell prelaunch failure: {type(exc).__name__}: {exc}",
      file=errors)
    return 2

  try:
    invoked = True
    previous_environment = _set_fixed_parent_gpu_environment()
    try:
      envelope = guarded_stage(
        config=config,
        runtime_builder=build_production_candidate_prefix_runtime,
        timeout_seconds=timeout)
      if {
          key: os.environ.get(key)
          for key in ("DEV", "AMD_AQL", "PROFILE")} != {
            "DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}:
        raise ValueError(
          "parent GPU environment changed during guarded diagnostic")
    finally:
      _restore_parent_gpu_environment(previous_environment)
    if not isinstance(envelope, Mapping):
      raise ValueError("guarded diagnostic returned no envelope")
    validated = forensic_validator(envelope, config=config)
    if validated.get("status") not in ("PASS", "BLOCKED"):
      raise ValueError("guarded diagnostic status is neither PASS nor BLOCKED")
    forensic_publisher(output_path, validated)
    loaded = forensic_validator(
      json.loads(output_path.read_bytes()), config=config)
    if loaded != validated:
      raise ValueError("forensic no-doorbell envelope round-trip differs")
    _fsync_file(output_path)
    _fsync_directory(output_path.parent)
    _emit_receipt(validated, output_path, receipts)
    status = validated["status"]
    _release_claim(claim)
    return 0 if status == "PASS" else 1
  except BaseException as exc:
    assert invoked
    print(
      f"PM4 no-doorbell postlaunch failure (claim retained at {claim}): "
      f"{type(exc).__name__}: {exc}", file=errors)
    return 3


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Capture exactly one guarded PM4 ffn_gate_up command without "
      "submitting it"))
  parser.add_argument("--frozen-bundle", required=True)
  parser.add_argument("--staged-family-manifest", required=True)
  parser.add_argument("--execution-fixture-v2", required=True)
  parser.add_argument("--pm4-c4", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument(
    "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
  return parser


def main(
    argv: Sequence[str] | None = None, *,
    runner: Callable[..., int] = run_pm4_no_doorbell,
    ) -> int:
  args = _parser().parse_args(argv)
  return runner(
    frozen_bundle=args.frozen_bundle,
    staged_family_manifest=args.staged_family_manifest,
    execution_fixture_v2=args.execution_fixture_v2,
    pm4_c4=args.pm4_c4, output=args.output,
    timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = [
  "CLAIM_SCHEMA", "DEFAULT_TIMEOUT_SECONDS", "PM4_QUEUE_MODE",
  "PREFIX_EPOCHS", "RECEIPT_SCHEMA", "SUBMIT_POLICY", "main",
  "run_pm4_no_doorbell",
  "validate_pm4_no_doorbell_forensic_envelope",
]
