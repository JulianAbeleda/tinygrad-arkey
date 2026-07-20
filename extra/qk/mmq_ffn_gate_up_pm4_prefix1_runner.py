"""Single-attempt production runner for guarded PM4 ``ffn_gate_up`` prefix-1.

This is intentionally not a general correctness CLI.  It admits exactly one
PM4 prefix-1 guarded stage, with no predecessor evidence and the production
runtime builder.  A sibling claim excludes concurrent attempts.  PASS
envelopes are published through the admissible frozen-evidence API; BLOCKED
envelopes are retained only as non-admissible forensic JSON.
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
import tempfile
from typing import Any, TextIO

from extra.qk.mmq_ffn_gate_up_guarded_correctness import (
  CANDIDATE_SCHEMA, ENVELOPE_SCHEMA, SCHEMA,
  build_production_candidate_prefix_runtime,
  ffn_gate_up_candidate_executable_identity,
  freeze_correctness_evidence,
  load_frozen_correctness_evidence,
  run_guarded_candidate_prefix,
  validate_guarded_envelope,
)


PM4_QUEUE_MODE = "PM4"
PREFIX_EPOCHS = 1
DEFAULT_TIMEOUT_SECONDS = 900.0
CLAIM_SCHEMA = "tinygrad.mmq_q4k_q8_1.ffn_gate_up_pm4_prefix1_claim.v1"
_HEX = frozenset("0123456789abcdef")
_ENVELOPE_KEYS = {
  "schema", "status", "exact_blocker", "queue_mode", "operation_schema",
  "health_before", "health_after", "kernel_faults",
  "kernel_fault_evidence", "launched", "spawn_count", "child_status",
  "timed_out", "error", "elapsed_seconds", "result", "no_retry",
  "retry_count", "no_queue_fallback", "promotion_evidence_eligible",
  "request_identity", "config_identity", "evidence_identity",
}


def _canonical(value: Any) -> bytes:
  return json.dumps(
    value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _identity(value: Any) -> str:
  return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _content_identity(value: Any, label: str) -> str:
  if not isinstance(value, str) or not value.startswith("sha256:") or \
     len(value) != 71 or any(char not in _HEX for char in value[7:]):
    raise ValueError(f"{label} must be a sha256 content identity")
  return value


def _resolve_input(path: str | Path, label: str, *, directory: bool) -> Path:
  resolved = Path(path).expanduser().resolve(strict=True)
  if directory:
    if not resolved.is_dir():
      raise ValueError(f"{label} must be a directory")
    descriptor = os.open(resolved, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    os.close(descriptor)
  else:
    if not resolved.is_file():
      raise ValueError(f"{label} must be a regular file")
    with resolved.open("rb") as handle:
      handle.read(1)
  return resolved


def _resolve_output(path: str | Path) -> Path:
  raw = Path(path).expanduser()
  if not raw.name:
    raise ValueError("output must name a file")
  parent = raw.parent.resolve(strict=True)
  if not parent.is_dir():
    raise ValueError("output parent must be a directory")
  output = parent / raw.name
  if os.path.lexists(output):
    raise FileExistsError(f"refusing to replace output {output}")
  return output


def _claim_path(output: Path) -> Path:
  return output.with_name(f"{output.name}.claim")


def _fsync_directory(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _fsync_file(path: Path) -> None:
  descriptor = os.open(path, os.O_RDONLY)
  try:
    os.fsync(descriptor)
  finally:
    os.close(descriptor)


def _acquire_claim(output: Path) -> Path:
  claim = _claim_path(output)
  payload = {
    "schema": CLAIM_SCHEMA, "pid": os.getpid(), "output": str(output),
    "queue_mode": PM4_QUEUE_MODE, "prefix_epochs": PREFIX_EPOCHS,
  }
  descriptor = os.open(
    claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(descriptor, "wb") as handle:
      handle.write(_canonical(payload) + b"\n")
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


def derive_candidate_executable_identity(
    frozen_bundle: str | Path, staged_family_manifest: str | Path,
    ) -> str:
  """Load the frozen family without opening a Device and derive its identity."""
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  from extra.qk.mmq_frozen_staged_family import \
    load_frozen_staged_family_manifest

  role = exact_role_spec("ffn_gate_up")
  family = load_frozen_staged_family_manifest(
    staged_family_manifest, role_spec=role, frozen_bundle=frozen_bundle)
  return ffn_gate_up_candidate_executable_identity(family)


def validate_pm4_prefix1_semantic_preflight(
    frozen_bundle: str | Path, staged_family_manifest: str | Path,
    execution_fixture_v2: str | Path, pm4_c4: str | Path,
    ) -> str:
  """CPU-only validation of every artifact needed by the first PM4 dispatch."""
  from extra.qk.mmq_attn_qo_c6_binding import read_json
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  from extra.qk.mmq_ffn_gate_up_c8_runtime import \
    rebuild_ffn_gate_up_v2_fixture
  from extra.qk.mmq_frozen_staged_family import \
    load_frozen_staged_family_manifest
  from extra.qk.mmq_frozen_staged_family_execution import \
    validate_frozen_staged_runtime_canary_isolation

  role = exact_role_spec("ffn_gate_up")
  family = load_frozen_staged_family_manifest(
    staged_family_manifest, role_spec=role, frozen_bundle=frozen_bundle)
  rebuild_ffn_gate_up_v2_fixture(
    role, read_json(execution_fixture_v2, "execution fixture v2"))
  validate_frozen_staged_runtime_canary_isolation(
    read_json(pm4_c4, "PM4 C4 canary"), family, queue_mode=PM4_QUEUE_MODE)
  return ffn_gate_up_candidate_executable_identity(family)


def _candidate_request_identity(config: Mapping[str, Any]) -> str:
  return _identity({
    "schema": f"{SCHEMA}.candidate_request",
    "queue_mode": PM4_QUEUE_MODE, "prefix_epochs": PREFIX_EPOCHS,
    "config_identity": _identity(dict(config)),
    "prior_evidence_identity": None,
    "cross_queue_admission_identity": None,
  })


def validate_blocked_forensic_envelope(
    value: Any, *, config: Mapping[str, Any],
    ) -> dict[str, Any]:
  """Validate a complete BLOCKED envelope without making it admissible."""
  if not isinstance(value, Mapping):
    raise ValueError("blocked forensic envelope must be a mapping")
  row = dict(value)
  if set(row) != _ENVELOPE_KEYS or \
     row.get("evidence_identity") != _identity({
       key: item for key, item in row.items() if key != "evidence_identity"}):
    raise ValueError("blocked forensic envelope fields/identity differ")
  if row.get("schema") != ENVELOPE_SCHEMA or row.get("status") != "BLOCKED" or \
     not isinstance(row.get("exact_blocker"), str) or \
     not row["exact_blocker"] or \
     row.get("queue_mode") != PM4_QUEUE_MODE or \
     row.get("operation_schema") != CANDIDATE_SCHEMA:
    raise ValueError("blocked forensic envelope operation/state differs")
  if row.get("no_retry") is not True or row.get("retry_count") != 0 or \
     row.get("no_queue_fallback") is not True or \
     row.get("promotion_evidence_eligible") is not False:
    raise ValueError("blocked forensic envelope safety contract differs")
  if type(row.get("health_before")) is not bool or \
     type(row.get("health_after")) is not bool or \
     type(row.get("launched")) is not bool or \
     type(row.get("timed_out")) is not bool:
    raise ValueError("blocked forensic envelope boolean facts differ")
  if row.get("spawn_count") not in (0, 1) or \
     row["launched"] is not (row["spawn_count"] == 1):
    raise ValueError("blocked forensic envelope spawn facts differ")
  if not isinstance(row.get("kernel_faults"), list) or \
     any(not isinstance(item, str) for item in row["kernel_faults"]) or \
     not isinstance(row.get("kernel_fault_evidence"), Mapping):
    raise ValueError("blocked forensic envelope fault facts differ")
  elapsed = row.get("elapsed_seconds")
  if elapsed is not None and (
      not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or
      not math.isfinite(elapsed) or elapsed < 0):
    raise ValueError("blocked forensic envelope elapsed time differs")
  if row.get("result") is not None and not isinstance(row["result"], Mapping):
    raise ValueError("blocked forensic envelope child result differs")
  config_identity = _identity(dict(config))
  if row.get("config_identity") != config_identity or \
     row.get("request_identity") != _candidate_request_identity(config):
    raise ValueError("blocked forensic envelope request/config binding differs")
  _content_identity(row["config_identity"], "forensic config identity")
  _content_identity(row["request_identity"], "forensic request identity")
  return row


def _publish_forensic_envelope(
    path: Path, value: Mapping[str, Any],
    ) -> None:
  encoded = (
    json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) +
    "\n").encode()
  descriptor, temporary = tempfile.mkstemp(
    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  temporary_path = Path(temporary)
  try:
    with os.fdopen(descriptor, "wb") as handle:
      handle.write(encoded)
      handle.flush()
      os.fsync(handle.fileno())
    try:
      os.link(temporary_path, path)
    except FileExistsError as exc:
      raise FileExistsError(
        f"refusing to replace forensic evidence {path}") from exc
    _fsync_directory(path.parent)
  finally:
    try: temporary_path.unlink()
    except FileNotFoundError: pass


def _round_trip_forensic(path: Path, *, config: Mapping[str, Any]) -> None:
  validate_blocked_forensic_envelope(
    json.loads(path.read_bytes()), config=config)


def _set_fixed_parent_gpu_environment() -> dict[str, str | None]:
  previous = {key: os.environ.get(key) for key in ("DEV", "AMD_AQL")}
  os.environ.update({"DEV": "AMD", "AMD_AQL": "0"})
  return previous


def _restore_parent_gpu_environment(previous: Mapping[str, str | None]) -> None:
  for key in ("DEV", "AMD_AQL"):
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
  try:
    if nested_identity is not None:
      _content_identity(nested_identity, "nested stage evidence identity")
  except ValueError:
    nested_identity = None
  receipt = {
    "status": envelope["status"], "output": str(output),
    "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "outer_evidence_identity": envelope["evidence_identity"],
    "nested_stage_evidence_identity": nested_identity,
    "launched": envelope["launched"],
    "spawn_count": envelope["spawn_count"],
    "blocker": envelope["exact_blocker"],
  }
  stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
  stream.flush()


def run_pm4_prefix1(
    *, frozen_bundle: str | Path, staged_family_manifest: str | Path,
    execution_fixture_v2: str | Path, pm4_c4: str | Path,
    output: str | Path, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    semantic_preflight: Callable[
      [str | Path, str | Path, str | Path, str | Path], str] =
      validate_pm4_prefix1_semantic_preflight,
    claim_acquirer: Callable[[Path], Path] = _acquire_claim,
    guarded_stage: Callable[..., Mapping[str, Any]] =
      run_guarded_candidate_prefix,
    pass_validator: Callable[[Any], dict[str, Any]] =
      validate_guarded_envelope,
    pass_freezer: Callable[..., Any] = freeze_correctness_evidence,
    pass_loader: Callable[[Any], dict[str, Any]] =
      load_frozen_correctness_evidence,
    forensic_validator: Callable[..., dict[str, Any]] =
      validate_blocked_forensic_envelope,
    forensic_publisher: Callable[[Path, Mapping[str, Any]], None] =
      _publish_forensic_envelope,
    error_stream: TextIO | None = None,
    receipt_stream: TextIO | None = None,
    ) -> int:
  """Run and durably retain exactly one guarded PM4 prefix-1 attempt.

  Exit codes: 0 PASS, 1 retained BLOCKED forensics, 2 prelaunch failure,
  and 3 post-invocation validation/persistence failure.  Exit 3 deliberately
  leaves the exclusive claim in place.
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
      f"PM4 prefix-1 prelaunch failure: {type(exc).__name__}: {exc}",
      file=errors)
    return 2

  try:
    invoked = True
    previous_environment = _set_fixed_parent_gpu_environment()
    try:
      envelope = guarded_stage(
        config=config, queue_mode=PM4_QUEUE_MODE,
        prefix_epochs=PREFIX_EPOCHS,
        runtime_builder=build_production_candidate_prefix_runtime,
        prior_evidence=None, cross_queue_admission=None,
        timeout_seconds=timeout)
      if os.environ.get("DEV") != "AMD" or \
         os.environ.get("AMD_AQL") != "0":
        raise ValueError("parent GPU environment changed during guarded stage")
    finally:
      _restore_parent_gpu_environment(previous_environment)
    if not isinstance(envelope, Mapping):
      raise ValueError("guarded stage returned no envelope")
    status = envelope.get("status")
    if status == "PASS":
      validated = pass_validator(envelope)
      reference = pass_freezer(output_path, validated)
      loaded = pass_loader(reference)
      if loaded != validated:
        raise ValueError("frozen PASS envelope round-trip differs")
      _fsync_file(output_path)
      _fsync_directory(output_path.parent)
      _emit_receipt(validated, output_path, receipts)
      _release_claim(claim)
      return 0
    if status == "BLOCKED":
      validated = forensic_validator(envelope, config=config)
      forensic_publisher(output_path, validated)
      _round_trip_forensic(output_path, config=config)
      _fsync_file(output_path)
      _fsync_directory(output_path.parent)
      _emit_receipt(validated, output_path, receipts)
      _release_claim(claim)
      return 1
    raise ValueError("guarded envelope status is neither PASS nor BLOCKED")
  except BaseException as exc:
    assert invoked
    print(
      f"PM4 prefix-1 postlaunch failure (claim retained at {claim}): "
      f"{type(exc).__name__}: {exc}", file=errors)
    return 3


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run exactly one guarded PM4 ffn_gate_up prefix-1 attempt")
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
    runner: Callable[..., int] = run_pm4_prefix1,
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
  "PREFIX_EPOCHS", "derive_candidate_executable_identity", "main",
  "run_pm4_prefix1", "validate_blocked_forensic_envelope",
  "validate_pm4_prefix1_semantic_preflight",
]
