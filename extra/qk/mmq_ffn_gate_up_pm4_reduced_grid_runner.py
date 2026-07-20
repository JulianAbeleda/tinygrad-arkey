"""One-shot PM4 reduced-grid diagnostic runner for frozen ``ffn_gate_up``.

The runner admits only the review-bounded origin-anchored grid ladder.  It
reuses the exact prefix-1 semantic preflight and production runtime builder,
then retains either PASS or BLOCKED as immutable, promotion-ineligible
forensics.  A reduced-grid PASS is not full-output correctness evidence.
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
  ENVELOPE_SCHEMA, FFN_REDUCED_GRID_SCHEMA, SCHEMA,
  build_production_candidate_prefix_runtime, validate_guarded_envelope,
  validate_ffn_reduced_grid_evidence,
)
from extra.qk.mmq_ffn_gate_up_pm4_prefix1_runner import (
  _content_identity, _fsync_directory, _fsync_file, _identity,
  _publish_forensic_envelope, _resolve_input, _resolve_output,
  validate_pm4_prefix1_semantic_preflight,
)


PM4_QUEUE_MODE = "PM4"
PREFIX_EPOCHS = 1
DEFAULT_TIMEOUT_SECONDS = 900.0
FROZEN_FULL_GLOBAL_SIZE = (136, 4, 1)
FROZEN_LOCAL_SIZE = (256, 1, 1)
OUTPUT_TILE_COLUMNS = 128
OUTPUT_TILE_ROWS = 128
ALLOWED_DIAGNOSTIC_GLOBAL_SIZES = (
  (1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 4, 1), (8, 4, 1),
  (32, 4, 1), (40, 4, 1), (41, 4, 1), (136, 1, 1),
  # Boundary-search / deconfound rows for the (8,4,1)-pass -> (32,4,1)-fault
  # transition: separate max column index (gidx0) from total workgroup count.
  (16, 1, 1), (32, 1, 1), (64, 1, 1), (16, 4, 1),
)
CLAIM_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_pm4_reduced_grid_claim.v1"
RECEIPT_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.ffn_gate_up_pm4_reduced_grid_runner_receipt.v1"
_ENVELOPE_KEYS = {
  "schema", "status", "exact_blocker", "queue_mode", "operation_schema",
  "health_before", "health_after", "kernel_faults",
  "kernel_fault_evidence", "launched", "spawn_count", "child_status",
  "timed_out", "error", "elapsed_seconds", "result", "no_retry",
  "retry_count", "no_queue_fallback", "promotion_evidence_eligible",
  "request_identity", "config_identity", "evidence_identity",
}


def _default_guarded_stage(**kwargs: Any) -> Mapping[str, Any]:
  from extra.qk.mmq_ffn_gate_up_guarded_correctness import \
    run_guarded_ffn_reduced_grid
  return run_guarded_ffn_reduced_grid(**kwargs)


def validate_diagnostic_global_size(
    grid_x: Any, grid_y: Any,
    ) -> tuple[int, int, int]:
  """Return one exact allowlisted grid; reject booleans and coercions."""
  if type(grid_x) is not int or type(grid_y) is not int:
    raise ValueError("grid-x and grid-y must be integers (not booleans)")
  grid = (grid_x, grid_y, 1)
  if grid not in ALLOWED_DIAGNOSTIC_GLOBAL_SIZES:
    raise ValueError(
      f"diagnostic global size {grid!r} is not in the exact allowlist")
  return grid


def touched_output_rectangle(
    diagnostic_global_size: Sequence[int],
    ) -> dict[str, int]:
  grid = tuple(diagnostic_global_size)
  if grid not in ALLOWED_DIAGNOSTIC_GLOBAL_SIZES:
    raise ValueError("touched output rectangle requires an allowlisted grid")
  rows = grid[1] * OUTPUT_TILE_ROWS
  columns = grid[0] * OUTPUT_TILE_COLUMNS
  return {
    "row_start": 0, "row_stop_exclusive": rows,
    "column_start": 0, "column_stop_exclusive": columns,
    "row_count": rows, "column_count": columns,
    "element_count": rows * columns,
  }


def _request_identity(
    config: Mapping[str, Any], diagnostic_global_size: Sequence[int],
    ) -> str:
  return _identity({
    "schema": f"{SCHEMA}.ffn_reduced_grid_request",
    "queue_mode": PM4_QUEUE_MODE, "prefix_epochs": PREFIX_EPOCHS,
    "diagnostic_global_size": list(diagnostic_global_size),
    "config_identity": _identity(dict(config)),
  })


def _claim_path(output: Path) -> Path:
  return output.with_name(f"{output.name}.claim")


def _acquire_claim(
    output: Path, *, diagnostic_global_size: Sequence[int],
    ) -> Path:
  claim = _claim_path(output)
  payload = {
    "schema": CLAIM_SCHEMA, "pid": os.getpid(), "output": str(output),
    "queue_mode": PM4_QUEUE_MODE, "prefix_epochs": PREFIX_EPOCHS,
    "diagnostic_global_size": list(diagnostic_global_size),
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


def validate_pm4_reduced_grid_forensic_envelope(
    value: Any, *, config: Mapping[str, Any],
    diagnostic_global_size: Sequence[int],
    ) -> dict[str, Any]:
  """Validate PASS or BLOCKED reduced-grid evidence as forensics only."""
  grid = tuple(diagnostic_global_size)
  if grid not in ALLOWED_DIAGNOSTIC_GLOBAL_SIZES:
    raise ValueError("forensic reduced grid is not allowlisted")
  if not isinstance(value, Mapping):
    raise ValueError("PM4 reduced-grid forensic envelope must be a mapping")
  row = dict(value)
  if set(row) != _ENVELOPE_KEYS or \
     row.get("evidence_identity") != _identity({
       key: item for key, item in row.items()
       if key != "evidence_identity"}):
    raise ValueError("PM4 reduced-grid envelope fields/identity differ")
  if row.get("schema") != ENVELOPE_SCHEMA or \
     row.get("queue_mode") != PM4_QUEUE_MODE or \
     row.get("operation_schema") != FFN_REDUCED_GRID_SCHEMA:
    raise ValueError("PM4 reduced-grid forensic operation differs")
  if row.get("no_retry") is not True or row.get("retry_count") != 0 or \
     row.get("no_queue_fallback") is not True or \
     row.get("promotion_evidence_eligible") is not False:
    raise ValueError("PM4 reduced-grid forensic safety contract differs")
  if type(row.get("health_before")) is not bool or \
     type(row.get("health_after")) is not bool or \
     type(row.get("launched")) is not bool or \
     type(row.get("timed_out")) is not bool:
    raise ValueError("PM4 reduced-grid forensic boolean facts differ")
  if row.get("spawn_count") not in (0, 1) or \
     row["launched"] is not (row["spawn_count"] == 1):
    raise ValueError("PM4 reduced-grid forensic spawn facts differ")
  if not isinstance(row.get("kernel_faults"), list) or \
     any(not isinstance(item, str) for item in row["kernel_faults"]) or \
     not isinstance(row.get("kernel_fault_evidence"), Mapping):
    raise ValueError("PM4 reduced-grid forensic fault facts differ")
  elapsed = row.get("elapsed_seconds")
  if elapsed is not None and (
      not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool) or
      not math.isfinite(elapsed) or elapsed < 0):
    raise ValueError("PM4 reduced-grid forensic elapsed time differs")
  config_identity = _identity(dict(config))
  request_identity = _request_identity(config, grid)
  if row.get("config_identity") != config_identity or \
     row.get("request_identity") != request_identity:
    raise ValueError("PM4 reduced-grid forensic request/config binding differs")
  _content_identity(config_identity, "forensic config identity")
  _content_identity(request_identity, "forensic request identity")

  status, result = row.get("status"), row.get("result")
  if status == "PASS":
    if row.get("exact_blocker") is not None:
      raise ValueError("passing reduced-grid diagnostic has a blocker")
    validate_guarded_envelope(row)
    if not isinstance(result, Mapping):
      raise ValueError("passing reduced-grid diagnostic has no child result")
    effective = result.get(
      "diagnostic_global_size", result.get("effective_global_size"))
    if tuple(effective or ()) != grid or \
       result.get("target_dispatch_submitted") is not True or \
       result.get("promotion_evidence_eligible") is not False:
      raise ValueError("passing reduced-grid child grid/submission differs")
    return row
  if status != "BLOCKED" or \
     not isinstance(row.get("exact_blocker"), str) or \
     not row["exact_blocker"]:
    raise ValueError("PM4 reduced-grid forensic state differs")
  if result is not None:
    if not isinstance(result, Mapping):
      raise ValueError("blocked reduced-grid child result differs")
    child = dict(result)
    if child.get("status") == "PASS":
      validated_child = validate_ffn_reduced_grid_evidence(
        child, diagnostic_global_size=grid)
      if validated_child["config_identity"] != config_identity or \
         validated_child["request_identity"] != request_identity:
        raise ValueError("blocked outer envelope child binding differs")
    else:
      if child.get("evidence_identity") != _identity({
          key: item for key, item in child.items()
          if key != "evidence_identity"}) or \
         child.get("schema") != FFN_REDUCED_GRID_SCHEMA or \
         child.get("status") != "BLOCKED" or \
         not isinstance(child.get("exact_blocker"), str) or \
         not child["exact_blocker"] or \
         child.get("queue_mode") != PM4_QUEUE_MODE or \
         child.get("prefix_epochs") != PREFIX_EPOCHS or \
         tuple(child.get("diagnostic_global_size", ())) != grid or \
         child.get("no_retry") is not True or \
         child.get("retry_count") != 0 or \
         child.get("no_fallback") is not True or \
         child.get("compile_performed") is not False or \
         child.get("requires_recompile") is not False or \
         child.get("promotion_evidence_eligible") is not False or \
         child.get("config_identity") != config_identity or \
         child.get("request_identity") != request_identity:
        raise ValueError("blocked reduced-grid child safety/binding differs")
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
    envelope: Mapping[str, Any], output: Path,
    diagnostic_global_size: Sequence[int], stream: TextIO,
    ) -> None:
  nested = envelope.get("result")
  submitted = nested.get("target_dispatch_submitted") \
    if isinstance(nested, Mapping) else None
  diagnostic_receipt = nested.get("diagnostic_receipt") \
    if isinstance(nested, Mapping) else None
  pre_submit = diagnostic_receipt.get("pre_submit") \
    if isinstance(diagnostic_receipt, Mapping) else None
  pm4_user_data = pre_submit.get("pm4_kernarg_user_data") \
    if isinstance(pre_submit, Mapping) else None
  receipt = {
    "schema": RECEIPT_SCHEMA,
    "diagnostic": "PM4_REDUCED_GRID",
    "status": envelope["status"], "output": str(output),
    "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "outer_evidence_identity": envelope["evidence_identity"],
    "nested_diagnostic_evidence_identity":
      nested.get("evidence_identity") if isinstance(nested, Mapping) else None,
    "diagnostic_global_size": list(diagnostic_global_size),
    "frozen_full_global_size": list(FROZEN_FULL_GLOBAL_SIZE),
    "frozen_local_size": list(FROZEN_LOCAL_SIZE),
    "touched_output_rectangle":
      touched_output_rectangle(diagnostic_global_size),
    "health_before": envelope["health_before"],
    "health_after": envelope["health_after"],
    "kernel_faults": envelope["kernel_faults"],
    "kernel_fault_evidence_status":
      envelope["kernel_fault_evidence"].get("status"),
    "launched": envelope["launched"],
    "spawn_count": envelope["spawn_count"],
    "target_dispatch_submitted": submitted,
    "pre_submit_pm4_sha256":
      pre_submit.get("pm4_sha256")
      if isinstance(pre_submit, Mapping) else None,
    "pre_submit_kernarg_va":
      pre_submit.get("kernarg_va")
      if isinstance(pre_submit, Mapping) else None,
    "pre_submit_kernarg_qwords":
      pre_submit.get("kernarg_qwords")
      if isinstance(pre_submit, Mapping) else None,
    "pre_submit_user_data_pointer":
      pm4_user_data.get("pointer")
      if isinstance(pm4_user_data, Mapping) else None,
    "blocker": envelope["exact_blocker"],
    "promotion_evidence_eligible": False,
    "full_grid_correctness_claimed": False,
  }
  stream.write(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
  stream.flush()


def run_pm4_reduced_grid(
    *, frozen_bundle: str | Path, staged_family_manifest: str | Path,
    execution_fixture_v2: str | Path, pm4_c4: str | Path,
    output: str | Path, grid_x: int, grid_y: int,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    semantic_preflight: Callable[
      [str | Path, str | Path, str | Path, str | Path], str] =
      validate_pm4_prefix1_semantic_preflight,
    claim_acquirer: Callable[..., Path] = _acquire_claim,
    guarded_stage: Callable[..., Mapping[str, Any]] =
      _default_guarded_stage,
    forensic_validator: Callable[..., dict[str, Any]] =
      validate_pm4_reduced_grid_forensic_envelope,
    forensic_publisher: Callable[[Path, Mapping[str, Any]], None] =
      _publish_forensic_envelope,
    error_stream: TextIO | None = None,
    receipt_stream: TextIO | None = None,
    ) -> int:
  """Run and retain one reduced-grid diagnostic, with no retry/fallback."""
  errors = sys.stderr if error_stream is None else error_stream
  receipts = sys.stdout if receipt_stream is None else receipt_stream
  claim: Path | None = None
  invoked = False
  try:
    grid = validate_diagnostic_global_size(grid_x, grid_y)
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
    claim = claim_acquirer(
      output_path, diagnostic_global_size=grid)
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
      f"PM4 reduced-grid prelaunch failure: {type(exc).__name__}: {exc}",
      file=errors)
    return 2

  try:
    invoked = True
    previous_environment = _set_fixed_parent_gpu_environment()
    try:
      envelope = guarded_stage(
        config=config,
        runtime_builder=build_production_candidate_prefix_runtime,
        diagnostic_global_size=grid, timeout_seconds=timeout)
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
    validated = forensic_validator(
      envelope, config=config, diagnostic_global_size=grid)
    if validated.get("status") not in ("PASS", "BLOCKED"):
      raise ValueError("guarded diagnostic status is neither PASS nor BLOCKED")
    forensic_publisher(output_path, validated)
    loaded = forensic_validator(
      json.loads(output_path.read_bytes()), config=config,
      diagnostic_global_size=grid)
    if loaded != validated:
      raise ValueError("forensic reduced-grid envelope round-trip differs")
    _fsync_file(output_path)
    _fsync_directory(output_path.parent)
    _emit_receipt(validated, output_path, grid, receipts)
    status = validated["status"]
    _release_claim(claim)
    return 0 if status == "PASS" else 1
  except BaseException as exc:
    assert invoked
    print(
      f"PM4 reduced-grid postlaunch failure (claim retained at {claim}): "
      f"{type(exc).__name__}: {exc}", file=errors)
    return 3


def _parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Run one allowlisted guarded PM4 ffn_gate_up reduced-grid diagnostic"))
  parser.add_argument("--frozen-bundle", required=True)
  parser.add_argument("--staged-family-manifest", required=True)
  parser.add_argument("--execution-fixture-v2", required=True)
  parser.add_argument("--pm4-c4", required=True)
  parser.add_argument("--output", required=True)
  parser.add_argument("--grid-x", required=True, type=int)
  parser.add_argument("--grid-y", required=True, type=int)
  parser.add_argument(
    "--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
  return parser


def main(
    argv: Sequence[str] | None = None, *,
    runner: Callable[..., int] = run_pm4_reduced_grid,
    ) -> int:
  args = _parser().parse_args(argv)
  return runner(
    frozen_bundle=args.frozen_bundle,
    staged_family_manifest=args.staged_family_manifest,
    execution_fixture_v2=args.execution_fixture_v2,
    pm4_c4=args.pm4_c4, output=args.output,
    grid_x=args.grid_x, grid_y=args.grid_y,
    timeout_seconds=args.timeout_seconds)


if __name__ == "__main__":
  raise SystemExit(main())


__all__ = [
  "ALLOWED_DIAGNOSTIC_GLOBAL_SIZES", "CLAIM_SCHEMA",
  "DEFAULT_TIMEOUT_SECONDS", "FROZEN_FULL_GLOBAL_SIZE",
  "FROZEN_LOCAL_SIZE", "PM4_QUEUE_MODE", "PREFIX_EPOCHS",
  "RECEIPT_SCHEMA", "main", "run_pm4_reduced_grid",
  "touched_output_rectangle", "validate_diagnostic_global_size",
  "validate_pm4_reduced_grid_forensic_envelope",
]
