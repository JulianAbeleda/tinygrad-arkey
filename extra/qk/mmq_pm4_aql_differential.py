"""Fault-contained PM4-versus-AQL launcher differential for a frozen MMQ PROGRAM.

This module does not compile or dispatch a PROGRAM itself.  It validates one
frozen target bundle CPU-only, then asks the existing isolated target-role
probe to consume that exact PROGRAM in fresh child processes.  The only
intentional environment difference between the two modes is ``AMD_AQL``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from extra.qk.mmq_frozen_target_artifact import (
  ACCUMULATION, FILE_NAMES, FrozenTargetArtifact, load_frozen_target_artifact,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.pm4_aql_frozen_differential.v1"
EPOCH_PREFIXES = (1, 3)
N_CHUNK_TILES = 136
MODE_VALUES = (("pm4", "0"), ("aql", "1"))
SHARED_LAYER_CAVEAT = (
  "PM4 and AQL are not wholly independent implementations: they share the "
  "frozen generated ISA/code object, KFD/amdgpu, allocations, fixture, and "
  "substantial tinygrad runtime code.  A matching outcome does not isolate "
  "faults within those shared layers."
)

Runner = Callable[..., dict[str, Any]]
Loader = Callable[[str | Path], FrozenTargetArtifact]


def _default_runner() -> Runner:
  from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_target_role_probe_isolated
  return run_full_grid_target_role_probe_isolated


def _canonical_sha256(value: Any) -> str:
  encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return hashlib.sha256(encoded).hexdigest()


def _environment_identity(environ: Mapping[str, str]) -> str:
  """Hash, but never disclose, the ambient environment except AMD_AQL."""
  return _canonical_sha256(sorted((key, value) for key, value in environ.items() if key != "AMD_AQL"))


def _frozen_identity(artifact: FrozenTargetArtifact) -> dict[str, Any]:
  manifest, fixture = artifact.manifest, artifact.fixture
  program = manifest["program"]
  artifacts = manifest["artifacts"]
  return {
    "manifest_schema": manifest["schema"],
    "state": manifest["state"],
    "program_key": program["key"],
    "function": program["function"],
    "source_sha256": artifacts["source_sha256"],
    "binary_sha256": artifacts["binary_sha256"],
    "serialized_program_sha256": artifacts["serialized_program_sha256"],
    "fixture_schema": fixture["schema"],
    "fixture_sha256": manifest["files"][FILE_NAMES["fixture"]]["sha256"],
  }


def _blocked(reason: str, *, bundle: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
  return {
    "schema": SCHEMA, "status": "BLOCKED", "passed": False,
    "exact_blocker": reason, "bundle": bundle or {},
    "epoch_prefixes": list(EPOCH_PREFIXES), "modes": [],
    "classification": "INCONCLUSIVE", "shared_layer_caveat": SHARED_LAYER_CAVEAT,
    **extra,
  }


def _mapping(value: Any) -> Mapping[str, Any]:
  return value if isinstance(value, Mapping) else {}


def _validate_frozen_run(result: Any, *, mode: str, amd_aql: str, epoch_prefix: int,
                         expected: Mapping[str, Any]) -> list[str]:
  errors: list[str] = []
  if not isinstance(result, dict): return ["isolated runner returned no structured result"]
  artifacts = _mapping(result.get("artifacts"))
  timing = _mapping(result.get("timing"))
  correctness = _mapping(result.get("correctness"))
  comparison = _mapping(correctness.get("comparison"))
  frozen = _mapping(artifacts.get("frozen_bundle"))
  runtime = _mapping(result.get("runtime_evidence"))
  health_mode = _mapping(result.get("health_mode"))

  if result.get("status") != "PASS": errors.append(f"target status is {result.get('status')!r}")
  if correctness.get("status") != "PASS" or comparison.get("status") != "pass":
    errors.append("numeric correctness did not pass")
  if result.get("kernel_faults") != []: errors.append("kernel fault/reset evidence is nonempty")
  if result.get("health_before") is not True or result.get("health_after") is not True:
    errors.append("mode-specific pre/post health did not both pass")
  if health_mode.get("amd_aql_env") != amd_aql:
    errors.append("health canary lacks selected AMD_AQL identity")
  if health_mode.get("before") is not True or health_mode.get("after") is not True:
    errors.append("selected-mode health attestation did not pass")

  if result.get("accumulation") != ACCUMULATION or artifacts.get("accumulation") != ACCUMULATION:
    errors.append("run did not use frozen in-kernel accumulation")
  if timing.get("persistent_buffers") is not True or timing.get("preloaded_epochs") is not True:
    errors.append("persistent preloaded buffer lifecycle was not attested")
  if timing.get("stable_metadata_staging") is not True:
    errors.append("stable metadata staging was not attested")
  if timing.get("k_epoch_launches") != epoch_prefix:
    errors.append("launch count differs from requested epoch prefix")
  if timing.get("epoch_checks") not in ([], ()):
    errors.append("intermediate epoch readback was observed")
  if artifacts.get("no_fallback") is not True or result.get("no_fallback") is not True:
    errors.append("no-fallback evidence is missing")

  if artifacts.get("source_sha256") != expected["source_sha256"]:
    errors.append("source identity differs from frozen bundle")
  if artifacts.get("binary_sha256") != expected["binary_sha256"]:
    errors.append("binary identity differs from frozen bundle")
  for key in ("manifest_schema", "state", "program_key", "serialized_program_sha256", "fixture_sha256"):
    if frozen.get(key) != expected[key]: errors.append(f"frozen bundle {key} identity mismatch")
  if frozen.get("requires_recompile") is not False or frozen.get("compile_performed") is not False:
    errors.append("zero-recompile consumption was not attested")

  if str(runtime.get("amd_aql_env")) != amd_aql:
    errors.append("target runtime lacks selected AMD_AQL identity")
  if runtime.get("amd_aql_effective") is not (amd_aql == "1"):
    errors.append("effective target queue selection differs from AMD_AQL")
  expected_mode = "AQL" if mode == "aql" else "PM4"
  if str(runtime.get("queue_mode", "")).upper() != expected_mode:
    errors.append(f"runtime queue mode is not {expected_mode}")
  if runtime.get("launch_count") != epoch_prefix:
    errors.append("runtime launch evidence differs from epoch prefix")
  if runtime.get("intermediate_readback") is not False:
    errors.append("runtime did not attest absence of intermediate readback")
  if runtime.get("external_accumulation_add") is not False:
    errors.append("runtime did not attest absence of an external accumulation add")
  return errors


def _classification(mode_rows: list[dict[str, Any]]) -> str:
  if all(row["status"] == "PASS" for row in mode_rows): return "NO_DIFFERENTIAL_FAILURE"
  attempts = {row["mode"]: {x["epoch_prefix"]: x["status"] for x in row["attempts"]} for row in mode_rows}
  failed = {mode for mode, rows in attempts.items() if "BLOCKED" in rows.values()}
  if failed == {"pm4"} and any(
      status == "BLOCKED" and attempts["aql"].get(prefix) == "PASS"
      for prefix, status in attempts["pm4"].items()): return "PM4_ONLY_FAILURE"
  if failed == {"aql"} and any(
      status == "BLOCKED" and attempts["pm4"].get(prefix) == "PASS"
      for prefix, status in attempts["aql"].items()): return "AQL_ONLY_FAILURE"
  if failed == {"pm4", "aql"}: return "BOTH_MODES_FAILED_SHARED_OR_KERNEL_LAYER"
  return "INCONCLUSIVE"


def run_pm4_aql_frozen_differential(bundle_path: str | Path, *, timeout_seconds: float = 900.0,
                                     runner: Runner | None = None,
                                     loader: Loader = load_frozen_target_artifact,
                                     environ: MutableMapping[str, str] | None = None) -> dict[str, Any]:
  """Run the fixed 1-then-3 epoch differential without compiling.

  Each mode stops at its first failed prefix.  The other mode is still run so
  the result can distinguish a queue-mode-specific failure from a failure
  shared by both launch paths.
  """
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  path = Path(bundle_path).expanduser().resolve()
  try:
    artifact = loader(path)  # The sole CPU-only validation/load boundary.
    identity = _frozen_identity(artifact)
  except BaseException as exc:
    return _blocked(f"frozen bundle validation failed: {type(exc).__name__}: {exc}")
  manifest = artifact.manifest
  if manifest.get("compile_calls") != 1 or manifest.get("consumer", {}).get("requires_recompile") is not False:
    return _blocked("frozen bundle does not prove one compile and zero-recompile consumption", bundle=identity)
  if manifest.get("accumulation") != ACCUMULATION or manifest.get("accumulate") is not True:
    return _blocked("frozen bundle is not the in-kernel accumulating target PROGRAM", bundle=identity)

  selected_runner, selected_environ = runner or _default_runner(), environ if environ is not None else os.environ
  base_env = _environment_identity(selected_environ)
  mode_rows = [{
    "mode": mode, "AMD_AQL": amd_aql, "status": "NOT_RUN",
    "stopped_after_failure": False, "attempts": [],
  } for mode, amd_aql in MODE_VALUES]
  common_kwargs = {
    "timeout_seconds": timeout_seconds, "warmups": 0, "rounds": 1,
    "n_chunk_tiles": N_CHUNK_TILES, "epoch_start": 0,
    "host_accumulate": False, "in_kernel_accumulate": True, "per_epoch_check": False,
    "persistent_buffers": True, "preloaded_epochs": True, "sync_each_epoch": False,
    "stable_metadata_staging": True, "frozen_bundle": str(path),
  }
  escalation_stop_reason = None
  unsafe_stop = False
  # Prefix-major ordering is intentional: establish the matched one-epoch
  # baseline before either queue mode is exposed to the three-epoch sequence.
  for prefix in EPOCH_PREFIXES:
    for row in mode_rows:
      mode, amd_aql = row["mode"], row["AMD_AQL"]
      if _environment_identity(selected_environ) != base_env:
        row["attempts"].append({"epoch_prefix": prefix, "status": "BLOCKED",
                                "validation_errors": ["ambient environment changed between modes"]})
        unsafe_stop, escalation_stop_reason = True, "ambient environment changed during differential"
        break
      try:
        result = selected_runner(
          **common_kwargs, epoch_limit=prefix,
          child_env_overrides={"AMD_AQL": amd_aql},
        )
      except BaseException as exc:
        result = {"status": "BLOCKED", "exact_blocker": f"isolated runner raised {type(exc).__name__}: {exc}"}
      errors = _validate_frozen_run(result, mode=mode, amd_aql=amd_aql,
                                    epoch_prefix=prefix, expected=identity)
      row["attempts"].append({
        "epoch_prefix": prefix, "status": "PASS" if not errors else "BLOCKED",
        "validation_errors": errors, "result": result,
      })
      row["stopped_after_failure"] = bool(errors)
      # Never submit another target after an uncontained health failure.  A
      # fault with recovered post-health may still be matched at this prefix,
      # but it prevents escalation to any larger prefix below.
      if isinstance(result, dict) and result.get("health_after") is not True:
        unsafe_stop = True
        escalation_stop_reason = f"{mode}-{prefix} did not leave a healthy {mode}-selected canary"
        break
    if unsafe_stop: break
    current = [
      next((attempt for attempt in row["attempts"] if attempt["epoch_prefix"] == prefix), None)
      for row in mode_rows
    ]
    if any(attempt is None or attempt["status"] != "PASS" for attempt in current):
      escalation_stop_reason = (
        f"matched prefix {prefix} did not pass in both modes; larger prefixes were not submitted")
      break
  for row in mode_rows:
    row["status"] = ("PASS" if [x["epoch_prefix"] for x in row["attempts"]] == list(EPOCH_PREFIXES)
                     and all(x["status"] == "PASS" for x in row["attempts"])
                     else "BLOCKED" if any(x["status"] == "BLOCKED" for x in row["attempts"])
                     else "INCOMPLETE")

  classification = _classification(mode_rows)
  passed = classification == "NO_DIFFERENTIAL_FAILURE"
  return {
    "schema": SCHEMA, "status": "PASS" if passed else "BLOCKED", "passed": passed,
    "exact_blocker": None if passed else "PM4/AQL frozen differential did not pass all prefixes",
    "bundle": identity, "bundle_validations": 1, "compile_performed": False,
    "epoch_prefixes": list(EPOCH_PREFIXES), "base_environment_sha256": base_env,
    "intentional_environment_difference": {"key": "AMD_AQL", "pm4": "0", "aql": "1"},
    "forced_lifecycle": {
      "in_kernel_accumulate": True, "persistent_buffers": True, "preloaded_epochs": True,
      "stable_metadata_staging": True, "per_epoch_check": False,
      "intermediate_readback": False, "external_accumulation_add": False,
    },
    "sequence_policy": "prefix-major: PM4-1,AQL-1; PM4-3,AQL-3 only after a matched clean prefix",
    "escalation_stop_reason": escalation_stop_reason,
    "modes": mode_rows, "classification": classification, "shared_layer_caveat": SHARED_LAYER_CAVEAT,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("bundle", type=Path)
  parser.add_argument("--timeout-seconds", type=float, default=900.0)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args(argv)
  result = run_pm4_aql_frozen_differential(args.bundle, timeout_seconds=args.timeout_seconds)
  encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
  print(encoded, end="")
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())


__all__ = ["EPOCH_PREFIXES", "SCHEMA", "run_pm4_aql_frozen_differential"]
