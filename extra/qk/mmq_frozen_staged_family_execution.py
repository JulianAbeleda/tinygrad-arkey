"""Fail-closed certification runner for one immutable compact staged family.

The GPU work remains owned by the existing five-buffer harness and tinygrad's
native AMD queues.  This module validates the staged-family manifest, constrains
the existing guarded runner to the exact compact fixed-VA execution mode, and
admits only the monotonic 1 -> 3 -> full prefix ladder.  It deliberately does
not infer phase isolation from launch flags: C4 and per-epoch phase receipts
must be present in the returned evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Mapping

from extra.qk.mmq_exact_role_spec import (
  DEFAULT_INVENTORY, ExactRoleSpec, admit_exact_role_spec, exact_role_spec,
)
from extra.qk.mmq_frozen_staged_family import (
  FrozenStagedFamily, load_frozen_staged_family_manifest,
)


SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_family_execution.v1"
LADDER_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_family_ladder.v1"
RUNTIME_CANARY_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_runtime_canary.v1"
PHASE_SCHEMA = "tinygrad.mmq_q4k_q8_1.frozen_staged_phase_isolation.v1"
INTEGRATION_CAPABILITY_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_integration_capability.v1"
ISOLATED_C4_SCHEMA = \
  "tinygrad.mmq_q4k_q8_1.frozen_staged_runtime_canary_isolation.v1"
QUEUE_MODES = ("PM4", "AQL")
_BUFFER_NAMES = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
_NUMERIC_AUTHORITY = "same_session_fp16_rounded_ds4_reference"


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"schema": SCHEMA, "status": "BLOCKED", "exact_blocker": reason, **evidence}


def _queue_mode(queue_mode: str) -> str:
  if queue_mode not in QUEUE_MODES:
    raise ValueError(f"queue_mode must be one of {QUEUE_MODES!r}")
  return queue_mode


def _prefix_ladder(role_spec: ExactRoleSpec) -> tuple[int, ...]:
  return tuple(dict.fromkeys((1, 3, role_spec.epochs)))


def _prefix(role_spec: ExactRoleSpec, prefix_epochs: int) -> int:
  allowed = _prefix_ladder(role_spec)
  if type(prefix_epochs) is not int or prefix_epochs not in allowed:
    raise ValueError(
      f"staged family prefix_epochs must be one of {allowed!r} for role {role_spec.role!r}")
  return prefix_epochs


def _gate(role_spec: ExactRoleSpec, prefix_epochs: int) -> str:
  return "C6" if prefix_epochs == role_spec.epochs else "C5"


def _probe_integration_capability(
    probe_runner: Callable[..., Mapping[str, Any]] | None,
    ) -> tuple[Callable[..., Mapping[str, Any]] | None, dict[str, Any]]:
  """Resolve the default harness only when it advertises the required receipt."""
  if probe_runner is not None:
    return probe_runner, {
      "schema": INTEGRATION_CAPABILITY_SCHEMA,
      "runner_mode": "injected",
      "runner": f"{getattr(probe_runner, '__module__', type(probe_runner).__module__)}."
                f"{getattr(probe_runner, '__qualname__', type(probe_runner).__qualname__)}",
      "required_phase_receipt_schema": PHASE_SCHEMA,
      "advertised_phase_receipt_schema": None,
      "default_live_harness": False,
      "default_live_integration_qualified": False,
      "injected_runner_contract_validation_only": True,
      "live_gpu_execution_eligible": False,
      "reason": "an injected runner validates the evidence contract but does not qualify the default live harness",
    }
  from extra.qk import mmq_llama_five_buffer_gpu_harness as harness
  advertised = getattr(harness, "FROZEN_STAGED_PHASE_RECEIPT_SCHEMA", None)
  supported = advertised == PHASE_SCHEMA
  evidence = {
    "schema": INTEGRATION_CAPABILITY_SCHEMA,
    "runner_mode": "default_live_harness",
    "runner": "extra.qk.mmq_llama_five_buffer_gpu_harness."
              "run_full_grid_target_role_probe_isolated",
    "required_phase_receipt_schema": PHASE_SCHEMA,
    "advertised_phase_receipt_schema": advertised,
    "default_live_harness": True,
    "default_live_integration_qualified": supported,
    "injected_runner_contract_validation_only": False,
    "live_gpu_execution_eligible": supported,
    "reason": None if supported else
      "the default full-grid harness does not advertise the required staged phase-isolation receipt",
  }
  return (
    harness.run_full_grid_target_role_probe_isolated if supported else None,
    evidence,
  )


def validate_frozen_staged_runtime_canary(
    evidence: Mapping[str, Any], family: FrozenStagedFamily, *, queue_mode: str,
    ) -> dict[str, Any]:
  """Validate the no-target C4 fact for the exact family and queue mode."""
  queue_mode = _queue_mode(queue_mode)
  if not isinstance(evidence, Mapping) or evidence.get("schema") != RUNTIME_CANARY_SCHEMA:
    raise ValueError("staged runtime canary schema is missing or invalid")
  binding, manifest = family.binding, family.manifest
  checks = {
    "status_pass": evidence.get("status") == "PASS",
    "family_identity_matches": evidence.get("family_identity") == family.family_identity,
    "program_key_matches": evidence.get("program_key") == binding.program_key,
    "binary_sha256_matches": evidence.get("binary_sha256") == binding.binary_sha256,
    "queue_mode_matches": evidence.get("queue_mode") == queue_mode,
    "effective_queue_mode_attested":
      type(evidence.get("amd_aql_effective")) is bool and
      evidence["amd_aql_effective"] == (queue_mode == "AQL"),
    "runtime_class_attested":
      isinstance(evidence.get("runtime_class"), str) and bool(evidence["runtime_class"]),
    "queue_class_attested":
      isinstance(evidence.get("queue_class"), str) and bool(evidence["queue_class"]),
    "runtime_count_exact":
      type(evidence.get("runtime_count")) is int and evidence["runtime_count"] == 1,
    "target_dispatch_count_zero":
      type(evidence.get("target_dispatch_count")) is int and
      evidence["target_dispatch_count"] == 0,
    "runtime_cache_binding_exact": evidence.get("runtime_cache_binding_exact") is True,
    "code_ranges_valid": evidence.get("code_ranges_valid") is True,
    "timeline_clean": evidence.get("timeline_clean") is True,
    "health_before": evidence.get("health_before") is True,
    "health_after": evidence.get("health_after") is True,
    "kernel_faults_clean":
      isinstance(evidence.get("kernel_faults"), list) and not evidence["kernel_faults"],
    "compile_performed_false": evidence.get("compile_performed") is False,
    "requires_recompile_false": evidence.get("requires_recompile") is False,
  }
  if manifest["program"]["program_count"] != 1:
    checks["manifest_has_one_compact_program"] = False
  else:
    checks["manifest_has_one_compact_program"] = True
  if not all(checks.values()):
    failed = sorted(key for key, value in checks.items() if not value)
    raise ValueError(f"staged runtime canary failed checks: {failed!r}")
  return {**dict(evidence), "checks": checks, "all_checks_pass": True}


def _validate_phase_isolation(
    evidence: Any, *, prefix_epochs: int, program_key: str,
    ) -> dict[str, Any]:
  """Require explicit stage/target completion receipts; never infer them."""
  if not isinstance(evidence, Mapping) or evidence.get("schema") != PHASE_SCHEMA:
    raise ValueError("staged phase-isolation evidence schema is missing or invalid")
  preparation, epochs = evidence.get("preparation"), evidence.get("epochs")
  if not isinstance(preparation, Mapping) or not isinstance(epochs, list):
    raise ValueError("staged phase-isolation preparation or epoch receipts are missing")
  preparation_checks = {
    "producer_and_output_initialization_only":
      preparation.get("phase") == "producer_and_output_initialization",
    "status_pass": preparation.get("status") == "PASS",
    "target_dispatch_count_zero":
      type(preparation.get("target_dispatch_count")) is int and
      preparation["target_dispatch_count"] == 0,
    "synchronize_returned": preparation.get("synchronize_returned") is True,
    "target_allowed_only_after_synchronize":
      preparation.get("target_allowed_only_after_synchronize") is True,
  }
  if not all(preparation_checks.values()):
    raise ValueError("staged producer/output preparation phase is incomplete")
  if len(epochs) != prefix_epochs:
    raise ValueError("staged phase-isolation epoch receipt count differs")

  normalized_epochs = []
  prior_target_completion = None
  for epoch, row in enumerate(epochs):
    if not isinstance(row, Mapping):
      raise ValueError("staged phase-isolation epoch receipt must be a mapping")
    checks = {
      "epoch_exact":
        type(row.get("epoch")) is int and row["epoch"] == epoch,
      "program_key_exact": row.get("program_key") == program_key,
      "stage_completion_returned": row.get("stage_completion_returned") is True,
      "target_submitted_after_stage_completion":
        row.get("target_submitted_after_stage_completion") is True,
      "target_dispatch_returned": row.get("target_dispatch_returned") is True,
      "target_synchronize_returned": row.get("target_synchronize_returned") is True,
      "overwrite_allowed_only_after_target_completion":
        row.get("overwrite_allowed_only_after_target_completion") is True,
      "prior_target_completion_observed":
        (epoch == 0 and row.get("prior_target_completion_epoch") is None) or
        (epoch > 0 and type(row.get("prior_target_completion_epoch")) is int and
         row["prior_target_completion_epoch"] == epoch - 1),
    }
    if not all(checks.values()):
      failed = sorted(key for key, value in checks.items() if not value)
      raise ValueError(f"staged epoch {epoch} phase receipt failed checks: {failed!r}")
    prior_target_completion = epoch
    normalized_epochs.append({**dict(row), "checks": checks, "all_checks_pass": True})
  return {
    "schema": PHASE_SCHEMA,
    "preparation": {**dict(preparation), "checks": preparation_checks, "all_checks_pass": True},
    "epochs": normalized_epochs,
    "last_completed_target_epoch": prior_target_completion,
    "all_checks_pass": True,
  }


def _validate_stage_vas(
    result: Mapping[str, Any], family: FrozenStagedFamily, *, prefix_epochs: int,
    ) -> dict[str, Any]:
  epoch_staging, metadata_staging = result.get("epoch_staging"), result.get("metadata_staging")
  if not isinstance(epoch_staging, Mapping) or not isinstance(metadata_staging, Mapping):
    raise ValueError("fixed-stage execution evidence is missing")
  if epoch_staging.get("mode") != "all_inputs_fixed_va_gpu_sdma" or \
     metadata_staging.get("mode") != "fixed_va_gpu_sdma" or \
     epoch_staging.get("fixed_va") is not True or metadata_staging.get("fixed_va") is not True or \
     epoch_staging.get("transfer") != "gpu_sdma" or metadata_staging.get("transfer") != "gpu_sdma":
    raise ValueError("execution did not use the exact all-input fixed-VA staged mode")
  epoch_rows, metadata_rows = epoch_staging.get("per_epoch_vas"), metadata_staging.get("per_epoch_vas")
  if not isinstance(epoch_rows, list) or not isinstance(metadata_rows, list) or \
     len(epoch_rows) != prefix_epochs or len(metadata_rows) != prefix_epochs:
    raise ValueError("fixed-stage VA census does not cover the exact prefix")
  if any(not isinstance(row, Mapping) or type(row.get("epoch")) is not int
         for row in [*epoch_rows, *metadata_rows]) or \
     [row["epoch"] for row in epoch_rows] != list(range(prefix_epochs)) or \
     [row["epoch"] for row in metadata_rows] != list(range(prefix_epochs)):
    raise ValueError("fixed-stage VA census epoch order differs")
  stage_columns = (
    (epoch_rows, "stage_q4_va"), (epoch_rows, "stage_values_va"),
    (metadata_rows, "stage_scales_va"), (metadata_rows, "stage_sums_va"),
  )
  if any(any(type(row.get(column)) is not int or row[column] <= 0 for row in rows)
         for rows, column in stage_columns):
    raise ValueError("fixed-stage VA census contains an invalid stage address")
  if any(len({row[column] for row in rows}) != 1 for rows, column in stage_columns):
    raise ValueError("fixed-stage destination VA changed across the prefix")
  stage_vas = [rows[0][column] for rows, column in stage_columns]
  if len(set(stage_vas)) != 4:
    raise ValueError("fixed-stage input destinations alias")
  expected_bytes = {
    row["name"]: row["nbytes"] for row in family.manifest["program"]["abi"][1:]
  }
  per_epoch_stage_vas = [{
    "epoch": epoch,
    "slots_1_4": [
      epoch_rows[epoch]["stage_q4_va"], epoch_rows[epoch]["stage_values_va"],
      metadata_rows[epoch]["stage_scales_va"], metadata_rows[epoch]["stage_sums_va"],
    ],
  } for epoch in range(prefix_epochs)]
  return {
    "mode": "all_inputs_fixed_va_gpu_sdma",
    "prefix_epochs": prefix_epochs,
    "stage_vas": dict(zip(_BUFFER_NAMES[1:], stage_vas)),
    "stage_nbytes": expected_bytes,
    "per_epoch_stage_vas": per_epoch_stage_vas,
    "all_stage_vas_fixed": True,
    "all_stage_vas_distinct": True,
  }


def _validate_runtime_launches(
    result: Mapping[str, Any], family: FrozenStagedFamily, *, prefix_epochs: int,
    queue_mode: str,
    ) -> dict[str, Any]:
  runtime = result.get("runtime_evidence")
  if not isinstance(runtime, Mapping):
    raise ValueError("native runtime evidence is missing")
  if runtime.get("queue_mode") != queue_mode or \
     type(runtime.get("amd_aql_effective")) is not bool or \
     runtime["amd_aql_effective"] != (queue_mode == "AQL"):
    raise ValueError("effective native queue mode differs from the qualified mode")
  if runtime.get("binary_sha256") != family.binding.binary_sha256:
    raise ValueError("runtime binary identity differs from the staged family")
  launches = runtime.get("launches")
  if type(runtime.get("launch_count")) is not int or \
     runtime["launch_count"] != prefix_epochs or \
     not isinstance(launches, list) or len(launches) != prefix_epochs:
    raise ValueError("native launch census does not cover the exact prefix")
  expected_abi = family.manifest["program"]["abi"]
  first_vas = None
  normalized = []
  for epoch, launch in enumerate(launches):
    if not isinstance(launch, Mapping) or \
       type(launch.get("epoch")) is not int or launch["epoch"] != epoch:
      raise ValueError("native launch census epoch order differs")
    global_size, local_size = launch.get("global_size"), launch.get("local_size")
    if not isinstance(global_size, list) or not isinstance(local_size, list) or \
       any(type(value) is not int for value in [*global_size, *local_size]) or \
       global_size != family.manifest["program"]["grid"] or \
       local_size != family.manifest["program"]["local_size"]:
      raise ValueError("native launch geometry differs from the staged family")
    arguments, kernarg = launch.get("arguments"), launch.get("kernarg")
    if not isinstance(arguments, list) or len(arguments) != 5 or not isinstance(kernarg, Mapping):
      raise ValueError("native five-pointer launch census is incomplete")
    rows = []
    for slot, (argument, expected) in enumerate(zip(arguments, expected_abi)):
      if not isinstance(argument, Mapping) or \
         type(argument.get("slot")) is not int or argument["slot"] != slot or \
         argument.get("name") != expected["name"] or \
         type(argument.get("nbytes")) is not int or \
         argument["nbytes"] != expected["nbytes"] or \
         type(argument.get("va")) is not int or argument["va"] <= 0 or \
         argument.get("va_matches_base_offset") is not True:
        raise ValueError("native launch argument differs from the exact staged ABI")
      rows.append(dict(argument))
    vas = [row["va"] for row in rows]
    if len(set(vas)) != 5:
      raise ValueError("native launch arguments alias")
    if first_vas is None: first_vas = vas
    elif vas != first_vas:
      raise ValueError("native launch argument VAs changed across the staged prefix")
    if kernarg.get("pointer_words_match_bound") is not True or \
       kernarg.get("pointer_words") != vas or kernarg.get("bound_pointer_words") != vas:
      raise ValueError("native kernarg pointer census differs from the five launch buffers")
    normalized.append({**dict(launch), "arguments": rows})
  return {
    "queue_mode": queue_mode, "launch_count": prefix_epochs,
    "program_key": family.binding.program_key,
    "fixed_five_vas": first_vas, "launches": normalized,
  }


def _validate_probe_result(
    result: Mapping[str, Any], family: FrozenStagedFamily, *, prefix_epochs: int,
    queue_mode: str, frozen_bundle: str | Path,
    require_guard_evidence: bool = True,
    ) -> dict[str, Any]:
  role_spec, manifest = family.binding.role_spec, family.manifest
  if not isinstance(result, Mapping):
    raise ValueError("guarded staged probe returned no mapping")
  if result.get("status") != "PASS":
    raise ValueError(f"guarded staged probe did not pass: {result.get('exact_blocker')!r}")
  checks = {
    "probe_schema_exact":
      result.get("schema") == "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1",
    "role_exact": result.get("role") == role_spec.role,
    "shape_exact": result.get("shape") == list(role_spec.shape),
    "target_in_place_accumulation":
      result.get("accumulation") == "target_in_place_fp32_add",
    "compile_performed_false": result.get("compile_performed") is False,
    "requires_recompile_false": result.get("requires_recompile") is False,
    "no_fallback": result.get("no_fallback") is True,
  }
  if require_guard_evidence:
    checks.update({
      "health_before": result.get("health_before") is True,
      "health_after": result.get("health_after") is True,
      "mode_health_before": result.get("mode_health_before") is True,
      "mode_health_after": result.get("mode_health_after") is True,
      "isolated_child_queue_request_exact":
        result.get("child_env_overrides") ==
        {"AMD_AQL": "1" if queue_mode == "AQL" else "0"},
      "kernel_faults_clean":
        isinstance(result.get("kernel_faults"), list) and not result["kernel_faults"],
    })
  elif any(key in result for key in (
      "health_before", "health_after", "mode_health_before", "mode_health_after",
      "child_env_overrides", "kernel_faults", "kernel_fault_evidence")):
    raise ValueError(
      "persistent-session direct probe must not supply parent-owned guard evidence")
  correctness = result.get("correctness")
  comparison = correctness.get("comparison") if isinstance(correctness, Mapping) else None
  expected_values = role_spec.m * role_spec.n
  checks.update({
    "correctness_pass":
      isinstance(correctness, Mapping) and correctness.get("status") == "PASS" and
      isinstance(comparison, Mapping) and comparison.get("status") == "pass",
    "numeric_authority_exact":
      isinstance(correctness, Mapping) and
      correctness.get("authority") == _NUMERIC_AUTHORITY,
    "full_output_compared":
      isinstance(comparison, Mapping) and
      type(comparison.get("got_size")) is int and
      comparison["got_size"] == expected_values and
      type(comparison.get("reference_size")) is int and
      comparison["reference_size"] == expected_values and
      type(comparison.get("mismatch_count")) is int and
      comparison["mismatch_count"] == 0,
    "finite_output_and_reference":
      isinstance(comparison, Mapping) and
      all(type(comparison.get(name)) is int
          for name in ("nan_got", "nan_reference", "inf_got", "inf_reference", "joint_finite")) and
      comparison["nan_got"] == comparison["nan_reference"] == 0 and
      comparison["inf_got"] == comparison["inf_reference"] == 0 and
      comparison["joint_finite"] == expected_values,
  })
  artifact = result.get("artifacts")
  frozen = artifact.get("frozen_bundle") if isinstance(artifact, Mapping) else None
  checks.update({
    "artifact_binary_exact":
      isinstance(artifact, Mapping) and artifact.get("binary_sha256") == family.binding.binary_sha256,
    "artifact_source_exact":
      isinstance(artifact, Mapping) and artifact.get("source_sha256") == family.binding.source_sha256,
    "bundle_path_exact":
      isinstance(frozen, Mapping) and
      frozen.get("path") == str(Path(frozen_bundle).resolve()),
    "bundle_program_key_exact":
      isinstance(frozen, Mapping) and frozen.get("program_key") == family.binding.program_key,
    "bundle_execution_role_exact":
      isinstance(frozen, Mapping) and frozen.get("execution_role") == role_spec.role,
    "bundle_program_grid_exact":
      isinstance(frozen, Mapping) and frozen.get("program_grid") == manifest["program"]["grid"],
  })
  if not all(checks.values()):
    failed = sorted(key for key, value in checks.items() if not value)
    raise ValueError(f"guarded staged probe failed identity/correctness checks: {failed!r}")
  stage = _validate_stage_vas(result, family, prefix_epochs=prefix_epochs)
  runtime = _validate_runtime_launches(
    result, family, prefix_epochs=prefix_epochs, queue_mode=queue_mode)
  epoch_crosschecks = []
  for epoch, (stage_row, launch) in enumerate(
      zip(stage["per_epoch_stage_vas"], runtime["launches"])):
    stage_input_vas = stage_row["slots_1_4"]
    launch_input_vas = [row["va"] for row in launch["arguments"][1:]]
    matches = stage_input_vas == launch_input_vas
    epoch_crosschecks.append({
      "epoch": epoch, "stage_destination_vas_slots_1_4": stage_input_vas,
      "launch_kernarg_vas_slots_1_4": launch_input_vas, "all_slots_match": matches,
    })
  if len(epoch_crosschecks) != prefix_epochs or \
     not all(row["all_slots_match"] for row in epoch_crosschecks):
    raise ValueError(
      "fixed stage destination VAs do not match native launch/kernarg input VAs")
  stage_launch_crosscheck = {
    "prefix_epochs": prefix_epochs, "epochs": epoch_crosschecks,
    "all_stage_destinations_match_launch_kernargs": True,
  }
  phase = _validate_phase_isolation(
    result.get("phase_isolation"), prefix_epochs=prefix_epochs,
    program_key=family.binding.program_key)
  return {
    "checks": checks, "stage": stage, "runtime": runtime,
    "stage_launch_crosscheck": stage_launch_crosscheck, "phase_isolation": phase,
    "correctness": dict(correctness), "all_checks_pass": True,
  }


def run_frozen_staged_family_prefix_probe(
    *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    staged_family_manifest: str | Path, prefix_epochs: int, queue_mode: str,
    runtime_canary: Mapping[str, Any], timeout_seconds: float = 900.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    family_loader: Callable[..., FrozenStagedFamily] = load_frozen_staged_family_manifest,
    probe_runner: Callable[..., Mapping[str, Any]] | None = None,
    persistent_session_parent_containment: bool = False,
    ) -> dict[str, Any]:
  """Run one exact staged prefix through the existing guarded full-grid path."""
  try:
    role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
    prefix_epochs = _prefix(role_spec, prefix_epochs)
    queue_mode = _queue_mode(queue_mode)
    if timeout_seconds <= 0: raise ValueError("timeout_seconds must be positive")
    family = family_loader(
      staged_family_manifest, role_spec=role_spec, frozen_bundle=frozen_bundle,
      inventory=inventory)
    if not isinstance(family, FrozenStagedFamily) or family.binding.role_spec != role_spec:
      raise ValueError("staged family loader returned a mismatched family")
    canary = validate_frozen_staged_runtime_canary(
      runtime_canary, family, queue_mode=queue_mode)
  except BaseException as exc:
    return _blocked(
      "staged family admission or C4 runtime canary failed",
      exception=type(exc).__name__, error=str(exc),
      role=getattr(role_spec, "role", None), prefix_epochs=prefix_epochs,
      queue_mode=queue_mode)

  probe_runner, integration_capability = _probe_integration_capability(probe_runner)
  if probe_runner is None:
    return _blocked(
      "default staged live integration lacks required phase-isolation receipts",
      role=role_spec.role, shape=list(role_spec.shape), prefix_epochs=prefix_epochs,
      queue_mode=queue_mode, gate=_gate(role_spec, prefix_epochs),
      family_identity=family.family_identity, program_key=family.binding.program_key,
      c4_runtime_canary=canary, integration_capability=integration_capability,
      target_dispatch_attempted=False)
  kwargs = {
    "timeout_seconds": timeout_seconds, "role_spec": role_spec,
    "warmups": 0, "rounds": 1, "epoch_limit": prefix_epochs,
    "n_chunk_tiles": role_spec.program.grid[0], "epoch_start": 0,
    "host_accumulate": False, "in_kernel_accumulate": True,
    "per_epoch_check": False, "persistent_buffers": True,
    "preloaded_epochs": True, "sync_each_epoch": True,
    "stable_metadata_staging": True, "stable_epoch_staging": True,
    "wait_each_dispatch": True, "frozen_bundle": frozen_bundle,
    "child_env_overrides": {"AMD_AQL": "1" if queue_mode == "AQL" else "0"},
  }
  try:
    raw = dict(probe_runner(**kwargs))
  except BaseException as exc:
    return _blocked(
      "guarded staged prefix runner raised",
      exception=type(exc).__name__, error=str(exc), role=role_spec.role,
      shape=list(role_spec.shape), prefix_epochs=prefix_epochs,
      queue_mode=queue_mode, family_identity=family.family_identity,
      c4_runtime_canary=canary, integration_capability=integration_capability)
  try:
    validated = _validate_probe_result(
      raw, family, prefix_epochs=prefix_epochs, queue_mode=queue_mode,
      frozen_bundle=frozen_bundle,
      require_guard_evidence=not persistent_session_parent_containment)
  except BaseException as exc:
    return _blocked(
      "guarded staged prefix evidence failed closed",
      exception=type(exc).__name__, error=str(exc), role=role_spec.role,
      shape=list(role_spec.shape), prefix_epochs=prefix_epochs,
      queue_mode=queue_mode, gate=_gate(role_spec, prefix_epochs),
      family_identity=family.family_identity, program_key=family.binding.program_key,
      c4_runtime_canary=canary, integration_capability=integration_capability,
      raw_probe=raw)
  return {
    "schema": SCHEMA, "status": "PASS", "exact_blocker": None,
    "gate": _gate(role_spec, prefix_epochs), "role": role_spec.role,
    "shape": list(role_spec.shape), "prefix_epochs": prefix_epochs,
    "allowed_prefix_ladder": list(_prefix_ladder(role_spec)),
    "queue_mode": queue_mode, "queue_qualification_is_separate": True,
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "frozen_bundle": str(Path(frozen_bundle).resolve()),
    "staged_family_manifest": str(Path(staged_family_manifest).resolve()),
    "integration_capability": integration_capability,
    "containment_authority":
      "persistent_c8_session_parent" if persistent_session_parent_containment
      else "per_probe_guard_evidence",
    "c4_runtime_canary": canary, "validation": validated,
    "raw_probe": raw, "compile_performed": False,
    "requires_recompile": False, "hip_used": False, "no_fallback": True,
  }


def run_frozen_staged_family_ladder(
    *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    staged_family_manifest: str | Path, queue_mode: str,
    runtime_canary: Mapping[str, Any], timeout_seconds: float = 900.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    family_loader: Callable[..., FrozenStagedFamily] = load_frozen_staged_family_manifest,
    probe_runner: Callable[..., Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
  """Run 1 -> 3 -> full, stopping before the first unqualified escalation."""
  try:
    role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
    queue_mode = _queue_mode(queue_mode)
  except BaseException as exc:
    return {
      "schema": LADDER_SCHEMA, "status": "BLOCKED",
      "exact_blocker": "staged family ladder admission failed",
      "exception": type(exc).__name__, "error": str(exc),
    }
  rows = []
  for prefix_epochs in _prefix_ladder(role_spec):
    row = run_frozen_staged_family_prefix_probe(
      role_spec=role_spec, frozen_bundle=frozen_bundle,
      staged_family_manifest=staged_family_manifest,
      prefix_epochs=prefix_epochs, queue_mode=queue_mode,
      runtime_canary=runtime_canary, timeout_seconds=timeout_seconds,
      inventory=inventory, family_loader=family_loader, probe_runner=probe_runner)
    rows.append(row)
    if row.get("status") != "PASS":
      return {
        "schema": LADDER_SCHEMA, "status": "BLOCKED",
        "exact_blocker": f"staged family ladder stopped at prefix {prefix_epochs}",
        "role": role_spec.role, "shape": list(role_spec.shape),
        "queue_mode": queue_mode, "queue_qualification_is_separate": True,
        "allowed_prefix_ladder": list(_prefix_ladder(role_spec)),
        "completed_prefixes": [
          item["prefix_epochs"] for item in rows[:-1] if item.get("status") == "PASS"],
        "attempts": rows,
      }
  return {
    "schema": LADDER_SCHEMA, "status": "PASS", "exact_blocker": None,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "queue_mode": queue_mode, "queue_qualification_is_separate": True,
    "allowed_prefix_ladder": list(_prefix_ladder(role_spec)),
    "completed_prefixes": list(_prefix_ladder(role_spec)), "attempts": rows,
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
  }


def _c4_blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {
    "schema": RUNTIME_CANARY_SCHEMA, "status": "BLOCKED",
    "exact_blocker": reason, "target_dispatch_count": 0,
    "compile_performed": False, "requires_recompile": False,
    **evidence,
  }


def _isolated_c4_blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {
    "schema": ISOLATED_C4_SCHEMA, "status": "BLOCKED",
    "exact_blocker": reason,
    "containment_authority": "outer_parent_fresh_process_guards",
    "launched": False, "child_status": "not_launched",
    "timed_out": False, "error": None,
    "target_dispatch_count": 0,
    "health_before": None, "health_after": None,
    "kernel_faults": [], "runtime_canary": None,
    "compile_performed": False, "requires_recompile": False,
    **evidence,
  }


def _queue_environment(queue_mode: str) -> dict[str, str]:
  queue_mode = _queue_mode(queue_mode)
  expected = "1" if queue_mode == "AQL" else "0"
  if os.environ.get("AMD_AQL") != expected:
    raise ValueError(
      f"AMD_AQL must be {expected!r} for requested staged queue mode {queue_mode!r}")
  return {"AMD_AQL": expected}


def _run_frozen_staged_c4_worker(
    role_spec: ExactRoleSpec, frozen_bundle: str, staged_family_manifest: str,
    queue_mode: str, inventory: str | Path | Mapping[str, Any],
    ) -> dict[str, Any]:
  """Spawn-only compact-family C4 entry; never constructs a target CALL."""
  os.environ.update({"AMD_AQL": "1" if queue_mode == "AQL" else "0", "DEV": "AMD"})
  from extra.qk.mmq_llama_five_buffer_gpu_harness import \
    run_frozen_staged_runtime_preconstruction_canary
  # ``run_isolated`` workers are daemon processes and therefore cannot spawn
  # the helper's default health children. The outer parent owns both real
  # health probes and the complete kernel-fault window; these callbacks keep
  # the compact helper focused solely on its no-dispatch runtime operation.
  return run_frozen_staged_runtime_preconstruction_canary(
    role_spec=role_spec, frozen_bundle=frozen_bundle,
    staged_family_manifest=staged_family_manifest, queue_mode=queue_mode,
    inventory=inventory, health_probe=lambda: True,
    fault_collector=lambda started: ([], {}))


def validate_frozen_staged_runtime_canary_isolation(
    evidence: Mapping[str, Any], family: FrozenStagedFamily, *, queue_mode: str,
    ) -> dict[str, Any]:
  """Require the persisted parent-containment envelope and its exact base C4."""
  queue_mode = _queue_mode(queue_mode)
  if not isinstance(evidence, Mapping) or evidence.get("schema") != ISOLATED_C4_SCHEMA:
    raise ValueError("isolated staged runtime canary schema is missing or invalid")
  nested = evidence.get("runtime_canary")
  if not isinstance(nested, Mapping):
    raise ValueError("isolated staged runtime canary lacks its nested base canary")
  validated = validate_frozen_staged_runtime_canary(
    nested, family, queue_mode=queue_mode)
  checks = {
    "status_pass": evidence.get("status") == "PASS",
    "outer_parent_authority_exact":
      evidence.get("containment_authority") ==
      "outer_parent_fresh_process_guards",
    "launched_true": evidence.get("launched") is True,
    "child_status_passed": evidence.get("child_status") == "passed",
    "timed_out_false": evidence.get("timed_out") is False,
    "error_none": evidence.get("error") is None,
    "role_matches":
      evidence.get("role") == family.binding.role_spec.role,
    "shape_matches":
      evidence.get("shape") == list(family.binding.role_spec.shape),
    "family_identity_matches":
      evidence.get("family_identity") == family.family_identity,
    "program_key_matches":
      evidence.get("program_key") == family.binding.program_key,
    "binary_sha256_matches":
      evidence.get("binary_sha256") == family.binding.binary_sha256,
    "queue_mode_matches": evidence.get("queue_mode") == queue_mode,
    "timeout_positive":
      isinstance(evidence.get("timeout_seconds"), (int, float)) and
      not isinstance(evidence["timeout_seconds"], bool) and
      evidence["timeout_seconds"] > 0,
    "elapsed_nonnegative":
      isinstance(evidence.get("elapsed_seconds"), (int, float)) and
      not isinstance(evidence["elapsed_seconds"], bool) and
      evidence["elapsed_seconds"] >= 0,
    "captured_output_strings":
      isinstance(evidence.get("stdout_tail"), str) and
      isinstance(evidence.get("stderr_tail"), str),
    "target_dispatch_count_zero":
      type(evidence.get("target_dispatch_count")) is int and
      evidence["target_dispatch_count"] == 0,
    "health_before": evidence.get("health_before") is True,
    "health_after": evidence.get("health_after") is True,
    "kernel_faults_clean":
      isinstance(evidence.get("kernel_faults"), list) and
      not evidence["kernel_faults"],
    "kernel_fault_evidence_attested":
      isinstance(evidence.get("kernel_fault_evidence"), Mapping),
    "compile_performed_false": evidence.get("compile_performed") is False,
    "requires_recompile_false": evidence.get("requires_recompile") is False,
    "nested_target_dispatch_count_matches":
      evidence.get("target_dispatch_count") ==
      validated.get("target_dispatch_count"),
    "nested_health_matches":
      evidence.get("health_before") == validated.get("health_before") and
      evidence.get("health_after") == validated.get("health_after"),
    "nested_faults_match":
      evidence.get("kernel_faults") == validated.get("kernel_faults"),
  }
  if not all(checks.values()):
    failed = sorted(key for key, value in checks.items() if not value)
    raise ValueError(
      f"isolated staged runtime canary failed checks: {failed!r}")
  return {
    **dict(evidence), "runtime_canary": validated,
    "checks": checks, "all_checks_pass": True,
  }


def run_frozen_staged_runtime_canary_isolated(
    *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    staged_family_manifest: str | Path, queue_mode: str,
    timeout_seconds: float = 300.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    family_loader: Callable[..., FrozenStagedFamily] = load_frozen_staged_family_manifest,
    isolated_runner: Callable[..., Any] | None = None,
    health_probe: Callable[[Mapping[str, str]], bool] | None = None,
    fault_collector: Callable[[float], tuple[list[str], Mapping[str, Any]]] | None = None,
    ) -> dict[str, Any]:
  """Run compact one-PROGRAM C4 in a fresh child with parent-owned containment."""
  try:
    role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
    queue_mode = _queue_mode(queue_mode)
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or \
       timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive")
    env_overrides = _queue_environment(queue_mode)
    family = family_loader(
      staged_family_manifest, role_spec=role_spec, frozen_bundle=frozen_bundle,
      inventory=inventory)
    if not isinstance(family, FrozenStagedFamily) or family.binding.role_spec != role_spec:
      raise ValueError("staged family loader returned a mismatched family")
  except BaseException as exc:
    return _isolated_c4_blocked(
      "isolated staged runtime canary admission failed",
      exception=type(exc).__name__, error=str(exc),
      role=getattr(role_spec, "role", None), queue_mode=queue_mode)

  if isolated_runner is None:
    from tinygrad.runtime.process_isolated import run_isolated
    isolated_runner = run_isolated
  if health_probe is None:
    from extra.qk.mmq_target_epoch_orchestrator import spawned_tiny_health_probe
    health_probe = spawned_tiny_health_probe
  if fault_collector is None:
    from extra.qk.mmq_target_epoch_orchestrator import collect_kernel_fault_evidence
    fault_collector = collect_kernel_fault_evidence

  # The fault window begins before the first parent health dispatch and closes
  # only after the second. Thus either guard's fault/reset is attributable.
  started = time.time()
  try: health_before = bool(health_probe(env_overrides))
  except BaseException: health_before = False
  isolated, runner_error = None, None
  if health_before:
    try:
      isolated = isolated_runner(
        _run_frozen_staged_c4_worker,
        args=(
          role_spec, str(Path(frozen_bundle).resolve()),
          str(Path(staged_family_manifest).resolve()), queue_mode, inventory,
        ),
        timeout_seconds=float(timeout_seconds), start_method="spawn")
    except BaseException as exc:
      runner_error = f"{type(exc).__name__}: {exc}"
  try: health_after = bool(health_probe(env_overrides))
  except BaseException: health_after = False
  try:
    kernel_faults, kernel_fault_evidence = fault_collector(started)
    kernel_faults = list(kernel_faults)
    kernel_fault_evidence = dict(kernel_fault_evidence)
  except BaseException as exc:
    kernel_faults = [
      f"kernel fault collection failed: {type(exc).__name__}: {exc}"]
    kernel_fault_evidence = {}

  launched = health_before
  child_status = (
    "not_launched" if not launched else
    "runner_error" if runner_error is not None else
    getattr(isolated, "status", None))
  timed_out = bool(getattr(isolated, "timed_out", False))
  child_error = runner_error or getattr(isolated, "error", None)
  child = getattr(isolated, "result", None)
  base = dict(child) if isinstance(child, Mapping) else _c4_blocked(
    "isolated staged runtime canary child returned no structured result")
  child_faults = base.get("kernel_faults")
  combined_faults = list(dict.fromkeys(
    [*(child_faults if isinstance(child_faults, list) else []), *kernel_faults]))
  base.update({
    # The base helper's in-child health callbacks are deliberately inert; the
    # independent outer-parent probes are the sole persisted authority.
    "health_before": health_before,
    "health_after": health_after,
    "kernel_faults": combined_faults,
    "kernel_fault_evidence": kernel_fault_evidence,
    "target_dispatch_count": base.get("target_dispatch_count", 0),
    "compile_performed": False, "requires_recompile": False,
  })

  blocker = None
  if not health_before:
    blocker = "isolated staged runtime canary preflight health failed"
  elif timed_out:
    blocker = "isolated staged runtime canary child timed out"
    base.update({
      "status": "BLOCKED",
      "timeout": True, "timeout_seconds": timeout_seconds,
    })
  elif child_status != "passed":
    blocker = child_error or "isolated staged runtime canary child failed"
  elif combined_faults:
    blocker = \
      "kernel fault/reset marker observed during isolated staged runtime canary"
  elif base["health_before"] is not True or base["health_after"] is not True:
    blocker = "isolated staged runtime canary health containment failed"

  if blocker is None and base.get("status") == "PASS":
    try:
      base = validate_frozen_staged_runtime_canary(
        base, family, queue_mode=queue_mode)
    except BaseException as exc:
      blocker = "isolated staged runtime canary evidence failed closed"
      base.update({
        "status": "BLOCKED", "exception": type(exc).__name__,
        "error": str(exc),
      })
  elif blocker is None:
    blocker = base.get("exact_blocker") or \
      "isolated staged runtime canary child did not pass"
  if type(base.get("target_dispatch_count")) is not int or \
     base["target_dispatch_count"] != 0:
    blocker = "isolated staged runtime canary reported an invalid target dispatch count"
    base["status"] = "BLOCKED"

  envelope = {
    "schema": ISOLATED_C4_SCHEMA,
    "status": "PASS" if blocker is None else "BLOCKED",
    "exact_blocker": blocker,
    "containment_authority": "outer_parent_fresh_process_guards",
    "role": role_spec.role, "shape": list(role_spec.shape),
    "queue_mode": queue_mode,
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "launched": launched, "child_status": child_status,
    "timed_out": timed_out, "error": child_error,
    "elapsed_seconds": getattr(isolated, "elapsed_seconds", None),
    "stdout_tail": str(getattr(isolated, "stdout", ""))[-2000:],
    "stderr_tail": str(getattr(isolated, "stderr", ""))[-2000:],
    "timeout_seconds": timeout_seconds,
    "target_dispatch_count": base.get("target_dispatch_count"),
    "health_before": base.get("health_before"),
    "health_after": base.get("health_after"),
    "kernel_faults": combined_faults,
    "kernel_fault_evidence": kernel_fault_evidence,
    "runtime_canary": base,
    "compile_performed": False, "requires_recompile": False,
  }
  if envelope["status"] == "PASS":
    try:
      envelope = validate_frozen_staged_runtime_canary_isolation(
        envelope, family, queue_mode=queue_mode)
    except BaseException as exc:
      envelope.update({
        "status": "BLOCKED",
        "exact_blocker":
          "isolated staged runtime canary envelope failed closed",
        "exception": type(exc).__name__, "validation_error": str(exc),
      })
  return envelope


def run_frozen_staged_family_prefix_from_canary_file(
    *, role_spec: ExactRoleSpec, frozen_bundle: str | Path,
    staged_family_manifest: str | Path, runtime_canary_path: str | Path,
    prefix_epochs: int, queue_mode: str, timeout_seconds: float = 300.0,
    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY,
    family_loader: Callable[..., FrozenStagedFamily] = load_frozen_staged_family_manifest,
    probe_runner: Callable[..., Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
  """Admit an exact persisted C4 result before invoking the approved prefix wrapper."""
  try:
    role_spec = admit_exact_role_spec(role_spec, inventory=inventory)
    queue_mode = _queue_mode(queue_mode)
    _queue_environment(queue_mode)
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or \
       timeout_seconds <= 0:
      raise ValueError("timeout_seconds must be positive")
    family = family_loader(
      staged_family_manifest, role_spec=role_spec, frozen_bundle=frozen_bundle,
      inventory=inventory)
    if not isinstance(family, FrozenStagedFamily) or family.binding.role_spec != role_spec:
      raise ValueError("staged family loader returned a mismatched family")
    persisted = json.loads(Path(runtime_canary_path).read_text())
    if not isinstance(persisted, Mapping):
      raise ValueError("persisted isolated staged runtime canary must be a JSON object")
    isolated_canary = validate_frozen_staged_runtime_canary_isolation(
      persisted, family, queue_mode=queue_mode)
    runtime_canary = isolated_canary["runtime_canary"]
  except BaseException as exc:
    return _blocked(
      "persisted staged C4 runtime canary admission failed",
      exception=type(exc).__name__, error=str(exc),
      role=getattr(role_spec, "role", None), prefix_epochs=prefix_epochs,
      queue_mode=queue_mode, target_dispatch_attempted=False,
      runtime_canary_path=str(Path(runtime_canary_path).resolve()))
  result = run_frozen_staged_family_prefix_probe(
    role_spec=role_spec, frozen_bundle=frozen_bundle,
    staged_family_manifest=staged_family_manifest,
    prefix_epochs=prefix_epochs, queue_mode=queue_mode,
    runtime_canary=runtime_canary, timeout_seconds=timeout_seconds,
    inventory=inventory, family_loader=family_loader, probe_runner=probe_runner)
  return {**result, "c4_runtime_canary_isolation": isolated_canary}


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
  output = Path(path)
  output.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(
    prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
  try:
    with os.fdopen(fd, "w") as handle:
      json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, output)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def main(argv: list[str] | None = None) -> int:
  common = argparse.ArgumentParser(add_help=False)
  common.add_argument("--role", required=True)
  common.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
  common.add_argument("--frozen-bundle", type=Path, required=True)
  common.add_argument("--staged-family-manifest", type=Path, required=True)
  common.add_argument("--queue-mode", choices=QUEUE_MODES, required=True)
  common.add_argument("--timeout-seconds", type=float, default=300.0)
  common.add_argument("--output", type=Path, required=True)
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  subparsers.add_parser(
    "c4", parents=[common],
    help="run one compact staged runtime preconstruction with no target dispatch")
  prefix = subparsers.add_parser(
    "prefix", parents=[common],
    help="run one admitted staged prefix using an exact persisted C4 result")
  prefix.add_argument("--prefix-epochs", type=int, required=True)
  prefix.add_argument("--runtime-canary", type=Path, required=True)
  args = parser.parse_args(argv)

  try:
    role_spec = exact_role_spec(args.role, inventory=args.inventory)
    if args.command == "c4":
      result = run_frozen_staged_runtime_canary_isolated(
        role_spec=role_spec, frozen_bundle=args.frozen_bundle,
        staged_family_manifest=args.staged_family_manifest,
        queue_mode=args.queue_mode, timeout_seconds=args.timeout_seconds,
        inventory=args.inventory)
    else:
      result = run_frozen_staged_family_prefix_from_canary_file(
        role_spec=role_spec, frozen_bundle=args.frozen_bundle,
        staged_family_manifest=args.staged_family_manifest,
        runtime_canary_path=args.runtime_canary,
        prefix_epochs=args.prefix_epochs, queue_mode=args.queue_mode,
        timeout_seconds=args.timeout_seconds, inventory=args.inventory)
  except BaseException as exc:
    result = (
      _isolated_c4_blocked(
        "staged CLI admission failed", exception=type(exc).__name__, error=str(exc))
      if args.command == "c4" else
      _blocked(
        "staged CLI admission failed", exception=type(exc).__name__, error=str(exc),
        target_dispatch_attempted=False)
    )
  _atomic_write_json(args.output, result)
  print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
  return 0 if result.get("status") == "PASS" else 1


__all__ = [
  "INTEGRATION_CAPABILITY_SCHEMA", "ISOLATED_C4_SCHEMA", "LADDER_SCHEMA",
  "PHASE_SCHEMA", "QUEUE_MODES",
  "RUNTIME_CANARY_SCHEMA", "SCHEMA",
  "run_frozen_staged_family_prefix_from_canary_file",
  "run_frozen_staged_family_ladder", "run_frozen_staged_family_prefix_probe",
  "run_frozen_staged_runtime_canary_isolated",
  "validate_frozen_staged_runtime_canary",
  "validate_frozen_staged_runtime_canary_isolation",
]


if __name__ == "__main__": raise SystemExit(main(sys.argv[1:]))
