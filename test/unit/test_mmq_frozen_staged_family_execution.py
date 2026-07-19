from __future__ import annotations

from pathlib import Path

import pytest

from extra.qk.mmq_exact_role_spec import DEFAULT_INVENTORY
from extra.qk.mmq_frozen_staged_family import load_frozen_staged_family_manifest
from extra.qk.mmq_frozen_staged_family_execution import (
  INTEGRATION_CAPABILITY_SCHEMA, LADDER_SCHEMA, PHASE_SCHEMA,
  RUNTIME_CANARY_SCHEMA, SCHEMA,
  run_frozen_staged_family_ladder, run_frozen_staged_family_prefix_probe,
  validate_frozen_staged_runtime_canary,
)
from test.unit.test_mmq_frozen_staged_family import _loader, _produce


def _family(tmp_path: Path):
  role_spec, binding, manifest_path, _ = _produce(tmp_path)

  def load(path, *, role_spec, frozen_bundle, inventory):
    return load_frozen_staged_family_manifest(
      path, role_spec=role_spec, frozen_bundle=frozen_bundle, inventory=inventory,
      binding_loader=_loader(binding))

  family = load(
    manifest_path, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    inventory=DEFAULT_INVENTORY)
  return role_spec, binding, manifest_path, family, load


def _canary(family, queue_mode="PM4"):
  return {
    "schema": RUNTIME_CANARY_SCHEMA, "status": "PASS",
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "queue_mode": queue_mode, "amd_aql_effective": queue_mode == "AQL",
    "runtime_class": "test.FakeAMDProgram", "queue_class": "test.FakeAMDQueue",
    "runtime_count": 1,
    "target_dispatch_count": 0, "runtime_cache_binding_exact": True,
    "code_ranges_valid": True, "timeline_clean": True,
    "health_before": True, "health_after": True, "kernel_faults": [],
    "compile_performed": False, "requires_recompile": False,
  }


def _phase(binding, prefix):
  return {
    "schema": PHASE_SCHEMA,
    "preparation": {
      "phase": "producer_and_output_initialization", "status": "PASS",
      "target_dispatch_count": 0, "synchronize_returned": True,
      "target_allowed_only_after_synchronize": True,
    },
    "epochs": [{
      "epoch": epoch, "program_key": binding.program_key,
      "stage_completion_returned": True,
      "target_submitted_after_stage_completion": True,
      "target_dispatch_returned": True, "target_synchronize_returned": True,
      "overwrite_allowed_only_after_target_completion": True,
      "prior_target_completion_epoch": None if epoch == 0 else epoch - 1,
    } for epoch in range(prefix)],
  }


def _probe_result(family, frozen_bundle, prefix, queue_mode="PM4"):
  manifest, binding = family.manifest, family.binding
  abi = manifest["program"]["abi"]
  vas = [0x100000 + slot * 0x100000 for slot in range(5)]
  epoch_stage = [{
    "epoch": epoch, "source_q4_va": 0x800000 + epoch * 0x1000,
    "source_values_va": 0x900000 + epoch * 0x1000,
    "stage_q4_va": vas[1], "stage_values_va": vas[2],
  } for epoch in range(prefix)]
  metadata_stage = [{
    "epoch": epoch, "source_scales_va": 0xA00000 + epoch * 0x1000,
    "source_sums_va": 0xB00000 + epoch * 0x1000,
    "stage_scales_va": vas[3], "stage_sums_va": vas[4],
  } for epoch in range(prefix)]
  launches = []
  for epoch in range(prefix):
    arguments = [{
      "call_index": slot, "slot": slot, "name": row["name"], "va": vas[slot],
      "base_va": vas[slot], "offset_bytes": 0, "nbytes": row["nbytes"],
      "base_nbytes": row["nbytes"], "va_matches_base_offset": True,
    } for slot, row in enumerate(abi)]
    launches.append({
      "epoch": epoch, "n0": 0, "n1": binding.role_spec.n,
      "tile_count": binding.role_spec.program.grid[0],
      "global_size": manifest["program"]["grid"],
      "local_size": manifest["program"]["local_size"],
      "arguments": arguments,
      "kernarg": {
        "va": 0xC00000 + epoch * 0x1000, "size": 40,
        "bound_pointer_words": vas, "pointer_words": vas,
        "pointer_words_match_bound": True,
      },
    })
  return {
    "schema": "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1",
    "status": "PASS", "exact_blocker": None,
    "role": binding.role_spec.role, "shape": list(binding.role_spec.shape),
    "accumulation": "target_in_place_fp32_add",
    "compile_performed": False, "requires_recompile": False,
    "no_fallback": True, "health_before": True, "health_after": True,
    "mode_health_before": True, "mode_health_after": True,
    "child_env_overrides": {"AMD_AQL": "1" if queue_mode == "AQL" else "0"},
    "kernel_faults": [],
    "correctness": {
      "status": "PASS", "comparison": {
        "status": "pass", "mismatch_count": 0,
        "got_size": binding.role_spec.m * binding.role_spec.n,
        "reference_size": binding.role_spec.m * binding.role_spec.n,
        "nan_got": 0, "nan_reference": 0, "inf_got": 0, "inf_reference": 0,
        "joint_finite": binding.role_spec.m * binding.role_spec.n,
      },
      "authority": "same_session_fp16_rounded_ds4_reference",
    },
    "artifacts": {
      "source_sha256": binding.source_sha256,
      "binary_sha256": binding.binary_sha256,
      "frozen_bundle": {
        "path": str(Path(frozen_bundle).resolve()),
        "program_key": binding.program_key,
        "execution_role": binding.role_spec.role,
        "program_grid": manifest["program"]["grid"],
      },
    },
    "epoch_staging": {
      "mode": "all_inputs_fixed_va_gpu_sdma", "fixed_va": True,
      "transfer": "gpu_sdma", "per_epoch_vas": epoch_stage,
    },
    "metadata_staging": {
      "mode": "fixed_va_gpu_sdma", "fixed_va": True,
      "transfer": "gpu_sdma", "per_epoch_vas": metadata_stage,
    },
    "runtime_evidence": {
      "queue_mode": queue_mode, "amd_aql_effective": queue_mode == "AQL",
      "binary_sha256": binding.binary_sha256,
      "launch_count": prefix, "launches": launches,
    },
    "phase_isolation": _phase(binding, prefix),
  }


def _run_kwargs(role_spec, manifest_path, family, loader, runner, *, prefix=1,
                queue_mode="PM4", canary=None):
  return run_frozen_staged_family_prefix_probe(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, prefix_epochs=prefix,
    queue_mode=queue_mode, runtime_canary=_canary(family, queue_mode) if canary is None else canary,
    family_loader=loader, probe_runner=runner)


def test_staged_prefix_uses_exact_existing_guarded_runner_contract(tmp_path: Path):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  calls = []

  def runner(**kwargs):
    calls.append(kwargs)
    return _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])

  result = _run_kwargs(role_spec, manifest_path, family, loader, runner)
  assert result["schema"] == SCHEMA and result["status"] == "PASS" and result["gate"] == "C5"
  assert result["allowed_prefix_ladder"] == [1, 3, role_spec.epochs]
  assert result["family_identity"] == family.family_identity
  assert result["program_key"] == family.binding.program_key
  assert result["validation"]["phase_isolation"]["all_checks_pass"] is True
  assert result["validation"]["stage"]["all_stage_vas_fixed"] is True
  assert result["validation"]["stage_launch_crosscheck"][
    "all_stage_destinations_match_launch_kernargs"] is True
  assert result["validation"]["stage_launch_crosscheck"]["epochs"] == [{
    "epoch": 0, "stage_destination_vas_slots_1_4": [
      0x200000, 0x300000, 0x400000, 0x500000],
    "launch_kernarg_vas_slots_1_4": [
      0x200000, 0x300000, 0x400000, 0x500000],
    "all_slots_match": True,
  }]
  assert result["validation"]["runtime"]["launch_count"] == 1
  assert result["integration_capability"]["runner_mode"] == "injected"
  assert result["integration_capability"]["default_live_integration_qualified"] is False
  assert result["integration_capability"]["injected_runner_contract_validation_only"] is True
  assert result["integration_capability"]["live_gpu_execution_eligible"] is False
  assert len(calls) == 1
  assert calls[0] == {
    "timeout_seconds": 900.0, "role_spec": role_spec,
    "warmups": 0, "rounds": 1, "epoch_limit": 1,
    "n_chunk_tiles": role_spec.program.grid[0], "epoch_start": 0,
    "host_accumulate": False, "in_kernel_accumulate": True,
    "per_epoch_check": False, "persistent_buffers": True,
    "preloaded_epochs": True, "sync_each_epoch": True,
    "stable_metadata_staging": True, "stable_epoch_staging": True,
    "wait_each_dispatch": True, "frozen_bundle": "/frozen/bundle",
    "child_env_overrides": {"AMD_AQL": "0"},
  }


def test_staged_prefix_does_not_infer_missing_phase_isolation(tmp_path: Path):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)

  def runner(**kwargs):
    result = _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])
    result.pop("phase_isolation")
    return result

  result = _run_kwargs(role_spec, manifest_path, family, loader, runner)
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "guarded staged prefix evidence failed closed"
  assert "phase-isolation evidence schema" in result["error"]


@pytest.mark.parametrize("mutation, message", [
  ("stage_va", "destination VA changed"),
  ("stage_launch_link", "stage destination VAs do not match"),
  ("kernarg", "kernarg pointer census"),
  ("queue", "effective native queue mode"),
  ("program", "bundle_program_key_exact"),
  ("authority", "numeric_authority_exact"),
])
def test_staged_prefix_rejects_va_kernarg_queue_or_program_drift(
    tmp_path: Path, mutation: str, message: str):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)

  def runner(**kwargs):
    result = _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])
    if mutation == "stage_va":
      result["epoch_staging"]["per_epoch_vas"][-1]["stage_q4_va"] += 4
    elif mutation == "stage_launch_link":
      for row in result["epoch_staging"]["per_epoch_vas"]:
        row["stage_q4_va"] += 4
    elif mutation == "kernarg":
      result["runtime_evidence"]["launches"][0]["kernarg"]["pointer_words"][0] += 4
    elif mutation == "queue":
      result["runtime_evidence"]["queue_mode"] = "AQL"
      result["runtime_evidence"]["amd_aql_effective"] = True
    elif mutation == "program":
      result["artifacts"]["frozen_bundle"]["program_key"] = "0" * 64
    elif mutation == "authority":
      result["correctness"]["authority"] = "arbitrary_reference"
    return result

  prefix = 3 if mutation == "stage_va" else 1
  result = _run_kwargs(
    role_spec, manifest_path, family, loader, runner, prefix=prefix)
  assert result["status"] == "BLOCKED"
  assert message in result["error"]


def test_default_live_integration_blocks_before_gpu_without_phase_receipt_capability(
    tmp_path: Path, monkeypatch):
  from extra.qk import mmq_llama_five_buffer_gpu_harness as harness
  monkeypatch.delattr(harness, "FROZEN_STAGED_PHASE_RECEIPT_SCHEMA", raising=False)
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  result = run_frozen_staged_family_prefix_probe(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, prefix_epochs=1,
    queue_mode="PM4", runtime_canary=_canary(family),
    family_loader=loader)
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == \
    "default staged live integration lacks required phase-isolation receipts"
  capability = result["integration_capability"]
  assert capability["schema"] == INTEGRATION_CAPABILITY_SCHEMA
  assert capability["runner_mode"] == "default_live_harness"
  assert capability["advertised_phase_receipt_schema"] is None
  assert capability["default_live_integration_qualified"] is False
  assert capability["live_gpu_execution_eligible"] is False
  assert result["target_dispatch_attempted"] is False


def test_default_live_integration_becomes_eligible_with_exact_advertised_receipts(
    tmp_path: Path, monkeypatch):
  from extra.qk import mmq_llama_five_buffer_gpu_harness as harness
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  calls = []

  def runner(**kwargs):
    calls.append(kwargs)
    return _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])

  monkeypatch.setattr(harness, "run_full_grid_target_role_probe_isolated", runner)
  result = run_frozen_staged_family_prefix_probe(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, prefix_epochs=1,
    queue_mode="PM4", runtime_canary=_canary(family),
    family_loader=loader)
  assert result["status"] == "PASS"
  capability = result["integration_capability"]
  assert capability["advertised_phase_receipt_schema"] == PHASE_SCHEMA
  assert capability["default_live_integration_qualified"] is True
  assert capability["live_gpu_execution_eligible"] is True
  assert capability["injected_runner_contract_validation_only"] is False
  assert len(calls) == 1


@pytest.mark.parametrize("boundary", [
  ("preparation", "synchronize_returned"),
  ("epoch", "stage_completion_returned"),
  ("epoch", "target_dispatch_returned"),
  ("epoch", "target_synchronize_returned"),
  ("epoch", "overwrite_allowed_only_after_target_completion"),
])
def test_default_live_integration_blocks_malformed_phase_boundary(
    tmp_path: Path, monkeypatch, boundary):
  from extra.qk import mmq_llama_five_buffer_gpu_harness as harness
  role_spec, _, manifest_path, family, loader = _family(tmp_path)

  def runner(**kwargs):
    result = _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])
    section, field = boundary
    if section == "preparation":
      result["phase_isolation"]["preparation"][field] = False
    else:
      result["phase_isolation"]["epochs"][0][field] = False
    return result

  monkeypatch.setattr(harness, "run_full_grid_target_role_probe_isolated", runner)
  result = run_frozen_staged_family_prefix_probe(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, prefix_epochs=1,
    queue_mode="PM4", runtime_canary=_canary(family),
    family_loader=loader)
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "guarded staged prefix evidence failed closed"


def test_c4_canary_is_queue_specific_and_must_dispatch_no_target(tmp_path: Path):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  canary = _canary(family, "PM4")
  canary["target_dispatch_count"] = 1
  called = False

  def runner(**kwargs):
    nonlocal called
    called = True
    return {}

  result = _run_kwargs(
    role_spec, manifest_path, family, loader, runner, canary=canary)
  assert result["status"] == "BLOCKED" and "target_dispatch_count_zero" in result["error"]
  assert called is False
  with pytest.raises(ValueError, match="queue_mode_matches"):
    validate_frozen_staged_runtime_canary(canary | {"target_dispatch_count": 0}, family, queue_mode="AQL")


def test_real_harness_c4_normalizer_uses_no_target_runtime_preconstruction(
    tmp_path: Path, monkeypatch):
  from extra.qk.mmq_llama_five_buffer_gpu_harness import (
    FROZEN_STAGED_RUNTIME_CANARY_SCHEMA,
    run_frozen_staged_runtime_preconstruction_canary,
  )
  role_spec, binding, manifest_path, family, _ = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  calls = []

  def preconstruct(programs, program_keys, identities, *, device):
    calls.append((programs, program_keys, identities, device))
    return {
      "status": "PASS", "count": 1,
      "ordered_program_keys": [binding.program_key],
      "no_compute_dispatch_during_preconstruction": True,
      "runtime_cache_retains_code_allocations": True,
      "runtimes": [{"all_checks_pass": True}],
      "all_checks_pass": True,
    }

  health = iter((True, True))
  result = run_frozen_staged_runtime_preconstruction_canary(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    family_loader=lambda *args, **kwargs: family,
    runtime_preconstructor=preconstruct,
    runtime_attestor=lambda program, evidence: {
      "runtime": object(), "amd_aql_effective": False,
      "queue_class": "test.FakePM4Queue",
      "runtime_class": "test.FakeAMDProgram",
      "runtime_cache_binding_exact": True,
    },
    health_probe=lambda: next(health),
    fault_collector=lambda started: ([], {"since": started}))
  assert result["schema"] == FROZEN_STAGED_RUNTIME_CANARY_SCHEMA == RUNTIME_CANARY_SCHEMA
  assert result["status"] == "PASS"
  assert result["target_dispatch_count"] == 0 and result["runtime_count"] == 1
  assert result["compile_performed"] is result["requires_recompile"] is False
  assert len(calls) == 1 and calls[0][1] == (binding.program_key,)
  assert calls[0][3] == "AMD"
  validated = validate_frozen_staged_runtime_canary(result, family, queue_mode="PM4")
  assert validated["all_checks_pass"] is True


def test_real_harness_c4_normalizer_blocks_incomplete_runtime_boundary(
    tmp_path: Path, monkeypatch):
  from extra.qk.mmq_llama_five_buffer_gpu_harness import \
    run_frozen_staged_runtime_preconstruction_canary
  role_spec, binding, manifest_path, family, _ = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  health_calls = []
  result = run_frozen_staged_runtime_preconstruction_canary(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    family_loader=lambda *args, **kwargs: family,
    runtime_preconstructor=lambda *args, **kwargs: {
      "status": "PASS", "count": 1,
      "ordered_program_keys": [binding.program_key],
      "no_compute_dispatch_during_preconstruction": True,
      "runtime_cache_retains_code_allocations": True,
      "runtimes": [], "all_checks_pass": True,
    },
    runtime_attestor=lambda program, evidence: {
      "runtime": object(), "amd_aql_effective": False,
      "queue_class": "test.FakePM4Queue",
      "runtime_class": "test.FakeAMDProgram",
      "runtime_cache_binding_exact": True,
    },
    health_probe=lambda: health_calls.append(True) or True,
    fault_collector=lambda started: ([], {}))
  assert result["status"] == "BLOCKED"
  assert result["target_dispatch_count"] == 0
  assert "preconstruction checks failed" in result["error"]
  assert len(health_calls) == 2


def test_aql_qualification_is_independent_and_effective_mode_is_attested(tmp_path: Path):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  calls = []

  def runner(**kwargs):
    calls.append(kwargs)
    return _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"], "AQL")

  result = _run_kwargs(
    role_spec, manifest_path, family, loader, runner, queue_mode="AQL")
  assert result["status"] == "PASS" and result["queue_mode"] == "AQL"
  assert result["queue_qualification_is_separate"] is True
  assert calls[0]["child_env_overrides"] == {"AMD_AQL": "1"}


def test_staged_ladder_runs_one_three_full_and_stops_on_first_failure(tmp_path: Path):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  prefixes = []

  def runner(**kwargs):
    prefixes.append(kwargs["epoch_limit"])
    return _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])

  result = run_frozen_staged_family_ladder(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    runtime_canary=_canary(family), family_loader=loader, probe_runner=runner)
  assert result["schema"] == LADDER_SCHEMA and result["status"] == "PASS"
  assert prefixes == [1, 3, role_spec.epochs]
  assert result["completed_prefixes"] == prefixes
  assert [row["gate"] for row in result["attempts"]] == ["C5", "C5", "C6"]

  prefixes.clear()

  def fails_at_three(**kwargs):
    prefixes.append(kwargs["epoch_limit"])
    result = _probe_result(family, kwargs["frozen_bundle"], kwargs["epoch_limit"])
    if kwargs["epoch_limit"] == 3: result["health_after"] = False
    return result

  blocked = run_frozen_staged_family_ladder(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    runtime_canary=_canary(family), family_loader=loader, probe_runner=fails_at_three)
  assert blocked["status"] == "BLOCKED" and prefixes == [1, 3]
  assert blocked["completed_prefixes"] == [1]


@pytest.mark.parametrize("prefix", [0, 2, 4, 21])
def test_staged_prefix_allowlist_is_exactly_one_three_full(tmp_path: Path, prefix: int):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  result = _run_kwargs(
    role_spec, manifest_path, family, loader,
    lambda **kwargs: pytest.fail("runner must not be called"), prefix=prefix)
  assert result["status"] == "BLOCKED"
  assert "must be one of" in result["error"]
