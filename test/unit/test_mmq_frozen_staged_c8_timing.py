from __future__ import annotations

import copy
import json
from contextlib import contextmanager

import pytest

from extra.qk.mmq_frozen_staged_c8_timing import (
  CANDIDATE_RECEIPT_SCHEMA, FALLBACK_RECEIPT_SCHEMA,
  PHASE_SEMANTICS, QueueTimingAuthority, QueueTimingRunners, SCHEMA,
  collect_staged_c8_timing, collect_staged_c8_timing_from_samples,
  make_direct_packed_fallback_runner, make_frozen_staged_candidate_runner,
)
from extra.qk.mmq_frozen_staged_family import (
  FrozenStagedFamily, load_frozen_staged_family_manifest,
)
from extra.qk.mmq_staged_c7_c8_contract import (
  staged_c8_randomized_orders, staged_logical_memory_requirements,
  validate_staged_certification_bundle,
)
from extra.qk.mmq_llama_five_buffer_gpu_harness import (
  FROZEN_STAGED_C8_TIMING_RECEIPT_SCHEMA,
  _frozen_staged_c8_timing_receipt, _notify_staged_observer,
  _staged_observer_allocation, run_full_grid_target_role_probe,
)
from test.unit.test_mmq_frozen_staged_family import _loader, _produce
from test.unit.test_mmq_staged_c7_c8_contract import _c6, _c7, _fallback


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  return load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle",
    binding_loader=_loader(binding))


def _candidate_receipt(
    family: FrozenStagedFamily, c6: dict, queue: str, *,
    clock: str, elapsed: float = 0.01,
    ) -> dict:
  logical = staged_logical_memory_requirements(family)
  compact = logical["components"]
  transfer_rows = [
    {"slot": 1, "name": "q4", "category": "compact_q4_stage",
     "nbytes": compact["compact_q4_stage"], "elapsed_ms": elapsed},
    {"slot": 2, "name": "q8_values", "category": "compact_q8_values_stage",
     "nbytes": compact["compact_q8_values_stage"], "elapsed_ms": elapsed},
    {"slot": 3, "name": "q8_scales", "category": "compact_q8_scales_stage",
     "nbytes": compact["compact_q8_scales_stage"], "elapsed_ms": elapsed},
    {"slot": 4, "name": "q8_original_sums", "category": "compact_q8_sums_stage",
     "nbytes": compact["compact_q8_sums_stage"], "elapsed_ms": elapsed},
  ]
  epochs = [{
    "ordinal": ordinal, "gather_ms": elapsed, "transfers": copy.deepcopy(transfer_rows),
    "staging_sync_ms": elapsed, "dispatch_ms": elapsed, "dispatch_sync_ms": elapsed,
  } for ordinal in range(logical["epoch_count"])]
  output_initialization_ms = final_sync_ms = elapsed
  per_epoch = elapsed * 8  # gather + four transfers + two syncs + dispatch
  complete_role_ms = output_initialization_ms + logical["epoch_count"] * per_epoch + final_sync_ms
  return {
    "schema": CANDIDATE_RECEIPT_SCHEMA, "status": "PASS",
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "candidate_executable_identity": c6["candidate_executable_identity"],
    "queue_mode": queue, "measurement_source": "synchronized_wall",
    "clock_identity": clock, "phase_semantics": PHASE_SEMANTICS,
    "output_initialization_ms": output_initialization_ms,
    "epochs": epochs, "final_sync_ms": final_sync_ms,
    "complete_role_ms": complete_role_ms,
  }


def _fallback_receipt(c6: dict, queue: str, *, clock: str, elapsed: float = 30.0) -> dict:
  return {
    "schema": FALLBACK_RECEIPT_SCHEMA, "status": "PASS",
    "queue_mode": queue, "measurement_source": "synchronized_wall",
    "clock_identity": clock, "complete_role_ms": elapsed,
    "fallback_evidence": _fallback(queue, c6),
  }


def _runners(
    family: FrozenStagedFamily, c6: dict, queue: str, events: list[tuple], *,
    candidate_mutator=None, fallback_mutator=None,
    ) -> QueueTimingRunners:
  clock = "clock-policy-0"

  def candidate(**kwargs):
    events.append((queue, kwargs["phase"], kwargs["pair_index"], "staged_candidate"))
    row = _candidate_receipt(family, c6, queue, clock=clock)
    if candidate_mutator is not None: candidate_mutator(row)
    return row

  def fallback(**kwargs):
    events.append((queue, kwargs["phase"], kwargs["pair_index"], "direct_packed"))
    row = _fallback_receipt(c6, queue, clock=clock)
    if fallback_mutator is not None: fallback_mutator(row)
    return row

  return QueueTimingRunners(candidate, fallback)


def _collect(family: FrozenStagedFamily, *, candidate_mutator=None, fallback_mutator=None,
             warmups: int = 3, rounds: int = 10, seed: int = 20260719):
  c6, events = _c6(family), []
  result = collect_staged_c8_timing(
    family=family, c7_memory_ledger=_c7(family), c6_correctness_evidence=c6,
    queue_authorities={
      "PM4": QueueTimingAuthority("PM4-session", "clock-policy-0"),
      "AQL": QueueTimingAuthority("AQL-session", "clock-policy-0"),
    },
    queue_runners={
      "PM4": _runners(
        family, c6, "PM4", events, candidate_mutator=candidate_mutator,
        fallback_mutator=fallback_mutator),
      "AQL": _runners(
        family, c6, "AQL", events, candidate_mutator=candidate_mutator,
        fallback_mutator=fallback_mutator),
    },
    warmups=warmups, rounds=rounds, seed=seed, required_speedup=1.05)
  return result, events


def test_collects_exhaustive_phases_and_seeded_matched_pairs_per_queue(family):
  result, events = _collect(family)
  assert result["schema"] == SCHEMA and result["status"] == "PASS"
  assert result["protocol"] == {
    "warmups_per_route": 3, "paired_rounds": 10, "seed": 20260719,
    "randomization_algorithm": "python_random_v1_balanced_pair_order",
    "measurement_source": "synchronized_wall",
    "queue_modes_measured_separately": True,
  }
  c8 = result["c8_contract"]
  assert validate_staged_certification_bundle(c8) == c8
  assert c8["decision"]["status"] == "CERTIFIED_WIN"
  for queue in ("PM4", "AQL"):
    collected = result["queue_collections"][queue]
    assert collected["invocation_counts"] == {
      "staged_candidate": 13, "direct_packed": 13}
    assert collected["orders"] == staged_c8_randomized_orders(
      seed=20260719, round_count=10)
    epoch = collected["paired_rounds"][0]["candidate"]["epochs"][0]
    assert [row["name"] for row in epoch["transfers"]] == [
      "q4", "q8_values", "q8_scales", "q8_original_sums"]
    assert epoch["transfer_ms"] == pytest.approx(0.04)
    assert collected["all_candidate_epoch_phases_retained"] is True
    queue_events = [row for row in events if row[0] == queue and row[1] == "round"]
    for pair_index, order in enumerate(collected["orders"]):
      observed = [row[3] for row in queue_events if row[2] == pair_index]
      assert observed == order
  assert result["queue_collections"]["PM4"]["session_identity"] != \
    result["queue_collections"]["AQL"]["session_identity"]


def test_fails_closed_on_candidate_identity_transfer_or_phase_sum_drift(family):
  with pytest.raises(ValueError, match="binary_sha256"):
    _collect(family, candidate_mutator=lambda row: row.__setitem__("binary_sha256", "0"*64))

  def remove_transfer(row):
    row["epochs"][0]["transfers"].pop()
    row["complete_role_ms"] -= 0.01
  with pytest.raises(ValueError, match="exactly four"):
    _collect(family, candidate_mutator=remove_transfer)

  with pytest.raises(ValueError, match="exhaustive phase sum"):
    _collect(
      family,
      candidate_mutator=lambda row: row.__setitem__(
        "complete_role_ms", row["complete_role_ms"] + 1.0))


def test_fails_closed_on_protocol_or_queue_session_alias(family):
  c6, events = _c6(family), []
  runners = {
    queue: _runners(family, c6, queue, events) for queue in ("PM4", "AQL")}
  with pytest.raises(ValueError, match="warmups"):
    collect_staged_c8_timing(
      family=family, c7_memory_ledger=_c7(family), c6_correctness_evidence=c6,
      queue_authorities={
        "PM4": QueueTimingAuthority("PM4-session", "clock-policy-0"),
        "AQL": QueueTimingAuthority("AQL-session", "clock-policy-0")},
      queue_runners=runners, warmups=2)
  with pytest.raises(ValueError, match="rounds"):
    collect_staged_c8_timing(
      family=family, c7_memory_ledger=_c7(family), c6_correctness_evidence=c6,
      queue_authorities={
        "PM4": QueueTimingAuthority("PM4-session", "clock-policy-0"),
        "AQL": QueueTimingAuthority("AQL-session", "clock-policy-0")},
      queue_runners=runners, rounds=9)
  with pytest.raises(ValueError, match="sessions must be distinct"):
    collect_staged_c8_timing(
      family=family, c7_memory_ledger=_c7(family), c6_correctness_evidence=c6,
      queue_authorities={
        "PM4": QueueTimingAuthority("same", "clock-policy-0"),
        "AQL": QueueTimingAuthority("same", "clock-policy-0")},
      queue_runners=runners)


def test_replays_persisted_live_receipts_and_rejects_order_drift(family):
  c6, seed, rounds, warmups = _c6(family), 77, 10, 3
  samples = {"warmups": warmups, "rounds": rounds, "seed": seed, "queues": {}}
  for queue in ("PM4", "AQL"):
    clock = "clock-policy-0"
    samples["queues"][queue] = {
      "session_identity": f"{queue}-session", "clock_identity": clock,
      "candidate_warmups": [
        _candidate_receipt(family, c6, queue, clock=clock) for _ in range(warmups)],
      "fallback_warmups": [
        _fallback_receipt(c6, queue, clock=clock) for _ in range(warmups)],
      "paired_rounds": [{
        "order": order,
        "candidate": _candidate_receipt(family, c6, queue, clock=clock),
        "fallback": _fallback_receipt(c6, queue, clock=clock),
      } for order in staged_c8_randomized_orders(seed=seed, round_count=rounds)],
    }
  result = collect_staged_c8_timing_from_samples(
    family=family, c7_memory_ledger=_c7(family),
    c6_correctness_evidence=c6, samples=samples, required_speedup=1.05)
  assert result["status"] == "PASS"
  assert result["c8_contract"]["queues"]["PM4"]["randomization_seed"] == seed

  drift = copy.deepcopy(samples)
  drift["queues"]["AQL"]["paired_rounds"][0]["order"].reverse()
  with pytest.raises(ValueError, match="seeded schedule"):
    collect_staged_c8_timing_from_samples(
      family=family, c7_memory_ledger=_c7(family),
      c6_correctness_evidence=c6, samples=drift)


def test_collection_is_canonical_json_and_contains_no_device_dependency(family):
  result, _ = _collect(family)
  encoded = json.dumps(result, sort_keys=True, allow_nan=False)
  assert json.loads(encoded) == result
  assert result["production_dispatch_changed"] is False


def test_harness_builds_gap_free_four_transfer_receipt():
  transfers = [{
    "slot": slot, "name": name, "category": category,
    "nbytes": 100 + slot, "elapsed_ns": 10 * slot,
  } for slot, name, category in zip(
    range(1, 5), ("q4", "q8_values", "q8_scales", "q8_original_sums"),
    ("compact_q4_stage", "compact_q8_values_stage",
     "compact_q8_scales_stage", "compact_q8_sums_stage"))]
  receipt = _frozen_staged_c8_timing_receipt(
    program_key="program", binary_sha256="a"*64, queue_mode="PM4",
    output_initialization_ns=100,
    epoch_rows=[{
      "ordinal": 0, "gather_ns": 20, "transfers": transfers,
      "staging_sync_ns": 30, "dispatch_ns": 40, "dispatch_sync_ns": 50,
    }],
    final_sync_ns=60)
  assert receipt["schema"] == FROZEN_STAGED_C8_TIMING_RECEIPT_SCHEMA
  assert receipt["gap_free_phase_partition"] is True
  assert receipt["phase_semantics"] == PHASE_SEMANTICS
  assert [row["elapsed_ms"] for row in receipt["epochs"][0]["transfers"]] == [
    0.00001, 0.00002, 0.00003, 0.00004]
  assert receipt["complete_role_ms"] == pytest.approx(0.0004)
  broken = copy.deepcopy(transfers)
  broken.pop()
  with pytest.raises(ValueError, match="four transfer"):
    _frozen_staged_c8_timing_receipt(
      program_key="program", binary_sha256="a"*64, queue_mode="PM4",
      output_initialization_ns=1,
      epoch_rows=[{
        "ordinal": 0, "gather_ns": 1, "transfers": broken,
        "staging_sync_ns": 1, "dispatch_ns": 1, "dispatch_sync_ns": 1,
      }], final_sync_ns=1)


def test_harness_rejects_prefix_timing_or_lifecycle_observation_before_device(family):
  role_spec = family.binding.role_spec
  exact = {
    "role_spec": role_spec, "warmups": 0, "rounds": 1, "epoch_limit": 1,
    "n_chunk_tiles": role_spec.program.grid[0], "epoch_start": 0,
    "host_accumulate": False, "in_kernel_accumulate": True,
    "per_epoch_check": False, "persistent_buffers": True,
    "preloaded_epochs": True, "sync_each_epoch": True,
    "stable_metadata_staging": True, "stable_epoch_staging": True,
    "wait_each_dispatch": True, "frozen_bundle": "/not-loaded",
  }
  with pytest.raises(ValueError, match="full-role"):
    run_full_grid_target_role_probe(**exact, c8_phase_timing=True)
  with pytest.raises(ValueError, match="full-role"):
    run_full_grid_target_role_probe(**exact, staged_lifecycle_observer=object())


def test_neutral_lifecycle_observer_is_default_off_and_preallocation_scoped():
  events = []

  class Observer:
    @contextmanager
    def allocation(self, category, name):
      events.append(("enter", category, name))
      yield
      events.append(("exit", category, name))
    def begin_route(self): events.append(("begin",))
    def end_route(self): events.append(("end",))

  observer = Observer()
  with _staged_observer_allocation(observer, "output", "persistent_partial"):
    events.append(("allocate",))
  _notify_staged_observer(observer, "begin_route")
  _notify_staged_observer(observer, "end_route")
  assert events == [
    ("enter", "output", "persistent_partial"), ("allocate",),
    ("exit", "output", "persistent_partial"), ("begin",), ("end",)]
  with _staged_observer_allocation(None, "output", "ignored"):
    pass
  _notify_staged_observer(None, "begin_route")


def test_direct_packed_live_seam_uses_real_boundary_contract_without_device(monkeypatch, family):
  c6, events = _c6(family), []
  evidence = {queue: _fallback(queue, c6) for queue in ("PM4", "AQL")}
  clock = iter((1_000_000, 3_500_000))

  class Output:
    def realize(self): events.append("realize")

  def execute(linear, value, spec):
    events.append(("execute", linear, value, spec))
    return Output()

  def attest(output, queue):
    events.append(("attest", output.__class__.__name__, queue))
    return evidence[queue]

  runner = make_direct_packed_fallback_runner(
    linear="linear", input_tensor="input", route_spec="spec",
    fallback_evidence_by_queue=evidence, execution_attestor=attest,
    synchronize=lambda: events.append("sync"), executor=execute,
    clock_ns=lambda: next(clock))
  monkeypatch.setenv("AMD_AQL", "0")
  receipt = runner(
    queue_mode="PM4", clock_identity="clock-policy-0",
    family=family, route="direct_packed", phase="round",
    invocation_index=0, pair_index=0, prefix_epochs=20,
    candidate_executable_identity=c6["candidate_executable_identity"])
  assert receipt["complete_role_ms"] == 2.5
  assert receipt["fallback_evidence"] == evidence["PM4"]
  assert events == [
    "sync", ("execute", "linear", "input", "spec"), "realize", "sync",
    ("attest", "Output", "PM4")]
  monkeypatch.setenv("AMD_AQL", "1")
  with pytest.raises(StopIteration):
    # The effective AQL mode is admitted; the exhausted injected clock proves
    # this call reached the same synchronized execution boundary.
    runner(
      queue_mode="AQL", clock_identity="clock-policy-0",
      family=family, route="direct_packed", phase="round",
      invocation_index=0, pair_index=0, prefix_epochs=20,
      candidate_executable_identity=c6["candidate_executable_identity"])


def test_direct_packed_live_seam_rejects_execution_evidence_drift(monkeypatch, family):
  c6 = _c6(family)
  evidence = {queue: _fallback(queue, c6) for queue in ("PM4", "AQL")}
  drifted = dict(evidence["PM4"])
  drifted["binary_sha256"] = "0" * 64
  runner = make_direct_packed_fallback_runner(
    linear="linear", input_tensor="input", route_spec="spec",
    fallback_evidence_by_queue=evidence,
    execution_attestor=lambda _output, _queue: drifted,
    synchronize=lambda: None, executor=lambda *_args: object(),
    realize_output=lambda _output: None, clock_ns=iter((1, 2)).__next__)
  monkeypatch.setenv("AMD_AQL", "0")
  with pytest.raises(ValueError, match="timed direct_packed executable differs"):
    runner(queue_mode="PM4", clock_identity="clock-policy-0")


def test_staged_live_seam_reuses_validated_prefix_executor_and_enriches_receipt(
    monkeypatch, family):
  c6, seen = _c6(family), []
  raw = _frozen_staged_c8_timing_receipt(
    program_key=family.binding.program_key,
    binary_sha256=family.binding.binary_sha256, queue_mode="PM4",
    output_initialization_ns=10,
    epoch_rows=[{
      "ordinal": ordinal, "gather_ns": 1,
      "transfers": [{
        "slot": slot, "name": name, "category": category,
        "nbytes": nbytes, "elapsed_ns": 1,
      } for slot, name, category, nbytes in zip(
        range(1, 5), ("q4", "q8_values", "q8_scales", "q8_original_sums"),
        ("compact_q4_stage", "compact_q8_values_stage",
         "compact_q8_scales_stage", "compact_q8_sums_stage"),
        (737280, 131072, 16384, 16384))],
      "staging_sync_ns": 1, "dispatch_ns": 1, "dispatch_sync_ns": 1,
    } for ordinal in range(family.binding.role_spec.epochs)],
    final_sync_ns=10)

  def probe_runner(**kwargs):
    assert "timeout_seconds" not in kwargs
    assert "child_env_overrides" not in kwargs
    seen.append(("probe", kwargs["c8_phase_timing"]))
    return {"c8_timing_receipt": raw}

  def prefix_executor(**kwargs):
    seen.append(("prefix", kwargs["prefix_epochs"], kwargs["queue_mode"]))
    probe = kwargs["probe_runner"](
      timeout_seconds=123.0,
      child_env_overrides={"AMD_AQL": "1" if kwargs["queue_mode"] == "AQL" else "0"})
    return {
      "status": "PASS", "family_identity": family.family_identity,
      "raw_probe": probe,
    }

  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_family_execution.run_frozen_staged_family_prefix_probe",
    prefix_executor)
  monkeypatch.setenv("AMD_AQL", "0")
  runner = make_frozen_staged_candidate_runner(
    role_spec=family.binding.role_spec, frozen_bundle="/bundle",
    staged_family_manifest="/family",
    runtime_canary_by_queue={"PM4": {"canary": True}, "AQL": {"canary": True}},
    probe_runner=probe_runner)
  receipt = runner(
    queue_mode="PM4", family=family, clock_identity="clock-policy-0",
    candidate_executable_identity=c6["candidate_executable_identity"])
  assert seen == [("prefix", family.binding.role_spec.epochs, "PM4"), ("probe", True)]
  assert receipt["schema"] == CANDIDATE_RECEIPT_SCHEMA
  assert receipt["family_identity"] == family.family_identity
  assert receipt["phase_semantics"] == PHASE_SEMANTICS


def test_staged_live_seam_rejects_fresh_process_per_timing_invocation():
  from extra.qk.mmq_llama_five_buffer_gpu_harness import \
    run_full_grid_target_role_probe_isolated
  with pytest.raises(ValueError, match="cannot spawn one isolated child"):
    make_frozen_staged_candidate_runner(
      role_spec=object(), frozen_bundle="/bundle",
      staged_family_manifest="/family", runtime_canary_by_queue={},
      probe_runner=run_full_grid_target_role_probe_isolated)
