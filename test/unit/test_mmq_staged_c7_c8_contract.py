from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json

import pytest

from extra.qk.mmq_frozen_staged_family import FrozenStagedFamily, load_frozen_staged_family_manifest
from extra.qk.mmq_staged_c7_c8_contract import (
  C7_SCHEMA, C8_SCHEMA, build_staged_c7_memory_ledger, build_staged_c8_timing_contract,
  physical_lifetime_rows, staged_c7_budget_identity, staged_c7_census_identity, staged_logical_memory_requirements,
  staged_c8_randomized_orders, validate_staged_c7_memory_ledger, validate_staged_certification_bundle,
)
from extra.qk.physical_memory_ledger import AllocationLifetime, PhysicalMemoryEvidence
from test.unit.test_mmq_frozen_staged_family import _loader, _produce
from tinygrad.llm.physical_memory_ledger import AllocationOwner


def _identity(payload: dict) -> str:
  data = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
  return "sha256:" + hashlib.sha256(data).hexdigest()


def _evidence(payload: dict) -> dict:
  return {**payload, "evidence_identity": _identity(payload)}


@pytest.fixture
def family(tmp_path) -> FrozenStagedFamily:
  role_spec, binding, output, _ = _produce(tmp_path)
  return load_frozen_staged_family_manifest(
    output, role_spec=role_spec, frozen_bundle="/frozen/bundle", binding_loader=_loader(binding))


def _authority(*, budget: int = 10**9, provenance: str = "live device admission scan") -> dict:
  authority = {
    "device_identity": "gfx1100-device-0", "software_identity": "tinygrad-revision",
    "allocator_identity": "amd-allocator-instance-0", "allocation_granularity_bytes": 65536,
  }
  return {**authority, "budget_identity": staged_c7_budget_identity(
    **authority, admitted_budget_bytes=budget, budget_provenance=provenance)}


def _aligned(nbytes: int, granularity: int = 65536) -> int:
  return (nbytes + granularity - 1) // granularity * granularity


def _memory_rows(family: FrozenStagedFamily, *, queue_state_bytes: int = 4096,
                 temporary_transfer_bytes: int = 0) -> list[dict]:
  requirements = staged_logical_memory_requirements(family)["components"]
  source_identity = "sha256:" + "e"*64
  rows = []
  for index, (category, nbytes) in enumerate(requirements.items()):
    rows.append({
      "allocation_id": f"logical:{index}", "physical_base_identity": f"AMD:base:{index}",
      "category": category, "requested_bytes": nbytes, "physical_bytes": _aligned(nbytes),
      "live_from": 0, "live_until": 100, "provenance": f"runtime_allocation_census:{category}",
      "source": "runtime_allocation_census", "source_evidence_identity": source_identity,
    })
  for index, (category, nbytes) in enumerate((
      ("code_object", 64*1024), ("runtime", 32*1024),
      ("kernarg", 4096), ("queue_state", queue_state_bytes)), start=100):
    rows.append({
      "allocation_id": f"infra:{category}", "physical_base_identity": f"AMD:base:{index}",
      "category": category, "requested_bytes": nbytes, "physical_bytes": _aligned(nbytes),
      "live_from": 0, "live_until": 100, "provenance": f"runtime_allocation_census:{category}",
      "source": "runtime_allocation_census", "source_evidence_identity": source_identity,
    })
  rows.append({
    "allocation_id": "temp:gather", "physical_base_identity": None, "category": "temporary_gather",
    "requested_bytes": 0, "physical_bytes": 0, "live_from": 0, "live_until": 100,
    "provenance": "explicit_zero_measurement:temporary_gather",
    "source": "explicit_zero_measurement", "source_evidence_identity": source_identity,
  })
  rows.append({
    "allocation_id": "temp:transfer",
    "physical_base_identity": None if temporary_transfer_bytes == 0 else "AMD:base:transfer",
    "category": "temporary_transfer", "requested_bytes": temporary_transfer_bytes,
    "physical_bytes": 0 if temporary_transfer_bytes == 0 else _aligned(temporary_transfer_bytes),
    "live_from": 40, "live_until": 60, "provenance": "explicit_transfer_census",
    "source": "explicit_zero_measurement" if temporary_transfer_bytes == 0 else "runtime_allocation_census",
    "source_evidence_identity": source_identity,
  })
  return rows


def _queue_memory(rows: list[dict], *, authority: dict | None = None) -> dict:
  authority = _authority() if authority is None else authority
  return {
    "route_start": 0, "route_end": 100,
    "allocation_census_identity": staged_c7_census_identity(
      authority=authority, route_start=0, route_end=100, lifetimes=rows),
    "allocation_census_complete": True, "dense_fp16_weight_materialization": False,
    "authority": authority, "lifetimes": rows,
  }


def _c7(family: FrozenStagedFamily, *, budget: int = 10**9) -> dict:
  rows = _memory_rows(family)
  authority = _authority(budget=budget)
  return build_staged_c7_memory_ledger(
    family=family, queue_observations={
      "PM4": _queue_memory(rows, authority=authority), "AQL": _queue_memory(rows, authority=authority)},
    admitted_budget_bytes=budget, budget_provenance="live device admission scan",
    memory_authority=authority,
  )


def _c6(family: FrozenStagedFamily) -> dict:
  return _evidence({
    "schema": "tinygrad.mmq_q4k_q8_1.staged_c6_correctness_binding.v1", "status": "PASS",
    "family_identity": family.family_identity,
    "candidate_executable_identity": "sha256:" + "1"*64,
    "candidate_binary_sha256": family.manifest["program"]["binary_sha256"],
    "workload_identity": "attn_qo-m512-n5120-k5120",
    "input_identity": "sha256:" + "2"*64,
    "device_identity": _authority()["device_identity"],
    "software_identity": _authority()["software_identity"],
    "queue_correctness": {"PM4": "sha256:" + "3"*64, "AQL": "sha256:" + "4"*64},
    "queue_comparators": {"PM4": "sha256:" + "9"*64, "AQL": "sha256:" + "a"*64},
  })


def _candidate_round(family: FrozenStagedFamily, *, epoch_ms: float = 1.0) -> dict:
  logical = staged_logical_memory_requirements(family)
  stage_components = {name: nbytes for name, nbytes in logical["components"].items()
                      if name.startswith("compact_")}
  epochs = []
  for ordinal in range(logical["epoch_count"]):
    epochs.append({
      "ordinal": ordinal, "gather_ms": epoch_ms*0.1, "transfer_ms": epoch_ms*0.2,
      "staging_sync_ms": epoch_ms*0.1, "dispatch_ms": epoch_ms*0.5,
      "dispatch_sync_ms": epoch_ms*0.1, "staged_bytes": logical["compact_stage_bytes"],
      "staged_components": stage_components, "copy_count": 4,
    })
  return {
    "output_initialization_ms": 1.0, "epochs": epochs, "final_sync_ms": 1.0,
    "complete_role_ms": 2.0 + logical["epoch_count"]*epoch_ms,
  }


def _fallback(queue: str, c6: dict) -> dict:
  executable = "sha256:" + ("7" if queue == "PM4" else "8")*64
  comparator = c6["queue_comparators"][queue]
  binary = ("b" if queue == "PM4" else "c")*64
  artifact_schema = "tinygrad.direct_packed.frozen_executable.v1"
  artifact_identity = _identity({
    "artifact_schema": artifact_schema, "binary_sha256": binary,
    "executable_identity": executable, "comparator_identity": comparator,
    "queue_mode": queue, "workload_identity": c6["workload_identity"],
  })
  return _evidence({
    "schema": "tinygrad.direct_packed.complete_role_fallback.v1", "status": "PASS",
    "route_id": "direct_packed", "queue_mode": queue,
    "artifact_schema": artifact_schema, "artifact_identity": artifact_identity,
    "binary_sha256": binary, "executable_identity": executable, "comparator_identity": comparator,
    "workload_identity": c6["workload_identity"], "input_identity": c6["input_identity"],
    "device_identity": c6["device_identity"], "software_identity": c6["software_identity"],
    "clock_identity": "clock-policy-0",
  })


def _queue_timing(family: FrozenStagedFamily, queue: str, c6: dict, *,
                  candidate_epoch_ms: float = 1.0, fallback_ms: float = 30.0,
                  pairs: int = 10, warmups: int = 3) -> dict:
  orders = staged_c8_randomized_orders(seed=20260719, round_count=pairs) if pairs >= 10 else \
    [["staged_candidate", "direct_packed"] if index % 2 == 0
     else ["direct_packed", "staged_candidate"] for index in range(pairs)]
  paired = [{
    "pair_index": index, "order": order,
    "candidate": _candidate_round(family, epoch_ms=candidate_epoch_ms),
    "fallback": {"complete_role_ms": fallback_ms},
  } for index, order in enumerate(orders)]
  seed = 20260719
  return {
    "family_identity": family.family_identity,
    "candidate_executable_identity": c6["candidate_executable_identity"],
    "candidate_c6_queue_evidence_identity": c6["queue_correctness"][queue],
    "candidate_comparator_identity": c6["queue_comparators"][queue],
    "fallback_evidence": _fallback(queue, c6), "session_identity": f"{queue}-session-0",
    "workload_identity": c6["workload_identity"], "input_identity": c6["input_identity"],
    "device_identity": c6["device_identity"], "software_identity": c6["software_identity"],
    "clock_identity": "clock-policy-0", "measurement_source": "synchronized_wall",
    "candidate_warmups": warmups, "fallback_warmups": warmups, "randomization_seed": seed,
    "randomization_algorithm": "python_random_v1_balanced_pair_order",
    "randomized_order_identity": _identity({"seed": seed, "orders": orders}),
    "paired_rounds": paired,
  }


def _c8(family: FrozenStagedFamily, *, c7: dict | None = None, c6: dict | None = None,
        pm4: dict | None = None, aql: dict | None = None) -> dict:
  c7 = _c7(family) if c7 is None else c7
  c6 = _c6(family) if c6 is None else c6
  return build_staged_c8_timing_contract(
    family=family, c7_memory_ledger=c7, c6_correctness_evidence=c6,
    queue_observations={
      "PM4": _queue_timing(family, "PM4", c6) if pm4 is None else pm4,
      "AQL": _queue_timing(family, "AQL", c6) if aql is None else aql,
    }, required_speedup=1.05,
  )


def test_family_admission_recomputes_full_manifest_abi_and_content_identity(family):
  logical = staged_logical_memory_requirements(family)
  assert logical["compact_stage_bytes"] == 901_120
  assert logical["full_source_bytes"] == 18_022_400
  assert logical["output_bytes"] == 10_485_760

  tampered = copy.deepcopy(family.manifest)
  tampered["staging"]["inputs"][0]["source"]["elements"] += 1
  with pytest.raises(ValueError, match="content differs"):
    staged_logical_memory_requirements(FrozenStagedFamily(tampered, family.binding, family.family_identity))
  tampered = copy.deepcopy(family.manifest)
  tampered["program"]["abi"][1]["nbytes"] += 4
  with pytest.raises(ValueError, match="content differs|ABI"):
    staged_logical_memory_requirements(FrozenStagedFamily(tampered, family.binding, family.family_identity))


def test_c7_exact_per_queue_peak_authority_and_deep_validation(family):
  pm4 = _memory_rows(family, temporary_transfer_bytes=8192)
  aql = _memory_rows(family)
  peak = max(sum(row["physical_bytes"] for row in pm4), sum(row["physical_bytes"] for row in aql))
  authority = _authority(budget=peak, provenance="live scan")
  report = build_staged_c7_memory_ledger(
    family=family, queue_observations={
      "PM4": _queue_memory(pm4, authority=authority), "AQL": _queue_memory(aql, authority=authority)},
    admitted_budget_bytes=peak, budget_provenance="live scan", memory_authority=authority)
  assert report["schema"] == C7_SCHEMA and report["status"] == "PASS"
  assert report["queues"]["PM4"]["peak_physical_bytes"] == sum(row["physical_bytes"] for row in pm4)
  assert validate_staged_c7_memory_ledger(report, family=family) == report

  for mutate in (
      lambda row: row["logical_requirements"]["components"].__setitem__("compact_q4_stage", 1),
      lambda row: row.__setitem__("epoch_count", 1),
      lambda row: row["queues"]["PM4"].__setitem__("admitted", False),
      lambda row: row["queues"]["PM4"].__setitem__("peak_physical_bytes", 1)):
    tampered = copy.deepcopy(report)
    mutate(tampered)
    payload = {key: value for key, value in tampered.items() if key != "evidence_identity"}
    tampered["evidence_identity"] = _identity(payload)
    with pytest.raises(ValueError, match="recomputed"):
      validate_staged_c7_memory_ledger(tampered, family=family)
    with pytest.raises(ValueError, match="recomputed"):
      _c8(family, c7=tampered)


def test_c7_rejects_unbound_authority_alignment_duplicate_alias_and_raw_provenance(family):
  rows = _memory_rows(family)
  correct = _authority(provenance="scan")
  wrong_authority = {**correct, "allocator_identity": "different"}
  observation = _queue_memory(rows, authority=wrong_authority)
  with pytest.raises(ValueError, match="authority differs"):
    build_staged_c7_memory_ledger(
      family=family, queue_observations={"PM4": observation, "AQL": observation},
      admitted_budget_bytes=10**9, budget_provenance="scan", memory_authority=correct)

  unaligned = copy.deepcopy(rows)
  unaligned[0]["physical_bytes"] += 1
  observation = _queue_memory(unaligned, authority=correct)
  with pytest.raises(ValueError, match="aligned"):
    build_staged_c7_memory_ledger(
      family=family, queue_observations={"PM4": observation, "AQL": observation},
      admitted_budget_bytes=10**9, budget_provenance="scan", memory_authority=correct)

  aliased = copy.deepcopy(rows)
  aliased[1]["physical_base_identity"] = aliased[0]["physical_base_identity"]
  observation = _queue_memory(aliased, authority=correct)
  with pytest.raises(ValueError, match="duplicate/alias"):
    build_staged_c7_memory_ledger(
      family=family, queue_observations={"PM4": observation, "AQL": observation},
      admitted_budget_bytes=10**9, budget_provenance="scan", memory_authority=correct)

  invented = copy.deepcopy(rows)
  invented[0]["source"] = "invented_string"
  with pytest.raises(ValueError, match="not an admitted census authority"):
    _queue_memory(invented)


def test_physical_ledger_adapter_retains_base_and_evidence_identity(family):
  lifetime = AllocationLifetime(
    allocation_id=3, physical_base_id=2, device="AMD", alloc_sequence=2, free_sequence=8,
    requested_nbytes=4096, physical_nbytes=65536,
    owner=AllocationOwner("candidate_workspace", "candidate", semantic_owner_id="qo:q4_stage"),
    mapped=False,
  )
  evidence = PhysicalMemoryEvidence(
    "tinygrad.physical_memory_ledger.v1", True, (), (lifetime,), 65536, (("AMD", 65536),))
  rows = physical_lifetime_rows(
    evidence, category_by_semantic_owner={"qo:q4_stage": "compact_q4_stage"})
  assert rows[0]["physical_base_identity"] == "AMD:base:2"
  assert rows[0]["source_evidence_identity"] == _identity(evidence.to_json())
  with pytest.raises(ValueError, match="no explicit C7 semantic category"):
    physical_lifetime_rows(evidence, category_by_semantic_owner={})


def test_c8_binds_c6_candidate_and_exact_fallback_execution_identities(family):
  report = _c8(family)
  assert report["schema"] == C8_SCHEMA and report["decision"]["status"] == "CERTIFIED_WIN"
  assert report["queues"]["PM4"]["round_count"] == 10
  assert report["queues"]["PM4"]["warmups"] == {"staged_candidate": 3, "direct_packed": 3}
  assert validate_staged_certification_bundle(report) == report
  c6 = _c6(family)
  losing_aql = _queue_timing(family, "AQL", c6, fallback_ms=21.0)
  fallback = _c8(family, c6=c6, aql=losing_aql)
  assert fallback["decision"]["status"] == "CERTIFIED_FALLBACK"
  assert fallback["decision"]["selected_route"] == "direct_packed"

  pm4 = _queue_timing(family, "PM4", c6)
  pm4["candidate_executable_identity"] = "sha256:" + "f"*64
  with pytest.raises(ValueError, match="differs from C6"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6)
  fallback = copy.deepcopy(pm4["fallback_evidence"])
  fallback["artifact_identity"] = "sha256:" + "f"*64
  pm4["fallback_evidence"] = fallback
  with pytest.raises(ValueError, match="content identity differs"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6)
  fallback = copy.deepcopy(pm4["fallback_evidence"])
  fallback["binary_sha256"] = "f"*64
  payload = {key: value for key, value in fallback.items() if key != "evidence_identity"}
  fallback["evidence_identity"] = _identity(payload)
  pm4["fallback_evidence"] = fallback
  with pytest.raises(ValueError, match="artifact identity differs"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6)
  fallback = copy.deepcopy(pm4["fallback_evidence"])
  fallback["comparator_identity"] = "sha256:" + "f"*64
  fallback["artifact_identity"] = _identity({
    "artifact_schema": fallback["artifact_schema"], "binary_sha256": fallback["binary_sha256"],
    "executable_identity": fallback["executable_identity"],
    "comparator_identity": fallback["comparator_identity"], "queue_mode": "PM4",
    "workload_identity": fallback["workload_identity"],
  })
  payload = {key: value for key, value in fallback.items() if key != "evidence_identity"}
  fallback["evidence_identity"] = _identity(payload)
  pm4["fallback_evidence"] = fallback
  with pytest.raises(ValueError, match="comparator identity differs from candidate/C6"):
    _c8(family, c6=c6, pm4=pm4)

  c6_bad = copy.deepcopy(c6)
  c6_bad["candidate_binary_sha256"] = "0"*64
  payload = {key: value for key, value in c6_bad.items() if key != "evidence_identity"}
  c6_bad["evidence_identity"] = _identity(payload)
  with pytest.raises(ValueError, match="binary differs"):
    _c8(family, c6=c6_bad)


def test_c8_requires_three_equal_warmups_and_ten_randomized_paired_rounds(family):
  c6 = _c6(family)
  pm4 = _queue_timing(family, "PM4", c6, warmups=2)
  with pytest.raises(ValueError, match="at least three"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6, pairs=9)
  with pytest.raises(ValueError, match="at least ten"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6)
  pm4["paired_rounds"][0]["order"] = list(reversed(pm4["paired_rounds"][0]["order"]))
  orders = [pair["order"] for pair in pm4["paired_rounds"]]
  pm4["randomized_order_identity"] = _identity({"seed": pm4["randomization_seed"], "orders": orders})
  with pytest.raises(ValueError, match="seeded balanced randomization"):
    _c8(family, c6=c6, pm4=pm4)

  pm4 = _queue_timing(family, "PM4", c6)
  pm4["randomized_order_identity"] = "sha256:" + "0"*64
  with pytest.raises(ValueError, match="order content identity differs"):
    _c8(family, c6=c6, pm4=pm4)
