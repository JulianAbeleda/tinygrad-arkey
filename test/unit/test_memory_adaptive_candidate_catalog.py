from extra.qk.memory_adaptive_candidate_catalog import (CandidateSpec, build_candidate_catalog, derive_workload_policy_identity,
                                                         inventory_invocation_ids)
from extra.qk.memory_adaptive_evidence_runner import CandidateArtifacts, EvidenceAdapter, make_evidence_runner
from tinygrad.llm.prefill_memory_plan import Strategy


INVENTORY = {"inventory_identity": "inventory:sha256:" + "a" * 64,
             "rows": [{"invocation_id": "q", "role": "attention", "shape": [1, 2, 3]},
                      {"invocation_id": "f", "role": "ffn", "shape": [1, 4, 3]}]}


def test_catalog_contains_only_complete_structurally_available_policies():
  specs = [
    CandidateSpec("overlay", Strategy.FULL_RESIDENT_OVERLAY, ("q", "f"), target_requirements={"backend": "AMD"}),
    CandidateSpec("partial-bounded", Strategy.BOUNDED_PACKED_TILES, ("q",)),
    CandidateSpec("unproved-bounded", Strategy.BOUNDED_PACKED_TILES, ("q", "f"), evidence_available=False),
    CandidateSpec("direct", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f")),
  ]
  catalog = build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={"backend": "AMD"}, candidate_specs=specs)
  assert [x.candidate_id for x in catalog] == ["direct", "overlay"]
  assert all(set(x.memory.required_invocations) == set(x.memory.covered_invocations) for x in catalog)


def test_exact_scanned_architecture_and_geometry_are_structural_constraints():
  specs = [CandidateSpec("wave32", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
    target_requirements={"architecture": "gfx1100", "capabilities": {"wave_size": 32}},
    policy={"profile_id": "evidence-only", "geometry": [128, 128, 256]})]
  assert not build_candidate_catalog(selected_model_inventory=INVENTORY,
    target_capabilities={"architecture": "gfx1200", "capabilities": {"wave_size": 32}}, candidate_specs=specs)
  catalog = build_candidate_catalog(selected_model_inventory=INVENTORY,
    target_capabilities={"architecture": "gfx1100", "capabilities": {"wave_size": 32}}, candidate_specs=specs)
  assert catalog[0].policy_record()["geometry"] == [128, 128, 256]
  assert "profile_id" not in catalog[0].policy_record()


def test_catalog_uses_semantic_inventory_not_provenance_labels():
  a = inventory_invocation_ids({"inventory_identity": "i", "profile": "old", "rows": [{"role": "ffn", "shape": [1, 2, 3]}]})
  b = inventory_invocation_ids({"inventory_identity": "i", "profile": "new", "rows": [{"role": "ffn", "shape": [1, 2, 3]}]})
  assert a == b


def test_catalog_requires_explicit_parent_inventory_identity():
  try:
    build_candidate_catalog(selected_model_inventory={"profile": "not-an-identity", "rows": INVENTORY["rows"]},
      target_capabilities={}, candidate_specs=[])
  except ValueError as exc: assert "inventory_identity" in str(exc)
  else: raise AssertionError("model/profile labels must not replace parent inventory identity")


def test_whole_policy_identity_ignores_alias_and_declaration_order():
  terms = ({"name": "unknown", "bytes": None, "lifetime": "prefill_peak", "formula": "measured",
            "provenance": "allocation proof"},
           {"name": "workspace", "bytes": 32, "lifetime": "candidate_workspace", "formula": "2 * 16",
            "provenance": "route contract"})
  common = dict(strategy=Strategy.BOUNDED_PACKED_TILES, covered_invocations=("q", "f"), memory_terms=terms,
    target_requirements={"caps": {"wave": 32}}, full_m_values=(512, 256), correctness_m_values=(512, 256),
    invocation_bytes=({"m": 512, "activation_bytes": None, "scratch_bytes": 8},
                      {"m": 256, "activation_bytes": 4, "scratch_bytes": None}))
  first = build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={"caps": {"wave": 32}},
    candidate_specs=[CandidateSpec("alias-a", policy={"routes": {"q": "rq", "f": "rf"}, "tile": {"n": 8}}, **common)])[0]
  reordered = {**common, "memory_terms": tuple(reversed(terms)), "full_m_values": (256, 512),
               "correctness_m_values": (256, 512), "invocation_bytes": tuple(reversed(common["invocation_bytes"]))}
  second = build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={"caps": {"wave": 32}},
    candidate_specs=[CandidateSpec("alias-b", policy={"tile": {"n": 8}, "routes": {"f": "rf", "q": "rq"}}, **reordered)])[0]
  assert first.candidate_id != second.candidate_id
  assert first.policy["whole_policy_identity"] == second.policy["whole_policy_identity"]


def test_default_self_routes_ignore_operational_candidate_alias():
  def build(alias):
    return build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[
      CandidateSpec(alias, Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"))])[0]
  first, second = build("alias-a"), build("alias-b")
  assert first.policy["routes"] == {"q": "alias-a", "f": "alias-a"}
  assert second.policy["routes"] == {"q": "alias-b", "f": "alias-b"}
  assert first.policy["whole_policy_identity"] == second.policy["whole_policy_identity"]


def test_workload_identity_binds_choice_and_exact_term_but_not_alias():
  from tinygrad.llm.prefill_memory_plan import ByteLifetime, ByteTerm
  term16 = ByteTerm("prefill peak M=16", 12, "exact bytes", "activation + scratch", ByteLifetime.CANDIDATE_WORKSPACE)
  term32 = ByteTerm("prefill peak M=32", 24, "exact bytes", "activation + scratch", ByteLifetime.CANDIDATE_WORKSPACE)
  choice = {"candidate_id": "alias-a", "full_m": 16, "peak_incremental_bytes": 12, "feasible": True}
  identity = derive_workload_policy_identity(base_whole_policy_identity="whole-policy:sha256:base",
    workload_choice=choice, workload_memory_term=term16)
  assert identity == derive_workload_policy_identity(base_whole_policy_identity="whole-policy:sha256:base",
    workload_choice={**choice, "candidate_id": "alias-b"}, workload_memory_term=term16)
  assert identity != derive_workload_policy_identity(base_whole_policy_identity="whole-policy:sha256:base",
    workload_choice={**choice, "full_m": 32}, workload_memory_term=term16)
  assert identity != derive_workload_policy_identity(base_whole_policy_identity="whole-policy:sha256:base",
    workload_choice=choice, workload_memory_term=term32)


def test_changing_a_non_self_route_mutates_whole_policy_identity():
  def build(route):
    return build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[
      CandidateSpec("alias", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
                    policy={"routes": {"q": "alias", "f": route}})])[0].policy["whole_policy_identity"]
  assert build("external-a") != build("external-b")


def test_whole_policy_identity_mutates_for_each_semantic_slice():
  base = CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
    memory_terms=({"name": "workspace", "bytes": None, "lifetime": "prefill_peak", "formula": "observed",
                  "provenance": "allocator"},), target_requirements={"backend": "AMD"},
    policy={"routes": {"q": "rq", "f": "rf"}, "tile": 8}, full_m_values=(256,), evidence_available=True)
  def identity(spec=base, inventory=INVENTORY):
    return build_candidate_catalog(selected_model_inventory=inventory, target_capabilities={"backend": "AMD"},
                                   candidate_specs=[spec])[0].policy["whole_policy_identity"]
  original = identity()
  mutations = [
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements=base.target_requirements, policy={"routes": {"q": "changed", "f": "rf"}, "tile": 8}, full_m_values=(256,)),
    CandidateSpec("base", Strategy.FULL_RESIDENT_OVERLAY, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements=base.target_requirements, policy=base.policy, full_m_values=(256,)),
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
                  memory_terms=({**base.memory_terms[0], "bytes": 1},), target_requirements=base.target_requirements,
                  policy=base.policy, full_m_values=(256,)),
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements={"backend": "AMD", "wave": 32}, policy=base.policy, full_m_values=(256,)),
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements=base.target_requirements, policy={**base.policy, "tile": 16}, full_m_values=(256,)),
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements=base.target_requirements, policy=base.policy, full_m_values=(512,)),
    CandidateSpec("base", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"), memory_terms=base.memory_terms,
                  target_requirements=base.target_requirements, policy=base.policy, full_m_values=(256,), evidence_available=False),
  ]
  targets = ({"backend": "AMD"}, {"backend": "AMD"}, {"backend": "AMD"}, {"backend": "AMD", "wave": 32},
             {"backend": "AMD"}, {"backend": "AMD"}, {"backend": "AMD"})
  assert all(build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities=target,
    candidate_specs=[mutation])[0].policy["whole_policy_identity"] != original for mutation, target in zip(mutations, targets))
  assert identity(inventory={**INVENTORY, "inventory_identity": "inventory:sha256:" + "b" * 64}) != original


def test_duplicate_semantic_identity_is_rejected_even_with_unique_aliases():
  specs = [CandidateSpec(alias, Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
                         policy={"routes": {"q": "same-q", "f": "same-f"}}) for alias in ("a", "b")]
  try: build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=specs)
  except ValueError as exc: assert "whole_policy_identity" in str(exc)
  else: raise AssertionError("duplicate semantic policies must be rejected")


def test_producer_cannot_supply_whole_policy_identity():
  for spec in ({"candidate_id": "bad", "strategy": Strategy.DIRECT_PACKED_FALLBACK,
                "covered_invocations": ("q", "f"), "whole_policy_identity": "forged"},
               CandidateSpec("bad", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
                             policy={"whole_policy_identity": "forged"})):
    try: build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[spec])
    except ValueError as exc: assert "must not supply" in str(exc)
    else: raise AssertionError("catalog identity must be catalog-owned")


def test_intrinsic_size_fact_is_semantic_not_confused_with_size_label():
  catalog = build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[
    CandidateSpec("direct", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
                  policy={"tile": {"size": 128}, "size_label": "14B"})])
  assert catalog[0].policy["tile"]["size"] == 128
  assert "size_label" not in catalog[0].policy


def test_complete_policy_can_bind_different_routes_per_inventory_row():
  catalog = build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[
    CandidateSpec("mixed-policy", Strategy.BOUNDED_PACKED_TILES, ("q", "f"),
                  policy={"routes": {"q": "q4-bounded", "f": "q6-direct-fallback"}})])
  assert catalog[0].policy["routes"] == {"q": "q4-bounded", "f": "q6-direct-fallback"}


def test_per_row_route_policy_fails_closed_on_partial_or_empty_bindings():
  for routes in ({"q": "q4-bounded"}, {"q": "q4-bounded", "f": ""}):
    try:
      build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={}, candidate_specs=[
        CandidateSpec("bad", Strategy.BOUNDED_PACKED_TILES, ("q", "f"), policy={"routes": routes})])
    except ValueError: pass
    else: raise AssertionError("incomplete per-row route policy must fail closed")


def test_candidate_spec_publishes_exact_kernel_m_facts():
  spec = CandidateSpec("m-facts", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"),
    full_m_values=(256, 512), tail_m_values=(44,), correctness_m_values=(44, 256, 512),
    remainder_mappings=({"logical_m": 32, "physical_m": 512},),
    invocation_bytes=({"m": 44, "activation_bytes": 88, "scratch_bytes": 44},
                      {"m": 256, "activation_bytes": 512, "scratch_bytes": 256},
                      {"m": 512, "activation_bytes": 1024, "scratch_bytes": 512}))
  capability = spec.kernel_capability()
  assert capability.full_m_values == (256, 512)
  assert capability.remainder_mappings[0].physical_m == 512
  assert capability.invocation_bytes[-1].peak_bytes == 1536


def _candidate():
  return build_candidate_catalog(selected_model_inventory=INVENTORY, target_capabilities={},
    candidate_specs=[CandidateSpec("direct", Strategy.DIRECT_PACKED_FALLBACK, ("q", "f"))])[0]


def _artifacts():
  phases = {"phases": [
    {"phase": "compile", "status": "passed", "evidence": {"binary_sha256": "abc"}},
    {"phase": "execution", "status": "passed", "evidence": {"dispatch_state": "completed",
      "health": {"preflight": True, "postflight": True, "device_fault": False}}},
    {"phase": "correctness", "status": "passed", "evidence": {"full_output_compared": True,
      "numerical_passed": True, "finite_output": True, "inputs_unchanged": True}},
  ]}
  return CandidateArtifacts(phases, {"passed": True},
    {"status": "PASS", "complete": True, "covered_invocations": ["q", "f"]},
    {"scope": "end_to_end", "metric": "tok_s", "samples": [10.0, 10.1, 9.9]})


def test_adapter_translates_existing_artifacts_to_strict_autoscan_schema():
  proof = EvidenceAdapter().translate(_candidate(), _artifacts())
  assert all(proof[x]["status"] == "PASS" for x in ("compile", "correctness", "resource", "gpu_health", "route_census"))
  assert proof["route_census"]["complete"] is True
  assert proof["end_to_end_timing"]["scope"] == "end_to_end"


def test_adapter_rejects_incomplete_census_and_never_promotes_kernel_timing():
  artifacts = _artifacts()
  bad = CandidateArtifacts(artifacts.execution, artifacts.resource,
    {"status": "PASS", "complete": True, "covered_invocations": ["q"]},
    {"scope": "kernel", "metric": "tok_s", "samples": [1000.0]})
  proof = EvidenceAdapter().translate(_candidate(), bad)
  assert proof["route_census"] == {"status": "FAIL", "artifact": bad.route_census, "complete": False}
  assert proof["end_to_end_timing"]["samples"] == []


def test_runner_is_injected_and_does_not_execute_by_itself():
  calls = []
  runner = make_evidence_runner(lambda candidate: calls.append(candidate.candidate_id) or _artifacts())
  assert runner(_candidate())["correctness"]["status"] == "PASS"
  assert calls == ["direct"]
