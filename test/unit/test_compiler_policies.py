import pytest
from extra.qk.compiler_policies import (PipelinePolicy, StoragePolicy, WaitPolicy, ResourcePlan, RegisterPipePlan,
  WaitDependency, amdllvm_wait_dependency, pipeline_policy_for_route, prove_wait_dependency_coverage)


def test_policy_contracts_accept_lds_barrier_and_estimate():
  assert StoragePolicy("lds", buffer_count=2, slot_bytes=20480).buffer_count == 2
  assert WaitPolicy("full_barrier").scope == "workgroup"
  assert ResourcePlan("host_estimate", lds_bytes=40960).vgpr is None


def test_policy_contracts_accept_register_resident_targeted_final():
  assert StoragePolicy("global_register_resident").slot_bytes == 0
  assert WaitPolicy("targeted_vmcnt", scope="per_stage").scope == "per_stage"
  assert ResourcePlan("final_program", vgpr=120, sgpr=32).vgpr == 120


def test_register_pipe_plan_is_two_stage_b128_zero_lds_and_unproven_resources():
  plan = RegisterPipePlan()
  assert (plan.stages, plan.global_load_bytes, plan.storage.slot_bytes) == (2, 16, 0)
  assert plan.wait.kind == "targeted_vmcnt" and plan.resources.stage == "host_estimate"
  assert plan.policy.storage_kind == "global_register_resident" and plan.policy.logical_stage_count == 2

def test_route_policy_factory_keeps_lds_and_register_storage_interchangeable():
  lds = pipeline_policy_for_route("lds", buffer_count=2, slot_bytes=20480)
  reg = pipeline_policy_for_route("pipe")
  assert isinstance(lds, PipelinePolicy) and lds.storage_kind == "lds" and lds.resources.lds_bytes == 40960
  assert reg.storage_kind == "global_register_resident" and reg.resources.lds_bytes == 0
  assert reg.logical_stage_count == 2 and reg.wait.kind == "targeted_vmcnt"

def test_wait_dependency_accepts_full_barrier_and_rejects_targeted_amdllvm():
  full = WaitDependency(WaitPolicy("full_barrier"), "produce", "consume", "A")
  assert amdllvm_wait_dependency(full) is full
  targeted = WaitDependency(WaitPolicy("targeted_vmcnt", scope="per_stage"), "produce", "consume", "A")
  with pytest.raises(ValueError, match="unsupported by pure AMDLLVM"): amdllvm_wait_dependency(targeted)

def test_wait_dependency_coverage_proves_typed_stage_edges():
  policy = pipeline_policy_for_route("pipe")
  deps = (WaitDependency(policy.wait, "load_a", "wmma", "A", 0, 1, "per_stage"),
          WaitDependency(policy.wait, "load_b", "wmma", "B", 0, 1, "per_stage"))
  proof = prove_wait_dependency_coverage(policy, deps, (("A", 0, 1), ("B", 0, 1)))
  assert proof.passed and proof.covered == (("A", 0, 1), ("B", 0, 1))

def test_wait_dependency_coverage_rejects_duplicate_and_missing_edges():
  policy = pipeline_policy_for_route("pipe")
  dep = WaitDependency(policy.wait, "load_a", "wmma", "A", 0, 1, "per_stage")
  proof = prove_wait_dependency_coverage(policy, (dep, dep), (("A", 0, 1), ("B", 0, 1)))
  assert not proof.passed
  assert any("duplicate wait edge" in error for error in proof.errors)
  assert any("missing wait coverage" in error for error in proof.errors)

def test_wait_dependency_coverage_rejects_unscoped_or_out_of_range_targeted_edges():
  policy = pipeline_policy_for_route("pipe")
  unscoped = WaitDependency(policy.wait, "load_a", "wmma", "A")
  out_of_range = WaitDependency(policy.wait, "load_b", "wmma", "B", 0, 2, "per_stage")
  proof = prove_wait_dependency_coverage(policy, (unscoped, out_of_range))
  assert not proof.passed
  assert any("requires producer and consumer stages" in error for error in proof.errors)
  assert any("outside policy range" in error for error in proof.errors)


@pytest.mark.parametrize("factory", (
  lambda: StoragePolicy("lds"),
  lambda: StoragePolicy("global_register_resident", slot_bytes=16),
  lambda: WaitPolicy("full_barrier", scope="per_stage"),
  lambda: WaitPolicy("targeted_vmcnt", scope="workgroup"),
  lambda: ResourcePlan("host_estimate", vgpr=1),
  lambda: ResourcePlan("final_program"),
  lambda: RegisterPipePlan(stages=1),
  lambda: RegisterPipePlan(global_load_bytes=8),
  lambda: RegisterPipePlan(wait=WaitPolicy("full_barrier")),
  lambda: RegisterPipePlan(resources=ResourcePlan("final_program", vgpr=1, sgpr=1)),
  lambda: pipeline_policy_for_route("pipe", stages=1),
))
def test_policy_contracts_fail_closed(factory):
  with pytest.raises(ValueError): factory()
