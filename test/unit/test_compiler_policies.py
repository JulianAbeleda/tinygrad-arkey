import pytest
from extra.qk.compiler_policies import StoragePolicy, WaitPolicy, ResourcePlan, RegisterPipePlan


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
))
def test_policy_contracts_fail_closed(factory):
  with pytest.raises(ValueError): factory()
