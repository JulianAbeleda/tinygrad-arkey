import pytest
from extra.qk.compiler_policies import StoragePolicy, WaitPolicy, ResourcePlan


def test_policy_contracts_accept_lds_barrier_and_estimate():
  assert StoragePolicy("lds", buffer_count=2, slot_bytes=20480).buffer_count == 2
  assert WaitPolicy("full_barrier").scope == "workgroup"
  assert ResourcePlan("host_estimate", lds_bytes=40960).vgpr is None


def test_policy_contracts_accept_register_resident_targeted_final():
  assert StoragePolicy("global_register_resident").slot_bytes == 0
  assert WaitPolicy("targeted_vmcnt", scope="per_stage").scope == "per_stage"
  assert ResourcePlan("final_program", vgpr=120, sgpr=32).vgpr == 120


@pytest.mark.parametrize("factory", (
  lambda: StoragePolicy("lds"),
  lambda: StoragePolicy("global_register_resident", slot_bytes=16),
  lambda: WaitPolicy("full_barrier", scope="per_stage"),
  lambda: WaitPolicy("targeted_vmcnt", scope="workgroup"),
  lambda: ResourcePlan("host_estimate", vgpr=1),
  lambda: ResourcePlan("final_program"),
))
def test_policy_contracts_fail_closed(factory):
  with pytest.raises(ValueError): factory()
