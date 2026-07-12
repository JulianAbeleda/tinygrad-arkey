from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan, storage_policy_from_stage1
from extra.qk.compiler_policies import RegisterPipePlan, ResourcePlan, StoragePolicy, WaitPolicy

def test_stage1_plan_maps_to_proved_lds_policy():
  p = storage_policy_from_stage1(KernelStage1PipelinePlan(2, 4096))
  assert p.kind == "lds" and p.buffer_count == 2 and p.slot_bytes == 4096

def test_register_pipe_policy_is_no_lds_and_host_only():
  p = RegisterPipePlan()
  assert p.stages == 2 and p.storage.kind == "global_register_resident"
  assert p.storage.buffer_count == 1 and p.storage.slot_bytes == 0
  assert p.wait.kind == "targeted_vmcnt" and p.wait.scope == "per_stage"
  assert p.resources.stage == "host_estimate" and p.resources.vgpr is None

def test_register_pipe_rejects_lds_or_final_register_claims():
  try:
    RegisterPipePlan(storage=StoragePolicy("lds", 2, 20480))
  except ValueError as exc:
    assert "register pipe storage" in str(exc)
  else: raise AssertionError("register pipe must reject LDS storage")
  try:
    RegisterPipePlan(resources=ResourcePlan("final_program", vgpr=100, sgpr=20))
  except ValueError as exc:
    assert "final resources" in str(exc)
  else: raise AssertionError("host-only register pipe must reject final resource claims")
