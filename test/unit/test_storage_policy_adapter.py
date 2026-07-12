from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan, storage_policy_from_stage1

def test_stage1_plan_maps_to_proved_lds_policy():
  p = storage_policy_from_stage1(KernelStage1PipelinePlan(2, 4096))
  assert p.kind == "lds" and p.active_bytes == 8192 and p.resource_status == "proved"
