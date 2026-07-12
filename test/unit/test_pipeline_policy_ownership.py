from extra.qk import compiler_policies as compat
from extra.qk.prefill_schedule_spec import PrefillGEMMScheduleSpec
from extra.qk.wmma_pipe_spec import WMMAPipeIR, WMMAPipeSpec, extract_wmma_pipe_spec
from tinygrad.codegen.opt import compiler_policies as core


def _schedule(route_family: str) -> PrefillGEMMScheduleSpec:
  return PrefillGEMMScheduleSpec(
    m=512, n=4096, k=4096, route_family=route_family, tile_m=128, tile_n=128, tile_k=32,
    waves_m=2, waves_n=2, wm=4, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=128,
    dbuf=1, pad=16, role="attn_qo")


def test_compatibility_policy_module_reexports_core_identity():
  for name in ("StoragePolicy", "WaitPolicy", "ResourcePlan", "PipelinePolicy", "RegisterPipePlan",
               "WaitCount", "WaitDependency", "WaitDependencyCoverage"):
    assert getattr(compat, name) is getattr(core, name)


def test_route_specs_materialize_the_same_typed_policy_contract():
  pipe = _schedule("pipe")
  pipe_policy = pipe.pipeline_policy
  primitive = extract_wmma_pipe_spec(pipe)
  assert primitive is not None
  assert primitive.pipeline_policy == pipe_policy
  assert type(primitive.pipeline_policy.storage) is core.StoragePolicy
  assert type(primitive.pipeline_policy.wait) is core.WaitPolicy
  assert type(primitive.pipeline_policy.resources) is core.ResourcePlan

  lds = _schedule("lds")
  lds_policy = lds.pipeline_policy
  assert isinstance(lds_policy, core.PipelinePolicy)
  assert lds_policy.storage_kind == "lds" and lds_policy.resources.lds_bytes == 40960


def test_pipe_spec_and_ir_share_policy_identity_and_no_lds_claim():
  spec = WMMAPipeSpec(m=512, n=4096, k=4096, tile_m=128, tile_n=128, role="attn_qo")
  ir = WMMAPipeIR("attn_qo", (512, 4096, 4096), 2, spec.loads_per_stage, "targeted_vmcnt")
  assert spec.pipeline_policy == ir.pipeline_policy
  assert type(spec.pipeline_policy.storage) is type(ir.pipeline_policy.storage) is core.StoragePolicy
  assert spec.pipeline_policy.resources.lds_bytes == 0


def test_unknown_route_has_no_policy_identity():
  try:
    core.pipeline_policy_for_route("unknown")
  except ValueError as exc:
    assert "unsupported pipeline route family" in str(exc)
  else:
    raise AssertionError("unknown route must fail before policy construction")
