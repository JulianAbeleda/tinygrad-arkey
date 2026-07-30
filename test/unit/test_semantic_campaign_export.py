from extra.llm_research.semantic_campaign_export import export_request
from extra.llm_research.semantic_campaign_export import split_coupled_rows

def test_export_uses_canonical_futuresight_dimension_mapping():
  workload={"shape":{"m":1,"n":64,"k":256},"operands":{"a":{},"b":{},"c":{}}}
  schedule={"plan_kind":"tinygrad_opt_sequence.v1","transforms":[],"launch":{"threads":32},"tile":{"m":1,"n":64,"k":256},"static_constraints":{"max_local_memory_bytes":None}}
  out=export_request(semantic_workload=workload,schedule=schedule,compiler_facts={"compiler_transforms":["UPCAST"],"supported_plan_kinds":["tinygrad_opt_sequence.v1"],"generic_control_plan_kind":None},axis_choices={"schedule.launch.threads":[32]},legal_coupled_rows=[{"schedule.launch.threads":32}],resolved_target={},gguf_path="x",execution={},request_id="r",run_id="r",timestamp="t",budget={})
  assert out["dimensions"] == {"schedule.launch.threads":(32,)} and out["legal_coupled_rows"] == [{"schedule.launch.threads":32}]

def test_coupled_rows_are_split_before_boltbeam_candidate_hashing():
  baseline={"schedule":{"plan_kind":"tinygrad_opt_sequence.v1","transforms":[],"launch":{"threads":32},"tile":{"m":1,"n":2,"k":256},"static_constraints":{"max_local_memory_bytes":None}}}
  legal,rejected=split_coupled_rows([{"schedule.launch.threads":32},{"schedule.launch.threads":128}],{"max_threads_per_threadgroup":64,"compiler_transforms":[],"supported_plan_kinds":["tinygrad_opt_sequence.v1"],"generic_control_plan_kind":None},baseline,{"shape":{"m":1,"n":2,"k":256}})
  assert legal == [{"schedule.launch.threads":32}]
  assert rejected[0]["reason"] == "over_threads" and rejected[0]["row"] == {"schedule.launch.threads":128}
