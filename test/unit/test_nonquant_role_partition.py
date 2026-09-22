import json
from extra.llm_research.decode.nonquant_role_partition import _shapley_partition, build, native_roles

def test_shapley_partition_is_disjoint_and_priority_invariant():
  rows=[{"start":0,"end":10_000,"class":"mmq"},{"start":0,"end":20_000,"class":"rope"},
        {"start":5_000,"end":15_000,"class":"rms_norm"}]
  exposed,hidden,bounds=_shapley_partition(rows)
  assert sum(exposed.values()) == 10.0 and sum(hidden.values()) == 10.0
  assert hidden == {"rms_norm":2.5,"rope":7.5}
  assert bounds["rope"]["exposed_lower_us"] == 5.0

def test_native_partition_fails_closed_on_wrong_topology(tmp_path):
  dag=tmp_path/"dag.json";profile=tmp_path/"p.jsonl"
  dag.write_text(json.dumps({"nodes":[]}));profile.write_text("")
  try: native_roles(dag,profile,1.0)
  except ValueError as e: assert "948" in str(e)
  else: raise AssertionError("must fail closed")

def test_checked_in_support_gap_splits_raw_topology_from_hidden_overlap():
  result=json.load(open("docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json"))
  split=result["support_gap_mechanism_split"]
  assert abs(split["fusion_dataflow_and_body_attribution_us"] + split["llama_hidden_overlap_us"] - split["total_nonquantized_delta_us"]) < 0.002
  assert abs(split["total_nonquantized_delta_us"] - 1108.082) < 0.002
  assert abs(split["llama_hidden_overlap_us"] - 445.954) < 0.002
