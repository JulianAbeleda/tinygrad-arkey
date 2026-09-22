import json
from types import SimpleNamespace

from extra.llm_research.decode.native_semantic_profile_ledger import semantic_class, load_capture
from extra.llm_research.decode.full_token_dag_capture import _admission_metadata


def test_semantic_class_prefers_exact_tensor_role():
  md={"semantic":[{"tensor_name":"blk.7.attn_v.weight"}]}
  assert semantic_class("q6k_gen_partial_1024_4096_4_deadbeef",md) == "quantized_core/attn_v"
  assert semantic_class("flash_block_x",None) == "non_quantized/flash_score"
  assert semantic_class("r_16_256",None) == "non_quantized/reduction"
  assert semantic_class("E_32_32_4",None) == "non_quantized/elementwise"


def test_capture_requires_exact_groups_and_alignment(tmp_path):
  nodes=[];records=[];index=0
  for size in (32,64,128,256,468):
    entries=[]
    for _ in range(size):
      name=f"E_{index}"
      entries.append({"name":name,"duration":"1.0"});nodes.append({"name":name,"metadata":None});index+=1
    records.append({"entries":entries})
  path=tmp_path/"profile.jsonl";path.write_text("\n".join(json.dumps(x) for x in records)+"\n")
  out=load_capture(path,nodes)
  assert out["raw_total_us"] == 948.0
  assert out["counts"] == {"non_quantized/elementwise":948}


def test_admission_metadata_preserves_program_identity():
  item=SimpleNamespace(name="attn_qo",caller="",backward=False,phase="decode",tensor_name="blk.0.attn_q.weight",
                       module_path="blk.0.attn_q",role="attn_qo",logical_m=1,logical_n=4096,logical_k=4096,
                       source_quant_storage="Q4_K",source_layout="packed",module_representation="adapter",
                       input_dtype="half",output_dtype="float",accumulator_dtype="float")
  out=_admission_metadata([SimpleNamespace(call_index=7,metadata=(item,))])
  assert out[7][0]["tensor_name"] == "blk.0.attn_q.weight"
  assert out[7][0]["logical_n"] == 4096
