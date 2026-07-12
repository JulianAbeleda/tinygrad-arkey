import json, os, subprocess, sys

def test_dynamic_64_candidate_compile_only_proof_and_negative_mutations():
  script=r'''
import json,os
from test.unit.test_runtime_specs import _single_buffer_anchor_candidate,_strict_full_kernel_candidate
p=_single_buffer_anchor_candidate().full_kernel_candidate
p["schedule"]["tile"]={"m":64,"n":64,"k":32};p["schedule"]["waves"]={"m":2,"n":2};p["schedule"]["threads"]=128
p["schedule"]["lds"]["windows"]={"a":[0,5120],"b":[5120,10240]};p["schedule"]["lds"]["strides"]={"a":80,"b":80}
c=_strict_full_kernel_candidate(full_kernel_candidate=p)
os.environ.update(BOLTBEAM_FULL_KERNEL_CANDIDATE_JSON=json.dumps(p,separators=(",",":")),BOLTBEAM_FULL_KERNEL_CANDIDATE_HASH=c.canonical_identity,
  PREFILL_GRAPH_GEMM="1",PREFILL_WMMA_LDS_PRIMITIVE="1",PREFILL_DBUF="0")
from tinygrad import Tensor,dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
from extra.qk.prefill_graph_gemm_route import route_pf16_graph_gemm
from extra.qk.prefill.single_buffer_execution_authority import _compiler_lds_truth,_structural_binding
class L:bias=None;_prefill_graph_role="ffn_gate_up"
u=next(x for x in compile_linear(route_pf16_graph_gemm(L(),Tensor.empty(512,4096,dtype=dtypes.half),
  w=Tensor.empty(12288,4096,dtype=dtypes.half)).schedule_linear()).toposort()
  if x.op is Ops.PROGRAM and getattr(x.src[0].arg,"candidate_context",None))
lds=_compiler_lds_truth(u);good=_structural_binding(p,u,lds)
bad=json.loads(json.dumps(p));bad["schedule"]["lds"]["windows"]={"a":[0,4096],"b":[4096,8192]};bad["schedule"]["lds"]["strides"]={"a":64,"b":64}
wrong=_structural_binding(bad,u,lds)
print(json.dumps({"identity":c.canonical_identity,"local_size":u.arg.local_size,"good":good,"wrong_errors":wrong["errors"]}))
'''
  env={**os.environ,"PYTHONPATH":os.getcwd(),"DEV":"AMD"}
  proc=subprocess.run([sys.executable,"-c",script],text=True,capture_output=True,env=env,check=True)
  row=json.loads(proc.stdout.strip().splitlines()[-1])
  assert row["local_size"]==[32,2,2] and row["good"]["pre_gpu_eligible"] is True
  assert row["good"]["emitted_proof"]["tile"]=={"m":64,"n":64,"k":32}
  assert row["good"]["emitted_proof"]["producer_data_elements"]==4096
  assert "lds_windows: emitted structure differs from payload" in row["wrong_errors"]
  assert "lds_strides: emitted structure differs from payload" in row["wrong_errors"]
