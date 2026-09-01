#!/usr/bin/env python3
"""Live qualification of the exact BoltBeam Q6 vocab reducer."""
from __future__ import annotations
import argparse,hashlib,json,pathlib,statistics,subprocess
import numpy as np
from tinygrad import Device,Tensor,dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.boltbeam_authority import lower_authorized_candidate
from tinygrad.llm.decode_kernels import q6k_spec_for_role
from tinygrad.uop.ops import UOp

def main()->int:
  p=argparse.ArgumentParser();p.add_argument("--reps",type=int,default=11);p.add_argument("--out",required=True);a=p.parse_args()
  spec=q6k_spec_for_role(131072,256);parts=np.random.default_rng(202609016).normal(0,.2,(spec.rows,spec.partial_axis_extent)).astype(np.float32)
  reference=parts[:,0].copy()
  for i in range(1,spec.partial_axis_extent):reference=(reference+parts[:,i]).astype(np.float32)
  out=Tensor.empty(spec.rows,dtype=dtypes.float32,device="NV").realize();src=Tensor(parts.copy(),dtype=dtypes.float32,device="NV").realize()
  candidate={"family":"q6_vocab_reduce_route.v1","rows":spec.rows,"k":spec.k,"row_tile":spec.row_tile,
    "reduction":spec.reduction,"epilogue":spec.epilogue,"spec_binding":"q6_spec"}
  emitter,tickets=lower_authorized_candidate(candidate,(("decode_q6k_coop_generated","q6_vocab_reduce"),),lowering_bindings={"q6_spec":spec})
  ast=emitter(UOp.placeholder((spec.rows,),dtypes.float32,0),
    UOp.placeholder((spec.rows,spec.partial_axis_extent),dtypes.float32,1));program=to_program(ast,Device["NV"].renderer)
  runtime=get_runtime("NV",program);gs,ls=program.arg.launch_dims({});buffers=[t.uop.buffer.get_buf("NV") for t in (out,src)]
  def run(wait=False):return runtime(*[buffers[i] for i in program.arg.globals],global_size=gs,local_size=ls,vals=(),wait=wait)
  run(True);got=out.numpy();timing=[run(True)*1e6 for _ in range(a.reps)];mismatch=int(np.count_nonzero(got.view(np.uint32)!=reference.view(np.uint32)))
  report={"schema":"tinygrad.boltbeam_q6_vocab_reduce_qualification.v1","device":str(Device.DEFAULT),
    "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"route_id":"decode_q6k_coop_generated",
    "component":"q6_vocab_reduce","tickets":[x.to_dict() for x in tickets.tickets],"payload_sha256":hashlib.sha256(parts.tobytes()).hexdigest(),
    "correctness":{"pass":mismatch==0,"bitwise_identical":mismatch==0,"mismatched_words":mismatch},
    "timing_us":{"samples":timing,"median":statistics.median(timing)}}
  text=json.dumps(report,indent=2,sort_keys=True);path=pathlib.Path(a.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text+"\n");print(text)
  return 0 if mismatch==0 else 1
if __name__=="__main__":raise SystemExit(main())
