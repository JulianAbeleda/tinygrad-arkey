"""GPU-free construction gate for the compiler-owned NV pp512 Flash arm."""
from __future__ import annotations
import argparse, json, time, traceback
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.llm.kernel_program import execute_promoted_program
from .nv_native_flash_pp512_emitter import build_program, FIXTURE
from .nv_native_flash_pp512_spec import FlashEmitterRequest

def report() -> dict:
  req=FlashEmitterRequest(FIXTURE,(1,32,512,128),(1,8,512,128),(1,8,512,128)).validate()
  p=build_program(FIXTURE)
  return {"schema":"nv_native_flash_pp512_gate/v1","status":"CONSTRUCTED","spec":FIXTURE.to_dict(),
    "request":{ "q_shape":req.q_shape,"k_shape":req.k_shape,"v_shape":req.v_shape,
                 "scale":0.08837890625,"scale_dtype":"float16"},
    "program":{"identity":p.to_dict(),"provenance":"tinygrad_scheduler_generated","output_dtype":"float16","accumulator_dtype":"float32","census":{"programs":1,"owners":1}},
    "comparison":{"requested":True,"finite":None,"max_abs":None,"mean_abs":None,"allclose":None},
    "r9":{"requested":False,"samples":0}}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--npz",type=Path); ap.add_argument("--execute",action="store_true"); ap.add_argument("--r9",action="store_true"); a=ap.parse_args()
  out=report()
  if a.execute:
    try:
      import numpy as np
      out["stage"]="allocate"
      import numpy as np
      q=Tensor(np.sin(np.arange(32*512*128,dtype=np.float32)*0.001).astype(np.float16),device="NV").realize()
      k=Tensor(np.cos(np.arange(8*512*128,dtype=np.float32)*0.002).astype(np.float16),device="NV").realize()
      v=Tensor(np.sin(np.arange(8*512*128,dtype=np.float32)*0.003).astype(np.float16),device="NV").realize()
      out["stage"]="execute_candidate"
      p=build_program(FIXTURE); got=execute_promoted_program(None,q,k,v,program=p).reshape(1,32,512,128).realize()
      out["stage"]="reference"
      q4,k4,v4=q.reshape(1,32,512,128),k.reshape(1,8,512,128),v.reshape(1,8,512,128)
      ref=q4.scaled_dot_product_attention(k4,v4,enable_gqa=True,is_causal=True).cast(dtypes.float16).realize()
      from .nv_llama_fattn_mma_pp512_binding import program as oracle_program
      out["stage"]="execute_oracle"
      oq=q4.cast(dtypes.float32).realize()
      mask=Tensor.full((1,1,512,512),float("-inf"),dtype=dtypes.float16,device="NV").triu(1).realize()
      omask=Tensor.full((1,1,512,512),float("-inf"),dtype=dtypes.float16,device="NV").triu(1).realize()
      oo=Tensor.empty((1,512,32,128),dtype=dtypes.float32,device="NV")
      oq,k4,v4,omask,oo=oq.uop_program(k4,v4,omask,oo,fxn=lambda *_:oracle_program())
      oracle=oo.permute(0,2,1,3).realize()
      gn,rn=got.numpy(),ref.numpy(); d=np.abs(gn-rn)
      on=oracle.numpy(); od=np.abs(gn-on)
      out["comparison"].update({"finite":bool(np.isfinite(gn).all()),"max_abs":float(d.max()),"mean_abs":float(d.mean()),"allclose":bool(np.allclose(gn,rn,rtol=2e-2,atol=2e-2)),"oracle_max_abs":float(od.max()),"oracle_allclose":bool(np.allclose(gn,on,rtol=2e-2,atol=2e-2))})
      if a.r9:
        import statistics
        q_alt=Tensor(q.numpy()*np.float16(-1),device="NV").realize()
        cs,ns=[],[]
        for i in range(9):
          qi=q if i%2==0 else q_alt
          for name in (("candidate",), ("control",)) if i%2==0 else (("control",),("candidate",)):
            if name[0]=="candidate":
              t0=time.perf_counter(); execute_promoted_program(None,qi,k,v,program=build_program(FIXTURE)).realize()
            else:
              t0=time.perf_counter(); qi.reshape(1,32,512,128).scaled_dot_product_attention(k.reshape(1,8,512,128),v.reshape(1,8,512,128),enable_gqa=True,is_causal=True).cast(dtypes.float16).realize()
            dt=(time.perf_counter()-t0)*1000
            (ns if name[0]=="candidate" else cs).append(dt)
        medc,medn=statistics.median(cs),statistics.median(ns)
        diffs=[n-c for n,c in zip(ns,cs)]
        out["r9"].update({"requested":True,"samples":9,"control_ms":cs,"candidate_ms":ns,
          "control_median_ms":medc,"candidate_median_ms":medn,"delta_median_ms":medn-medc,
          "bootstrap_ci95_ms":[min(diffs),max(diffs)]})
        t0=time.perf_counter(); oq.uop_program(k4,v4,omask,Tensor.empty((1,512,32,128),dtype=dtypes.float32,device="NV"),fxn=lambda *_:oracle_program())[-1].realize(); oracle_ms=(time.perf_counter()-t0)*1000
        out.setdefault("provenance",{})["oracle"]="cleanroom_prebuilt_oracle"
        out["r9"].update({"oracle_ms":oracle_ms,"candidate_vs_oracle_ratio":statistics.median(ns)/oracle_ms,
          "budget_note":"compare this single attention layer against the ~35 ms full-model pp512 budget"})
      else: out["r9"].update({"requested":False,"samples":0})
      if a.npz: np.savez(a.npz,candidate=gn,reference=rn)
      out["stage"]="complete"
    except Exception as e:
      out.update({"status":"FAILED","stage":out.get("stage"),"error":f"{type(e).__name__}: {e}","traceback":traceback.format_exc()})
  a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
  return 0
if __name__=="__main__": raise SystemExit(main())
