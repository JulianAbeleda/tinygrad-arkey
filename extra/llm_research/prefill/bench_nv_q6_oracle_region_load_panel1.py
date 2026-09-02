#!/usr/bin/env python3
"""Fail-closed RegionLoad Q8 panel-1 source/SASS admission harness."""
from __future__ import annotations

import argparse, hashlib, json, os, re, signal, time
from pathlib import Path

from tinygrad.uop.ops import Ops, PostBarrierRegion, RegionLoad
import extra.llm_research.prefill.bench_nv_q6_oracle_true_late_panel1 as gate
import extra.llm_research.prefill.nv_q6_oracle_broad_cta as impl

ANCHOR_SHA="6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137"
FIXUP_SHA="483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514"
EXPECTED={"IMMA":256,"LDSM":32,"LDS":176,"LDG":109,"STS":73,"STG":64,"BAR":4,
          "I2FP":1024,"FMUL":1544,"FADD":1024,"FFMA":0}


def _builder(*args,q8_panel1_schedule="early",**kwargs):
  return impl.q6_oracle_broad_cta_kernel(*args,region_load_q8_panel1=(q8_panel1_schedule != "early"),**kwargs)


def _semantic(ast):
  topo=ast.toposort()
  regions=[x for x in topo if x.op is Ops.IF and isinstance(x.arg,PostBarrierRegion)]
  markers=[x for x in topo if x.op is Ops.AFTER and isinstance(x.arg,RegionLoad)]
  loads=[x for x in topo if x.op is Ops.LOAD and any(y in markers for y in x.src[1:])]
  publications=[x for x in topo if x.op is Ops.STORE and len(x.src)>=2 and x.src[1] in loads]
  ends=[x for x in topo if x.op is Ops.ENDIF and isinstance(x.arg,PostBarrierRegion)]
  owners={x.src[0].src[0] for x in loads}
  written={p for x in publications for p in x.src[0].backward_slice_with_self if p.op is Ops.PARAM}
  return {"regions":len(regions),"markers":len(markers),"loads":len(loads),"publications":len(publications),
          "ends":len(ends),"publication_roots":len(ends[0].src)-1 if len(ends)==1 else -1,
          "global_immutable":len(owners)==1 and next(iter(owners)).op is Ops.PARAM and owners.isdisjoint(written),
          "index_markers":sum(any(y in x.src[0].backward_slice_with_self for y in markers) for x in loads)}


def _compile_bounded(ast,label,root,bound):
  def timeout(_signum,_frame): raise TimeoutError(f"compile exceeded {bound}s")
  old=signal.signal(signal.SIGALRM,timeout); signal.alarm(bound); start=time.monotonic()
  try: gate._compile_ast(ast,label,root)
  finally: signal.alarm(0); signal.signal(signal.SIGALRM,old)
  return time.monotonic()-start


def _artifact(root,label):
  base=root/label; sass=next((base/"sass").glob("*.sass.json")); dis=next((base/"sass").glob("*.nvdisasm"))
  return base, json.loads(sass.read_text()), dis.read_text()


def _write(path:Path,data): path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")


def main() -> int:
  ap=argparse.ArgumentParser()
  ap.add_argument("--out",default="docs/task_workflow/evidence/nv-q6-region-load-panel1-gate11-20260831/pre-sass.json")
  ap.add_argument("--artifacts",default="docs/task_workflow/evidence/nv-q6-region-load-panel1-gate11-20260831/artifacts")
  ap.add_argument("--compile-bound",type=int,default=240)
  args=ap.parse_args(); out=Path(args.out); root=Path(args.artifacts); root.mkdir(parents=True,exist_ok=True)
  result={"schema":"tinygrad.nv_q6_region_load_panel1_pre_sass.v1","anchor_sha256":ANCHOR_SHA,"fixup_sha256":FIXUP_SHA,
          "gpu_lock_held":os.getenv("NV_Q6_GPU_LOCK_HELD")=="1","gpu_work_started":False}
  if not result["gpu_lock_held"]:
    result|={"verdict":"BLOCKED_GPU_LOCK_NOT_HELD","promotion_eligible":False}; _write(out,result); return 2
  original=gate.q6_oracle_broad_cta_kernel; gate.q6_oracle_broad_cta_kernel=_builder
  try:
    anchor_ast=gate._ast("early"); candidate_ast=gate._ast("true_late_tail")
    result["semantic"]=_semantic(candidate_ast)
    result["compile_elapsed_s"]={"anchor":_compile_bounded(anchor_ast,"anchor",root,args.compile_bound),
                                  "candidate":_compile_bounded(candidate_ast,"candidate",root,args.compile_bound)}
  except Exception as exc:
    result|={"verdict":"REJECT_COMPILE","promotion_eligible":False,"error":f"{type(exc).__name__}: {exc}"}; _write(out,result); return 2
  finally: gate.q6_oracle_broad_cta_kernel=original
  abase,ajson,_=_artifact(root,"anchor"); cbase,cjson,cdis=_artifact(root,"candidate")
  asha=hashlib.sha256((abase/"anchor.cubin").read_bytes()).hexdigest(); csha=hashlib.sha256((cbase/"candidate.cubin").read_bytes()).hexdigest()
  panel=gate._classify_q8_panel1(cdis); fam=cjson["families"]; resources=cjson["resources"]
  result|={"anchor_rebuild_sha256":asha,"candidate_cubin_sha256":csha,"candidate_source_sha256":hashlib.sha256((cbase/"candidate.cu").read_bytes()).hexdigest(),
           "candidate_symbol":cjson["symbol"],"instruction_total":cjson["instruction_total"],"families":fam,"resources":resources,"panel":panel}
  gates={"anchor_byte_identical":asha==ANCHOR_SHA,
         "semantic_region_exact":result["semantic"]=={"regions":1,"markers":1,"loads":18,"publications":18,"ends":1,
                                                        "publication_roots":18,"global_immutable":True,"index_markers":0},
         "panel_exact":panel.get("classified") and panel.get("panel1_loads")==18 and panel.get("panel1_stores")==18,
         "span":panel.get("panel1_load_to_store_span_instructions",10**9)<=160,
         "families":all(fam.get(k,0)==v for k,v in EXPECTED.items()),
         "no_forbidden":all(fam.get(k,0)==0 for k in ("MEMBAR","ATOM","LDL","STL")),
         "resources":resources.get("stack_bytes")==0 and resources.get("local_static_bytes")==0,
         "no_schedule_lop3":fam.get("LOP3")==ajson["families"].get("LOP3") and 'xor.b32' not in (cbase/"candidate.cu").read_text()}
  result["gates"]=gates; result["promotion_eligible"]=all(gates.values()); result["verdict"]="PASS_SOURCE_SASS" if result["promotion_eligible"] else "REJECT_SOURCE_SASS"
  _write(out,result); print(json.dumps({"verdict":result["verdict"],"gates":gates,"panel":panel},sort_keys=True))
  return 0 if result["promotion_eligible"] else 2


if __name__ == "__main__": raise SystemExit(main())
