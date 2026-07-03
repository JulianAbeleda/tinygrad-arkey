"""Q6K-0 residual proof gate (AUDIT-ONLY, no kernels). Decide whether a direct/warp Q6_K route is justified BEFORE
implementing it. Reuses the eager-PROFILE per-kernel capture (one ProfileRangeEvent per kernel; same method as the
weight-path route attribution) to get the FULL per-kernel GPU-time breakdown, then:

  - p_q6k_proven(ctx) = q6k_gemv + lm_head  (the q6k coop_partial GEMV kernels)
  - role-resolve reduce_partial: split the r_* reduce kernels into q6k-coop-reduce (the reduce that sums a q6k
    coop_partial GEMV's partials -- matched by output-dim) vs rmsnorm/flash/other.
  - a(ctx) = q6k-coop-reduce time / total reduce_partial time   (MEASURED, plus the scope's {0,.25,.5,1} sensitivity grid)
  - Amdahl: S = 1/(1 - p_q6k_total*r) for r in {.25,.5,1}, ctx in {512,4096}; R_new = R0*S.

The lever is ROUTE efficiency (single-pass warp like Q4_K G3, eliminating coop partials+sum), NOT quant demotion
(sub-4-bit / Q6_K->lower is dNLL-quality-refuted). 820 GB/s memcpy is NOT the full-decode target.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/q6k_residual_math_gate.py
Writes: bench/amd-isa-backend-q6k-residual-math/{latest.json,summary.md,reduce_role_split.json,amdahl_sensitivity.json,q6k_route_candidates.json}
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-q6k-residual-math"
R0 = {"512": 103.9, "1024": 102.0, "2048": 99.6, "4096": 94.4}   # g3-promotion W==D (median)

# --- capture child: full per-kernel eager GPU-time (same model build/PROFILE method as weight-path route attribution;
#     run under BUBBLEBEAM_FUTURESIGHT=1 = the PROMOTED Q4_K G3 route, so the system matches the current best) ---
CHILD = r'''
import os, json, re
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk.harness_contract import DEFAULT_MODEL
from extra.llm.generate import load_model_and_tokenizer
MAXC=4608; CTX=int(os.environ["Q6K_CTX"]); NSTEPS=int(os.environ.get("Q6K_NSTEPS","6"))
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for lin in (getattr(m,"_q4k_linears",None).linears if getattr(m,"_q4k_linears",None) else []): lin.decode_enabled=True
for b in m.blk: b._use_flash,b._prefill_v2=True,False
v=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); jit=TinyJit(m.forward); tk=Tensor([[100]],dtype="int32").contiguous()
for i in range(4): jit(tk,v.bind(CTX+i),temp).realize().item()
import tinygrad.runtime.ops_amd
Compiled.profile_events=[]
with Context(PROFILE=1):
  for i in range(NSTEPS): m.forward(tk,v.bind(CTX+i),temp).realize().item()
agg={}; calls={}
for e in Compiled.profile_events:
  if isinstance(e,ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name,"name",None) or str(e.name); agg[nm]=agg.get(nm,0.0)+float(e.en-e.st); calls[nm]=calls.get(nm,0)+1
print("@@"+json.dumps({"ctx":CTX,"per_kernel":{k:{"dur":round(agg[k]/NSTEPS,4),"calls":round(calls[k]/NSTEPS,2)} for k in agg}}))
'''

def capture(ctx):
  env={**os.environ,"DEV":"AMD","PYTHONPATH":str(ROOT),"Q6K_CTX":str(ctx),"BUBBLEBEAM_FUTURESIGHT":"1"}
  out=subprocess.run([sys.executable,"-c",CHILD],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=900).stdout
  ln=[l for l in out.splitlines() if l.startswith("@@")]
  if not ln: raise RuntimeError("capture failed: "+out[-1200:])
  return json.loads(ln[-1][2:])["per_kernel"]

def _dims(name):  # trailing integer groups in a kernel name, e.g. q6k_coop_partial_151936_4096 -> [151936,4096]
  return [int(x) for x in re.findall(r'\d+', name)]

import math
MATDIMS={4096,12288,151936,1024}  # weight matrix dims of interest (used to strip the spurious "6" from "q6k")
def _mat(name): return [x for x in _dims(name) if x in MATDIMS]

def role_resolve(pk):
  """Classify every kernel; split reduce_partial (r_*) into FIRMLY-attributable q6k-coop-reduce vs ambiguous.
  A q6k coop GEMV (q6k_coop_partial_OUT_IN) sums its partials in a separate r_* whose dim-product == the GEMV output.
  Only the lm_head reduce (prod==151936) is uniquely attributable; the per-layer prod==4096 reduces are AMBIGUOUS
  (RMSNorm vs q6k gate_up coop, same dim) -> kept in 'ambiguous', surfaced via the a-grid, NOT silently credited."""
  tot=sum(v["dur"] for v in pk.values()) or 1e-9
  q6k_gemv=lm_head=q4k_gemv=elementwise=attn=0.0
  q6k_out_dims=set()
  for nm,v in pk.items():
    n=nm.lower()
    if "q6k_coop_partial" in n or "q6k_gemv_partial" in n:
      md=_mat(nm)
      if 151936 in md or "151936" in n: lm_head+=v["dur"]; q6k_out_dims.add(151936)
      else: q6k_gemv+=v["dur"]; q6k_out_dims.update(md)   # e.g. {4096,12288} for gate_up
    elif "q4k_gemv" in n or "q4k" in n: q4k_gemv+=v["dur"]
    elif "flash" in n or "combine" in n or "gmax" in n: attn+=v["dur"]
    elif n.startswith("e_") or "cast" in n or "copy" in n: elementwise+=v["dur"]
  red_firm_q6k=red_ambig=red_other=red_total=0.0; red_rows=[]
  for nm,v in pk.items():
    n=nm.lower()
    if not n.startswith("r_"): continue
    red_total+=v["dur"]; d=_dims(nm); prod=math.prod(d) if d else 0
    if prod==151936: cls="q6k_lm_head_reduce_FIRM"; red_firm_q6k+=v["dur"]    # uniquely lm_head's coop reduce
    elif prod in (12288,) or prod in {x for x in q6k_out_dims if x not in (4096,)}: cls="q6k_coop_reduce_likely"; red_firm_q6k+=v["dur"]
    elif prod==4096: cls="ambiguous_rmsnorm_or_q6k_gateup"; red_ambig+=v["dur"]  # 4096 = hidden (RMSNorm) AND a q6k in-dim
    else: cls="other_reduce"; red_other+=v["dur"]
    red_rows.append({"kernel":nm,"dur":round(v["dur"],3),"calls":v["calls"],"dims":d,"prod":prod,"class":cls})
  return {"total":tot,"q6k_gemv":q6k_gemv,"lm_head":lm_head,"q4k_gemv":q4k_gemv,"attn":attn,"elementwise":elementwise,
          "reduce_total":red_total,"reduce_q6k_firm":red_firm_q6k,"reduce_ambiguous":red_ambig,"reduce_other":red_other,
          "pct":{k:round(100*val/tot,2) for k,val in [("q6k_gemv",q6k_gemv),("lm_head",lm_head),("q4k_gemv",q4k_gemv),
                 ("attn",attn),("elementwise",elementwise),("reduce_total",red_total),("reduce_q6k_firm",red_firm_q6k),("reduce_ambiguous",red_ambig)]},
          "reduce_rows":sorted(red_rows,key=lambda r:-r["dur"])[:25],"q6k_out_dims":sorted(q6k_out_dims)}

def main():
  OUT.mkdir(parents=True,exist_ok=True)
  Q6K_BW, Q4K_G3_BW = 503.0, 650.0   # measured eff bw (system-residual): a direct warp Q6_K route matches Q4_K G3 efficiency
  rr={}
  for ctx in (512,4096): rr[str(ctx)]=role_resolve(capture(ctx))
  # a(ctx) = FIRM q6k-coop reduce / reduce_total (lm_head + likely-q6k); ambiguous reduce is the a-grid upside.
  a_meas={c:(rr[c]["reduce_q6k_firm"]/rr[c]["reduce_total"] if rr[c]["reduce_total"] else 0.0) for c in rr}
  a_max ={c:((rr[c]["reduce_q6k_firm"]+rr[c]["reduce_ambiguous"])/rr[c]["reduce_total"] if rr[c]["reduce_total"] else 0.0) for c in rr}
  def p_proven(c): return (rr[c]["q6k_gemv"]+rr[c]["lm_head"])/rr[c]["total"]
  def p_total(c,a): return p_proven(c)+a*(rr[c]["reduce_total"]/rr[c]["total"])
  AGRID=[0.0,0.25,0.5,1.0]; RGRID=[0.25,0.5,1.0]
  sens={}
  for c in rr:
    sens[c]={"R0":R0[c],"p_q6k_proven_pct":round(100*p_proven(c),2),"a_measured":round(a_meas[c],3),
             "reduce_total_pct":rr[c]["pct"]["reduce_total"],"grid":{}}
    for a in AGRID:
      pt=p_total(c,a)
      sens[c]["grid"][f"a={a}"]={"p_q6k_total_pct":round(100*pt,2),
        **{f"r={r}":{"speedup":round(1/(1-pt*r),4),"R_new":round(R0[c]/(1-pt*r),1),"gain_pct":round(100*(1/(1-pt*r)-1),1)} for r in RGRID}}
  # ---- FIRM removable share (does not rely on the ambiguous per-layer reduces) ----
  # (1) q6k_gemv bw gap: a direct warp route matches Q4_K G3 efficiency -> time *= Q6K_BW/Q4K_G3_BW (removes the rest).
  # (2) lm_head coop reduce + likely-q6k reduce: PURE overhead a single-pass route eliminates (r=1.0 on that slice).
  def removable_firm(c):
    gemv_gap = rr[c]["q6k_gemv"]*(1 - Q6K_BW/Q4K_G3_BW)   # us removed by matching Q4_K efficiency
    reduce_firm = rr[c]["reduce_q6k_firm"]                 # coop reduce eliminated
    return (gemv_gap+reduce_firm)/rr[c]["total"]
  proven={c:p_proven(c) for c in rr}
  rem_firm={c:removable_firm(c) for c in rr}
  affected_pct=max(100*proven[c] for c in rr)
  # gain from FIRM removables alone (conservative); + ambiguous-reduce upside via the a-grid
  gain_firm={c:round(100*(1/(1-rem_firm[c])-1),1) for c in rr}
  removable_25_of_affected = any(rem_firm[c] >= 0.25*proven[c] for c in rr)   # firm removable >=25% of affected share
  if affected_pct>=10 and removable_25_of_affected:
    verdict="AMD_ISA_Q6K_RESIDUAL_PASS_DIRECT_ROUTE_JUSTIFIED"
  elif affected_pct>=10:
    verdict="AMD_ISA_Q6K_RESIDUAL_INCONCLUSIVE_REDUCE_NOT_ROLE_RESOLVED"
  else:
    verdict="AMD_ISA_Q6K_RESIDUAL_PASS_RECLASSIFY_TARGET"
  # route candidates
  cand={c:{"current_q6k_route":{"q6k_coop_gemv_pct":rr[c]["pct"]["q6k_gemv"],"q6k_gemv_eff_bw":Q6K_BW,"lm_head_pct":rr[c]["pct"]["lm_head"],
            "reduce_q6k_firm_pct":rr[c]["pct"]["reduce_q6k_firm"],"reduce_ambiguous_pct":rr[c]["pct"]["reduce_ambiguous"],
            "q6k_out_dims":rr[c]["q6k_out_dims"],"pattern":"q6k_coop_partial GEMV (503 GB/s) + separate r_* reduce (partials+sum)"},
           "candidate_direct_route":{"replaces":["q6k_coop_partial_* GEMV (->650 GB/s warp)","its r_* coop reduce (eliminated)"],
            "single_pass_like_q4k_g3":True,"preserves_quant_semantics":True,"is_quant_demotion":False,
            "firm_removable_pct_gpu":round(100*rem_firm[c],2),"removed_fraction_r_modeled":RGRID}} for c in rr}
  rec={"verdict":verdict,"R0_wd":R0,
       "p_q6k_proven_pct":{c:round(100*proven[c],2) for c in rr},
       "a_measured_firm":{c:round(a_meas[c],3) for c in a_meas}, "a_max_incl_ambiguous":{c:round(a_max[c],3) for c in a_max},
       "reduce_q6k_firm_pct":{c:rr[c]["pct"]["reduce_q6k_firm"] for c in rr},
       "reduce_ambiguous_pct":{c:rr[c]["pct"]["reduce_ambiguous"] for c in rr},
       "firm_removable_pct_gpu":{c:round(100*rem_firm[c],2) for c in rr},
       "gain_from_firm_removables_pct":gain_firm,
       "decision":{"affected_share_pct":round(affected_pct,1),"firm_removable_>=25pct_of_affected":bool(removable_25_of_affected),
         "expected_gain_conservative_pct":gain_firm,
         "lever":"direct/warp single-pass Q6_K route eliminating coop partials+sum (NOT quant demotion -- quality-refuted)",
         "note_lm_head":"lm_head coop GEMV is bw-efficient (761); its FIRM removable part is the coop REDUCE (prod==151936), not the GEMV.",
         "note_ambiguous":"the per-layer prod==4096 reduces are NOT credited to q6k (could be RMSNorm) -- they are ADDITIONAL upside via the a-grid, not the basis of the verdict."},
       "sensitivity":sens,"reduce_role_split":{c:{"a_measured_firm":round(a_meas[c],3),"a_max":round(a_max[c],3),
         "reduce_total_pct":rr[c]["pct"]["reduce_total"],"reduce_q6k_firm_pct":rr[c]["pct"]["reduce_q6k_firm"],
         "reduce_ambiguous_pct":rr[c]["pct"]["reduce_ambiguous"],"rows":rr[c]["reduce_rows"]} for c in rr},
       "route_candidates":cand,
       "caveats":["FIRM q6k-coop reduce = r_* with dim-product==151936 (lm_head) or a non-4096 q6k output dim; per-layer prod==4096 reduces are AMBIGUOUS (RMSNorm vs q6k gate_up coop, identical dim) and are NOT credited -- only surfaced as a-grid upside",
                  "verdict rests on FIRM removables: q6k_gemv bw gap (503->650, matching Q4_K G3) + lm_head coop reduce -- not on the ambiguous reduce attribution",
                  "W==D R0 is g3-promotion median (wall spread ~52% auto-clock confound); GPU-time %s are the reliable attribution",
                  "820 GB/s memcpy is NOT the full-decode ceiling (dequant-GEMV intrinsic tax)"]}
  json.dump(rec,open(OUT/"latest.json","w"),indent=2)
  json.dump(rec["reduce_role_split"],open(OUT/"reduce_role_split.json","w"),indent=2)
  json.dump(sens,open(OUT/"amdahl_sensitivity.json","w"),indent=2)
  json.dump(cand,open(OUT/"q6k_route_candidates.json","w"),indent=2)
  md=[f"# Q6K-0 residual proof gate\n\n**Verdict:** {verdict}\n",
      f"affected (q6k_gemv+lm_head) share = {round(affected_pct,1)}% GPU-time; FIRM a (uniquely-q6k fraction of reduce) = {rec['a_measured_firm']}, a_max (incl. ambiguous) = {rec['a_max_incl_ambiguous']}; FIRM q6k-coop reduce = {rec['reduce_q6k_firm_pct']}%, ambiguous reduce = {rec['reduce_ambiguous_pct']}% GPU-time.\n",
      f"**FIRM removable** (q6k_gemv 503->650 bw gap + lm_head coop reduce) = {rec['firm_removable_pct_gpu']}% GPU-time -> conservative W==D gain {rec['gain_from_firm_removables_pct']}.\n",
      "## Amdahl sensitivity (R_new = R0 / (1 - p_q6k_total*r); a = fraction of reduce_partial credited to q6k)\n| ctx | a | p_q6k_total% | r=0.25 | r=0.5 | r=1.0 |","|---|---|---|---|---|---|"]
  for c in ("512","4096"):
    for a in ("a=0.0","a=0.25","a=0.5","a=1.0"):
      g=sens[c]["grid"][a]; md.append(f"| {c} | {a[2:]} | {g['p_q6k_total_pct']} | +{g['r=0.25']['gain_pct']}% ({g['r=0.25']['R_new']}) | +{g['r=0.5']['gain_pct']}% ({g['r=0.5']['R_new']}) | +{g['r=1.0']['gain_pct']}% ({g['r=1.0']['R_new']}) |")
  md+=["\n## Decision\n"+rec["decision"]["lever"],
       f"\naffected share {round(affected_pct,1)}% (>=10%); FIRM removable >=25% of affected: {rec['decision']['firm_removable_>=25pct_of_affected']}; conservative gain {rec['gain_from_firm_removables_pct']}.",
       "\n## Caveats\n"+"\n".join(f"- {x}" for x in rec["caveats"])]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__=="__main__":
  rec=main()
  print(json.dumps({"verdict":rec["verdict"],"affected_pct":rec["decision"]["affected_share_pct"],
                    "a_measured_firm":rec["a_measured_firm"],"a_max":rec["a_max_incl_ambiguous"],
                    "reduce_q6k_firm_pct":rec["reduce_q6k_firm_pct"],"firm_removable_pct":rec["firm_removable_pct_gpu"],
                    "gain_conservative_pct":rec["gain_from_firm_removables_pct"]},indent=2))
  print("\nQ6K-0",rec["verdict"])
