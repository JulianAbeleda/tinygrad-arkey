"""Q6K-2 Stage-2: in-model lm_head correctness/route gate for the half-warp direct route (Q6K_DIRECT_ROUTE).
Compares DETERMINISTIC logits (m.logits, no gumbel/argmax) at ctx512 with the flag OFF (baseline coop+sum) vs ON
(half-warp direct): argmax(token) match + logit aggregates (maxval/sum/sumsq) match. Cold-profiles each arm (no warmup
=> no buffer caching) for route names, then set-diffs to show what the route REMOVED (q6k_coop_partial_151936 + its
sum) vs what is intrinsic (the gumbel-argmax r_32_4_1187 lives in forward, NOT in m.logits, so it does not appear here).

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_q6k_lmhead_token_gate.py
Writes: bench/amd-isa-backend-q6k-direct-correctness/{token_gate.json, route_attribution.json}
"""
import os, sys, json, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-q6k-direct-correctness"

CHILD = r'''
import os, json
from tinygrad import Tensor, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
import tinygrad.runtime.ops_amd
MAXC=4608; CTX=int(os.environ["CTX"])
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for lin in m._q4k_linears.linears: lin.decode_enabled=True   # registry holds q4k AND q6k linears
for b in m.blk: b._use_flash,b._prefill_v2=True,False
v=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); tk=Tensor([[100]],dtype="int32").contiguous()
# route NAMES from a COLD m.forward (the lm_head GEMV is mid-graph here -> appears by kernel name; in m.logits it is
# the final-output custom_kernel and gets wrapped in a graph TracingKey, hidden). r_32_4_1187 here = gumbel-argmax.
Compiled.profile_events=[]
with Context(PROFILE=1): m.forward(tk,v.bind(CTX),temp).realize()
names=sorted({ (getattr(e.name,"name",None) or str(e.name)) for e in Compiled.profile_events if isinstance(e,ProfileRangeEvent)})
# correctness from DETERMINISTIC logits (no gumbel/argmax) -> stable across the two flag subprocesses
lg=m.logits(tk,v.bind(CTX)).realize()
f=lg.reshape(-1)
rec={"argmax":int(f.argmax().item()),"n":int(f.numel()),"maxval":float(f.max().item()),
     "sum":float(f.sum().item()),"sumsq":float((f*f).sum().item()),"names":names}
print("@@"+json.dumps(rec))
'''

def run(flag, ctx):
  env={**os.environ,"DEV":"AMD","PYTHONPATH":str(ROOT),"CTX":str(ctx),"Q6K_DIRECT_ROUTE":str(flag)}
  r=subprocess.run([sys.executable,"-c",CHILD],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=900)
  ln=[l for l in r.stdout.splitlines() if l.startswith("@@")]
  if not ln: raise RuntimeError(f"flag={flag} failed:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
  return json.loads(ln[-1][2:])

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  base=run(0,512); cand=run(1,512)
  bn,cn=set(base["names"]),set(cand["names"])
  removed=sorted(bn-cn); added=sorted(cn-bn)
  def rel(a,b): return abs(a-b)/(abs(a)+abs(b)+1e-9)
  token_match = base["argmax"]==cand["argmax"]
  numeric_match = max(rel(base[k],cand[k]) for k in("maxval","sum","sumsq"))<1e-3
  route_bound = any("q6k_halfwarp_partition" in n for n in cn)
  cand_no_coop_lmhead = not any("q6k_coop_partial" in n and any(int(x)>=100000 for x in __import__("re").findall(r"\d+",n)) for n in cn)
  base_used_coop_lmhead = any("q6k_coop_partial" in n and any(int(x)>=100000 for x in __import__("re").findall(r"\d+",n)) for n in bn)
  ok = token_match and numeric_match and route_bound and cand_no_coop_lmhead and base_used_coop_lmhead
  verdict = ("Q6K2_PASS_LMHEAD_TOKEN_GATE" if ok else
    "Q6K2_BLOCKED_TOKEN_MISMATCH" if not(token_match and numeric_match) else
    "Q6K2_BLOCKED_ROUTE_BINDING" if not route_bound else
    "Q6K2_BLOCKED_BASELINE_NOT_COOP" if not base_used_coop_lmhead else
    "Q6K2_BLOCKED_COOP_STILL_PRESENT")
  rec={"verdict":verdict,"ctx":512,"compared":"m.logits (deterministic, no gumbel/argmax)",
       "token_match":token_match,"numeric_match":numeric_match,
       "argmax":{"baseline":base["argmax"],"candidate":cand["argmax"]},
       "logits_agg":{"baseline":{k:base[k] for k in("maxval","sum","sumsq")},"candidate":{k:cand[k] for k in("maxval","sum","sumsq")}},
       "route_bound_halfwarp":route_bound,"halfwarp_kernels":[n for n in cn if "q6k_halfwarp_partition" in n],
       "baseline_used_coop_lmhead":base_used_coop_lmhead,"candidate_dropped_coop_lmhead":cand_no_coop_lmhead,
       "kernels_removed_by_route":removed,"kernels_added_by_route":added,
       "note":"r_32_4_1187 (gumbel-argmax over vocab) is in forward, NOT m.logits, so it is correctly absent here -- it is the SAMPLING reduce, intrinsic to decode, NOT the lm_head coop reduce. The route removes q6k_coop_partial_151936 + its partials .sum; what's eliminated is in kernels_removed_by_route."}
  json.dump(rec, open(OUT/"token_gate.json","w"), indent=2)
  json.dump({"baseline_names":base["names"],"candidate_names":cand["names"]}, open(OUT/"route_attribution.json","w"), indent=2)
  return rec

if __name__=="__main__":
  rec=main()
  print(json.dumps({k:rec[k] for k in("verdict","token_match","numeric_match","route_bound_halfwarp","candidate_dropped_coop_lmhead","argmax","kernels_removed_by_route")}, indent=2))
  print("\nQ6K2-S2", rec["verdict"])
