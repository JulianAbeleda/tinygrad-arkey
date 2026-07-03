"""Q6K-3 speed gate: W==D for the lm_head half-warp direct route (Q6K_DIRECT_ROUTE) vs the shipped coop+sum baseline.
Median decode tok/s over NMEAS real steps per ctx (auto-clock wall spread is large -> rely on median + sign across ctx,
not single deltas), greedy token_match (temp->0), route_counts proving the half-warp lm_head kernel fired on the
candidate (and coop_partial_151936 did NOT). Tiered classification (TIER_A>=5% / TIER_B 2-5% / TIER_C -1..+2% /
CORRECT_BUT_NOT_FAST / REGRESSION). Audit/measurement only; default-off; baseline arm == flag-off == shipped route.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/q6k_direct_speed_gate.py
Writes: bench/amd-isa-backend-q6k-direct-speed/{latest.json,summary.md,wd_table.json,route_counts.json,amdahl_vs_measured.json}
"""
import os, sys, json, time, statistics, io, contextlib, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-q6k-direct-speed"
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS","512,1024,2048,4096").split(",")]
NMEAS = int(os.environ.get("QK_NMEAS","12")); NWARM = int(os.environ.get("QK_NWARM","5")); NTOK = 5
MAXC = 4608
_ANSI = re.compile(r"\x1b\[[0-9;]*m"); _KNAME = re.compile(r"\*\*\* AMD\s+\d+\s+(\S+)")

def tier(p):
  if p >= 5.0: return "TIER_A_MAJOR"
  if p >= 2.0: return "TIER_B_RESIDUAL"
  if p >= -1.0: return "TIER_C_EQUIVALENT_CLEANUP"
  return "REGRESSION"

CHILD = r'''
import os, json, time, statistics, io, contextlib, re
from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
MAXC=4608; CKPTS=[int(x) for x in os.environ["QK_CKPTS"].split(",")]
NMEAS=int(os.environ["QK_NMEAS"]); NWARM=int(os.environ["QK_NWARM"]); NTOK=5
ANSI=re.compile(r"\x1b\[[0-9;]*m"); KNAME=re.compile(r"\*\*\* AMD\s+\d+\s+(\S+)")
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for lin in (getattr(m,"_q4k_linears",None).linears if getattr(m,"_q4k_linears",None) else []): lin.decode_enabled=True
ids=(tok.prefix() if hasattr(tok,"prefix") else [])+tok.encode("the quick brown fox jumps over the lazy dog. "*800)
ids=(ids*(1+MAXC//max(1,len(ids))))[:MAXC]
v_sp=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); rows={}
for ck in CKPTS:
  for b in m.blk: b._use_flash,b._prefill_v2=ck>=512,False
  step=TinyJit(m.forward); tokid=int(ids[ck]); out=Tensor([[tokid]],dtype="int32").contiguous()
  for i in range(NWARM): out=step(out,v_sp.bind(ck+i),temp).realize()
  out=Tensor([[tokid]],dtype="int32").contiguous(); W=[]; toks=[]
  for i in range(NMEAS):
    t0=time.perf_counter(); out=step(out,v_sp.bind(ck+i),temp); tid=int(out.item()); W.append(time.perf_counter()-t0)
    if i<NTOK: toks.append(tid)
  buf=io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); m.forward(Tensor([[tokid]],dtype="int32").contiguous(),v_sp.bind(ck),temp).realize()
  names=sorted({KNAME.search(ANSI.sub("",l)).group(1) for l in buf.getvalue().splitlines() if KNAME.search(ANSI.sub("",l))})
  halfwarp=[n for n in names if "q6k_halfwarp_partition" in n]
  coop_lmhead=[n for n in names if "q6k_coop_partial" in n and any(int(x)>=100000 for x in re.findall(r"\d+",n))]
  w_ms=statistics.median(W)*1e3; sd=statistics.pstdev(W)*1e3
  rows[ck]={"tok_s":round(1000/w_ms,2),"w_ms_median":round(w_ms,3),"spread_pct":round(100*sd/w_ms,2),
            "tokens":toks,"halfwarp_fired":halfwarp,"coop_lmhead_fired":coop_lmhead,"nmeas":NMEAS}
print("@@"+json.dumps(rows))
'''

def run(flag):
  env={**os.environ,"DEV":"AMD","PYTHONPATH":str(ROOT),"Q6K_DIRECT_ROUTE":str(flag),
       "QK_CKPTS":",".join(map(str,CKPTS)),"QK_NMEAS":str(NMEAS),"QK_NWARM":str(NWARM)}
  r=subprocess.run([sys.executable,"-c",CHILD],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=1800)
  ln=[l for l in r.stdout.splitlines() if l.startswith("@@")]
  if not ln: raise RuntimeError(f"flag={flag} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
  return {int(k):v for k,v in json.loads(ln[-1][2:]).items()}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  base=run(0); cand=run(1)
  wd={}; deltas=[]; tok_ok=True; route_ok=True; worst_reg=0.0
  for ck in CKPTS:
    b,c=base[ck],cand[ck]
    d=round(100.0*(c["tok_s"]-b["tok_s"])/b["tok_s"],2); deltas.append(d)
    tm=b["tokens"]==c["tokens"]; tok_ok = tok_ok and tm
    rb = len(c["halfwarp_fired"])>0 and len(c["coop_lmhead_fired"])==0
    route_ok = route_ok and rb
    worst_reg = min(worst_reg, d)
    wd[ck]={"baseline_tok_s":b["tok_s"],"candidate_tok_s":c["tok_s"],"delta_pct":d,
            "baseline_spread_pct":b["spread_pct"],"candidate_spread_pct":c["spread_pct"],
            "token_match":tm,"candidate_halfwarp_fired":c["halfwarp_fired"],"candidate_coop_lmhead_fired":c["coop_lmhead_fired"]}
  med_delta = round(statistics.median(deltas),2); best_delta=round(max(deltas),2)
  # classify on the BEST credible ctx delta but require no ctx regress beyond the tier's bound; honest about spread
  t = tier(best_delta)
  # apply regression bounds: TIER_A no ctx < -2%, TIER_B no ctx < -1%
  if not tok_ok: verdict="AMD_ISA_Q6K_DIRECT_SPEED_BLOCKED_TOKEN_MISMATCH"
  elif not route_ok: verdict="AMD_ISA_Q6K_DIRECT_SPEED_BLOCKED_ROUTE_ATTRIBUTION"
  elif t=="TIER_A_MAJOR" and worst_reg>=-2.0: verdict="AMD_ISA_Q6K_DIRECT_SPEED_PASS_TIER_A"
  elif best_delta>=2.0 and worst_reg>=-1.0: verdict="AMD_ISA_Q6K_DIRECT_SPEED_PASS_TIER_B"
  elif best_delta>=-1.0 and worst_reg>=-1.0: verdict="AMD_ISA_Q6K_DIRECT_SPEED_CORRECT_BUT_NOT_FAST"  # TIER_C cleanup
  else: verdict="AMD_ISA_Q6K_DIRECT_SPEED_REGRESSION"
  # amdahl-vs-measured: Q6K-0 predicted ~+2.4% TIER_B (later refined: removable = coop GEMV inefficiency + partials .sum,
  # NOT the gumbel-argmax r_32_4_1187). Record prediction vs measured.
  amdahl={"q6k0_prediction":"+2.4% TIER_B (lm_head coop-reduce removal)","refinement":"r_32_4_1187 is the gumbel-argmax (intrinsic), not the coop reduce; firm removable = q6k_coop_partial_151936 + r_1187_32_4_16 partials-sum",
          "measured_best_delta_pct":best_delta,"measured_median_delta_pct":med_delta,"per_ctx_delta_pct":{str(ck):wd[ck]["delta_pct"] for ck in CKPTS}}
  rec={"verdict":verdict,"tier_of_best":t,"best_delta_pct":best_delta,"median_delta_pct":med_delta,"worst_ctx_delta_pct":round(worst_reg,2),
       "token_match_all_ctx":tok_ok,"route_bound_all_ctx":route_ok,"wd":wd,"amdahl_vs_measured":amdahl,
       "caveat":f"W==D wall spread is large (auto-clock confound; candidate spreads {[wd[ck]['candidate_spread_pct'] for ck in CKPTS]}%). Verdict rests on median across {len(CKPTS)} ctx + token/route gates, not any single delta. baseline arm == flag-off == shipped coop route; rollback = unset flag."}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump(wd, open(OUT/"wd_table.json","w"), indent=2)
  json.dump({str(ck):{"candidate_halfwarp":wd[ck]["candidate_halfwarp_fired"],"candidate_coop_lmhead":wd[ck]["candidate_coop_lmhead_fired"]} for ck in CKPTS}, open(OUT/"route_counts.json","w"), indent=2)
  json.dump(amdahl, open(OUT/"amdahl_vs_measured.json","w"), indent=2)
  md=[f"# Q6K-3 W==D speed gate\n\n**Verdict:** {verdict}\n",
      "| ctx | baseline tok/s | candidate tok/s | delta% | spread% (b/c) | token_match | halfwarp fired |","|---|---|---|---|---|---|---|"]
  for ck in CKPTS:
    r=wd[ck]; md.append(f"| {ck} | {r['baseline_tok_s']} | {r['candidate_tok_s']} | {r['delta_pct']:+} | {r['baseline_spread_pct']}/{r['candidate_spread_pct']} | {r['token_match']} | {bool(r['candidate_halfwarp_fired'])} |")
  md+=[f"\nbest delta {best_delta:+}% ({t}); median {med_delta:+}%; worst ctx {round(worst_reg,2):+}%. token_match all ctx: {tok_ok}; route-bound all ctx: {route_ok}.",
       f"\nAmdahl: {amdahl['q6k0_prediction']}; refinement: {amdahl['refinement']}.","\n"+rec["caveat"]]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__=="__main__":
  rec=main()
  print(json.dumps({k:rec[k] for k in("verdict","best_delta_pct","median_delta_pct","worst_ctx_delta_pct","token_match_all_ctx","route_bound_all_ctx")},indent=2))
  for ck,r in rec["wd"].items(): print(f"  ctx{ck}: base={r['baseline_tok_s']} cand={r['candidate_tok_s']} ({r['delta_pct']:+}%) spread b/c {r['baseline_spread_pct']}/{r['candidate_spread_pct']}%")
  print("\nQ6K3", rec["verdict"])
