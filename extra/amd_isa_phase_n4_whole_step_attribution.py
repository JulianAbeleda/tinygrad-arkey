"""AMD ISA backend — Phase N4: whole-decode-step per-kernel attribution (native-dynamic-S vs owned/default).

N3F proved the attention tile is only ~10% of the decode step. N4 measures EVERY kernel in the decode step to decide
where the remaining wall-clock is (attention tile residual / gmax+combine / Q4_K GEMV+FFN / projections / small ops),
which selects the single N5 branch. Per-kernel GPU ms via DEBUG=2 program_ms (averaged over several JIT-replay steps),
captured in fresh subprocesses (getenv memoizes) at ctx512 and ctx4096 for both routes.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n4_whole_step_attribution.py
Writes: bench/amd-isa-backend-phase-n4/{latest.json, summary.md}
"""
import os, sys, json, pathlib, subprocess, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-phase-n4"
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NSTEPS = int(os.environ.get("QK_N4_STEPS", "4"))   # eager steps (per-kernel GPU timestamps are clean; few suffice)
ROUTES = {
  "native_dyn_s": {"DECODE_ATTN_AMDGCN_TILE":"0","DECODE_ATTN_GENERATED_WHOLECACHE":"1","DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE":"1",
                   "DECODE_ATTN_BLOCK_TILE":"1","DECODE_ATTN_NATIVE_ISA_BLOCK_TILE":"1"},
  "owned": {"DECODE_ATTN_AMDGCN_TILE":"1"},
}
_ZERO = ["DECODE_ATTN_AMDGCN_TILE","DECODE_ATTN_GENERATED_WHOLECACHE","DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE",
         "DECODE_ATTN_BLOCK_TILE","DECODE_ATTN_BLOCK_TILE_FIXED_S","DECODE_ATTN_NATIVE_ISA_BLOCK_TILE"]

def _owner(name:str) -> str:
  if name.startswith("native_block_tile"): return "native_attn_tile"
  if name.startswith("owned_flash_tile"): return "owned_attn_tile"
  if name.startswith("flash_state_gmax"): return "attn_gmax"
  if name.startswith(("flash_state_combine","flash_fused_state_combine")): return "attn_combine"
  if name.startswith(("q4k_","q6k_","q8")) or "gemv" in name or "mmvq" in name: return "q4k_gemv_ffn"
  if name.startswith("r_"): return "generated_reduce"
  if name.startswith("E_"): return "generated_elementwise"
  if name.startswith(("copy","cast")) or "copy" in name: return "copy_cast"
  return "other"

CHILD = r'''
import os, json
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
MAXC=4608; CTX=int(os.environ["N4_CTX"]); NSTEPS=int(os.environ["N4_STEPS"])
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for lin in (getattr(m,"_q4k_linears",None).linears if getattr(m,"_q4k_linears",None) else []): lin.decode_enabled=True
for b in m.blk: b._use_flash,b._prefill_v2=True,False
v=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); jit=TinyJit(m.forward)
tk=Tensor([[100]],dtype="int32").contiguous(); toks=[]
for i in range(4): toks.append(int(jit(tk,v.bind(CTX+i),temp).realize().item()))   # warmup/compile (clean tokens for the gate)
import tinygrad.runtime.ops_amd  # noqa
# EAGER forward under PROFILE: each kernel dispatch is its own ProfileRangeEvent with GPU HW timestamps (st,en) =
# CLEAN per-kernel GPU COMPUTE time (not wall). The JIT graph profiles as ONE range (no per-kernel breakdown), so we
# use eager. Per-kernel GPU-compute share is valid for native-vs-owned attribution; absolute != JIT W==D wall envelope.
Compiled.profile_events=[]
with Context(PROFILE=1):
  for i in range(NSTEPS): out=m.forward(tk,v.bind(CTX+i),temp).realize(); toks.append(int(out.item()))
agg={}; calls={}; nsteps=NSTEPS
for e in Compiled.profile_events:
  if isinstance(e, ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name,"name",None) or str(e.name)
    agg[nm]=agg.get(nm,0.0)+float(e.en-e.st); calls[nm]=calls.get(nm,0)+1
n=max(1,nsteps)
per_kernel={k:{"dur_per_step":round(agg[k]/n,4),"calls_per_step":round(calls[k]/n,2)} for k in agg}
print("@@"+json.dumps({"ctx":CTX,"per_kernel":per_kernel,"warm_tokens":toks[1:6],"nsteps":n}))
'''

def _capture(route, ctx):
  env={**os.environ,"DEV":"AMD","JIT":"1","PROFILE":"1","N4_CTX":str(ctx),"N4_STEPS":str(NSTEPS),"PYTHONPATH":str(ROOT)}
  for k in _ZERO: env.pop(k,None)
  env.update(ROUTES[route])
  out=subprocess.run([sys.executable,"-c",CHILD],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=1800).stdout
  line=[l for l in out.splitlines() if l.startswith("@@")]
  return json.loads(line[-1][2:]) if line else {"failed":True}

def _step_table(cap):
  pk=cap["per_kernel"]; tot=sum(v["dur_per_step"] for v in pk.values()) or 1e-9
  byowner={}
  for k,v in pk.items():
    o=_owner(k); e=byowner.setdefault(o,{"owner":o,"dur":0.0,"kernels":[]}); e["dur"]+=v["dur_per_step"]; e["kernels"].append(k)
  rows=sorted(byowner.values(), key=lambda r:-r["dur"])
  for r in rows: r["pct_of_step"]=round(100*r["dur"]/tot,1); r["dur"]=round(r["dur"],4); r["n_kernels"]=len(r["kernels"]); r.pop("kernels")
  return {"total_step_dur":round(tot,4), "by_owner":rows}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rec={"verdict":None,"scope":"Phase N4: whole-decode-step per-kernel attribution (native dynamic-S vs owned)","ckpts":CKPTS}
  data={}
  for ctx in CKPTS:
    for route in ROUTES:
      cap=_capture(route, ctx); data[f"{route}_ctx{ctx}"]=cap
  rec["raw"]={k:(v.get("per_kernel") and _step_table(v) or v) for k,v in data.items()}
  # token gates (native vs owned tokens per ctx)
  gates={}
  for ctx in CKPTS:
    n=data.get(f"native_dyn_s_ctx{ctx}",{}).get("warm_tokens"); o=data.get(f"owned_ctx{ctx}",{}).get("warm_tokens")
    gates[f"ctx{ctx}_token_match"]= (n==o and n is not None)
  rec["gates"]=gates
  # dominant remaining delta on the native route (largest owner by pct), per ctx
  rec["native_step_breakdown"]={f"ctx{ctx}": rec["raw"].get(f"native_dyn_s_ctx{ctx}") for ctx in CKPTS}
  rec["owned_step_breakdown"]={f"ctx{ctx}": rec["raw"].get(f"owned_ctx{ctx}") for ctx in CKPTS}
  def top(route,ctx):
    t=rec["raw"].get(f"{route}_ctx{ctx}")
    return t["by_owner"][0] if (t and t.get("by_owner")) else None
  rec["top_owner_native"]={f"ctx{ctx}": top("native_dyn_s",ctx) for ctx in CKPTS}
  # NATIVE-vs-OWNED delta by owner (ctx512 = primary decode regime). The biggest POSITIVE delta (native spends MORE
  # GPU-compute than owned in that class) is the highest-value target. Shared classes (GEMV/FFN) ~cancel -> not a delta.
  pc=CKPTS[0]
  def owner_dur(route, ctx):
    t=rec["raw"].get(f"{route}_ctx{ctx}"); return {r["owner"]:r["dur"] for r in t["by_owner"]} if (t and t.get("by_owner")) else {}
  nd, od = owner_dur("native_dyn_s", pc), owner_dur("owned", pc)
  deltas=sorted([{"owner":o,"native":round(nd.get(o,0),3),"owned":round(od.get(o,0),3),"delta":round(nd.get(o,0)-od.get(o,0),3)}
                 for o in set(nd)|set(od)], key=lambda r:-r["delta"])
  rec["native_minus_owned_by_owner_ctx%d"%pc]=deltas
  tn=top("native_dyn_s",pc)
  m={"native_attn_tile":"N5A","owned_attn_tile":"N5A","attn_gmax":"N5B","attn_combine":"N5B","q4k_gemv_ffn":"N5C","copy_cast":"N5D","generated_reduce":"N5D","generated_elementwise":"N5D","other":"N5D"}
  ok=all(gates.values()) and tn is not None and deltas
  if not ok: rec["verdict"]="AMD_ISA_PHASE_N4_BLOCKED_TOKEN_MATCH" if not all(gates.values()) else "AMD_ISA_PHASE_N4_BLOCKED_ROUTE_ATTRIBUTION"
  else:
    rec["verdict"]="AMD_ISA_PHASE_N4_PASS_WHOLE_STEP_ATTRIBUTION_PINNED"
    top_delta=deltas[0]
    rec["selected_n5_branch"]=m.get(top_delta["owner"],"N5E")
    rec["top_bottleneck"]=(f"largest native-vs-owned GPU-compute delta at ctx{pc}: {top_delta['owner']} "
      f"(native {top_delta['native']} vs owned {top_delta['owned']}, delta {top_delta['delta']}). "
      f"Native top single owner: {tn['owner']} {tn['pct_of_step']}% of native GPU-compute.")
    rec["caveat"]="eager per-kernel GPU-COMPUTE share != JIT W==D WALL share (N3F: tile ~10% of wall step). The delta selects WHERE native does more GPU work than owned; verify W==D moves in N5."
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  md=[f"# Phase N4 whole-step attribution\n",f"**Verdict:** {rec['verdict']}  ",f"**Selected N5 branch:** {rec.get('selected_n5_branch','-')}  ",f"**Top:** {rec.get('top_bottleneck','-')}\n"]
  for ctx in CKPTS:
    md.append(f"\n## ctx{ctx} native dynamic-S (by owner)\n| owner | dur/step | % | n |\n|---|---|---|---|")
    t=rec["raw"].get(f"native_dyn_s_ctx{ctx}")
    if t:
      for r in t["by_owner"]: md.append(f"| {r['owner']} | {r['dur']} | {r['pct_of_step']}% | {r['n_kernels']} |")
      md.append(f"| **total** | **{t['total_step_dur']}** | | |")
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec=main(); print(json.dumps({k:rec.get(k) for k in ("verdict","selected_n5_branch","top_bottleneck","gates")},indent=2)); print("\nPHASE_N4",rec["verdict"])
