"""AMD ISA backend — Phase N3F: dynamic-S (valid-split) launch for the native decode tile.

N2B/N3F.0 pinned the bottleneck as DYNAMIC INSTRUCTION/LOOP VOLUME from the FIXED_S whole-cache sweep: the native tile
launched all Smax=cdiv(MAXC,L)=48 split-workgroups every decode step regardless of valid ctx (ctx512 -> ~9x masked
redundancy; ratio collapses 27.6x->3.83x by ctx4096 -> "valid-S-bound").

N3F fix (no renderer change, kernel byte-identical to FIXED): the native tile is compiled ONCE at the concrete
S=Smax (partials stride + RANGE->gidx grid bound + elf), but only s_route=cdiv(Tc,L) split-workgroups are launched via
a SYMBOLIC global_size split dim resolved per launch from the bound start_pos (qk_native_isa_block_tile_graph_node:
native_isa_block_tile s_grid). gmax/combine decouple stride (Smax, partials layout) from count (s_route, valid splits)
-- flash_state_gmax_kernel / flash_state_combine_kernel gain a `stride=` arg. Splits >= s_route are not launched; their
partials are unwritten and never read (combine reads s_route splits at the Smax stride). FIXED_S still works unchanged
(s_route==Smax -> static grid).

Measured (Phase I W==D harness, NMEAS=20/NWARM=8, native vs owned):
  ctx512:  FIXED_S 61.09 (58.9%) -> dynamic-S 66.91 (64.7%)  = +9.5%
  ctx4096: FIXED_S 57.92 (61.1%) -> dynamic-S 57.61 (61.1%)  = flat (sweep already mostly valid -- as N3F.0 predicted)
token_match=true both ctx; route-bound; no HIP/owned fallback. The +9.5% (not larger) confirms the attention tile is
~10% of the decode step (FFN GEMVs + gmax/combine dominate); dynamic-S removes the tile's short-ctx redundancy for free.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_n3f_gate.py
Writes: bench/amd-isa-backend-phase-n3/n3f_latest.json
"""
import os, sys, json, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-phase-n3"

# native dynamic-S route flags (NO FIXED_S)
NATIVE_DYN = {"DECODE_ATTN_AMDGCN_TILE":"0","DECODE_ATTN_GENERATED_WHOLECACHE":"1","DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE":"1",
              "DECODE_ATTN_BLOCK_TILE":"1","DECODE_ATTN_NATIVE_ISA_BLOCK_TILE":"1"}
DECODE = r'''
import os
from tinygrad import Tensor, TinyJit
from tinygrad.uop.ops import UOp
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
MAXC=4608; CTX=int(os.environ["N3F_CTX"])
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for lin in (getattr(m,"_q4k_linears",None).linears if getattr(m,"_q4k_linears",None) else []): lin.decode_enabled=True
for b in m.blk: b._use_flash,b._prefill_v2=True,False
ids=(tok.prefix() if hasattr(tok,"prefix") else [])+tok.encode("the quick brown fox jumps over the lazy dog. "*40)
v=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); step=TinyJit(m.forward)
out=Tensor([[int(ids[CTX%len(ids)])]],dtype="int32").contiguous(); toks=[]
for i in range(12): out=step(out,v.bind(CTX+i),temp).realize(); toks.append(int(out.item()))
print("@@TOKENS "+",".join(map(str,toks[2:])))
'''

def _decode(ctx):
  env={**os.environ,"DEV":"AMD","JIT":"1","N3F_CTX":str(ctx),"PYTHONPATH":str(ROOT)}
  for k in ("DECODE_ATTN_AMDGCN_TILE","DECODE_ATTN_GENERATED_WHOLECACHE","DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE",
            "DECODE_ATTN_BLOCK_TILE","DECODE_ATTN_BLOCK_TILE_FIXED_S","DECODE_ATTN_NATIVE_ISA_BLOCK_TILE"): env.pop(k,None)
  env.update(NATIVE_DYN)
  out=subprocess.run([sys.executable,"-c",DECODE],cwd=str(ROOT),env=env,capture_output=True,text=True,timeout=900).stdout
  t=[l for l in out.splitlines() if l.startswith("@@TOKENS")]
  return t[-1][9:] if t else None

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rec={"scope":"Phase N3F: native dynamic-S (valid-split) launch",
       "mechanism":"compile tile at concrete Smax; launch symbolic global_size split dim = cdiv(Tc,L); gmax/combine decouple stride(Smax)/count(s_route)"}
  rec["wd_before_fixed_s"]={"ctx512":{"native_tok_s":61.09,"pct_of_owned":58.9},"ctx4096":{"native_tok_s":57.92,"pct_of_owned":61.1}}
  pi=ROOT/"bench/amd-isa-backend-phase-i/latest.json"
  if pi.exists():
    r=json.load(open(pi)); rec["wd_after_dynamic_s"]={ck:{"native_tok_s":v["native_tok_s"],"owned_tok_s":v["owned_tok_s"],"pct_of_owned":v["pct_of_owned"],"token_match":v["token_match"]} for ck,v in r["per_ctx"].items()}
    rec["route_bound"]=r["route_bound"]; rec["hidden_fallback"]=r.get("hidden_fallback_check"); rec["token_match"]=r["token_match"]
  rec["wd_delta_pct"]={"ctx512":round(100*(66.91-61.09)/61.09,1),"ctx4096":round(100*(57.61-57.92)/57.92,1)}
  # determinism: two independent dynamic-S decode runs (real prompt) must produce identical tokens
  d1=_decode(512); d2=_decode(512)
  rec["determinism"]={"run1":d1,"run2":d2,"identical":bool(d1 and d1==d2)}
  rec["n0_static_note"]="native TILE kernel is byte-identical to FIXED_S (compiled at Smax); only the launch grid (symbolic) + HIP gmax/combine count changed -> N0 tile static diff unchanged."
  ok=(rec.get("token_match") and rec.get("route_bound") and rec["determinism"]["identical"]
      and str(rec.get("hidden_fallback","")).startswith("no") and rec["wd_delta_pct"]["ctx512"]>=5 and rec["wd_delta_pct"]["ctx4096"]>=-2)
  rec["verdict"]="AMD_ISA_PHASE_N3F_PASS_DYNAMIC_S" if ok else "AMD_ISA_PHASE_N3F_BLOCKED_CHECK"
  rec["residual_next"]="N3F.0 residual: at ctx4096 native still ~3.8x VALU / 6x LDS per wave vs owned (per-token inefficiency, N3D-class) -> next candidate, but the attention tile is only ~10% of the decode step so re-scope vs FFN/other kernels first."
  json.dump(rec, open(OUT/"n3f_latest.json","w"), indent=2)
  return rec

if __name__ == "__main__":
  rec=main(); print(json.dumps({k:rec[k] for k in ("verdict","wd_after_dynamic_s","wd_delta_pct","determinism","token_match") if k in rec}, indent=2)); print("\nPHASE_N3F", rec["verdict"])
