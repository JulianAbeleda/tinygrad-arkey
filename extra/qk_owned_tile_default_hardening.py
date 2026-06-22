#!/usr/bin/env python3
"""Step 1 default-decision hardening (docs/owned-tile-post-promotion-four-step-scope-20260623.md).
True IN-PROCESS A/B: one model, two jits (route off vs on, toggled via getenv.cache_clear at capture),
interleaved timing -> tightest spread. Plus route-fired + fallback + default-unchanged checks."""
import json, os, pathlib, statistics, time, sys
os.environ.setdefault("DEV","AMD"); os.environ.setdefault("JIT","1")
os.environ.setdefault("Q4K_GEMV_WARP","1"); os.environ.setdefault("Q4K_GEMV_WARP_DOWN","1")
from tinygrad import Tensor, UOp, TinyJit
from tinygrad.helpers import getenv
from tinygrad.uop.ops import Ops
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL
OUT=pathlib.Path("bench/qk-owned-amdgcn-tile-post-promotion"); OUT.mkdir(parents=True,exist_ok=True)
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for lin in (getattr(m,'_q4k_linears',None).linears if getattr(m,'_q4k_linears',None) else []): lin.decode_enabled=True
for b in m.blk: b._use_flash,b._prefill_v2=True,False
v=UOp.variable("start_pos",0,4607); temp=Tensor([0.0]); tk=Tensor([[100]],dtype="int32").contiguous()
def build(flag, ck):
    os.environ["DECODE_ATTN_AMDGCN_TILE"]=flag; getenv.cache_clear()
    j=TinyJit(m.forward)
    for _ in range(10): j(tk, v.bind(ck), temp).realize().item()  # warm + capture
    n=sum('owned_flash' in str(getattr(u.arg,'name','')) for u in j.captured.linear.toposort() if u.op is Ops.PROGRAM) if j.captured else -1
    return j, n
res={"date":"2026-06-23","phase":"OWNER_DEFAULT_HARDENING","gpu":"gfx1100","method":"in-process A/B, interleaved, 60 samples/mode/ctx","rows":[]}
for ck in (1024,4096):
    joff,noff=build("0",ck); jon,non=build("1",ck)
    toff,ton=[],[]
    for i in range(60):
        t0=time.perf_counter(); joff(tk,v.bind(ck),temp).realize().item(); toff.append(time.perf_counter()-t0)
        t0=time.perf_counter(); jon(tk,v.bind(ck),temp).realize().item(); ton.append(time.perf_counter()-t0)
    moff,mon=statistics.median(toff),statistics.median(ton)
    sp_off=(max(toff[5:])-min(toff[5:]))/moff*100; sp_on=(max(ton[5:])-min(ton[5:]))/mon*100
    delta=100*(1/mon-1/moff)/(1/moff)
    res["rows"].append({"ctx":ck,"gqa_tok_s":round(1/moff,1),"owned_tok_s":round(1/mon,1),"delta_pct":round(delta,1),
                        "owned_flash_nodes_off":noff,"owned_flash_nodes_on":non,"route_fired":non>0 and noff==0,
                        "spread_off_pct":round(sp_off,1),"spread_on_pct":round(sp_on,1)})
    print(f"ctx{ck}: gqa={1/moff:.1f} owned={1/mon:.1f} tok/s delta={delta:+.1f}% | route_fired={non>0 and noff==0} (off={noff},on={non}) spread off/on={sp_off:.0f}/{sp_on:.0f}%", file=sys.stderr)
# fallback: wrong head config (simulate unsupported) -> route must NOT fire (guard B/Hd/Hq). Use ctx but flag on with a shape guard miss is hard in-model;
# instead assert the guard: at ctx<512 the route must not fire (falls back).
joff2,nlow=build("1",256)
res["fallback_ctx256_route_off"]= (nlow==0)
print(f"fallback: ctx256 (below MIN_CTX) route fired nodes={nlow} -> falls back={nlow==0}", file=sys.stderr)
allfired=all(r["route_fired"] for r in res["rows"]); gates=all((r["ctx"]==1024 and r["delta_pct"]>=5) or (r["ctx"]==4096 and r["delta_pct"]>=7) for r in res["rows"])
res["verdict"]="OWNER_DEFAULT_READY" if (allfired and gates and res["fallback_ctx256_route_off"]) else "OWNER_DEFAULT_KEEP_OFF_PENDING_MORE_BAKE"
res["default_flip_authorized"]=False; res["default_on"]=False
(OUT/"default_decision.json").write_text(json.dumps(res,indent=2))
print("VERDICT:",res["verdict"],"(default_on stays false; no flip authorized)", file=sys.stderr)
