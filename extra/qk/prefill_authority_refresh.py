"""Prefill P1 — authority baseline refresh, REBUILT on the synced whole-prefill methodology (extra/qk/prefill_whole_synced.py,
the recovered authority that reproduces ~3595/3503/3252/2822 @512/1024/2048/4096).

Methodology (NOT end-to-end generate, NOT a non-JIT eager loop): per arm (env flags set in a FRESH subprocess, since the
graph-gemm route is chosen at compile time), TinyJit(m.forward) + synced bursts (dev.synchronize, K=8, min-of-3) at
concrete start_pos {0,512,1024,2048,3584}; whole-prefill@L = sum of (interpolated) chunk times for chunks 0..L step 512.

Arms: current_default (PREFILL_V2=1), eightwave_off (PREFILL_GEMM_8WAVE=0), pipe_tm2_tn2 (PREFILL_GEMM_PIPELINE=1 TM=2 TN=2).
ctx8192 exceeds the harness max_context=4608 (would need start_pos 7680) -> recorded unsupported, not measured.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/prefill_authority_refresh.py
Writes: bench/qk-prefill-authority-refresh/{latest,summary.md,current_default_by_ctx,eightwave_guard,noise_profile,route_attribution}.json
"""
import os, sys, json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-prefill-authority-refresh"
ARMS = {
  "current_default": {},
  "eightwave_off": {"PREFILL_GEMM_8WAVE": "0"},
  "pipe_tm2_tn2": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "2", "PREFILL_GEMM_PIPELINE_TN": "2"},
}
CTXS = [512, 1024, 2048, 4096]   # 8192 needs max_context>=8192 (harness loads 4608) -> unsupported here

# TIMING child: clean synced bursts, NO profiling (profiling inflates tok/s).
TIMING_CHILD = r'''
import os, json, time, bisect
os.environ.setdefault("PREFILL_V2","1")
from tinygrad import Tensor, Device, TinyJit
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
from tinygrad.llm.model import PREFILL_GRAPH_GEMM
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
temp=Tensor([0.0]); N=512
chunk=Tensor([[(i*7)%1000 for i in range(N)]],dtype="int32").contiguous()
chunk_ms={}; spreads={}
for sp in [0,512,1024,2048,3584]:
  j=TinyJit(m.forward)
  for _ in range(4): j(chunk, sp, temp).realize()
  dev.synchronize()
  ts=[]
  for _ in range(3):
    dev.synchronize(); t0=time.perf_counter()
    for _ in range(8): j(chunk, sp, temp).realize()
    dev.synchronize(); ts.append((time.perf_counter()-t0)/8*1e3)
  chunk_ms[sp]=min(ts); spreads[sp]=round((max(ts)-min(ts))/min(ts)*100,1)
pts=sorted(chunk_ms.items()); xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
def interp(s):
  if s<=xs[0]: return ys[0]
  if s>=xs[-1]: return ys[-1]
  i=bisect.bisect_right(xs,s)-1; return ys[i]+(ys[i+1]-ys[i])*(s-xs[i])/(xs[i+1]-xs[i])
whole={L:round(L/sum(interp(s) for s in range(0,L,512))*1e3,1) for L in [512,1024,2048,4096]}
print("@@"+json.dumps({"graph_gemm":bool(PREFILL_GRAPH_GEMM),
  "chunk_ms":{str(k):round(v,2) for k,v in chunk_ms.items()},"chunk_spread_pct":{str(k):spreads[k] for k in spreads},
  "whole_prefill_tok_s":{str(k):v for k,v in whole.items()}}))
'''
# ROUTE child: PROFILE=1 (env, set at launch) -> ProfileRangeEvent kernel names for one warmed chunk forward.
ROUTE_CHILD = r'''
import os, json
os.environ.setdefault("PREFILL_V2","1")
from tinygrad import Tensor, Device
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
import tinygrad.runtime.ops_amd  # noqa
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
temp=Tensor([0.0]); chunk=Tensor([[(i*7)%1000 for i in range(512)]],dtype="int32").contiguous()
m.forward(chunk,0,temp).realize(); dev.synchronize()      # warm/compile
Compiled.profile_events=[]
m.forward(chunk,0,temp).realize(); dev.synchronize()
names=sorted({(getattr(e.name,"name",None) or str(e.name)) for e in Compiled.profile_events if isinstance(e,ProfileRangeEvent)})
names=[n for n in names if not n.startswith("TracingKey")]
print("@@"+json.dumps({"route_kernels":names,"n_route_kernels":len(names),"external_kernel_fallback":any("tensile" in n.lower() or "rocblas" in n.lower() for n in names)}))
'''

def _run(code, env_extra, profile):
  env = {**os.environ, "DEV": "AMD", "PYTHONPATH": str(ROOT), "PREFILL_V2": "1", **env_extra}
  if profile: env["PROFILE"] = "1"
  r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800)
  ln = [l for l in r.stdout.splitlines() if l.startswith("@@")]
  return json.loads(ln[-1][2:]) if ln else {"error": "no @@ line", "stderr_tail": r.stderr[-700:]}

def run_arm(arm, env_extra):
  t = _run(TIMING_CHILD, env_extra, profile=False)
  rt = _run(ROUTE_CHILD, env_extra, profile=True)
  return {**t, **{k: rt.get(k) for k in ("route_kernels", "n_route_kernels", "external_kernel_fallback")}}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  res = {arm: run_arm(arm, env) for arm, env in ARMS.items()}
  cur = res.get("current_default", {})
  cur512 = cur.get("whole_prefill_tok_s", {}).get("512")
  sane = isinstance(cur512, (int, float)) and cur512 >= 3000   # must be ~3595-scale, not ~217/~150
  route_nonempty = all(res[a].get("n_route_kernels", 0) > 0 for a in res if "error" not in res[a])
  reval = {}
  for L in CTXS:
    c = cur.get("whole_prefill_tok_s", {}).get(str(L)); p = res.get("pipe_tm2_tn2", {}).get("whole_prefill_tok_s", {}).get(str(L))
    reval[str(L)] = {"current": c, "pipe_tm2_tn2": p, "delta_pct": (round((p-c)/c*100, 1) if (c and p) else None)}
  if not sane: verdict = "PREFILL_P1_BLOCKED_NOISY_OR_STALE"
  elif not route_nonempty: verdict = "PREFILL_P1_BLOCKED_ROUTE_ATTRIBUTION"
  else: verdict = "PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED"
  rec = {"verdict": verdict, "methodology": "synced whole-prefill (qk_prefill_whole_synced) per-arm subprocess; TinyJit+synced bursts K=8 min-of-3; whole@L=sum interpolated chunk times",
    "ctx_note": "8192 unsupported: harness max_context=4608 (start_pos 7680 > 4607)", "authority_ref": {"512":3597,"1024":3504,"2048":3248,"4096":2803},
    "current_default_sane_vs_3595": sane, "route_attribution_nonempty": route_nonempty,
    "by_arm": res, "pipe_tm2_tn2_revalidation_vs_current": reval}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump({"by_ctx": cur.get("whole_prefill_tok_s", {}), "chunk_ms": cur.get("chunk_ms", {})}, open(OUT/"current_default_by_ctx.json","w"), indent=2)
  ew = res.get("eightwave_off", {})
  json.dump({"current_default": cur.get("whole_prefill_tok_s",{}), "eightwave_off": ew.get("whole_prefill_tok_s",{}),
    "eightwave_gain_pct": {str(L): (round((cur.get("whole_prefill_tok_s",{}).get(str(L),0)-ew.get("whole_prefill_tok_s",{}).get(str(L),0))/ew.get("whole_prefill_tok_s",{}).get(str(L),1)*100,1) if ew.get("whole_prefill_tok_s",{}).get(str(L)) else None) for L in CTXS}}, open(OUT/"eightwave_guard.json","w"), indent=2)
  json.dump({arm: res[arm].get("chunk_spread_pct") for arm in res}, open(OUT/"noise_profile.json","w"), indent=2)
  json.dump({arm: {"n":res[arm].get("n_route_kernels"),"kernels":res[arm].get("route_kernels",[])[:40]} for arm in res}, open(OUT/"route_attribution.json","w"), indent=2)
  lines = [f"# Prefill P1 authority baseline refresh\n\n**Verdict:** {verdict}\n",
    "## Whole-prefill tok/s (synced authority)\n| arm | ctx512 | ctx1024 | ctx2048 | ctx4096 | route kernels |", "|---|---|---|---|---|---|"]
  for arm in res:
    w = res[arm].get("whole_prefill_tok_s", {})
    lines.append(f"| {arm} | {w.get('512','—')} | {w.get('1024','—')} | {w.get('2048','—')} | {w.get('4096','—')} | {res[arm].get('n_route_kernels','—')} |")
  lines += ["\n## pipe_tm2_tn2 re-validation vs current_default\n| ctx | current | pipe_tm2_tn2 | Δ% |", "|---|---|---|---|"]
  for L in CTXS:
    r = reval[str(L)]; lines.append(f"| {L} | {r['current']} | {r['pipe_tm2_tn2']} | {r['delta_pct']} |")
  lines.append(f"\nsanity (current_default@512 >= 3000): {sane} (got {cur512}); route attribution non-empty: {route_nonempty}; 8192 unsupported (max_context=4608).")
  (OUT/"summary.md").write_text("\n".join(lines))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "current_default": rec["by_arm"].get("current_default",{}).get("whole_prefill_tok_s"),
    "pipe_reval": rec["pipe_tm2_tn2_revalidation_vs_current"], "sane": rec["current_default_sane_vs_3595"]}, indent=2))
  print("\nP1", rec["verdict"])
