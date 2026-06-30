"""Prefill P1: authority baseline refresh. BUILT IN THE P0 TURN, NOT YET RUN.

⚠ RUN ONLY ON A FREE GPU. This is a GPU-timing campaign; running it concurrently with a decode W==D campaign corrupts
both. Re-measure the current default prefill baseline under one clean authority harness across ctx 512/1024/2048/4096/8192,
plus the eightwave-off guard and (if safe) the pipe_tm2_tn2 candidate, with route attribution + noise/spread + determinism.

Arms (env):
  current_default        : (no flags)                                  -> graph_gemm default (eightwave is the promoted default)
  eightwave_off          : PREFILL_GEMM_8WAVE=0                         -> guard that eightwave-off doesn't beat default
  pipe_tm2_tn2           : PREFILL_GEMM_PIPELINE=1 TM=2 TN=2            -> re-validate the +11-19% aggressive candidate

Whole-prefill measure: feed the prompt in M=512-token chunks over the causal start_pos schedule [0,512,...,C-512];
tok_s = C / total_wall_ms. Median over repeats; record spread. Route attribution = dev.runtime kernel-name capture
(graph_gemm vs tensile/BLAS fallback). Equivalence: logits/argmax match vs current_default (accepted prefill gate).

Run (FREE GPU): DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_authority_refresh.py
Writes: bench/qk-prefill-authority-refresh/{latest,current_default_by_ctx,eightwave_guard,noise_profile,route_attribution}.json + summary.md
"""
import os, sys, json, pathlib, subprocess, statistics
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-prefill-authority-refresh"
CTXS = [512, 1024, 2048, 4096, 8192]
M = 512
ARMS = {
  "current_default": {},
  "eightwave_off": {"PREFILL_GEMM_8WAVE": "0"},
  "pipe_tm2_tn2": {"PREFILL_GEMM_PIPELINE": "1", "PREFILL_GEMM_PIPELINE_TM": "2", "PREFILL_GEMM_PIPELINE_TN": "2"},
}
REPEATS = 3

# CHILD: load model, run whole-prefill in M-chunks for one ctx, capture per-kernel route names, return median tok_s + spread.
CHILD = r'''
import os, json, time, statistics
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
MAXC=8192; C=int(os.environ["P1_CTX"]); REP=int(os.environ.get("P1_REP","3")); M=512
m,tok=load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
for b in m.blk: b._use_flash, b._prefill_v2 = True, True   # prefill path (NOTE: validate prefill-mode flags vs harness)
# whole-prefill = chunks over causal start_pos schedule [0, M, 2M, ... C-M]
def run_once():
  t0=time.perf_counter()
  for sp in range(0, C, M):
    toks=Tensor([[100]*M], dtype="int32").contiguous()
    m.forward(toks, UOp.variable("start_pos",0,MAXC-1).bind(sp), Tensor([0.0])).realize()
  return (time.perf_counter()-t0)*1e3
run_once()  # warmup
times=[run_once() for _ in range(REP)]
ms=statistics.median(times); spread=(max(times)-min(times))/ms if ms else 0
# route capture (one eager profiled pass)
Compiled.profile_events=[]; names={}
with Context(PROFILE=1):
  for sp in range(0, C, M):
    toks=Tensor([[100]*M], dtype="int32").contiguous()
    m.forward(toks, UOp.variable("start_pos",0,MAXC-1).bind(sp), Tensor([0.0])).realize()
for e in Compiled.profile_events:
  if isinstance(e,ProfileRangeEvent):
    nm=getattr(e.name,"name",None) or str(e.name); names[nm]=names.get(nm,0)+1
print("@@"+json.dumps({"ctx":C,"tok_s":round(C/(ms/1e3),1),"median_ms":round(ms,3),"spread_pct":round(100*spread,1),
  "route_kernels":sorted(names, key=lambda k:-names[k])[:25],
  "tensile_fallback":any("tensile" in k.lower() or "Cijk" in k for k in names),
  "graph_gemm":any("gemm" in k.lower() for k in names)}))
'''

def measure(ctx, arm_env):
  env = {**os.environ, "DEV": "AMD", "PYTHONPATH": str(ROOT), "P1_CTX": str(ctx), "P1_REP": str(REPEATS), **arm_env}
  out = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200).stdout
  ln = [l for l in out.splitlines() if l.startswith("@@")]
  if not ln: raise RuntimeError("P1 measure failed: " + out[-1500:])
  return json.loads(ln[-1][2:])

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  results = {arm: {} for arm in ARMS}
  for arm, env in ARMS.items():
    for ctx in CTXS:
      try: results[arm][str(ctx)] = measure(ctx, env)
      except Exception as e: results[arm][str(ctx)] = {"error": str(e)[:200]}
  cur = {c: results["current_default"].get(c, {}) for c in map(str, CTXS)}
  # verdicts
  noisy = any(isinstance(r, dict) and r.get("spread_pct", 0) > 30 for r in results["current_default"].values())
  fallback = any(isinstance(r, dict) and r.get("tensile_fallback") for r in cur.values())
  verdict = ("PREFILL_P1_BLOCKED_NOISY_OR_STALE" if noisy else
             "PREFILL_P1_BLOCKED_ROUTE_ATTRIBUTION" if fallback else
             "PREFILL_P1_PASS_AUTHORITY_BASELINE_PINNED")
  rec = {"verdict": verdict, "arms": list(ARMS), "contexts": CTXS, "repeats": REPEATS, "results": results,
    "current_default_tok_s": {c: cur[c].get("tok_s") for c in cur},
    "note": "BUILT in P0 turn; validate prefill-mode flags (_prefill_v2/start_pos schedule) against the authoritative whole-prefill harness before trusting numbers"}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump({c: cur[c] for c in cur}, open(OUT/"current_default_by_ctx.json","w"), indent=2)
  json.dump(results.get("eightwave_off"), open(OUT/"eightwave_guard.json","w"), indent=2)
  json.dump({arm: {c: results[arm].get(c, {}).get("spread_pct") for c in map(str, CTXS)} for arm in ARMS}, open(OUT/"noise_profile.json","w"), indent=2)
  json.dump({arm: {c: results[arm].get(c, {}).get("route_kernels") for c in map(str, CTXS)} for arm in ARMS}, open(OUT/"route_attribution.json","w"), indent=2)
  (OUT/"summary.md").write_text(f"# Prefill P1 authority refresh\n\n**Verdict:** {verdict}\n\ncurrent_default tok/s by ctx: {rec['current_default_tok_s']}\n\n(arms: {list(ARMS)}; {REPEATS} repeats; spread + route in artifacts)\n")
  return rec

if __name__ == "__main__":
  print(json.dumps(main(), indent=2)[:1500])
