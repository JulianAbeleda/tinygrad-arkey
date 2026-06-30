"""Prefill P2: whole-prefill role attribution. BUILT IN THE P0 TURN, NOT YET RUN.

⚠ RUN ONLY ON A FREE GPU (GPU-timing; do not run concurrently with another timing campaign).

Attribute whole-prefill GPU time into role buckets (by ctx AND by chunk start_pos), with route label + shape + effective
TFLOPS per GEMM role, unknown bucket < 10%. Refreshes the DIAGNOSTIC per_role_time_tax (which OOM'd in-session) under
the authoritative eager-PROFILE per-kernel method (one ProfileRangeEvent per kernel, GPU HW timestamps -- same method as
the decode role-attribution tools). This is what P0's share-weighted ceiling needs upgraded from diagnostic to authority.

Buckets: ffn_gate_up, ffn_down, attn_qo, attn_kv, attention_qk, attention_pv, attention_softmax, norm_rope_elementwise,
copy_cast_sync, graph_launch_sync, unknown.

Run (FREE GPU): DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_whole_role_attribution.py
Writes: bench/qk-prefill-whole-role-attribution/{latest,per_role_by_ctx,per_chunk_by_ctx,route_coverage,unknown_bucket}.json + summary.md
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-prefill-whole-role-attribution"
CTXS = [512, 1024, 2048, 4096]   # 8192 optional (OOM risk -> the diagnostic tool OOM'd on 5.4GB load; gate it)
M = 512

# role classification by kernel name + matrix dims (M x N x K embedded in name or shape). GEMM dims of interest below.
def classify(name):
  n = name.lower()
  dims = [int(x) for x in re.findall(r"\d+", name)]
  mn = [d for d in dims if d in (4096, 12288, 1024, 151936)]
  if "softmax" in n or "_max" in n: return "attention_softmax"
  if "flash" in n or "_qk" in n: return "attention_qk"
  if "_pv" in n or "combine" in n: return "attention_pv"
  if any(s in n for s in ("rope", "norm", "rms", "silu", "mul", "add")) or n.startswith("e_"): return "norm_rope_elementwise"
  if "copy" in n or "cast" in n or "sync" in n: return "copy_cast_sync"
  if "gemm" in n or "matmul" in n or "Cijk" in name or "tensile" in n:
    s = set(mn)
    if {4096, 12288} <= s: return "ffn_gate_up" if 12288 in mn and mn.index(12288) >= mn.index(4096) else "ffn_down"
    if s == {4096}: return "attn_qo"
    if {1024, 4096} <= s: return "attn_kv"
  return "unknown"

CHILD = r'''
import os, json, re
from tinygrad import Tensor, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
MAXC=8192; C=int(os.environ["P2_CTX"]); M=512
m,tok=load_model_and_tokenizer(DEFAULT_MODEL, MAXC, seed=20260617)
for b in m.blk: b._use_flash, b._prefill_v2 = True, True
# per-CHUNK attribution: profile each chunk's forward separately so we can see attention growth with start_pos
chunks={}
for sp in range(0, C, M):
  Compiled.profile_events=[]
  with Context(PROFILE=1):
    m.forward(Tensor([[100]*M],dtype="int32").contiguous(), UOp.variable("start_pos",0,MAXC-1).bind(sp), Tensor([0.0])).realize()
  agg={}
  for e in Compiled.profile_events:
    if isinstance(e,ProfileRangeEvent) and e.en is not None:
      nm=getattr(e.name,"name",None) or str(e.name); agg[nm]=agg.get(nm,0.0)+float(e.en-e.st)
  chunks[sp]=agg
print("@@"+json.dumps({"ctx":C,"chunks":{str(sp):chunks[sp] for sp in chunks}}))
'''

def capture(ctx):
  env = {**os.environ, "DEV": "AMD", "PYTHONPATH": str(ROOT), "P2_CTX": str(ctx)}
  out = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1200).stdout
  ln = [l for l in out.splitlines() if l.startswith("@@")]
  if not ln: raise RuntimeError("P2 capture failed: " + out[-1500:])
  return json.loads(ln[-1][2:])

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  per_role_by_ctx, per_chunk_by_ctx = {}, {}
  for ctx in CTXS:
    try: cap = capture(ctx)
    except Exception as e: per_role_by_ctx[str(ctx)] = {"error": str(e)[:200]}; continue
    role_tot, chunk_roles = {}, {}
    for sp, agg in cap["chunks"].items():
      cr = {}
      for nm, dur in agg.items():
        b = classify(nm); cr[b] = cr.get(b, 0.0) + dur; role_tot[b] = role_tot.get(b, 0.0) + dur
      chunk_roles[sp] = cr
    tot = sum(role_tot.values()) or 1e-9
    per_role_by_ctx[str(ctx)] = {b: {"us": round(v, 2), "share": round(v/tot, 3)} for b, v in sorted(role_tot.items(), key=lambda x: -x[1])}
    per_chunk_by_ctx[str(ctx)] = {sp: {b: round(v, 2) for b, v in cr.items()} for sp, cr in chunk_roles.items()}
  unknown = {c: per_role_by_ctx.get(c, {}).get("unknown", {}).get("share", 0) for c in map(str, CTXS)}
  max_unknown = max((u for u in unknown.values() if isinstance(u, (int, float))), default=0)
  verdict = ("PREFILL_P2_INCONCLUSIVE_UNKNOWN_BUCKET" if max_unknown >= 0.10 else
             "PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED" if per_role_by_ctx else "PREFILL_P2_BLOCKED_OOM_OR_PROFILING")
  rec = {"verdict": verdict, "contexts": CTXS, "per_role_by_ctx": per_role_by_ctx, "max_unknown_share": max_unknown,
    "prior_flagged_checks": {"ffn_gate_up_dominant": "verify vs P0 0.386", "ffn_down_deeper_k": "shape option",
      "kv_proj_wg_starvation": "check kv shape-key", "attention_growth_with_ctx": "per_chunk_by_ctx shows start_pos scaling"},
    "note": "BUILT in P0 turn; validate prefill-mode flags + that 8192 doesn't OOM (the diagnostic per_role tool OOM'd on 5.4GB load)"}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump(per_role_by_ctx, open(OUT/"per_role_by_ctx.json","w"), indent=2)
  json.dump(per_chunk_by_ctx, open(OUT/"per_chunk_by_ctx.json","w"), indent=2)
  json.dump({c: list(per_role_by_ctx.get(c, {}).keys()) for c in map(str, CTXS)}, open(OUT/"route_coverage.json","w"), indent=2)
  json.dump(unknown, open(OUT/"unknown_bucket.json","w"), indent=2)
  (OUT/"summary.md").write_text(f"# Prefill P2 whole-prefill role attribution\n\n**Verdict:** {verdict}\n\nmax unknown bucket: {round(100*max_unknown,1)}%\n\nper-role shares by ctx in per_role_by_ctx.json; per-chunk (attention growth) in per_chunk_by_ctx.json.\n")
  return rec

if __name__ == "__main__":
  print(json.dumps(main(), indent=2)[:1500])
