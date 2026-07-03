"""Prefill P2 — whole-prefill role attribution, REBUILT on env-PROFILE=1 ProfileRangeEvent capture (the only method that
works for the PREFILL_V2 path; Context(PROFILE=1) yields nothing). Profiles a warmed chunk forward at start_pos 0 and
3584 (to show ctx growth), aggregates per-kernel GPU time, buckets by the self-labeled prefill_graph_gemm_M_N_K names
+ E_/r_/flash kernels, computes effective TFLOPS per GEMM role vs the BLAS ceilings.

Role map (M=512 chunk): prefill_graph_gemm_512_12288_4096=ffn_gate_up, _512_4096_12288=ffn_down, _512_4096_4096=attn_qo,
_512_1024_4096=attn_kv.  Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk/prefill_whole_role_attribution.py
Writes: bench/qk-prefill-whole-role-attribution/{latest,summary.md,per_role_by_ctx,per_chunk_by_ctx,route_coverage,unknown_bucket}.json
"""
import os, sys, json, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-prefill-whole-role-attribution"
BLAS = {"ffn_gate_up": 69.8, "ffn_down": 70.9, "attn_qo": 76.7, "attn_kv": 51.8}
GEMM_NK = {"ffn_gate_up": (12288, 4096), "ffn_down": (4096, 12288), "attn_qo": (4096, 4096), "attn_kv": (1024, 4096)}

CHILD = r'''
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
SP=int(os.environ["P2_SP"])
m.forward(chunk,SP,temp).realize(); dev.synchronize()    # warm/compile
Compiled.profile_events=[]
m.forward(chunk,SP,temp).realize(); dev.synchronize()
agg={}; calls={}
for e in Compiled.profile_events:
  if isinstance(e,ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name,"name",None) or str(e.name)
    if nm.startswith("TracingKey"): continue
    agg[nm]=agg.get(nm,0.0)+float(e.en-e.st); calls[nm]=calls.get(nm,0)+1
print("@@"+json.dumps({"sp":SP,"per_kernel":{k:{"dur":round(agg[k],3),"calls":calls[k]} for k in agg}}))
'''

def _bucket(nm):
  n = nm.lower()
  if nm.startswith("prefill_graph_gemm_512_12288_4096"): return "ffn_gate_up"
  if nm.startswith("prefill_graph_gemm_512_4096_12288"): return "ffn_down"
  if nm.startswith("prefill_graph_gemm_512_4096_4096"): return "attn_qo"
  if nm.startswith("prefill_graph_gemm_512_1024_4096"): return "attn_kv"
  if "gemm" in n and ("12288" in n): return "ffn_gate_up" if n.index("12288") < n.rindex("_") else "ffn_down"
  if "flash" in n or "attn" in n:
    if "soft" in n or "max" in n or "sum" in n: return "attention_softmax"
    if "pv" in n or "_v" in n: return "attention_pv"
    return "attention_qk"
  if "copy" in n or "cast" in n or n.startswith("d_"): return "copy_cast_sync"
  if n.startswith("e_"): return "norm_rope_elementwise"     # elementwise: silu/gate-mul/rope/residual/norm-affine
  if n.startswith("r_"): return "norm_rope_elementwise"     # reductions: RMSNorm/softmax stats (prefill non-GEMM)
  return "unknown"

def run_sp(sp):
  env = {**os.environ, "DEV": "AMD", "PYTHONPATH": str(ROOT), "PREFILL_V2": "1", "PROFILE": "1", "P2_SP": str(sp)}
  r = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=900)
  ln = [l for l in r.stdout.splitlines() if l.startswith("@@")]
  return json.loads(ln[-1][2:]) if ln else {"sp": sp, "per_kernel": {}, "error": r.stderr[-700:]}

def attribute(per_kernel):
  buckets = {}
  for nm, v in per_kernel.items():
    b = _bucket(nm); e = buckets.setdefault(b, {"dur": 0.0, "calls": 0, "kernels": set()})
    e["dur"] += v["dur"]; e["calls"] += v["calls"]; e["kernels"].add(nm)
  tot = sum(e["dur"] for e in buckets.values()) or 1e-9
  out = {}
  for b, e in buckets.items():
    row = {"pct_gpu": round(100*e["dur"]/tot, 2), "dur_us": round(e["dur"], 2), "calls": e["calls"], "n_kernels": len(e["kernels"])}
    if b in GEMM_NK:
      n, k = GEMM_NK[b]; flop = 2*512*n*k
      # per-call time (us) -> TFLOPS; calls includes 36 layers (gate_up/down) or per-proj; use mean per call
      tflops = flop / (e["dur"]/max(1, e["calls"]) * 1e-6) / 1e12
      row["eff_tflops"] = round(tflops, 1); row["blas_ceiling"] = BLAS[b]; row["pct_of_blas"] = round(100*tflops/BLAS[b], 1)
    out[b] = row
  return {"total_dur_us": round(tot, 1), "buckets": out}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  caps = {sp: run_sp(sp) for sp in (0, 3584)}
  attr = {str(sp): attribute(caps[sp]["per_kernel"]) for sp in caps}
  unknown = {sp: attr[str(sp)]["buckets"].get("unknown", {}).get("pct_gpu", 0.0) for sp in caps}
  worst_unknown = max(unknown.values())
  gemm_pct = {sp: sum(attr[str(sp)]["buckets"].get(b, {}).get("pct_gpu", 0) for b in GEMM_NK) for sp in caps}
  gate_up_pct = {sp: attr[str(sp)]["buckets"].get("ffn_gate_up", {}).get("pct_gpu", 0) for sp in caps}
  gate_up_dominant = all(gate_up_pct[sp] == max(attr[str(sp)]["buckets"].get(b, {}).get("pct_gpu", 0) for b in GEMM_NK) for sp in caps)
  compute_bound = all(gemm_pct[sp] >= 50 for sp in caps)
  if any("error" in caps[sp] and not caps[sp]["per_kernel"] for sp in caps):
    verdict = "PREFILL_P2_BLOCKED_OOM_OR_PROFILING"
  elif worst_unknown >= 10:
    verdict = "PREFILL_P2_INCONCLUSIVE_UNKNOWN_BUCKET"
  else:
    verdict = "PREFILL_P2_PASS_ROLE_ATTRIBUTION_PINNED"
  rec = {"verdict": verdict, "method": "env PROFILE=1 ProfileRangeEvent per-kernel, chunk forward @sp=0 and 3584",
    "unknown_pct_by_sp": unknown, "gemm_pct_by_sp": {str(k): round(v,1) for k,v in gemm_pct.items()},
    "gate_up_dominant": gate_up_dominant, "compute_bound_measured": compute_bound,
    "p0_diagnostic_shares": {"ffn_gate_up": 0.386, "ffn_down": 0.217, "attn_qo": 0.148, "attn_kv": 0.096},
    "attribution_by_sp": attr,
    "prior_flag_checks": {
      "ffn_gate_up_dominant": gate_up_dominant,
      "attention_grows_with_ctx": (attr["3584"]["buckets"].get("attention_qk", {}).get("pct_gpu", 0) + attr["3584"]["buckets"].get("attention_pv", {}).get("pct_gpu", 0))
                                   > (attr["0"]["buckets"].get("attention_qk", {}).get("pct_gpu", 0) + attr["0"]["buckets"].get("attention_pv", {}).get("pct_gpu", 0)),
      "kv_proj_present": all("attn_kv" in attr[str(sp)]["buckets"] for sp in caps)},
    "p3_candidate_families": ["pipe_tm2_tn2 (revalidated +11-19% in P1)", "ffn_gate_up GEMM-tile/WMMA (dominant role)",
      "ffn_down deeper-K", "kv_proj WG-starvation", "wmma_mfma_tile_search", "native_isa_generated_gemm"]}
  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump({sp: {b: attr[str(sp)]["buckets"][b]["pct_gpu"] for b in attr[str(sp)]["buckets"]} for sp in caps}, open(OUT/"per_role_by_ctx.json","w"), indent=2)
  json.dump({str(sp): attr[str(sp)] for sp in caps}, open(OUT/"per_chunk_by_ctx.json","w"), indent=2)
  json.dump({sp: {b: list(attr[str(sp)]["buckets"][b].keys()) for b in attr[str(sp)]["buckets"]} for sp in caps}, open(OUT/"route_coverage.json","w"), indent=2)
  json.dump({"unknown_pct_by_sp": unknown, "worst": worst_unknown, "unknown_kernels": {str(sp): sorted(n for n in caps[sp]["per_kernel"] if _bucket(n)=="unknown")[:25] for sp in caps}}, open(OUT/"unknown_bucket.json","w"), indent=2)
  lines = [f"# Prefill P2 whole-prefill role attribution\n\n**Verdict:** {verdict}\n",
    f"compute-bound (GEMM>=50%): {compute_bound} ({ {str(k):round(v,1) for k,v in gemm_pct.items()} }); gate_up dominant: {gate_up_dominant}; worst unknown: {worst_unknown}%\n",
    "## Role wall-stack (% GPU time) + GEMM effective TFLOPS\n| role | sp=0 % | sp=3584 % | eff TFLOPS@0 | BLAS | % of BLAS |", "|---|---|---|---|---|---|"]
  allb = sorted(set(attr["0"]["buckets"]) | set(attr["3584"]["buckets"]), key=lambda b: -attr["0"]["buckets"].get(b, {}).get("pct_gpu", 0))
  for b in allb:
    r0 = attr["0"]["buckets"].get(b, {}); r1 = attr["3584"]["buckets"].get(b, {})
    lines.append(f"| {b} | {r0.get('pct_gpu','—')} | {r1.get('pct_gpu','—')} | {r0.get('eff_tflops','—')} | {r0.get('blas_ceiling','—')} | {r0.get('pct_of_blas','—')} |")
  (OUT/"summary.md").write_text("\n".join(lines))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "unknown_pct_by_sp": rec["unknown_pct_by_sp"], "gemm_pct_by_sp": rec["gemm_pct_by_sp"],
    "gate_up_dominant": rec["gate_up_dominant"], "compute_bound": rec["compute_bound_measured"]}, indent=2))
  print("\nP2", rec["verdict"])
