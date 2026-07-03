"""W1: weight-path route attribution + wall share. Per-kernel GPU time (eager PROFILE -> one ProfileRangeEvent per
kernel with GPU HW timestamps; the JIT graph profiles as one opaque range, so eager is required -- same method as
the decode-step attribution artifacts). Classifies each kernel through extra.qk.decode_role_profile, which
derives role / quant / shape facts from the selected GGUF tensor table instead of Qwen3-8B constants. Audit-only.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/weight_path_route_attribution.py
Run another model: QK_MODEL=/path/to/model.gguf DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/weight_path_route_attribution.py
Writes: bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
"""
import os, sys, json, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-weight-path-ceiling"
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NSTEPS = int(os.environ.get("QK_W1_STEPS", "4"))
sys.path.insert(0, str(ROOT))
from extra.qk.decode_role_profile import classify_kernel, profile_from_gguf, summarize_profile

CHILD = r'''
import os, json, re
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
MODEL=os.environ.get("QK_MODEL", DEFAULT_MODEL)
MAXC=int(os.environ.get("W1_MAX_CONTEXT", "4608")); CTX=int(os.environ["W1_CTX"]); NSTEPS=int(os.environ["W1_STEPS"])
m,tok=load_model_and_tokenizer(MODEL,MAXC,seed=20260617)
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
n=max(1,NSTEPS)
print("@@"+json.dumps({"ctx":CTX,"per_kernel":{k:{"dur_per_step":round(agg[k]/n,4),"calls_per_step":round(calls[k]/n,2)} for k in agg},"nsteps":n}))
'''

def capture(route_flags, ctx):
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "PROFILE": "1", "W1_CTX": str(ctx), "W1_STEPS": str(NSTEPS), "PYTHONPATH": str(ROOT)}
  env.update(route_flags)
  out = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=1800).stdout
  line = [l for l in out.splitlines() if l.startswith("@@")]
  return json.loads(line[-1][2:]) if line else {"failed": True}

def attribute(cap, ctx, profile):
  pk = cap["per_kernel"]; tot = sum(v["dur_per_step"] for v in pk.values()) or 1e-9
  # device-clock units: scale so total matches the measured decode wall (1/tok_s). Use unit-normalized % + relative.
  rows = []
  for nm, v in pk.items():
    c = classify_kernel(nm, profile); dur = v["dur_per_step"]
    eff_bw = None
    if c["is_weight"] and c["bytes_per_call"]:
      eff_bw = "see W2 (needs dur->seconds scale)"  # effective bw computed in W2 with the wall-time scale
    rows.append({"kernel": nm[:40], "role": c["role"], "quant": c["quant"], "route_class": c["route_class"],
                 "calls_per_step": v["calls_per_step"], "dur_per_step": dur, "pct_of_gpu_compute": round(100*dur/tot, 2),
                 "is_weight": c["is_weight"], "matdims": c["matdims"], "bytes_per_call": c["bytes_per_call"]})
  rows.sort(key=lambda r: -r["dur_per_step"])
  # aggregate by role
  byrole = {}
  for r in rows:
    e = byrole.setdefault(r["role"], {"role": r["role"], "dur": 0.0, "pct": 0.0, "quants": set(), "route_classes": set()})
    e["dur"] += r["dur_per_step"]; e["pct"] += r["pct_of_gpu_compute"]; e["quants"].add(r["quant"]); e["route_classes"].add(r["route_class"])
  for e in byrole.values(): e["dur"] = round(e["dur"], 3); e["pct"] = round(e["pct"], 1); e["quants"] = sorted(e["quants"]); e["route_classes"] = sorted(e["route_classes"])
  return {"total_dur": round(tot, 3), "by_kernel_top": rows[:20], "by_role": sorted(byrole.values(), key=lambda e: -e["dur"])}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  from extra.qk.harness_contract import DEFAULT_MODEL
  model_path = os.environ.get("QK_MODEL", DEFAULT_MODEL)
  profile = profile_from_gguf(model_path, pathlib.Path(model_path).stem)
  # shipped/default route = owned-warp Q4_K (Q4K_GEMV_WARP default-on). Capture per ctx.
  rec = {"scope": "W1 weight-path route attribution (eager PROFILE per-kernel GPU time, classified by name)",
         "route": "shipped_default (Q4K_GEMV_WARP owned-warp)", "model_profile": summarize_profile(profile),
         "classifier": "extra.qk.decode_role_profile (GGUF tensor-table driven; no 8B dimension constants)",
         "per_ctx": {}}
  for ctx in CKPTS:
    cap = capture({}, ctx)
    rec["per_ctx"][str(ctx)] = attribute(cap, ctx, profile) if "per_kernel" in cap else {"failed": cap}
  ok = all("by_role" in rec["per_ctx"][str(c)] for c in CKPTS)
  rec["verdict"] = "AMD_ISA_WEIGHT_W1_PASS_WALL_ATTRIBUTED" if ok else "AMD_ISA_WEIGHT_W1_BLOCKED_ROUTE_ATTRIBUTION"
  json.dump(rec, open(OUT / "route_attribution.json", "w"), indent=2)
  return rec

if __name__ == "__main__":
  rec = main()
  for c in CKPTS:
    t = rec["per_ctx"].get(str(c), {})
    if "by_role" in t: print(f"ctx{c} by role:", [(r["role"], r["pct"], r["quants"], r["route_classes"]) for r in t["by_role"][:6]])
  print("\nW1", rec["verdict"])
