"""W1: weight-path route attribution + wall share. Per-kernel GPU time (eager PROFILE -> one ProfileRangeEvent per
kernel with GPU HW timestamps; the JIT graph profiles as one opaque range, so eager is required -- same method as
extra/amd_isa_phase_n4_whole_step_attribution.py). Classifies each kernel by NAME -> role / quant / route_class, and
computes bytes_estimate (from the shape in the name) + effective_bandwidth. Audit-only.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_weight_path_route_attribution.py
Writes: bench/amd-isa-backend-weight-path-ceiling/route_attribution.json
"""
import os, sys, json, re, pathlib, subprocess
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-weight-path-ceiling"
CKPTS = [int(x) for x in os.environ.get("QK_CKPTS", "512,4096").split(",")]
NSTEPS = int(os.environ.get("QK_W1_STEPS", "4"))
Q_BPW = {"q4k": 4.5, "q6k": 6.5, "q8": 8.0, "fp16": 16.0}   # bits per weight (Q4_K_M avg ~4.5, Q6_K ~6.5)

def classify(name):
  nm = name.lower()
  quant = "q4k" if nm.startswith("q4k") else "q6k" if nm.startswith("q6k") else "q8" if nm.startswith("q8") else "fp16" if ("half" in nm or "f16" in nm) else "unknown"
  rc = ("owned_warp" if "warp" in nm else "coop" if "coop" in nm else "scheduler" if ("sched" in nm or "lanemap" in nm or "_g3" in nm) else
        "generated_g3" if ("lanemap" in nm or "futuresight" in nm) else ("gemv" if "gemv" in nm or "mmvq" in nm else None))
  dims = [int(x) for x in re.findall(r"_(\d+)", nm)]
  mn = [d for d in dims if d in (4096, 12288, 151936, 1024)]   # weight matrix dims of interest
  role = "other"
  if len(mn) >= 2:
    a, b = mn[0], mn[1]
    if {a, b} == {4096, 12288}: role = "ffn_gate_up" if b == 12288 else "ffn_down"
    elif a == 4096 and b == 4096: role = "attn_qkvo_proj"
    elif 151936 in (a, b): role = "lm_head"
  is_weight = quant in ("q4k", "q6k", "q8") and ("gemv" in nm or "mmvq" in nm or "coop" in nm) and len(mn) >= 2
  bytes_est = int(mn[0] * mn[1] * Q_BPW.get(quant, 16) / 8) if (is_weight and len(mn) >= 2) else 0
  return {"role": role, "quant": quant, "route_class": rc or "fallback_graph", "is_weight": is_weight,
          "matdims": mn[:2], "bytes_per_call": bytes_est}

CHILD = r'''
import os, json, re
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.qk_harness_contract import DEFAULT_MODEL
from extra.llm_generate import load_model_and_tokenizer
MAXC=4608; CTX=int(os.environ["W1_CTX"]); NSTEPS=int(os.environ["W1_STEPS"])
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
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

def attribute(cap, ctx):
  pk = cap["per_kernel"]; tot = sum(v["dur_per_step"] for v in pk.values()) or 1e-9
  # device-clock units: scale so total matches the measured decode wall (1/tok_s). Use unit-normalized % + relative.
  rows = []
  for nm, v in pk.items():
    c = classify(nm); dur = v["dur_per_step"]
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
  # shipped/default route = owned-warp Q4_K (Q4K_GEMV_WARP default-on). Capture per ctx.
  rec = {"scope": "W1 weight-path route attribution (eager PROFILE per-kernel GPU time, classified by name)",
         "route": "shipped_default (Q4K_GEMV_WARP owned-warp)", "per_ctx": {}}
  for ctx in CKPTS:
    cap = capture({}, ctx)
    rec["per_ctx"][str(ctx)] = attribute(cap, ctx) if "per_kernel" in cap else {"failed": cap}
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
