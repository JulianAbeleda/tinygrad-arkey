#!/usr/bin/env python3
"""Phase RSR0/RSR1: ordered reduce-source trace for the decode graph.

The aggregate attribution tool hides graph position. This captures Compiled.profile_events IN ORDER for one eager
decode step, then for each hot r_* reduce kernel records its nearest previous/next non-reduce kernels and a +/-5
window, so the reduce can be classified by what it sits next to (RMSNorm, attention, coop-partial GEMV combine,
elementwise), not by shape product alone.

Run: DEV=AMD JIT=1 DECODE_Q4K_G3_ANYSHAPE=1 PYTHONPATH=. python3 extra/qk/decode_reduce_source_trace.py \
       --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf --id qwen3-14b-g3anyshape --ctx 128
Writes bench/qwen-14b-32b-truegen/reduce_source_trace/{ordered_events,reduce_windows,latest}.json + summary.md
Verdict: RSR0_PASS_ORDERED_REDUCE_TRACE / RSR0_BLOCKED_*
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, pathlib, collections
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extra.qk.decode_role_profile import classify_kernel, profile_from_gguf
OUT = ROOT / "bench/qwen-14b-32b-truegen/reduce_source_trace"

CHILD = r'''
import json, os
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.llm.generate import load_model_and_tokenizer
MODEL=os.environ["QK_ATTR_MODEL"]; MAXC=int(os.environ.get("QK_ATTR_MAX_CONTEXT","4608")); CTX=int(os.environ["QK_ATTR_CTX"])
m,tok=load_model_and_tokenizer(MODEL, MAXC, seed=20260630)
for lin in (getattr(m,"_q4k_linears",None).linears if getattr(m,"_q4k_linears",None) else []): lin.decode_enabled=True
use_flash = CTX >= int(os.environ.get("FLASH_DECODE_THRESHOLD","512"))
for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
v=UOp.variable("start_pos",0,MAXC-1); temp=Tensor([0.0]); tk=Tensor([[100]],dtype="int32").contiguous()
step=TinyJit(m.forward)
for i in range(4): step(tk, v.bind(CTX+i), temp).realize().item()   # warm compile
Compiled.profile_events=[]
with Context(PROFILE=1):
  m.forward(tk, v.bind(CTX), temp).realize().item()                 # ONE eager step, ordered
events=[]
for e in Compiled.profile_events:
  if isinstance(e, ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name,"name",None) or str(e.name)
    events.append([nm, round(float(e.en-e.st),3)])
print("@@"+json.dumps({"ctx":CTX,"use_flash":use_flash,"events":events}))
'''

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", required=True)
  ap.add_argument("--ctx", type=int, default=128)
  args = ap.parse_args()

  env = {**os.environ, "DEV": os.environ.get("DEV", "AMD"), "JIT": os.environ.get("JIT", "1"), "PROFILE": "1",
         "PYTHONPATH": str(ROOT), "QK_ATTR_MODEL": str(pathlib.Path(args.model).expanduser()), "QK_ATTR_CTX": str(args.ctx),
         "QK_ATTR_MAX_CONTEXT": "4608"}
  p = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=2400)
  line = [l for l in p.stdout.splitlines() if l.startswith("@@")]
  if not line:
    print("RSR0_BLOCKED_CAPTURE_FAILED"); print(p.stderr[-2000:]); sys.exit(2)
  cap = json.loads(line[-1][2:])
  events = cap["events"]
  profile = profile_from_gguf(pathlib.Path(args.model).expanduser())

  def is_reduce(nm): return nm.startswith("r_")
  def bucket(nm): return classify_kernel(nm, profile).get("bucket", "?")
  # per-reduce: aggregate dur + calls, and record window context around each occurrence
  agg = collections.defaultdict(lambda: {"dur": 0.0, "calls": 0, "prev": collections.Counter(), "next": collections.Counter(),
                                          "windows": []})
  total = sum(d for _, d in events)
  for i, (nm, d) in enumerate(events):
    if not is_reduce(nm): continue
    a = agg[nm]; a["dur"] += d; a["calls"] += 1
    # nearest previous / next NON-reduce kernel
    pv = next((events[j][0] for j in range(i - 1, -1, -1) if not is_reduce(events[j][0])), None)
    nx = next((events[j][0] for j in range(i + 1, len(events)) if not is_reduce(events[j][0])), None)
    a["prev"][pv] += 1; a["next"][nx] += 1
    if len(a["windows"]) < 3:
      a["windows"].append([events[j][0] for j in range(max(0, i - 4), min(len(events), i + 5))])

  rows = []
  for nm, a in sorted(agg.items(), key=lambda x: -x[1]["dur"]):
    rows.append({"kernel": nm, "pct_gpu": round(100 * a["dur"] / total, 2), "calls_in_step": a["calls"],
                 "shape_factors": [int(x) for x in nm.replace("r_", "").replace("n1", "").split("start_pos")[0].split("_") if x.isdigit()],
                 "prev_nonreduce": a["prev"].most_common(3), "next_nonreduce": a["next"].most_common(3),
                 "example_window": a["windows"][0] if a["windows"] else []})

  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / "ordered_events.json").write_text(json.dumps(events))
  (OUT / "reduce_windows.json").write_text(json.dumps(rows, indent=2))
  (OUT / "latest.json").write_text(json.dumps({"id": args.id, "ctx": args.ctx, "verdict": "RSR0_PASS_ORDERED_REDUCE_TRACE",
    "n_events": len(events), "reduce_rows": rows}, indent=2))
  L = [f"# RSR0 ordered reduce trace — {args.id} ctx{args.ctx}", "", f"{len(events)} kernels in one decode step.", "",
       "| reduce kernel | %gpu | calls/step | prev non-reduce | next non-reduce |", "|---|---|---|---|---|"]
  for r in rows[:8]:
    L.append(f"| `{r['kernel']}` | {r['pct_gpu']}% | {r['calls_in_step']} | "
             f"{r['prev_nonreduce'][0][0] if r['prev_nonreduce'] else '-'} | {r['next_nonreduce'][0][0] if r['next_nonreduce'] else '-'} |")
  (OUT / "summary.md").write_text("\n".join(L) + "\n")

  print(f"RSR0_PASS_ORDERED_REDUCE_TRACE — {len(events)} kernels/step; top reduce rows:")
  for r in rows[:6]:
    print(f"  {r['kernel']:34} {r['pct_gpu']:5.2f}%  calls={r['calls_in_step']:3d}  prev={r['prev_nonreduce'][0][0] if r['prev_nonreduce'] else '-'}  next={r['next_nonreduce'][0][0] if r['next_nonreduce'] else '-'}")

if __name__ == "__main__":
  main()
