#!/usr/bin/env python3
"""Multi-process decode serving throughput (gfx1100). The clean, practical use of AMD hardware overlap.

In-process two-stream decode is blocked (HCQProgram.__call__ hardcodes ring 0 + one global timeline -- Phase 7a).
But CROSS-PROCESS overlap already works: each process has its own AMDDevice = its own timeline + compute ring, so
two decode processes overlap on the GPU with ZERO runtime/dispatch changes (the original +32% premise). This
benchmark measures it end-to-end: 1 process vs 2 concurrent processes, aggregate tok/s.

This is SERVING THROUGHPUT (aggregate tok/s across concurrent requests), NOT single-stream latency (per-process
tok/s drops when sharing; aggregate is what improves). No AMD_COMPUTE_RINGS, no runtime/model edits.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_multiprocess_decode_throughput.py
"""
from __future__ import annotations

import json, os, pathlib, subprocess, sys, time

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
NTOK = int(os.environ.get("NTOK", 128))  # long-ish window: short windows (e.g. 64) are noisy from process
                                          # load-time skew (the two measured windows must overlap to capture sharing)
WARM = 8
PROMPT = [5, 6, 7, 8, 9, 10, 11, 12]

def _worker(tag:str):
  os.environ.setdefault("JIT", "1")
  from tinygrad import Tensor
  from tinygrad.llm.model import Transformer
  Tensor.manual_seed(0)
  m, _ = Transformer.from_gguf(pathlib.Path(MODEL).expanduser(), 2048)
  got, t0 = [], None
  for i, tid in enumerate(m.generate(list(PROMPT), temperature=0.0)):
    got.append(int(tid))
    if i == WARM - 1: t0 = time.perf_counter()          # start timing AFTER warmup (steady-state decode)
    if i >= WARM + NTOK - 1: break
  dt = time.perf_counter() - t0
  print("@@W@@" + json.dumps({"tag": tag, "tok_s": round(NTOK / dt, 2), "n": NTOK,
                              "sample": got[WARM:WARM + 6], "warm_sample": got[:6]}))

def _vram_mb():
  for cmd in (["rocm-smi", "--showmeminfo", "vram", "--json"], ["amd-smi", "metric", "-m", "--json"]):
    try:
      r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
      if r.returncode == 0 and r.stdout.strip(): return r.stdout.strip()[:400]
    except Exception: pass
  return None

def _spawn(tag):
  return subprocess.Popen([sys.executable, __file__, "--worker", tag], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, env={**os.environ, "PYTHONPATH": "."})

def _collect(p):
  out, err = p.communicate(timeout=600)
  line = next((l for l in out.splitlines() if l.startswith("@@W@@")), None)
  if line is None: raise RuntimeError(f"worker failed:\n{err[-500:]}")
  return json.loads(line[5:])

def main():
  if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
    _worker(sys.argv[2]); return

  solo = _collect(_spawn("solo"))                       # 1 process baseline
  vram_before = _vram_mb()
  p0, p1 = _spawn("c0"), _spawn("c1")                   # 2 concurrent processes (started together)
  time.sleep(20); vram_2proc = _vram_mb()               # sample VRAM mid-run (after both have loaded)
  c0, c1 = _collect(p0), _collect(p1)

  agg = round(c0["tok_s"] + c1["tok_s"], 2)
  ratio = round(agg / solo["tok_s"], 3) if solo["tok_s"] else None
  # sanity: greedy + same prompt -> all three produce identical tokens (concurrency must not corrupt output)
  sane = bool(solo["sample"] == c0["sample"] == c1["sample"] and solo["warm_sample"] == c0["warm_sample"])
  passes = bool(ratio and ratio >= 1.2 and sane)
  out = {"arch": "gfx1100", "model": pathlib.Path(MODEL).name, "ntok": NTOK,
         "one_proc_tok_s": solo["tok_s"], "two_proc_per_tok_s": [c0["tok_s"], c1["tok_s"]],
         "two_proc_aggregate_tok_s": agg, "aggregate_speedup": ratio, "output_sane": sane,
         "vram_2proc": vram_2proc, "passes": passes,
         "note": "SERVING THROUGHPUT (aggregate tok/s across concurrent decode processes), NOT single-stream "
                 "latency -- per-process tok/s drops under sharing; aggregate is the metric. Cross-process "
                 "overlap = each process has its own AMDDevice/timeline/ring; no runtime or dispatch changes.",
         "verdict": (f"PASS: 2 processes aggregate {agg} tok/s vs 1-process {solo['tok_s']} tok/s = {ratio}x "
                     f"serving-throughput gain (per-proc {c0['tok_s']}/{c1['tok_s']}), output sane -> cross-process "
                     f"is the practical multi-ring payoff" if passes else
                     f"REFUTED: aggregate {agg} vs solo {solo['tok_s']} = {ratio}x (<1.2) or output not sane "
                     f"({sane}) -> cross-process decode doesn't materially overlap; bank the primitive only")}
  print(f"1 process            : {solo['tok_s']} tok/s")
  print(f"2 processes (per)    : {c0['tok_s']} + {c1['tok_s']} tok/s")
  print(f"2 processes aggregate: {agg} tok/s  -> {ratio}x  (output sane={sane})")
  print(out["verdict"])
  art = pathlib.Path("bench/amd-multiprocess-decode-throughput/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out, indent=2)); print(f"artifact: {art}")

if __name__ == "__main__":
  main()
