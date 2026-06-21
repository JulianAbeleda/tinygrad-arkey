#!/usr/bin/env python3
"""Prefill DEFAULT-POLICY evaluation (measurement only; NO new kernels).

Decision: should PREFILL_V2 and/or PREFILL_CONCRETE_KV be default-on / server-only / long-prompt-only / opt-in?
Product policy is about e2e TTFT, load time, peak VRAM, compile/precompile cost, prompt length, and
cold-one-shot vs warm/server amortization -- NOT the (settled) synced forward throughput.

Matrix: modes {0:PREFILL_V2=0 (true default), 1:PREFILL_V2=1, 2:PREFILL_V2=1 PREFILL_CONCRETE_KV=1}
        x prompt lengths {512,1024,2048,4096}. Fresh subprocess per (mode,len) so COLD (first gen, incl compile)
        is cleanly separated from WARM (2nd divergent-prompt gen, jits reused). OOM is recorded, not fatal.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_default_policy_eval.py
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys, time

MODEL = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
MODES = {0: {}, 1: {"PREFILL_V2": "1"}, 2: {"PREFILL_V2": "1", "PREFILL_CONCRETE_KV": "1"}}
LENS = [512, 1024, 2048, 4096]
DECODE_N = 16


def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)
def gpu_used_gb():
  try:
    out = subprocess.run(["rocm-smi", "--showmeminfo", "vram"], capture_output=True, text=True).stdout
    for ln in out.splitlines():
      if "Used Memory" in ln: return round(int(ln.split(":")[-1].strip()) / 1e9, 2)
  except Exception: pass
  return None


def worker(mode: int, plen: int):
  from tinygrad import Tensor, Device, GlobalCounters
  dev = Device["AMD"]
  maxc = plen + 256
  rocm_before = gpu_used_gb()
  t0 = time.perf_counter()
  from extra.llm_generate import load_model_and_tokenizer
  m, tok = load_model_and_tokenizer(MODEL, maxc, seed=0)
  load_s = time.perf_counter() - t0
  vram_after_load = GlobalCounters.mem_used_per_device["AMD"] / 1e9
  pfx = (tok.prefix() if hasattr(tok, "prefix") else [])
  filler = "the quick brown fox jumps over a lazy dog near rivers and hills while stars wheel overhead "
  A = (pfx + tok.encode("Alpha " + filler * 300))[:plen]
  B = (pfx + tok.encode("Zeta " + filler * 300))[:plen]
  T = type(m); orig = T.__call__; sched = []
  def tr(self, tokens, start_pos, *a, **k):
    sh = tokens.shape[1]; sched.append(("int" if isinstance(start_pos, int) else "sym") + str(sh if isinstance(sh, int) else "sym"))
    return orig(self, tokens, start_pos, *a, **k)
  T.__call__ = tr

  def gen_first(ids):
    sched.clear(); dev.synchronize(); s = time.perf_counter()
    g = m.generate(list(ids), chunk_size=32, temperature=0.0); t0_ = next(g); dev.synchronize()
    return (time.perf_counter() - s), t0_, list(sched), g

  perflevel("high")
  try:
    cold_s, tok0, sched_cold, _ = gen_first(A)
    vram_peak = max(vram_after_load, GlobalCounters.mem_used_per_device["AMD"] / 1e9)
    warm_s, tok0b, sched_warm, g = gen_first(B)
    # decode tok/s after the warm prefill (continue the same generator). Warm the decode jit first (the first
    # decode steps compile) so the rate isn't compile-contaminated.
    for _ in range(4): next(g)
    dev.synchronize(); ds = time.perf_counter(); nd = 0
    for _ in range(DECODE_N):
      next(g); nd += 1
    dev.synchronize(); dec_tps = round(nd / (time.perf_counter() - ds), 1)
  finally:
    perflevel("auto")
  print("@@R@@" + json.dumps({
    "mode": mode, "prompt_tokens": plen, "lenA": len(A), "max_context": maxc,
    "load_s": round(load_s, 1), "vram_after_load_gb": round(vram_after_load, 2), "peak_vram_gb": round(vram_peak, 2),
    "rocm_used_before_gb": rocm_before, "rocm_used_after_gb": gpu_used_gb(),
    "first_prefill_s": round(cold_s, 2), "warm_prefill_s": round(warm_s, 2), "tok0": tok0,
    "ttft_coldoneshot_s": round(load_s + cold_s, 2), "ttft_warm_s": round(warm_s, 2),
    "call_schedule": sched_cold, "decode_tok_s": dec_tps}))


def main() -> int:
  if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
    worker(int(sys.argv[2]), int(sys.argv[3])); return 0
  rows = []
  for mode, env in MODES.items():
    for plen in LENS:
      p = subprocess.run([sys.executable, __file__, "--worker", str(mode), str(plen)],
                         env={**os.environ, "DEV": "AMD", "PYTHONPATH": ".", **env},
                         capture_output=True, text=True, timeout=1800)
      line = next((l for l in p.stdout.splitlines() if l.startswith("@@R@@")), None)
      if line is None:
        oom = "MemoryError" in (p.stdout + p.stderr) or "out of memory" in (p.stdout + p.stderr).lower()
        rows.append({"mode": mode, "prompt_tokens": plen, "FAILED": "OOM" if oom else "ERR",
                     "err": (p.stderr or p.stdout)[-400:]})
        print(f"  mode{mode} len{plen}: {'OOM' if oom else 'ERR'}")
        continue
      r = json.loads(line[5:]); rows.append(r)
      print(f"  mode{mode} len{plen:5d}: load {r['load_s']:5.1f}s vram {r['peak_vram_gb']:5.2f}GB | "
            f"cold_prefill {r['first_prefill_s']:6.2f}s warm {r['warm_prefill_s']:6.2f}s | TTFT_cold {r['ttft_coldoneshot_s']:6.1f}s | "
            f"tok0 {r['tok0']} dec {r['decode_tok_s']} | {r['call_schedule']}")
  # tok0 match vs mode0 per length
  base = {r["prompt_tokens"]: r.get("tok0") for r in rows if r.get("mode") == 0 and "tok0" in r}
  for r in rows:
    if "tok0" in r: r["tok0_match_mode0"] = (r["tok0"] == base.get(r["prompt_tokens"]))
  out = pathlib.Path("bench/qk-prefill-default-policy-eval"); out.mkdir(parents=True, exist_ok=True)
  (out / "result.json").write_text(json.dumps({"date": "2026-06-20", "model": pathlib.Path(MODEL).name,
    "gpu": "gfx1100 RX 7900 XTX 24GB", "decode_n": DECODE_N, "rows": rows}, indent=2) + "\n")
  print(f"\nartifact: {out/'result.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
