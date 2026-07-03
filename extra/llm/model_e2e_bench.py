#!/usr/bin/env python3
"""End-to-end per-model benchmark (decode tok/s + prefill pp512 + VRAM) for the model-level bench docs.

This is a *whole-model* benchmark, not a kernel A/B gate. It uses the retained model-measurement authority:
clean W==D via `model.generate`, PROFILE=0, auto clock, warmup before
measuring, repeated steady-state samples with a median+spread band (extra.qk.harness_contract.repro_band), and a
git/hardware/env provenance stamp. Decode tok/s is the headline (HBM-bound); prefill pp512 is secondary and
measured on whatever prefill path is active by default (reported in env).

Usage:
  python extra/llm/model_e2e_bench.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b \
      --max_context 2048 --decode-tokens 96 --warmup-skip 16 --prefill 512 \
      --out bench/models/qwen/data/amd-gfx1100/qwen3-8b.json
"""
from __future__ import annotations
import os, sys, json, time, argparse, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tinygrad.helpers import getenv, GlobalCounters, fetch, Context, DEBUG
from tinygrad.llm.model import Transformer
import tinygrad.llm.model as _M
from tinygrad.llm.cli import SimpleTokenizer, models as BUILTIN, _quant_from_name, _device_target
from extra.qk.harness_contract import repro_band

def _git(*args):
  try: return subprocess.check_output(["git", *args], cwd=pathlib.Path(__file__).resolve().parents[2]).decode().strip()
  except Exception: return None

def measure_decode(model, n_tokens:int, skip:int):
  seed = 0
  gen = model.generate([seed])
  per_tok_us, per_tok_mem = [], []
  for _ in range(n_tokens):
    GlobalCounters.reset()
    t0 = time.perf_counter_ns()
    next(gen)
    dt = time.perf_counter_ns() - t0
    per_tok_us.append(dt / 1e3)
    per_tok_mem.append(GlobalCounters.global_mem)
  steady_us = per_tok_us[skip:]
  steady_mem = per_tok_mem[skip:]
  band = repro_band(steady_us)
  med_s = band["median"] / 1e6
  decode_tok_s = round(1.0 / med_s, 2) if med_s else None
  # HBM bandwidth proxy from steady-state bytes moved per token / median token time
  med_mem = sorted(steady_mem)[len(steady_mem) // 2] if steady_mem else 0
  decode_gb_s = round(med_mem / 1e9 / med_s, 1) if med_s else None
  # tok/s band edges (min time -> max tok/s)
  band_tok_s = {"median": decode_tok_s,
                "min": round(1e6 / band["max"], 2) if band["max"] else None,
                "max": round(1e6 / band["min"], 2) if band["min"] else None,
                "spread_pct": band["spread_pct"]}
  return {"n_measured": len(steady_us), "skipped": skip, "tok_s": band_tok_s, "gb_s": decode_gb_s,
          "per_token_us_band": band}

def measure_prefill(model, n_prompt:int):
  # time-to-first-token for an n_prompt-token prompt = prefill time; pp = n_prompt / ttft
  try:
    if getattr(model, "_cached_tokens", None): model._cached_tokens = []   # force full prefill (no prefix reuse)
    prompt = [(i % 1000) + 1 for i in range(n_prompt)]
    gen = model.generate(prompt)
    t0 = time.perf_counter_ns()
    next(gen)
    ttft_s = (time.perf_counter_ns() - t0) / 1e9
    return {"n_prompt": n_prompt, "ttft_s": round(ttft_s, 4), "prefill_tok_s": round(n_prompt / ttft_s, 1) if ttft_s else None}
  except Exception as e:
    return {"n_prompt": n_prompt, "error": str(e)}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True, help="GGUF path or built-in alias")
  ap.add_argument("--id", required=True, help="short model id for the artifact / table row")
  ap.add_argument("--max_context", type=int, default=2048)
  ap.add_argument("--decode-tokens", type=int, default=96)
  ap.add_argument("--warmup-skip", type=int, default=16, help="leading decode tokens dropped (clock ramp / first-token)")
  ap.add_argument("--prefill", type=int, default=512, help="prefill prompt length for pp (0 to skip)")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  source = BUILTIN.get(args.model, args.model)
  t_load0 = time.perf_counter()
  model, kv = Transformer.from_gguf(fetch(source), args.max_context)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  load_s = time.perf_counter() - t_load0

  import tinygrad.nn as nn
  params = sum(x.numel() for x in nn.state.get_parameters(model))

  # warmup: capture the JIT before measuring (clean W==D requires warm kernels). DEBUG forced to 0 so kernel
  # schedule logs don't flood the benchmark output.
  with Context(DEBUG=0):
    for _ in range(2): list(zip(range(2), model.generate([0])))
  vram_bytes = GlobalCounters.mem_used

  prefill = measure_prefill(model, args.prefill) if args.prefill else None
  decode = measure_decode(model, args.decode_tokens, args.warmup_skip)

  artifact = {
    "id": args.id,
    "arch": kv.get("general.architecture"),
    "model_name": kv.get("general.name") or kv.get("general.basename"),
    "quant": _quant_from_name(str(source)),
    "params": params,
    "file_bytes": pathlib.Path(source).stat().st_size if pathlib.Path(str(source)).exists() else None,
    "max_context": model.max_context,
    "vram_used_bytes": vram_bytes,
    "vram_used_gb": round(vram_bytes / 1e9, 2),
    "load_s": round(load_s, 2),
    "decode": decode,
    "prefill": prefill,
    "timing_authority": "clean W==D model.generate, PROFILE=0, auto clock (HARNESS_GUIDE.md)",
    "provenance": {
      "command": "python " + " ".join(sys.argv),
      "git_commit": _git("rev-parse", "--short", "HEAD"),
      "git_dirty": bool(_git("status", "--porcelain")),
      "hardware": "AMD Radeon RX 7900 XTX",
      "target": _device_target(),
      "perf_state": "auto clock (not pinned)",
      "warmups": "2x2-token generate JIT capture",
      "env": {"PREFILL_V2": bool(_M.PREFILL_V2), "PREFILL_CONCRETE_KV": bool(_M.PREFILL_CONCRETE_KV),
              "Q4K_PRIMITIVE": getenv("Q4K_PRIMITIVE", 1), "Q6K_PRIMITIVE": getenv("Q6K_PRIMITIVE", 1),
              "HALF": getenv("HALF", 1)},
    },
  }
  outp = pathlib.Path(args.out)
  outp.parent.mkdir(parents=True, exist_ok=True)
  outp.write_text(json.dumps(artifact, indent=2))
  d = decode["tok_s"]
  print(f"{args.id}: decode {d['median']} tok/s (band {d['min']}-{d['max']}, spread {d['spread_pct']}%) "
        f"| {decode['gb_s']} GB/s | prefill {prefill.get('prefill_tok_s') if prefill else None} tok/s "
        f"| VRAM {artifact['vram_used_gb']} GB | {artifact['quant']} | params {params:,}")
  print(f"wrote {outp}")

if __name__ == "__main__":
  main()
