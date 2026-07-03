#!/usr/bin/env python3
"""Run llama.cpp's llama-bench on a model and merge the numbers into the matching model_e2e_bench artifact.

This is the apples-to-apples reference: same GGUF file, same GPU. llama-bench reports pp512 (prefill) and tg128
(decode) tok/s with its own warmup + repeats. We store avg_ts/stddev so the per-model bench doc can show
tinygrad-vs-llama.cpp and the decode ratio.

Usage:
  python extra/llm/llama_cpp_bench.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b \
      --artifact bench/models/qwen/data/amd-gfx1100/qwen3-8b.json
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, pathlib

DEFAULT_BIN = "/home/ubuntu/env/llama.cpp/build/bin/llama-bench"

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", required=True)
  ap.add_argument("--artifact", required=True, help="model_e2e_bench JSON to merge llama numbers into")
  ap.add_argument("--bin", default=DEFAULT_BIN)
  ap.add_argument("--ngl", type=int, default=99)
  ap.add_argument("--prefill", type=int, default=512)
  ap.add_argument("--gen", type=int, default=128)
  ap.add_argument("--reps", type=int, default=5)
  args = ap.parse_args()

  cmd = [args.bin, "-m", args.model, "-ngl", str(args.ngl), "-p", str(args.prefill), "-n", str(args.gen),
         "-r", str(args.reps), "-o", "json"]
  out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
  rows = json.loads(out)
  pp = next((r for r in rows if r.get("n_prompt") and not r.get("n_gen")), None)
  tg = next((r for r in rows if r.get("n_gen") and not r.get("n_prompt")), None)
  llama = {
    "bin": args.bin,
    "build_commit": (tg or pp or {}).get("build_commit"),
    "gpu_info": (tg or pp or {}).get("gpu_info"),
    "reps": args.reps,
    "decode_tg_tok_s": round(tg["avg_ts"], 2) if tg else None,
    "decode_tg_stddev": round(tg["stddev_ts"], 2) if tg else None,
    "decode_n_gen": tg["n_gen"] if tg else None,
    "prefill_pp_tok_s": round(pp["avg_ts"], 1) if pp else None,
    "prefill_pp_stddev": round(pp["stddev_ts"], 1) if pp else None,
    "prefill_n_prompt": pp["n_prompt"] if pp else None,
  }
  art_path = pathlib.Path(args.artifact)
  art = json.loads(art_path.read_text())
  art["llama_cpp"] = llama
  # decode ratio: tinygrad / llama.cpp (headline comparison)
  tg_tinygrad = art.get("decode", {}).get("tok_s", {}).get("median")
  if tg_tinygrad and llama["decode_tg_tok_s"]:
    art["decode_ratio_tinygrad_over_llama"] = round(tg_tinygrad / llama["decode_tg_tok_s"], 3)
  art_path.write_text(json.dumps(art, indent=2))

  ratio = art.get("decode_ratio_tinygrad_over_llama")
  print(f"{args.id}: llama.cpp decode {llama['decode_tg_tok_s']} tok/s (±{llama['decode_tg_stddev']}), "
        f"prefill pp512 {llama['prefill_pp_tok_s']} tok/s | tinygrad/llama decode ratio = "
        f"{ratio} ({round(ratio*100)}%)" if ratio else f"{args.id}: llama.cpp decode {llama['decode_tg_tok_s']} tok/s")

if __name__ == "__main__":
  main()
