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
import json, argparse, pathlib

from extra.llm.llama_bench import (ARTIFACT_VERSION, LLAMA_BENCH_BIN as DEFAULT_BIN, atomic_write_json,
  build_llama_bench_cmd, model_identity, run_llama_bench_cmd, llama_pp_row, llama_tg_rows, summarize_row)

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", required=True)
  ap.add_argument("--artifact", required=True, help="model_e2e_bench JSON to merge llama numbers into")
  ap.add_argument("--out", required=True, help="new merged artifact (never mutates or rereads a shared default)")
  ap.add_argument("--bin", default=DEFAULT_BIN)
  ap.add_argument("--ngl", type=int, default=99)
  ap.add_argument("--prefill", type=int, default=512)
  ap.add_argument("--gen", type=int, default=128)
  ap.add_argument("--reps", type=int, default=5)
  args = ap.parse_args()

  cmd = build_llama_bench_cmd(args.model, ["-p", args.prefill, "-n", args.gen], bin=args.bin, ngl=args.ngl, reps=args.reps)
  rows = run_llama_bench_cmd(cmd)
  pp = llama_pp_row(rows)
  tg = next(iter(llama_tg_rows(rows)), None)
  tg_summary, pp_summary = (summarize_row(tg, args.reps) if tg else None), (summarize_row(pp, args.reps) if pp else None)
  llama = {
    "artifact_version": ARTIFACT_VERSION,
    "model_identity": model_identity(args.model),
    "bin": args.bin,
    "build_commit": (tg or pp or {}).get("build_commit"),
    "gpu_info": (tg or pp or {}).get("gpu_info"),
    "reps": args.reps,
    "decode_tg_tok_s": tg_summary["median_tok_s"] if tg else None,
    "decode_tg_stddev": round(tg["stddev_ts"], 2) if tg else None,
    "decode_n_gen": tg["n_gen"] if tg else None,
    "prefill_pp_tok_s": round(pp["avg_ts"], 1) if pp else None,
    "prefill_pp_stddev": round(pp["stddev_ts"], 1) if pp else None,
    "prefill_n_prompt": pp["n_prompt"] if pp else None,
    "decode": tg_summary, "prefill": pp_summary, "raw_rows": rows,
    "settings": {"depth": None, "decode_tokens": args.gen, "prefill_tokens": args.prefill,
                 "ngl": args.ngl, "kv": "llama.cpp default", "fa": "llama.cpp default"},
    "tool": {"identity": "llama-bench", "command": cmd},
  }
  art_path = pathlib.Path(args.artifact)
  if art_path.resolve() == pathlib.Path(args.out).resolve():
    raise ValueError("--out must differ from --artifact so the input artifact remains invocation-specific and immutable")
  art = json.loads(art_path.read_text())
  source_identity = art.get("model_identity")
  if source_identity and pathlib.Path(source_identity["path"]).resolve() != pathlib.Path(llama["model_identity"]["path"]):
    raise ValueError("tinygrad artifact and llama.cpp model paths do not match")
  if art.get("file_bytes") not in (None, llama["model_identity"]["size_bytes"]):
    raise ValueError("tinygrad artifact and llama.cpp model sizes do not match")
  if art.get("id") not in (None, args.id): raise ValueError("artifact id does not match --id")
  tiny_n = art.get("decode", {}).get("n_measured")
  if tiny_n is not None and tiny_n != args.gen: raise ValueError("decode token count mismatch")
  art["llama_cpp"] = llama
  # decode ratio: tinygrad / llama.cpp (headline comparison)
  tg_tinygrad = art.get("decode", {}).get("tok_s", {}).get("median")
  if tg_tinygrad and llama["decode_tg_tok_s"]:
    art["decode_ratio_tinygrad_over_llama"] = round(tg_tinygrad / llama["decode_tg_tok_s"], 3)
  art["matched_llama_artifact_version"] = ARTIFACT_VERSION
  atomic_write_json(args.out, art)

  ratio = art.get("decode_ratio_tinygrad_over_llama")
  print(f"{args.id}: llama.cpp decode {llama['decode_tg_tok_s']} tok/s (±{llama['decode_tg_stddev']}), "
        f"prefill pp512 {llama['prefill_pp_tok_s']} tok/s | tinygrad/llama decode ratio = "
        f"{ratio} ({round(ratio*100)}%)" if ratio else f"{args.id}: llama.cpp decode {llama['decode_tg_tok_s']} tok/s")

if __name__ == "__main__":
  main()
