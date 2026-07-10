#!/usr/bin/env python3
"""Authority decode benchmark + matched-context llama.cpp comparison for the per-model bench docs.

Produces fixed-context authority artifacts alongside the older generate-window E2E artifacts:
  - decode: extra/qk/decode_runtime_overhead.py (clean W==D, TinyJit, synced, NMEAS=40, fixed context,
    shipped FLASH_DECODE_THRESHOLD so the owned-attention route fires at ctx>=512). tok_s_W per ctx.
  - llama.cpp: llama-bench tg128 at MATCHED depth (-d <ctx>) so decode is compared at the same context, and
    pp512 for prefill reference.
Key difference from model_e2e_bench: decode is measured at FIXED contexts (128 and 512), not a median over a
growing-context generate window, and llama is compared at the same depth. Adopting this for README tables is a
methodology decision, not a mechanical replacement.

Writes bench/models/qwen/data/amd-gfx1100/<id>.authority.json

Usage:
  python extra/llm/model_authority_bench.py --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --id qwen3-8b \
      --out bench/models/qwen/data/amd-gfx1100/qwen3-8b.authority.json
"""
from __future__ import annotations
import os, sys, json, argparse, subprocess, pathlib

from extra.llm.llama_bench import build_llama_bench_cmd, run_llama_bench_cmd, llama_pp_row, llama_tg_rows

ROOT = pathlib.Path(__file__).resolve().parents[2]

def run_decode_authority(model:str, ckpts:str) -> dict:
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": str(ROOT), "QK_MODEL": model, "QK_CKPTS": ckpts}
  r = subprocess.run([sys.executable, "extra/qk/decode_runtime_overhead.py"], cwd=str(ROOT), env=env,
                     capture_output=True, text=True, timeout=1800)
  res = json.loads((ROOT / "bench/qk-decode-runtime-overhead/result.json").read_text())
  return res

def run_llama_matched(model:str, depths:list[int], reps:int=3) -> dict:
  # decode at matched depths + pp512
  d_arg = ",".join(str(d) for d in depths)
  cmd = build_llama_bench_cmd(model, ["-n", "128", "-d", d_arg, "-p", "512"], reps=reps)
  rows = run_llama_bench_cmd(cmd)
  decode_by_depth = {str(r.get("n_depth", 0)): {"tok_s": round(r["avg_ts"], 2), "stddev": round(r["stddev_ts"], 2)}
                     for r in llama_tg_rows(rows)}
  pp = llama_pp_row(rows)
  prefill = {"pp": pp["n_prompt"], "tok_s": round(pp["avg_ts"], 1), "stddev": round(pp["stddev_ts"], 1)} if pp else None
  return {"decode_by_depth": decode_by_depth, "prefill_pp512": prefill,
          "build_commit": (rows[0].get("build_commit") if rows else None)}

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--id", required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--ckpts", default="128,512")
  args = ap.parse_args()

  ckpt_ints = [int(x) for x in args.ckpts.split(",")]
  dec = run_decode_authority(args.model, args.ckpts)
  llama = run_llama_matched(args.model, ckpt_ints)

  # build matched-context comparison rows
  comp = []
  for row in dec["rows"]:
    ctx = row["ctx"]
    tg = row["tok_s_W"]
    lc = llama["decode_by_depth"].get(str(ctx), {}).get("tok_s")
    comp.append({"ctx": ctx, "flash_route": row["flash"], "tinygrad_tok_s_W": tg,
                 "llama_tok_s": lc, "ratio_pct": round(tg / lc * 100, 1) if (tg and lc) else None,
                 "host_sync_pct": row["host_sync_pct_of_wall"]})

  artifact = {
    "id": args.id,
    "model_id": dec.get("model_id"),
    "hardware": dec.get("hardware"),
    "decode_authority": {
      "method": dec.get("method"), "nmeas": dec.get("nmeas"),
      "rows": dec["rows"], "verdict": dec.get("verdict"),
    },
    "llama_cpp": llama,
    "decode_matched_comparison": comp,
    "timing_authority": "clean W==D qk_decode_runtime_overhead (TinyJit, synced, NMEAS=40, fixed ctx, shipped flash threshold)",
    "note": "decode compared to llama.cpp tg128 at MATCHED depth (-d). Prefill authority (tuned PREFILL_V2 path) is "
            "recorded separately where measured (8B: qk_prefill_authority_refresh); llama prefill is pp512.",
  }
  outp = pathlib.Path(args.out)
  outp.parent.mkdir(parents=True, exist_ok=True)
  outp.write_text(json.dumps(artifact, indent=2))
  for c in comp:
    r = f"{c['ratio_pct']}%" if c["ratio_pct"] else "—"
    print(f"{args.id} ctx{c['ctx']}{'F' if c['flash_route'] else ' '}: tinygrad {c['tinygrad_tok_s_W']} vs "
          f"llama {c['llama_tok_s']} = {r}  (host-sync {c['host_sync_pct']}%)")
  print(f"wrote {outp}")

if __name__ == "__main__":
  main()
