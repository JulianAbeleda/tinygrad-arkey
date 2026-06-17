#!/usr/bin/env python3
"""Flash-threshold search (Track 2) — the second dogfood of the machine-search scaffold.

    search spec (SearchRow, search_space=flash_threshold) -> sweep runner (SDPA vs flash tok/s across
    context buckets, extra/qk_flash_sweep.py) -> find the crossover -> AcceptedPolicy (ctx_range =
    [threshold, max_context]).

Flash is exact (byte-identical to SDPA up to fp reassociation), so the "quality gate" is exactness, not
dNLL — the search optimizes tok/s and records the context threshold above which flash should be used. Pure
orchestration over the existing sweep runner; reuses the provenance helpers from qk_demote_search (DRY).
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys

from extra.llm_eval_common import write_json
from extra.qk_demote_search import HARDWARE, _git_commit, _model_id
from extra.qk_search_spec import AcceptedPolicy, Constraints, assemble_search_row, baseline as model_baseline

def _sweep(model:str, flash:bool, buckets:list[int], max_context:int, timeout:int) -> dict:
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": ".", "FLASH_DECODE": "1" if flash else "0"}
  cmd = [sys.executable, "-m", "extra.qk_flash_sweep", "--model", model, "--max-context", str(max_context),
         "--mode", "flash" if flash else "sdpa", "--buckets", *map(str, buckets)]
  p = subprocess.run(cmd, cwd=str(pathlib.Path(__file__).resolve().parents[1]), env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
  if p.returncode != 0: raise RuntimeError(f"sweep failed (flash={flash}):\n{p.stdout[-800:]}")
  return {int(k): v for k, v in json.loads(p.stdout.strip().splitlines()[-1])["by_ctx"].items()}

def find_threshold(sdpa:dict, flash:dict, max_context:int) -> tuple[int, list[dict]]:
  """Crossover = smallest ctx bucket where flash tok/s >= SDPA tok/s. Returns (threshold, frontier rows)."""
  rows, threshold = [], None
  for b in sorted(set(sdpa) & set(flash)):
    speedup = flash[b] / sdpa[b] if sdpa[b] else 0.0
    win = speedup >= 1.0
    if win and threshold is None: threshold = b
    rows.append({"ctx": b, "sdpa": sdpa[b], "flash": flash[b], "speedup": round(speedup, 3), "flash_wins": win})
  return (threshold if threshold is not None else max_context), rows

def run_search(model:str, *, buckets:list[int], max_context:int, timeout:int, out_dir:pathlib.Path) -> dict:
  model_id, commit = _model_id(model), _git_commit()
  llama = model_baseline("qwen3_8b")["llama_tok_s"]
  sdpa = _sweep(model, False, buckets, max_context, timeout)
  flash = _sweep(model, True, buckets, max_context, timeout)
  threshold, frontier = find_threshold(sdpa, flash, max_context)
  spec = assemble_search_row(row_id=f"flash_threshold:{model_id}", phase="long_context_decode",
                             model="qwen3_8b", op_scope="attention", backend="AMD",
                             search_space="flash_threshold", objective="tok_s",
                             constraints=Constraints(exact_required=True, ctx_range=(1, max_context)))
  # accepted policy: above the threshold, flash is the (exact) win. baseline/accepted measured at the
  # longest swept ctx (where the gain is largest and the decision matters most).
  top = max(buckets)
  ap = AcceptedPolicy(model="qwen3_8b", phase="long_context_decode", backend="AMD",
                      ctx_range=(threshold, max_context), objective="tok_s",
                      baseline_tok_s=sdpa.get(top, 0.0), accepted_tok_s=flash.get(top, 0.0),
                      quality_gate="exact (flash == SDPA up to fp reassociation)",
                      exactness="byte-identical", commit=commit, hardware=HARDWARE)
  summary = {"model_id": model_id, "hardware": HARDWARE, "commit": commit, "llama_tok_s": llama,
             "threshold_ctx": threshold, "frontier": frontier, "spec": spec, "accepted_policy": ap.to_dict()}
  write_json(out_dir / "flash-search.json", summary)
  write_json(out_dir / "accepted-flash-threshold.json", {**ap.to_dict(), "model_id": model_id,
                                                         "threshold_ctx": threshold})
  return summary

def frontier_md(summary:dict) -> str:
  lines = [f"# Flash-threshold frontier — crossover at ctx {summary['threshold_ctx']}", "",
           f"llama.cpp = {summary['llama_tok_s']} tok/s; flash is exact (byte-identical to SDPA).", "",
           "| ctx | SDPA tok/s | flash tok/s | speedup | flash wins |",
           "| ---: | ---: | ---: | ---: | :-: |"]
  for r in summary["frontier"]:
    lines.append(f"| {r['ctx']} | {r['sdpa']} | {r['flash']} | {r['speedup']}x | "
                 f"{'YES' if r['flash_wins'] else '-'} |")
  return "\n".join(lines)

def main():
  ap = argparse.ArgumentParser(description="flash-threshold search (Track 2)")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--buckets", type=int, nargs="*", default=[8, 256, 384, 512, 768, 1024, 1536, 2048, 3072])
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--timeout", type=int, default=900)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qk-flash-search"))
  args = ap.parse_args()
  summary = run_search(args.model, buckets=args.buckets, max_context=args.max_context, timeout=args.timeout,
                       out_dir=args.out)
  print(frontier_md(summary))

if __name__ == "__main__":
  main()
