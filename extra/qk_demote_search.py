#!/usr/bin/env python3
"""B3 per-tensor Q6->Q4 demotion search — first end-to-end use of the qk_search_spec scaffold.

The loop the machine-search direction prescribes:
    search spec (qk_search_spec.SearchRow) -> candidate demotion sets -> isolated runner
    (tok/s via tinygrad.llm.cli, dNLL via extra.qk_nll_eval) -> scorer (quality gate) -> AcceptedPolicy.

Pure orchestration: it spawns the two EXISTING measurement CLIs per candidate (no new measurement
code) and consults the scaffold for the row/record shapes + the quality gate. A candidate is
accepted iff it is faster than baseline AND within the dNLL budget. Output: an accepted-policy
artifact per accepted set + a frontier table. Run on the AMD box (loads the gguf + requants).
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess, sys

from extra.llm_eval_common import write_json
from extra.qk_search_spec import Constraints, AcceptedPolicy, assemble_search_row, baseline as model_baseline

_TOK_RE = re.compile(r"([0-9]+\.[0-9]+) tok/s")

# Candidate demotion sets (cumulative ladder over the Q6_K tensors of Qwen3-8B-Q4_K_M):
# ffn_down already shipped (the free win); attn_v + output(lm_head) are the remaining Q6 tensors.
CANDIDATES = [
  ("baseline", ""),
  ("ffn_down", "ffn_down"),
  ("ffn_down+attn_v", "ffn_down,attn_v"),
  ("ffn_down+attn_v+output", "ffn_down,attn_v,output"),
]

def _run(cmd:list[str], targets:str, timeout:int) -> str:
  env = {**os.environ, "DEV": "AMD", "JIT": "1", "PYTHONPATH": ".", "QK_DEMOTE_TENSORS": targets}
  p = subprocess.run(cmd, cwd=str(pathlib.Path(__file__).resolve().parents[1]), env=env, text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
  if p.returncode != 0: raise RuntimeError(f"child failed ({cmd[-3:]}, demote={targets!r}):\n{p.stdout[-800:]}")
  return p.stdout

def measure(model:str, targets:str, *, bench:int, tokens:int, timeout:int) -> dict:
  """Measure steady-state tok/s + teacher-forced dNLL for one demotion set, each in isolation."""
  out = _run([sys.executable, "-m", "tinygrad.llm.cli", "-m", model, "--warmup", "--benchmark", str(bench)],
             targets, timeout)
  toks = [float(x) for x in _TOK_RE.findall(out)]
  if len(toks) < 5: raise RuntimeError(f"too few tok/s samples for demote={targets!r}")
  tok_s = statistics.median(sorted(toks)[3:])  # drop the first 3 (clock ramp)
  njson = _run([sys.executable, "-m", "extra.qk_nll_eval", "--model", model, "--tokens", str(tokens)],
               targets, timeout)
  nll = json.loads(njson.strip().splitlines()[-1])["nll"]
  return {"tok_s": tok_s, "nll": nll}

def run_search(model:str, *, epsilon:float, bench:int, tokens:int, timeout:int, out_dir:pathlib.Path) -> dict:
  llama = model_baseline("qwen3_8b")["llama_tok_s"]
  results, base = [], None
  for label, targets in CANDIDATES:
    spec = assemble_search_row(row_id=f"demote:{label or 'baseline'}", phase="decode", model="qwen3_8b",
                               op_scope="ffn_down" if "ffn_down" in targets else "attention", backend="AMD",
                               search_space="demotion", objective="tok_s",
                               constraints=Constraints(exact_required=False, dnll_epsilon=epsilon))
    try:
      m = measure(model, targets, bench=bench, tokens=tokens, timeout=timeout)
    except Exception as e:  # a candidate (e.g. a slow/large requant) may fail; record + keep going.
      if label == "baseline": raise RuntimeError(f"baseline measurement failed (no reference): {e}")
      results.append({"label": label, "targets": targets, "error": str(e)[:300], "accepted": False, "spec": spec})
      write_json(out_dir / "search.json", {"model": model, "epsilon": epsilon, "llama_tok_s": llama,
                                           "baseline": base, "results": results})
      continue
    if label == "baseline": base = m
    dnll = m["nll"] - base["nll"] if base else 0.0
    faster = base is not None and m["tok_s"] > base["tok_s"]
    within = dnll <= epsilon
    accepted = label != "baseline" and faster and within
    row = {"label": label, "targets": targets, **m, "dnll": round(dnll, 5),
           "pct_llama": round(100 * m["tok_s"] / llama, 1), "faster": faster, "within_quality": within,
           "accepted": accepted, "spec": spec}
    results.append(row)
    if accepted:
      ap = AcceptedPolicy(model="qwen3_8b", phase="decode", backend="AMD", ctx_range=(1, 4096),
                          objective="tok_s", baseline_tok_s=round(base["tok_s"], 2),
                          accepted_tok_s=round(m["tok_s"], 2), quality_gate=f"dNLL <= {epsilon}",
                          exactness=f"lossy: dNLL={dnll:+.5f}", commit="uncommitted", memory_cap_mb=None)
      write_json(out_dir / f"accepted-{label}.json", {**ap.to_dict(), "targets": targets})
    write_json(out_dir / "search.json", {"model": model, "epsilon": epsilon, "llama_tok_s": llama,
                                         "baseline": base, "results": results})  # incremental
  return {"model": model, "epsilon": epsilon, "llama_tok_s": llama, "baseline": base, "results": results}

def frontier_md(summary:dict) -> str:
  lines = ["# B3 demotion-search frontier", "",
           f"epsilon (dNLL budget) = {summary['epsilon']}; llama.cpp = {summary['llama_tok_s']} tok/s", "",
           "| set | tok/s | %llama | dNLL | faster | quality | accepted |",
           "| --- | ---: | ---: | ---: | :-: | :-: | :-: |"]
  for r in summary["results"]:
    if "error" in r:
      lines.append(f"| {r['label']} | ERR | - | - | - | - | - |  <!-- {r['error'][:80]} -->")
      continue
    lines.append(f"| {r['label']} | {r['tok_s']:.1f} | {r['pct_llama']} | {r['dnll']:+.5f} | "
                 f"{'Y' if r['faster'] else '-'} | {'pass' if r['within_quality'] else 'FAIL'} | "
                 f"{'ACCEPT' if r['accepted'] else '-'} |")
  return "\n".join(lines)

def main():
  ap = argparse.ArgumentParser(description="B3 Q6->Q4 demotion search")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--epsilon", type=float, default=0.01, help="dNLL budget (accept if dNLL <= epsilon)")
  ap.add_argument("--bench", type=int, default=24)
  ap.add_argument("--tokens", type=int, default=128)
  ap.add_argument("--timeout", type=int, default=700)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/qk-demote-search"))
  args = ap.parse_args()
  summary = run_search(args.model, epsilon=args.epsilon, bench=args.bench, tokens=args.tokens,
                       timeout=args.timeout, out_dir=args.out)
  print(frontier_md(summary))

if __name__ == "__main__":
  main()
