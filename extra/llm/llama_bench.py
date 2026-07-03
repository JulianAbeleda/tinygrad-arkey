#!/usr/bin/env python3
"""The single home for invoking llama.cpp's `llama-bench` and parsing its `-o json` output.

Three benches used to each rebuild the same argv (`-m/-ngl/.../-r/-o json`), hardcode the same binary path, and
re-derive the same prompt-processing / token-generation row classification: `llama_cpp_bench.py`,
`model_authority_bench.run_llama_matched`, `llama_kv_ctx_slope_bench.run_depth`. This module owns that shared job so
the binary path and the json contract live once. Callers pass only their workload flags (`-p/-n/-d/-ctk/-ctv/-fa`);
`-m/-ngl/-r/-o json` are standardized here. No GPU / tinygrad dependency.
"""
from __future__ import annotations
import json, subprocess

# The llama-bench binary (was hardcoded 3x under three different constant names).
LLAMA_BENCH_BIN = "/home/ubuntu/env/llama.cpp/build/bin/llama-bench"

def build_llama_bench_cmd(model, spec_argv, *, bin:str=LLAMA_BENCH_BIN, ngl:int=99, reps:int=5) -> list[str]:
  """Standardized llama-bench argv in json mode. `spec_argv` is the caller's workload (-p/-n/-d/-ctk/...)."""
  return [bin, "-m", str(model), "-ngl", str(ngl), *[str(a) for a in spec_argv], "-r", str(reps), "-o", "json"]

def run_llama_bench_cmd(cmd, *, timeout=None, merge_stderr:bool=False) -> list[dict]:
  """Run a built llama-bench cmd and return the parsed result rows. Tolerant parse: llama-bench occasionally
  prefixes non-json before the array, so we start at the first '['."""
  stderr = subprocess.STDOUT if merge_stderr else subprocess.DEVNULL
  raw = subprocess.check_output(cmd, stderr=stderr, timeout=timeout).decode()
  start = raw.find("[")
  return json.loads(raw[start:] if start >= 0 else raw)

def llama_pp_row(rows) -> dict | None:
  """The prompt-processing (prefill, pp512) row, or None."""
  return next((r for r in rows if r.get("n_prompt") and not r.get("n_gen")), None)

def llama_tg_rows(rows) -> list[dict]:
  """The token-generation (decode, tg) rows (>=1; a matched-depth run returns one per -d depth)."""
  return [r for r in rows if r.get("n_gen") and not r.get("n_prompt")]
