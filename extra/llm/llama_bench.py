#!/usr/bin/env python3
"""The single home for invoking llama.cpp's `llama-bench` and parsing its `-o json` output.

Three benches used to each rebuild the same argv (`-m/-ngl/.../-r/-o json`), hardcode the same binary path, and
re-derive the same prompt-processing / token-generation row classification: `llama_cpp_bench.py`,
`model_authority_bench.run_llama_matched`, `llama_kv_ctx_slope_bench.run_depth`. This module owns that shared job so
the binary path and the json contract live once. Callers pass only their workload flags (`-p/-n/-d/-ctk/-ctv/-fa`);
`-m/-ngl/-r/-o json` are standardized here. No GPU / tinygrad dependency.
"""
from __future__ import annotations
import json, os, pathlib, statistics, subprocess, tempfile

ARTIFACT_VERSION = 2

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

def atomic_write_json(path:str | pathlib.Path, value:dict) -> None:
  """Write an artifact without exposing a partial JSON file to readers."""
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(value, f, indent=2)
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  except BaseException:
    try: os.unlink(tmp)
    except FileNotFoundError: pass
    raise

def model_identity(model:str | pathlib.Path) -> dict:
  path = pathlib.Path(model).expanduser().resolve()
  stat = path.stat()
  return {"path": str(path), "filename": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}

def row_samples(row:dict) -> list[float]:
  """Return llama-bench's individual throughput samples (the stable JSON key is samples_ts)."""
  samples = row.get("samples_ts")
  if not isinstance(samples, list) or not samples:
    raise ValueError("llama-bench row has no raw per-rep samples_ts")
  return [float(x) for x in samples]

def summarize_row(row:dict, reps:int) -> dict:
  samples = row_samples(row)
  if len(samples) != reps:
    raise ValueError(f"llama-bench returned {len(samples)} samples, expected {reps}")
  return {"median_tok_s": round(statistics.median(samples), 2), "raw_tok_s": samples,
          "mean_tok_s": round(float(row["avg_ts"]), 2), "stddev_tok_s": round(float(row["stddev_ts"]), 2)}
