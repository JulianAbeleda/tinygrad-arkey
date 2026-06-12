#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, subprocess, sys, time
from dataclasses import dataclass

from extra.q4_k_safety import assert_q4k_native_sweep_allowed
from extra.qk_layout import GGML_Q6_K, read_metadata, tensor_shape

HEADER_RE = re.compile(r"^tensor=(?P<tensor>\S+) full_shape=\((?P<shape>[^)]*)\).*?quant_bytes=(?P<bytes>\d+)", re.MULTILINE)
BENCH_RE = re.compile(r"^(?P<name>q6k_(?:fused_graph|gemv_primitive_partial)): .*?device=(?P<ms>[0-9.]+) ms \((?P<gbs>[0-9.]+) quant-GB/s\)", re.MULTILINE)
GEMV_RE = re.compile(r"^correctness: max_abs=([0-9.eE+-]+)", re.MULTILINE)
UNPACK_RE = re.compile(r"^unpack_correctness: .* max_abs=([0-9.eE+-]+)", re.MULTILINE)

@dataclass(frozen=True)
class Candidate:
  name: str
  parts: int
  opts: tuple[str, ...]

DEFAULT_CANDIDATES = (
  Candidate("local16_p1", 1, ("LOCAL:0:16",)),
  Candidate("local32_p1", 1, ("LOCAL:0:32",)),
  Candidate("local64_p1", 1, ("LOCAL:0:64",)),
  Candidate("local128_p1", 1, ("LOCAL:0:128",)),
  Candidate("local32_p2", 2, ("LOCAL:0:32",)),
  Candidate("local64_p2", 2, ("LOCAL:0:64",)),
  Candidate("local32_p4", 4, ("LOCAL:0:32",)),
  Candidate("local64_p4", 4, ("LOCAL:0:64",)),
  Candidate("local64_upcast2_p1", 1, ("LOCAL:0:64", "UPCAST:0:2")),
)

def classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "KernelOptError" in out: return "illegal-opt"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def parse_run(out:str, candidate:str, status:str, elapsed:float, parts:int|None, opts:list[str], tail_lines:int) -> dict:
  header = HEADER_RE.search(out)
  benches = {m["name"]: {"device_ms": float(m["ms"]), "quant_gbs": float(m["gbs"])} for m in BENCH_RE.finditer(out)}
  row = benches.get("q6k_fused_graph" if candidate == "fused_graph" else "q6k_gemv_primitive_partial", {})
  quant_bytes = int(header["bytes"]) if header else None
  device_ms = row.get("device_ms")
  dot_tflops = None if quant_bytes is None or device_ms is None or header is None else _dot_tflops(header["shape"], device_ms)
  return {
    "tensor": header["tensor"] if header else None,
    "shape": header["shape"].replace(", ", "x") if header else None,
    "candidate": candidate,
    "status": status,
    "elapsed_s": round(elapsed, 3),
    "parts": parts,
    "opts": opts,
    "quant_bytes": quant_bytes,
    "device_ms": device_ms,
    "quant_gbs": row.get("quant_gbs"),
    "dot_tflops": dot_tflops,
    "gemv_max_abs": float(g.group(1)) if (g:=GEMV_RE.search(out)) else None,
    "unpack_max_abs": float(u.group(1)) if (u:=UNPACK_RE.search(out)) else None,
    "tail": "\n".join(out.strip().splitlines()[-tail_lines:]),
  }

def _dot_tflops(shape_s:str, device_ms:float) -> float|None:
  try:
    rows_s, k_s = [x.strip() for x in shape_s.split(",")]
    rows, k = int(rows_s), int(k_s)
  except Exception:
    return None
  return (2.0 * rows * k) / (device_ms / 1000.0) / 1e12

def run_candidate(args, tensor:str, cand:Candidate|None) -> dict:
  cmd = [sys.executable, "extra/q6_k_gemv_primitive.py", str(args.gguf), "--device", args.device,
         "--tensor", tensor, "--iters", str(args.iters), "--unpack-check-rows", str(args.unpack_check_rows),
         "--seed", str(args.seed)]
  if cand is not None:
    cmd += ["--parts", str(cand.parts)]
    for opt in cand.opts: cmd += ["--opt", opt]
  env = {**os.environ, "DEV": args.device, "DEBUG": str(args.debug), "PYTHONPATH": "."}
  st = time.perf_counter()
  timeout = False
  try:
    proc = subprocess.run(cmd, cwd=args.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
    rc, out = proc.returncode, proc.stdout
  except subprocess.TimeoutExpired as e:
    timeout, rc, out = True, 124, (e.stdout or "") + "\nTIMEOUT"
  elapsed = time.perf_counter() - st
  status = classify(rc, out, timeout)
  return parse_run(out, "fused_graph" if cand is None else cand.name, status, elapsed, None if cand is None else cand.parts,
                   [] if cand is None else list(cand.opts), args.tail_lines)

def summarize_policy(results:list[dict], min_gain:float) -> list[dict]:
  policy = []
  tensors = list(dict.fromkeys(r["tensor"] for r in results if r["tensor"] is not None))
  for tensor in tensors:
    rows = [r for r in results if r["tensor"] == tensor]
    fused = next((r for r in rows if r["candidate"] == "fused_graph" and r["status"] == "pass"), None)
    prims = [r for r in rows if r["candidate"] != "fused_graph" and r["status"] == "pass" and r["quant_gbs"] is not None]
    best = max(prims, key=lambda r: r["quant_gbs"], default=None)
    fused_gbs, best_gbs = (fused or {}).get("quant_gbs"), (best or {}).get("quant_gbs")
    use_primitive = bool(fused_gbs is not None and best_gbs is not None and best_gbs > fused_gbs * (1.0 + min_gain))
    policy.append({
      "tensor": tensor,
      "shape": (fused or best or {}).get("shape"),
      "fused_quant_gbs": fused_gbs,
      "best_primitive": None if best is None else best["candidate"],
      "best_primitive_quant_gbs": best_gbs,
      "ratio": None if fused_gbs in (None, 0) or best_gbs is None else best_gbs / fused_gbs,
      "choice": best["candidate"] if use_primitive and best is not None else "fused_graph",
      "parts": None if not use_primitive or best is None else best["parts"],
      "opts": [] if not use_primitive or best is None else best["opts"],
    })
  return policy

def print_results(results:list[dict]) -> None:
  print("| tensor | candidate | status | quant GB/s | device ms | dot TFLOP/s | gemv | opts |")
  print("|---|---|---|---:|---:|---:|---:|---|")
  for r in results:
    print(f"| {r['tensor'] or ''} | {r['candidate']} | {r['status']} | "
          f"{r['quant_gbs'] if r['quant_gbs'] is not None else ''} | "
          f"{r['device_ms'] if r['device_ms'] is not None else ''} | "
          f"{r['dot_tflops'] if r['dot_tflops'] is not None else ''} | "
          f"{r['gemv_max_abs'] if r['gemv_max_abs'] is not None else ''} | "
          f"{' '.join(r['opts']) if r['opts'] else ''} |")

def print_policy(policy:list[dict]) -> None:
  print("\n| tensor | shape | fused | best primitive | ratio | choice |")
  print("|---|---:|---:|---:|---:|---|")
  for p in policy:
    best = "" if p["best_primitive_quant_gbs"] is None else f"{p['best_primitive_quant_gbs']} ({p['best_primitive']})"
    print(f"| {p['tensor']} | {p['shape'] or ''} | {p['fused_quant_gbs'] if p['fused_quant_gbs'] is not None else ''} | "
          f"{best} | {p['ratio'] if p['ratio'] is not None else ''} | {p['choice']} |")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Shape-aware Q6_K primitive policy sweep")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--tensor", action="append", required=True)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--debug", type=int, default=2)
  parser.add_argument("--timeout", type=float, default=120)
  parser.add_argument("--unpack-check-rows", type=int, default=2)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--min-gain", type=float, default=0.05)
  parser.add_argument("--only", action="append", default=[])
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--json", type=pathlib.Path)
  args = parser.parse_args()
  assert_q4k_native_sweep_allowed(args.device, "Q6_K policy sweep")

  meta = read_metadata(args.gguf)
  for tensor in args.tensor:
    matches = [x for x in meta.infos if x.name == tensor]
    if not matches: raise ValueError(f"tensor {tensor!r} not found")
    if matches[0].typ != GGML_Q6_K or len(matches[0].dims) != 2:
      raise ValueError(f"{tensor!r} is not a 2D Q6_K tensor")
    print(f"target {tensor} shape={tensor_shape(matches[0])}", flush=True)

  candidates = [c for c in DEFAULT_CANDIDATES if not args.only or c.name in set(args.only)]
  results = []
  for tensor in args.tensor:
    print(f"=== {tensor} fused_graph ===", flush=True)
    res = run_candidate(args, tensor, None)
    results.append(res)
    print(f"{res['status']} quant_gbs={res.get('quant_gbs')}", flush=True)
    for cand in candidates:
      print(f"=== {tensor} {cand.name} ===", flush=True)
      res = run_candidate(args, tensor, cand)
      results.append(res)
      print(f"{res['status']} quant_gbs={res.get('quant_gbs')} opts={list(cand.opts)} parts={cand.parts}", flush=True)

  policy = summarize_policy(results, args.min_gain)
  print_results(results)
  print_policy(policy)
  if args.json:
    args.json.write_text(json.dumps({"min_gain": args.min_gain, "results": results, "policy": policy}, indent=2, sort_keys=True))
