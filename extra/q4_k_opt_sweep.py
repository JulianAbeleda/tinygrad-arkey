#!/usr/bin/env python3
import argparse, json, os, pathlib, re, subprocess, sys, time

DEFAULT_CANDIDATES: list[tuple[str, str, list[str]]] = [
  ("baseline", "none", []),
  ("auto", "auto", []),
  *[(f"local0_{x}", "none", [f"LOCAL:0:{x}"]) for x in (2, 4, 8, 16, 32)],
  *[(f"upcast0_{x}", "none", [f"UPCAST:0:{x}"]) for x in (2, 3, 4, 5)],
  *[(f"unroll{x}_{arg}", "none", [f"UNROLL:{x}:{arg}"]) for x in (0, 1, 2, 3) for arg in (0, 4)],
  *[(f"group0_{x}", "none", [f"GROUP:0:{x}"]) for x in (4, 8, 16)],
  *[(f"grouptop0_{x}", "none", [f"GROUPTOP:0:{x}"]) for x in (16, 32)],
  ("auto_like", "none", ["UPCAST:0:3", "UNROLL:2:0", "LOCAL:0:32"]),
  ("row_local", "none", ["LOCAL:0:32", "UPCAST:0:3"]),
  ("row4_local", "none", ["LOCAL:0:32", "UPCAST:0:4"]),
  ("upcast3_unroll2", "none", ["UPCAST:0:3", "UNROLL:2:0"]),
  ("local32_unroll2", "none", ["LOCAL:0:32", "UNROLL:2:0"]),
]

BENCH_RE = re.compile(r"device=([0-9.]+) ms \(([0-9.]+) Q4-GB/s\)")
GEMV_RE = re.compile(r"^correctness: max_abs=([0-9.eE+-]+)", re.MULTILINE)
UNPACK_RE = re.compile(r"^unpack_correctness: .* max_abs=([0-9.eE+-]+)", re.MULTILINE)

def classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "KernelOptError" in out: return "illegal-opt"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def run_candidate(args, name:str, schedule:str, opts:list[str]) -> dict:
  cmd = [sys.executable, "extra/q4_k_gemv_primitive.py", str(args.gguf),
         "--device", args.device, "--tensor", args.tensor, "--rows", str(args.rows),
         "--mode", args.mode, "--parts", str(args.parts), "--schedule", schedule,
         "--iters", str(args.iters), "--unpack-check-rows", str(args.unpack_check_rows)]
  for opt in opts: cmd += ["--opt", opt]
  env = {**os.environ, "DEV": args.device, "DEBUG": str(args.debug), "PYTHONPATH": "."}
  st = time.perf_counter()
  timeout = False
  try:
    proc = subprocess.run(cmd, cwd=args.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=args.timeout)
    rc, out = proc.returncode, proc.stdout
  except subprocess.TimeoutExpired as e:
    timeout, rc, out = True, 124, (e.stdout or "") + "\nTIMEOUT"
  elapsed = time.perf_counter() - st
  bench = BENCH_RE.search(out)
  gemv = GEMV_RE.search(out)
  unpack = UNPACK_RE.search(out)
  status = classify(rc, out, timeout)
  return {
    "name": name, "status": status, "schedule": schedule, "opts": opts, "elapsed_s": round(elapsed, 3),
    "device_ms": float(bench.group(1)) if bench else None,
    "q4_gbs": float(bench.group(2)) if bench else None,
    "gemv_max_abs": float(gemv.group(1)) if gemv else None,
    "unpack_max_abs": float(unpack.group(1)) if unpack else None,
    "tail": "\n".join(out.strip().splitlines()[-args.tail_lines:]),
  }

def print_table(rows:list[dict]) -> None:
  print("| candidate | status | q4_gbs | device_ms | gemv | unpack | opts |")
  print("|---|---|---:|---:|---:|---:|---|")
  for r in rows:
    print(f"| {r['name']} | {r['status']} | {r['q4_gbs'] if r['q4_gbs'] is not None else ''} | "
          f"{r['device_ms'] if r['device_ms'] is not None else ''} | "
          f"{r['gemv_max_abs'] if r['gemv_max_abs'] is not None else ''} | "
          f"{r['unpack_max_abs'] if r['unpack_max_abs'] is not None else ''} | "
          f"{' '.join(r['opts']) if r['opts'] else r['schedule']} |")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Sweep explicit schedule opts for the Q4_K GEMV primitive")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--rows", type=int, default=12288)
  parser.add_argument("--mode", choices=("serial", "partial"), default="partial")
  parser.add_argument("--parts", type=int, default=1)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--debug", type=int, default=2)
  parser.add_argument("--timeout", type=float, default=45)
  parser.add_argument("--unpack-check-rows", type=int, default=2)
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--only", action="append", default=[], help="candidate name to run, can repeat")
  parser.add_argument("--json", type=pathlib.Path, help="write full JSON results")
  args = parser.parse_args()

  selected = [c for c in DEFAULT_CANDIDATES if not args.only or c[0] in set(args.only)]
  results = []
  for name, schedule, opts in selected:
    print(f"=== {name} ===", flush=True)
    res = run_candidate(args, name, schedule, opts)
    results.append(res)
    print(f"{res['status']} q4_gbs={res['q4_gbs']} device_ms={res['device_ms']} opts={opts or [schedule]}", flush=True)
  print_table(results)
  if args.json:
    args.json.write_text(json.dumps(results, indent=2, sort_keys=True))
