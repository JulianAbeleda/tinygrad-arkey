#!/usr/bin/env python3
import argparse, json, os, pathlib, re, subprocess, sys, time
from dataclasses import dataclass

from extra.q4_k_bench import GGML_Q4_K, model_shape_targets, read_metadata, tensor_shape
from extra.q4_k_safety import assert_q4k_native_sweep_allowed

SUMMARY_RE = re.compile(
  r"^(?P<tensor>\S+) (?P<shape>\S+) (?P<name>\S+): (?P<ms>[0-9.]+) ms .*?"
  r"q4_eff=(?P<q4>[0-9.]+) GB/s device_q4_eff=(?P<dev>[0-9.]+ GB/s|n/a) "
  r"kernels=(?P<kernels>[0-9.]+)",
  re.MULTILINE,
)
GEMV_RE = re.compile(r"^primitive_gemv_correctness: PASS \S+ max_abs=([0-9.eE+-]+)", re.MULTILINE)
UNPACK_RE = re.compile(r"^primitive_unpack_correctness: PASS \S+ .* max_abs=([0-9.eE+-]+)", re.MULTILINE)

@dataclass(frozen=True)
class Candidate:
  name: str
  parts: int
  opts: tuple[str, ...]
  schedule: str = "none"

DEFAULT_CANDIDATES = [
  Candidate("local8_p1", 1, ("LOCAL:0:8",)),
  Candidate("local16_p1", 1, ("LOCAL:0:16",)),
  Candidate("local32_p1", 1, ("LOCAL:0:32",)),
  Candidate("local64_p1", 1, ("LOCAL:0:64",)),
  Candidate("local16_p2", 2, ("LOCAL:0:16",)),
  Candidate("local32_p2", 2, ("LOCAL:0:32",)),
  Candidate("local32_p4", 4, ("LOCAL:0:32",)),
  Candidate("local32_upcast2_p1", 1, ("LOCAL:0:32", "UPCAST:0:2")),
  Candidate("local32_upcast3_p1", 1, ("LOCAL:0:32", "UPCAST:0:3")),
]

def classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "KernelOptError" in out: return "illegal-opt"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def parse_summary(out:str) -> dict[str, dict]:
  rows = {}
  for m in SUMMARY_RE.finditer(out):
    dev = None if m["dev"] == "n/a" else float(m["dev"].split()[0])
    rows[m["name"]] = {
      "tensor": m["tensor"], "shape": m["shape"], "name": m["name"],
      "ms": float(m["ms"]), "q4_eff_gbs": float(m["q4"]),
      "device_q4_eff_gbs": dev, "kernels": float(m["kernels"]),
    }
  return rows

def run_bench(args, tensor:str, cand:Candidate|None) -> dict:
  cmd = [sys.executable, "extra/q4_k_bench.py", str(args.gguf), "--device", args.device,
         "--tensor", tensor, "--iters", str(args.iters), "--format", "text",
         "--activation", args.activation, "--seed", str(args.seed)]
  if cand is not None:
    cmd += ["--primitive", "--primitive-mode", "partial", "--primitive-parts", str(cand.parts),
            "--primitive-schedule", cand.schedule]
    for opt in cand.opts: cmd += ["--primitive-opt", opt]
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
  rows = parse_summary(out)
  row_name = "decode_q4_k_plus_matmul" if cand is None else "q4k_primitive_gemv"
  row = rows.get(row_name, {})
  gemv = GEMV_RE.search(out)
  unpack = UNPACK_RE.search(out)
  return {
    "tensor": tensor, "candidate": "fused_graph" if cand is None else cand.name,
    "status": status, "elapsed_s": round(elapsed, 3),
    "parts": None if cand is None else cand.parts,
    "opts": [] if cand is None else list(cand.opts),
    "schedule": None if cand is None else cand.schedule,
    "shape": row.get("shape"), "ms": row.get("ms"), "q4_eff_gbs": row.get("q4_eff_gbs"),
    "device_q4_eff_gbs": row.get("device_q4_eff_gbs"), "kernels": row.get("kernels"),
    "primitive_gemv_max_abs": float(gemv.group(1)) if gemv else None,
    "primitive_unpack_max_abs": float(unpack.group(1)) if unpack else None,
    "tail": "\n".join(out.strip().splitlines()[-args.tail_lines:]),
  }

def metric_value(row:dict, metric:str) -> float|None:
  return row.get(metric)

def summarize_policy(results:list[dict], metric:str, min_gain:float) -> list[dict]:
  policy = []
  tensors = list(dict.fromkeys(r["tensor"] for r in results))
  for tensor in tensors:
    rows = [r for r in results if r["tensor"] == tensor]
    fused = next((r for r in rows if r["candidate"] == "fused_graph" and r["status"] == "pass"), None)
    prims = [r for r in rows if r["candidate"] != "fused_graph" and r["status"] == "pass" and metric_value(r, metric) is not None]
    best = max(prims, key=lambda r: metric_value(r, metric), default=None)
    fused_metric = metric_value(fused, metric) if fused is not None else None
    best_metric = metric_value(best, metric) if best is not None else None
    use_primitive = bool(fused_metric is not None and best_metric is not None and best_metric > fused_metric * (1.0 + min_gain))
    policy.append({
      "tensor": tensor, "shape": (fused or best or {}).get("shape"),
      "metric": metric, "fused": fused_metric,
      "best_primitive": None if best is None else best["candidate"],
      "best_primitive_metric": best_metric,
      "ratio": None if fused_metric in (None, 0) or best_metric is None else best_metric / fused_metric,
      "choice": best["candidate"] if use_primitive and best is not None else "fused_graph",
      "opts": [] if not use_primitive or best is None else best["opts"],
      "parts": None if not use_primitive or best is None else best["parts"],
    })
  return policy

def print_results(results:list[dict]) -> None:
  print("| tensor | candidate | status | q4 GB/s | device Q4 GB/s | ms | gemv | opts |")
  print("|---|---|---|---:|---:|---:|---:|---|")
  for r in results:
    print(f"| {r['tensor']} | {r['candidate']} | {r['status']} | "
          f"{r['q4_eff_gbs'] if r['q4_eff_gbs'] is not None else ''} | "
          f"{r['device_q4_eff_gbs'] if r['device_q4_eff_gbs'] is not None else ''} | "
          f"{r['ms'] if r['ms'] is not None else ''} | "
          f"{r['primitive_gemv_max_abs'] if r['primitive_gemv_max_abs'] is not None else ''} | "
          f"{' '.join(r['opts']) if r['opts'] else ''} |")

def print_policy(policy:list[dict]) -> None:
  print("\n| tensor | shape | fused | best primitive | ratio | choice |")
  print("|---|---:|---:|---:|---:|---|")
  for p in policy:
    print(f"| {p['tensor']} | {p['shape'] or ''} | {p['fused'] if p['fused'] is not None else ''} | "
          f"{p['best_primitive_metric'] if p['best_primitive_metric'] is not None else ''} "
          f"({p['best_primitive'] or ''}) | {p['ratio'] if p['ratio'] is not None else ''} | {p['choice']} |")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Shape-aware Q4_K primitive policy sweep")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--tensor", action="append", help="exact tensor to sweep; default is representative shapes")
  parser.add_argument("--max-shapes", type=int, default=None)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--debug", type=int, default=2, help="use DEBUG=2 for device event timing")
  parser.add_argument("--timeout", type=float, default=60)
  parser.add_argument("--activation", choices=("random", "ones"), default="random")
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--metric", choices=("device_q4_eff_gbs", "q4_eff_gbs"), default="device_q4_eff_gbs")
  parser.add_argument("--min-gain", type=float, default=0.05)
  parser.add_argument("--only", action="append", default=[], help="candidate name to run, can repeat")
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--json", type=pathlib.Path, help="write full JSON results and policy")
  args = parser.parse_args()
  assert_q4k_native_sweep_allowed(args.device, "Q4_K policy sweep")

  meta = read_metadata(args.gguf)
  if args.tensor:
    targets = []
    for name in args.tensor:
      matches = [x for x in meta.infos if x.name == name]
      if not matches: raise ValueError(f"tensor {name!r} not found")
      if matches[0].typ != GGML_Q4_K or len(matches[0].dims) != 2: raise ValueError(f"{name!r} is not a 2D Q4_K tensor")
      targets.append(matches[0])
  else:
    targets = model_shape_targets(meta.infos, meta.kv, args.max_shapes)
  candidates = [c for c in DEFAULT_CANDIDATES if not args.only or c.name in set(args.only)]

  results = []
  for info in targets:
    tensor = info.name
    print(f"=== {tensor} {tensor_shape(info)} fused_graph ===", flush=True)
    res = run_bench(args, tensor, None)
    results.append(res)
    print(f"{res['status']} {args.metric}={res.get(args.metric)}", flush=True)
    for cand in candidates:
      print(f"=== {tensor} {tensor_shape(info)} {cand.name} ===", flush=True)
      res = run_bench(args, tensor, cand)
      results.append(res)
      print(f"{res['status']} {args.metric}={res.get(args.metric)} opts={list(cand.opts)} parts={cand.parts}", flush=True)

  policy = summarize_policy(results, args.metric, args.min_gain)
  print_results(results)
  print_policy(policy)
  if args.json:
    args.json.write_text(json.dumps({"metric": args.metric, "min_gain": args.min_gain, "results": results, "policy": policy},
                                    indent=2, sort_keys=True))
