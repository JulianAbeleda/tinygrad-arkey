#!/usr/bin/env python3
import argparse, json, os, pathlib, subprocess, sys, time

def classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "KernelOptError" in out: return "illegal-opt"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def run_cmd(args, label:str, cmd:list[str], debug:int, timeout_s:float) -> dict:
  env = {**os.environ, "DEV": args.device, "DEBUG": str(debug), "PYTHONPATH": ".",
         "PARALLEL": "0", "BEAM_DEBUG": "2", "BEAM_STRICT_MODE": "0", "BEAM_DEV_TIMEOUT": "1"}
  st = time.perf_counter()
  timeout = False
  try:
    proc = subprocess.run(cmd, cwd=args.repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout_s)
    rc, out = proc.returncode, proc.stdout
  except subprocess.TimeoutExpired as e:
    timeout, rc, out = True, 124, (e.stdout or "") + "\nTIMEOUT"
  if args.log_dir:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    (args.log_dir / f"{label}.log").write_text(out)
  return {"label": label, "status": classify(rc, out, timeout), "returncode": rc, "elapsed_s": round(time.perf_counter()-st, 3),
          "tail": "\n".join(out.strip().splitlines()[-args.tail_lines:])}

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Contain risky Q4_K scheduler/BEAM paths and verify GPU health afterward")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--rows", type=int, default=12288)
  parser.add_argument("--timeout", type=float, default=60)
  parser.add_argument("--log-dir", type=pathlib.Path)
  parser.add_argument("--tail-lines", type=int, default=12)
  args = parser.parse_args()

  base = [sys.executable, "extra/q4_k_gemv_primitive.py", str(args.gguf), "--device", args.device,
          "--tensor", args.tensor, "--rows", str(args.rows), "--mode", "partial", "--parts", "1",
          "--iters", "1", "--unpack-check-rows", "2"]
  risky = run_cmd(args, "risky_schedule_auto", base + ["--schedule", "auto"], debug=2, timeout_s=args.timeout)
  health = run_cmd(args, "health_local64", base + ["--schedule", "none", "--opt", "LOCAL:0:64"], debug=2, timeout_s=args.timeout)
  result = {"risky": risky, "health": health, "contained": risky["status"] in ("pass", "compile-fail", "illegal-opt", "wrong") and health["status"] == "pass"}
  print(json.dumps(result, indent=2, sort_keys=True))
  raise SystemExit(0 if result["contained"] else 1)
