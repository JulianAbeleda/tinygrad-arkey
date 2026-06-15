#!/usr/bin/env python3
"""Phase B0: batch-size efficiency curve for Q4_K matmul.

Batch-1 decode GEMV has zero weight reuse and is latency-bound (~20-47% of peak bandwidth).
Batching (B>1) makes it a GEMM where each dequantized weight is reused B times. This sweeps
--seq-len and measures, per batch B, the per-token device latency and achieved FLOPS (as a
fraction of the measured fp16 compute roof) for:
  - decode_q4_k_plus_matmul: fused dequant+matmul (the real quantized GEMM)
  - matmul_decoded: weights pre-dequantized to fp16 then dense matmul (compute ceiling)
to quantify the amortization and locate the memory->compute crossover.
"""
from __future__ import annotations

import argparse, json, os, pathlib, re, statistics, subprocess
from typing import Any

DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
DEFAULT_ARTIFACT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614/batched-b0")
DEFAULT_TENSORS = ("blk.20.attn_q.weight", "blk.13.ffn_gate.weight")
SEQ_LENS = (1, 2, 4, 8, 16, 32, 64, 128)
Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES = 256, 144

LINE_RE = re.compile(r"^(?P<tensor>\S+) (?P<shape>\d+x\d+) (?P<name>matmul_decoded|decode_q4_k_plus_matmul): "
                     r".*device_q4_eff=(?P<dev>[0-9.]+) GB/s", re.MULTILINE)

def _q4_bytes(m:int, k:int) -> int:
  return (m * k // Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_BYTES

def measure_fp16_compute_peak(device:str) -> float:
  # Large square fp16 matmul; report best device TFLOPS over warm iters.
  code = (
    "from tinygrad import Tensor, dtypes\n"
    "n=4096\n"
    "a=Tensor.randn(n,n,dtype=dtypes.float16).realize(); b=Tensor.randn(n,n,dtype=dtypes.float16).realize()\n"
    "for _ in range(8): (a@b).realize()\n"
  )
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  out = subprocess.run([".venv/bin/python", "-c", code], cwd=".", env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180).stdout
  gflops = [float(x) for x in re.findall(r"\(\s*([0-9.]+) GFLOPS", out)]
  return round(max(gflops) / 1000.0, 2) if gflops else 0.0  # TFLOPS

def _run(repo:pathlib.Path, model:pathlib.Path, tensor:str, seq_len:int, *, device:str, iters:int, runs:int) -> dict[str, float]:
  cmd = [".venv/bin/python", "extra/q4_k_bench.py", str(model), "--device", device, "--tensor", tensor,
         "--iters", str(iters), "--seq-len", str(seq_len), "--format", "text", "--activation", "random", "--seed", "1337"]
  env = {**os.environ, "DEV": device, "DEBUG": "2", "PYTHONPATH": "."}
  dev = {"matmul_decoded": [], "decode_q4_k_plus_matmul": []}
  for _ in range(runs):
    out = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240).stdout
    for m in LINE_RE.finditer(out):
      dev[m["name"]].append(float(m["dev"]))
  return {name: (statistics.median(v) if v else 0.0) for name, v in dev.items()}

def run_b0(repo:pathlib.Path, model:pathlib.Path, tensors:tuple[str, ...], artifact:pathlib.Path, *,
           device:str="AMD", iters:int=5, runs:int=3) -> dict[str, Any]:
  repo, model = repo.resolve(), model.expanduser().resolve()
  compute_peak_tflops = measure_fp16_compute_peak(device)
  per_tensor = {}
  for tensor in tensors:
    m, k = (4096, 4096) if "attn_q" in tensor else (12288, 4096)
    q4b = _q4_bytes(m, k)
    curve = []
    for b in SEQ_LENS:
      dev = _run(repo, model, tensor, b, device=device, iters=iters, runs=runs)
      row = {"batch": b}
      for name, dev_gbs in dev.items():
        if dev_gbs <= 0: continue
        device_time_s = q4b / (dev_gbs * 1e9)  # device_q4_eff = q4_bytes / device_time
        tflops = 2 * m * k * b / device_time_s / 1e12
        row[name] = {
          "device_time_us": round(device_time_s * 1e6, 2),
          "per_token_us": round(device_time_s * 1e6 / b, 3),
          "achieved_tflops": round(tflops, 2),
          "pct_compute_peak": round(tflops / compute_peak_tflops * 100, 2) if compute_peak_tflops else None,
        }
      curve.append(row)
    per_tensor[tensor] = {"shape": [m, k], "curve": curve}
  summary = _summarize(per_tensor, compute_peak_tflops)
  artifact.mkdir(parents=True, exist_ok=True)
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def _summarize(per_tensor:dict[str, Any], compute_peak_tflops:float) -> dict[str, Any]:
  for tensor, e in per_tensor.items():
    curve = e["curve"]
    fused = [(r["batch"], r["decode_q4_k_plus_matmul"]) for r in curve if "decode_q4_k_plus_matmul" in r]
    dense = {r["batch"]: r["matmul_decoded"] for r in curve if "matmul_decoded" in r}
    b1 = next((m["per_token_us"] for b, m in fused if b == 1), None)
    bmax, bmax_m = fused[-1] if fused else (None, None)
    # Robust: ratio of fused to dense throughput AT THE LARGEST BATCH (not a single noisy point).
    ratio_at_bmax = (round(bmax_m["achieved_tflops"] / dense[bmax]["achieved_tflops"], 3)
                     if (bmax in dense and dense[bmax]["achieved_tflops"]) else None)
    e["fused_per_token_speedup_b1_to_bmax"] = round(b1 / bmax_m["per_token_us"], 2) if (b1 and bmax_m) else None
    e["fused_vs_dense_throughput_at_bmax"] = ratio_at_bmax
    e["fused_pct_compute_peak_at_bmax"] = bmax_m.get("pct_compute_peak") if bmax_m else None
    e["dense_pct_compute_peak_at_bmax"] = dense.get(bmax, {}).get("pct_compute_peak")
  speedups = [e["fused_per_token_speedup_b1_to_bmax"] for e in per_tensor.values() if e["fused_per_token_speedup_b1_to_bmax"]]
  ratios = [e["fused_vs_dense_throughput_at_bmax"] for e in per_tensor.values() if e["fused_vs_dense_throughput_at_bmax"] is not None]
  steep = bool(speedups) and min(speedups) >= 3.0
  near_dense = bool(ratios) and min(ratios) >= 0.9
  if not steep:
    conclusion = "batching_gives_limited_per_token_amortization_re_examine"
  elif near_dense:
    conclusion = "batching_confirmed_lever_fused_path_approaches_dense_ceiling"
  else:
    conclusion = "batching_amortizes_dequant_strongly_but_fused_stays_below_dense_ceiling_motivates_b1_fused_gemm"
  return {
    "kind": "qk_flywheel_batched_b0", "phase": "Phase B0", "conclusion": conclusion,
    "fp16_compute_peak_tflops": compute_peak_tflops, "seq_lens": list(SEQ_LENS),
    "per_tensor": per_tensor,
    "metric": "per-token device latency and achieved FLOPS / measured fp16 compute roof",
    "note": ("decode_q4_k_plus_matmul = fused dequant+matmul (real quantized GEMM, dequant materialized to "
             "fp16 then matmul); matmul_decoded = weights pre-dequantized to fp16 then dense matmul (ceiling). "
             "Batching raises throughput/prefill, not single-stream per-token latency. The B=4 point is a noisy "
             "outlier; the robust verdict uses the fused-vs-dense ratio at the largest batch."),
  }

def rescore_b0(artifact:pathlib.Path, compute_peak_tflops:float|None=None) -> dict[str, Any]:
  d = json.loads((artifact / "summary.json").read_text())
  peak = compute_peak_tflops if compute_peak_tflops is not None else d["fp16_compute_peak_tflops"]
  per_tensor = {t: {"shape": e["shape"], "curve": e["curve"]} for t, e in d["per_tensor"].items()}
  summary = _summarize(per_tensor, peak)
  (artifact / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def main() -> int:
  p = argparse.ArgumentParser(description="Phase B0 batch-size efficiency curve")
  p.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  p.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  p.add_argument("--tensor", action="append", default=None)
  p.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT)
  p.add_argument("--device", default="AMD")
  p.add_argument("--rescore", action="store_true", help="recompute the verdict from existing summary.json, no GPU")
  args = p.parse_args()
  tensors = tuple(args.tensor) if args.tensor else DEFAULT_TENSORS
  if args.rescore:
    print(json.dumps(rescore_b0(args.artifact), indent=2, sort_keys=True))
  else:
    print(json.dumps(run_b0(args.repo, args.model, tensors, args.artifact, device=args.device), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
