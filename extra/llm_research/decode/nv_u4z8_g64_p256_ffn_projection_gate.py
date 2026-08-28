#!/usr/bin/env python3
"""First performance gate for U4Z8 on one 12288x4096 FFN projection.

Gate and up have the same matrix shape and representation, so this deliberately
tests one projection before implementing the paired SiLU(gate)*up topology.  It
reuses the independently qualified packet oracle and CUDA harness from the O
gate, changing only the row extent and allocation constants.  No production
route or model artifact is modified.
"""
from __future__ import annotations

import argparse, json, os, re, shutil, statistics, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nv_u4z8_g64_p256_o_microgate as base

ROWS, K, ROTATIONS = 12288, 4096, 16
KBLOCKS = K // 256
CW, UW = ROWS * KBLOCKS * 36, ROWS * KBLOCKS * 34
CB, UB = CW * 4, UW * 4
CONTROL = f"q4k_g3_lanemap_gemv_vec_epi_resadd_{ROWS}_{K}"
CANDIDATE = f"u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_{ROWS}_{K}"
CUDA_BIN = "/usr/local/cuda-13.2/bin"


def render() -> tuple[str, str]:
  old = (base.ROWS, base.K, base.K_BLOCKS, base.CONTROL_WORDS, base.CANDIDATE_WORDS,
         base.CONTROL_WEIGHT_BYTES, base.CANDIDATE_WEIGHT_BYTES, base.CONTROL, base.CANDIDATE)
  try:
    base.ROWS, base.K, base.K_BLOCKS = ROWS, K, KBLOCKS
    base.CONTROL_WORDS, base.CANDIDATE_WORDS = CW, UW
    base.CONTROL_WEIGHT_BYTES, base.CANDIDATE_WEIGHT_BYTES = CB, UB
    base.CONTROL, base.CANDIDATE = CONTROL, CANDIDATE
    return base._render()
  finally:
    (base.ROWS, base.K, base.K_BLOCKS, base.CONTROL_WORDS, base.CANDIDATE_WORDS,
     base.CONTROL_WEIGHT_BYTES, base.CANDIDATE_WEIGHT_BYTES, base.CONTROL, base.CANDIDATE) = old


def source() -> str:
  control, candidate = render()
  text = base.HARNESS.replace("__CONTROL_SOURCE__", control).replace("__CANDIDATE_SOURCE__", candidate)
  replacements = {
    "#define ROWS 4096": f"#define ROWS {ROWS}",
    "#define CONTROL_WORDS 2359296": f"#define CONTROL_WORDS {CW}",
    "#define CANDIDATE_WORDS 2228224": f"#define CANDIDATE_WORDS {UW}",
    "q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096": CONTROL,
    "u4z8_g64_p256_lanemap_gemv_vec_epi_resadd_4096_4096": CANDIDATE,
  }
  for before, after in replacements.items():
    if before not in text: raise RuntimeError(f"missing harness marker: {before}")
    text = text.replace(before, after)
  return text


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--hot-passes", type=int, default=160)
  ap.add_argument("--cold-passes", type=int, default=32)
  ap.add_argument("--reps", type=int, default=9)
  ap.add_argument("--threshold-us", type=float, default=0.30)
  ap.add_argument("--out", type=Path, required=True)
  ap.add_argument("--artifact-dir", type=Path)
  args = ap.parse_args()
  with tempfile.TemporaryDirectory(prefix="nv_u4z8_ffn_") as td:
    src, binary = Path(td)/"gate.cu", Path(td)/"gate"
    src.write_text(source())
    env = {**os.environ, "PATH": f"{CUDA_BIN}:" + os.environ.get("PATH", "")}
    build = subprocess.run(["nvcc", "-arch=sm_120a", "-O3", "-std=c++17", "--ptxas-options=-v",
      str(src), "-o", str(binary)], capture_output=True, text=True, env=env)
    if build.returncode:
      print(build.stderr[-12000:], file=sys.stderr); return 3
    if args.artifact_dir:
      args.artifact_dir.mkdir(parents=True, exist_ok=True)
      shutil.copy2(src, args.artifact_dir/"gate.cu")
      (args.artifact_dir/"ptxas.txt").write_text(build.stderr)
    run = subprocess.run([str(binary), str(args.hot_passes), str(args.cold_passes), str(args.reps), "batch"],
      capture_output=True, text=True)
    print(run.stdout.strip())
    if run.returncode not in (0, 5): print(run.stderr[-8000:], file=sys.stderr); return 4
    fixtures = [dict(fixture=int(m[1]), finite=bool(int(m[2])), guards=bool(int(m[3])), readonly=bool(int(m[4])),
      bad=int(m[5]), max_abs=float(m[6]), max_rel=float(m[7])) for m in re.finditer(
      r"fixture=(\d+) finite=(\d+) guards=(\d+) readonly=(\d+) bad=(\d+) max_abs=([0-9.eE+-]+) max_rel=([0-9.eE+-]+)", run.stdout)]
    samples = {k: [] for k in ("hot_control", "hot_candidate", "cold_control", "cold_candidate")}
    pat = re.compile(r"hot_control_us=([0-9.]+) hot_candidate_us=([0-9.]+) cold_control_us=([0-9.]+) cold_candidate_us=([0-9.]+)")
    for m in pat.finditer(run.stdout):
      for key, val in zip(samples, m.groups()): samples[key].append(float(val))
    med = {k: statistics.median(v) for k, v in samples.items()}
    correct = len(fixtures) == 3 and all(x["finite"] and x["guards"] and x["readonly"] and x["bad"] == 0 for x in fixtures)
    recovery = med["cold_control"] - med["cold_candidate"]
    result = {"schema": "tinygrad.nv_u4z8_g64_p256_ffn_projection_gate.v1",
      "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
      "method": "single gate/up-shaped projection; continuous 16-copy rotated-cold service; no production edits",
      "shape": {"rows": ROWS, "k": K, "control_weight_bytes": CB, "candidate_weight_bytes": UB,
        "byte_reduction_fraction": 1-UB/CB}, "correctness": {"fixtures": fixtures, "pass": correct},
      "timing": {"unit": "us_per_projection", "samples": samples, "medians": med,
        "hot_recovery_us": med["hot_control"]-med["hot_candidate"], "cold_recovery_us": recovery,
        "control_cold_rate_tb_s": CB/med["cold_control"]/1e6, "candidate_cold_rate_tb_s": UB/med["cold_candidate"]/1e6},
      "sass": {"control": base._sass(binary, CONTROL), "candidate": base._sass(binary, CANDIDATE)},
      "threshold": {"cold_recovery_us_per_projection": args.threshold_us},
      "verdict": "PASS" if correct and recovery >= args.threshold_us else "STOP"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if correct else 5


if __name__ == "__main__": raise SystemExit(main())
