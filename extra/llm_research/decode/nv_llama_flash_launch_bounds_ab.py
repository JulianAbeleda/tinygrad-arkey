#!/usr/bin/env python3
"""Ablate llama's two-argument launch bound without changing Flash source semantics.

Builds the pinned llama Flash decode probe twice:
  control:  __launch_bounds__(128, 1), as shipped by llama
  ablated:  __launch_bounds__(128), removing only the min-blocks compiler contract

The copied header is placed first on the include path, so the llama checkout is
never edited.  Results include ptxas resources, SASS, repeated hot/cold CUDA
event timing, and an optional cold Nsight Compute pass.
"""
from __future__ import annotations

import argparse, csv, io, json, os, re, statistics, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LLAMA = Path("/home/ubuntu/env/llama.cpp")
NVCC = Path("/usr/local/cuda-13.2/bin/nvcc")
CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
NCU = Path("/usr/local/bin/ncu")
SOURCE = ROOT / "extra/llm_research/microbench/llama_fattn_vec_iso.cu"
HEADER = LLAMA / "ggml/src/ggml-cuda/fattn-vec.cuh"
BOUND = "__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device(), 1)"
ABLATION = "__launch_bounds__(ggml_cuda_fattn_vec_get_nthreads_device())"


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, text=True, capture_output=True, check=check)


def parse_us(stdout: str) -> float:
  match = re.search(r"sequence_us_per_replay=([0-9.]+)", stdout)
  if not match: raise RuntimeError(f"timing output missing: {stdout}")
  return float(match.group(1))


def resources(log: str) -> dict[str, int | None]:
  reg = re.findall(r"Used (\d+) registers", log)
  spill_store = re.findall(r"(\d+) bytes spill stores", log)
  spill_load = re.findall(r"(\d+) bytes spill loads", log)
  return {
    # The probe also contains the low-register cache conditioner.  Flash is the
    # maximum-register entry in this deliberately two-kernel binary.
    "registers_per_thread": max(map(int, reg), default=None),
    "spill_stores_bytes": max(map(int, spill_store), default=0),
    "spill_loads_bytes": max(map(int, spill_load), default=0),
  }


def sass_summary(sass: str) -> dict[str, int]:
  # NCU source views start with a runtime address; cuobjdump source lines have
  # an address comment followed by the instruction.  Ignore encoding-only
  # comments so distances are instruction slots rather than text lines.
  lines = [line for line in sass.splitlines() if re.match(r"^0x[0-9a-f]+", line)]
  if not lines:
    lines = [line for line in sass.splitlines() if re.search(r"/\* [0-9a-fx]+ \*/\s+\S", line)]
  ordinary = [i for i, line in enumerate(lines) if "LDG.E.128 " in line and "CONSTANT" not in line]
  return {
    "instruction_lines": len(lines),
    "ldg_e_128": sum("LDG.E.128" in line for line in lines),
    "ldg_e": sum("LDG.E" in line for line in lines),
    "ordinary_ldg_e_128": len(ordinary),
    "first_16_load_span_instructions": ordinary[15] - ordinary[0] + 1 if len(ordinary) >= 16 else 0,
    "second_16_load_span_instructions": ordinary[31] - ordinary[16] + 1 if len(ordinary) >= 32 else 0,
  }


def ncu_metrics(binary: Path, condition_mib: int, report: Path, gridy: int, tc: int) -> dict:
  report_file = report.with_suffix(".ncu-rep")
  metrics = ",".join([
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
  ])
  # Match the retained tinygrad cold-counter protocol.  The explicit flush is
  # stronger and less topology-dependent than inferring eviction from a read
  # conditioner.
  cmd = ["sudo", "-n", str(NCU), "--target-processes", "all", "--cache-control", "all",
         "--kernel-name", "regex:flash_attn_ext_vec", "--launch-skip", "0", "--launch-count", "1",
         "--metrics", metrics, "--force-overwrite", "--export", str(report), "--csv",
         str(binary), "--warmup", "0", "--replays", "1",
         "--gridy", str(gridy), "--tc", str(tc), "--condition-mib", str(condition_mib)]
  cp = run(cmd)
  raw = run([str(NCU), "--import", str(report_file), "--csv", "--page", "raw"]).stdout
  table = list(csv.reader(io.StringIO(raw)))
  rows: dict[str, str] = {}
  if len(table) >= 3:
    header, units, values = table[:3]
    for metric in metrics.split(","):
      if metric in header:
        idx = header.index(metric)
        rows[metric] = f"{values[idx]} {units[idx]}".strip()
  source = run([str(NCU), "--import", str(report_file), "--page", "source", "--print-source", "sass"]).stdout
  source_path = report.with_suffix(".sass.txt")
  source_path.write_text(source)
  return {"report": str(report_file), "source_view": str(source_path), "parsed": rows, "sass": sass_summary(source)}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--runs", type=int, default=7)
  ap.add_argument("--replays", type=int, default=800)
  ap.add_argument("--condition-mib", type=int, default=96)
  ap.add_argument("--gridy", type=int, default=6)
  ap.add_argument("--tc", type=int, default=768)
  ap.add_argument("--max-tc", type=int, default=768)
  ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--out", type=Path, required=True)
  args = ap.parse_args()

  original = HEADER.read_text()
  if original.count(BOUND) != 1: raise RuntimeError("pinned llama launch-bound spelling changed")
  result: dict = {
    "schema": "tinygrad.nv_llama_flash_launch_bounds_ab.v1",
    "llama_commit": run(["git", "-C", str(LLAMA), "rev-parse", "HEAD"]).stdout.strip(),
    "invariant": "same source/math/bytes/grid; only minBlocksPerMultiprocessor compiler contract removed",
    "arms": {},
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.TemporaryDirectory(prefix="nv_llama_flash_lb_") as tmp_name:
    tmp = Path(tmp_name)
    for name, header in (("control_min_blocks_1", original), ("ablated_one_arg", original.replace(BOUND, ABLATION))):
      overlay = tmp / name / "ggml-cuda"
      overlay.mkdir(parents=True)
      (overlay / "fattn-vec.cuh").write_text(header)
      binary = tmp / f"llama_flash_{name}"
      cmd = [str(NVCC), "-O3", "-std=c++17", f"-DLLAMA_FLASH_MAX_TC={args.max_tc}",
             "--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]",
             "--ptxas-options=-v", "-I", str(overlay.parent),
             "-I", str(LLAMA / "ggml/src/ggml-cuda"), "-I", str(LLAMA / "ggml/src"),
             "-I", str(LLAMA / "ggml/include"), "-isystem", "/usr/local/cuda/targets/x86_64-linux/include",
             str(SOURCE), "-o", str(binary)]
      build = run(cmd)
      nvdisasm = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
      disasm_env = dict(os.environ, NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{os.environ.get('PATH', '')}")
      disasm = subprocess.run([str(CUOBJDUMP), "--dump-sass", str(binary)], text=True, capture_output=True, env=disasm_env)
      sass = disasm.stdout
      base_run = [str(binary), "--replays", str(args.replays), "--warmup", "20", "--gridy", str(args.gridy), "--tc", str(args.tc)]
      hot = [parse_us(run(base_run).stdout) for _ in range(args.runs)]
      cold = [parse_us(run(base_run + ["--condition-mib", str(args.condition_mib)]).stdout) for _ in range(args.runs)]
      arm = {
        "build_command": cmd,
        "build_log": build.stdout + build.stderr,
        "resources": resources(build.stdout + build.stderr),
        "sass": sass_summary(sass),
        "sass_disassembly_error": disasm.stderr.strip() if disasm.returncode else None,
        "hot_sequence_us": hot,
        "hot_sequence_median_us": statistics.median(hot),
        "cold_sequence_us": cold,
        "cold_sequence_median_us": statistics.median(cold),
      }
      if args.ncu:
        report = args.out.parent / f"llama-launch-bounds-{name}"
        arm["ncu_cold_second_flash"] = ncu_metrics(binary, args.condition_mib, report, args.gridy, args.tc)
      result["arms"][name] = arm
    control = result["arms"]["control_min_blocks_1"]
    ablated = result["arms"]["ablated_one_arg"]
    result["deltas_ablated_minus_control"] = {
      "hot_sequence_us": ablated["hot_sequence_median_us"] - control["hot_sequence_median_us"],
      "cold_sequence_us": ablated["cold_sequence_median_us"] - control["cold_sequence_median_us"],
      "registers_per_thread": ablated["resources"]["registers_per_thread"] - control["resources"]["registers_per_thread"],
    }
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
