#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import re
import shutil
import subprocess
import textwrap
import os
from typing import Any

from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/qk-decode-primitive-transfer/oracle_hip_runner"
RESULT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_hip_runner_result.json"
ROCPROF = pathlib.Path("/opt/rocm/bin/rocprofv3")
HIPCC = "/opt/rocm/bin/hipcc" if pathlib.Path("/opt/rocm/bin/hipcc").exists() else (shutil.which("hipcc") or "hipcc")
ROCM_ENV = {
  **os.environ,
  "LD_LIBRARY_PATH": "/opt/rocm/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
  "PATH": "/opt/rocm/bin:" + os.environ.get("PATH", ""),
}


def run(cmd: list[str], timeout: int = 120, env: dict[str, str] | None = None) -> dict[str, Any]:
  p = subprocess.run(cmd, cwd=ROOT, env=env or ROCM_ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout}


def runner_source() -> str:
  return HIP_MMVQ_GATEUP_SOURCE + r"""

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static void ck(hipError_t e, const char *what) {
  if (e != hipSuccess) {
    std::cerr << "{\"error\":\"" << what << "\",\"hip\":\"" << hipGetErrorString(e) << "\"}" << std::endl;
    std::exit(2);
  }
}

int main(int argc, char **argv) {
  int iters = 8;
  int warmups = 2;
  for (int i = 1; i < argc; i++) {
    std::string a(argv[i]);
    if (a == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--warmups" && i + 1 < argc) warmups = std::atoi(argv[++i]);
  }

  constexpr int rows = 12288;
  constexpr int dim = 4096;
  constexpr int nbpr = dim / QK_K;
  constexpr int y_blocks = nbpr * 8;
  const size_t x_blocks = static_cast<size_t>(rows) * nbpr;

  std::vector<block_q4_K> hx0(x_blocks), hx1(x_blocks);
  std::vector<block_q8_1> hy(y_blocks);
  std::vector<float> hd0(rows, 0.0f), hd1(rows, 0.0f);

  for (size_t i = 0; i < x_blocks; i++) {
    block_q4_K &a = hx0[i], &b = hx1[i];
    a.d = __float2half(0.03125f);
    a.dmin = __float2half(0.001953125f);
    b.d = __float2half(0.02734375f);
    b.dmin = __float2half(0.00146484375f);
    for (int j = 0; j < 12; j++) {
      a.scales[j] = static_cast<unsigned char>((i + j * 7) & 63);
      b.scales[j] = static_cast<unsigned char>((i * 3 + j * 5) & 63);
    }
    for (int j = 0; j < 128; j++) {
      a.qs[j] = static_cast<unsigned char>((i + j * 13) & 255);
      b.qs[j] = static_cast<unsigned char>((i * 11 + j * 17) & 255);
    }
  }
  for (int i = 0; i < y_blocks; i++) {
    hy[i].d = __float2half(0.015625f);
    hy[i].s = __float2half(0.0f);
    for (int j = 0; j < 32; j++) hy[i].qs[j] = static_cast<signed char>(((i * 19 + j * 23) & 255) - 128);
  }

  block_q4_K *dx0 = nullptr, *dx1 = nullptr;
  block_q8_1 *dy = nullptr;
  float *dd0 = nullptr, *dd1 = nullptr;
  ck(hipMalloc(&dx0, hx0.size() * sizeof(block_q4_K)), "hipMalloc dx0");
  ck(hipMalloc(&dx1, hx1.size() * sizeof(block_q4_K)), "hipMalloc dx1");
  ck(hipMalloc(&dy, hy.size() * sizeof(block_q8_1)), "hipMalloc dy");
  ck(hipMalloc(&dd0, hd0.size() * sizeof(float)), "hipMalloc dd0");
  ck(hipMalloc(&dd1, hd1.size() * sizeof(float)), "hipMalloc dd1");
  ck(hipMemcpy(dx0, hx0.data(), hx0.size() * sizeof(block_q4_K), hipMemcpyHostToDevice), "copy dx0");
  ck(hipMemcpy(dx1, hx1.data(), hx1.size() * sizeof(block_q4_K), hipMemcpyHostToDevice), "copy dx1");
  ck(hipMemcpy(dy, hy.data(), hy.size() * sizeof(block_q8_1), hipMemcpyHostToDevice), "copy dy");
  ck(hipMemset(dd0, 0, hd0.size() * sizeof(float)), "zero dd0");
  ck(hipMemset(dd1, 0, hd1.size() * sizeof(float)), "zero dd1");

  dim3 grid(rows, 2, 1);
  dim3 block(32, 4, 1);
  for (int i = 0; i < warmups; i++) {
    hipLaunchKernelGGL(q8_mmvq_gateup, grid, block, 0, 0, dd0, dd1, dx0, dx1, dy);
  }
  ck(hipDeviceSynchronize(), "warmup sync");

  hipEvent_t start, stop;
  ck(hipEventCreate(&start), "event create start");
  ck(hipEventCreate(&stop), "event create stop");
  ck(hipEventRecord(start), "event start");
  for (int i = 0; i < iters; i++) {
    hipLaunchKernelGGL(q8_mmvq_gateup, grid, block, 0, 0, dd0, dd1, dx0, dx1, dy);
  }
  ck(hipEventRecord(stop), "event stop");
  ck(hipEventSynchronize(stop), "event sync");
  float ms = 0.0f;
  ck(hipEventElapsedTime(&ms, start, stop), "event elapsed");
  ck(hipMemcpy(hd0.data(), dd0, hd0.size() * sizeof(float), hipMemcpyDeviceToHost), "copy out d0");
  ck(hipMemcpy(hd1.data(), dd1, hd1.size() * sizeof(float), hipMemcpyDeviceToHost), "copy out d1");

  double checksum = 0.0;
  bool finite = true;
  for (int idx : {0, 1, 17, 511, 4096, 12287}) {
    checksum += static_cast<double>(hd0[idx]) * 0.125 + static_cast<double>(hd1[idx]) * 0.25;
    finite = finite && std::isfinite(hd0[idx]) && std::isfinite(hd1[idx]);
  }

  std::cout
    << "{\"kernel\":\"q8_mmvq_gateup\""
    << ",\"rows\":" << rows
    << ",\"grid\":[12288,2,1]"
    << ",\"block\":[32,4,1]"
    << ",\"iters\":" << iters
    << ",\"warmups\":" << warmups
    << ",\"event_ms_total\":" << ms
    << ",\"event_us_per_kernel\":" << ((ms * 1000.0f) / static_cast<float>(iters))
    << ",\"finite\":" << (finite ? "true" : "false")
    << ",\"checksum\":" << checksum
    << "}" << std::endl;

  hipFree(dx0); hipFree(dx1); hipFree(dy); hipFree(dd0); hipFree(dd1);
  return finite ? 0 : 3;
}
"""


def parse_runner_json(text: str) -> dict[str, Any]:
  for line in reversed(text.splitlines()):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
      try:
        return json.loads(line)
      except json.JSONDecodeError:
        pass
  return {}


def find_csvs(path: pathlib.Path) -> list[pathlib.Path]:
  return sorted(path.rglob("*.csv"))


def parse_kernel_rows(csvs: list[pathlib.Path]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in csvs:
    try:
      text = path.read_text(errors="ignore")
    except OSError:
      continue
    if "q8_mmvq_gateup" not in text:
      continue
    for raw in text.splitlines():
      if "q8_mmvq_gateup" not in raw:
        continue
      rows.append({"path": str(path.relative_to(ROOT)), "raw": raw})
  return rows


def parse_kernel_trace_csv(csvs: list[pathlib.Path]) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for path in csvs:
    try:
      with path.open(newline="") as f:
        for row in csv.DictReader(f):
          if row.get("Kernel_Name") != "q8_mmvq_gateup":
            continue
          rec: dict[str, Any] = {"path": str(path.relative_to(ROOT))}
          for k, v in row.items():
            if v is None:
              rec[k] = v
            elif re.fullmatch(r"-?\d+", v):
              rec[k] = int(v)
            elif re.fullmatch(r"-?\d+\.\d+", v):
              rec[k] = float(v)
            else:
              rec[k] = v
          if isinstance(rec.get("Start_Timestamp"), int) and isinstance(rec.get("End_Timestamp"), int):
            rec["Duration_ns"] = rec["End_Timestamp"] - rec["Start_Timestamp"]
          rows.append(rec)
    except OSError:
      continue
  return rows


def summarize_kernel_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    return {}
  durations = [row["Duration_ns"] for row in rows if isinstance(row.get("Duration_ns"), int)]
  first = rows[0]
  return {
    "dispatches": len(rows),
    "duration_ns_min": min(durations) if durations else None,
    "duration_ns_max": max(durations) if durations else None,
    "duration_ns_avg": (sum(durations) / len(durations)) if durations else None,
    "resource_fields_first": {
      "LDS_Block_Size": first.get("LDS_Block_Size"),
      "Scratch_Size": first.get("Scratch_Size"),
      "VGPR_Count": first.get("VGPR_Count"),
      "Accum_VGPR_Count": first.get("Accum_VGPR_Count"),
      "SGPR_Count": first.get("SGPR_Count"),
      "Workgroup_Size_X": first.get("Workgroup_Size_X"),
      "Workgroup_Size_Y": first.get("Workgroup_Size_Y"),
      "Workgroup_Size_Z": first.get("Workgroup_Size_Z"),
      "Grid_Size_X": first.get("Grid_Size_X"),
      "Grid_Size_Y": first.get("Grid_Size_Y"),
      "Grid_Size_Z": first.get("Grid_Size_Z"),
    },
  }


def infer_trace_fields(rows: list[dict[str, Any]]) -> dict[str, Any]:
  if not rows:
    return {}
  raw = rows[0]["raw"]
  fields = next(csv.reader([raw]))
  numeric = []
  for item in fields:
    s = item.strip()
    if re.fullmatch(r"-?\d+(\.\d+)?", s):
      numeric.append(float(s) if "." in s else int(s))
  return {"first_row_fields": fields, "numeric_values": numeric[:20]}


def main() -> int:
  OUTDIR.mkdir(parents=True, exist_ok=True)
  cpp = OUTDIR / "q8_mmvq_gateup_runner.cpp"
  exe = OUTDIR / "q8_mmvq_gateup_runner"
  prof = OUTDIR / "rocprof_kernel_trace"
  cpp.write_text(runner_source())

  compile_cmd = [
    str(HIPCC), "-O3", "--offload-arch=gfx1100", "--rocm-path=/opt/rocm",
    "-isystem", "/opt/rocm/include", "-L/opt/rocm/lib", "-Wl,-rpath,/opt/rocm/lib",
    str(cpp), "-o", str(exe),
  ]
  compile_res = run(compile_cmd, timeout=240)
  direct_res = {"returncode": None, "stdout": ""}
  rocprof_res = {"returncode": None, "stdout": ""}
  kernel_rows: list[dict[str, Any]] = []
  kernel_records: list[dict[str, Any]] = []
  if compile_res["returncode"] == 0:
    direct_res = run([str(exe), "--warmups", "2", "--iters", "5"], timeout=120)
    if direct_res["returncode"] == 0 and ROCPROF.exists():
      if prof.exists():
        shutil.rmtree(prof)
      prof.mkdir(parents=True, exist_ok=True)
      rocprof_res = run([
        str(ROCPROF), "--kernel-trace", "-f", "csv", "-d", str(prof), "-o", "q8_gateup",
        "--", str(exe), "--warmups", "1", "--iters", "3",
      ], timeout=180)
      csvs = find_csvs(prof)
      kernel_rows = parse_kernel_rows(csvs)
      kernel_records = parse_kernel_trace_csv(csvs)

  runner = parse_runner_json(direct_res.get("stdout", ""))
  prof_runner = parse_runner_json(rocprof_res.get("stdout", ""))
  gates = {
    "hipcc_present": pathlib.Path(str(HIPCC)).exists() or shutil.which(str(HIPCC)) is not None,
    "rocprofv3_present": ROCPROF.exists(),
    "compiled": compile_res["returncode"] == 0,
    "direct_run_passed": direct_res.get("returncode") == 0 and runner.get("finite") is True,
    "rocprof_run_passed": rocprof_res.get("returncode") == 0,
    "kernel_trace_has_gateup": bool(kernel_rows),
  }
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_HIP_RUNNER_OES5_SURFACE",
    "schema": "decode_oracle_hip_runner_probe_v1",
    "verdict": "PASS_DECODE_ORACLE_ROCPROF_VISIBLE_HIP_RUNNER" if all(gates.values()) else "BLOCKED_DECODE_ORACLE_HIP_RUNNER",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "paths": {
      "cpp": str(cpp.relative_to(ROOT)),
      "exe": str(exe.relative_to(ROOT)),
      "rocprof_dir": str(prof.relative_to(ROOT)),
    },
    "runner": runner,
    "profiler_runner": prof_runner,
    "kernel_trace_rows": kernel_rows[:20],
    "kernel_trace_inferred": infer_trace_fields(kernel_rows),
    "kernel_trace_records": kernel_records[:20],
    "kernel_trace_summary": summarize_kernel_trace(kernel_records),
    "commands": {
      "compile": compile_cmd,
      "direct": direct_res.get("cmd"),
      "rocprof": rocprof_res.get("cmd"),
    },
    "environment": {
      "hipcc": str(HIPCC),
      "rocprofv3": str(ROCPROF),
      "ld_library_path_prefix": "/opt/rocm/lib",
    },
    "returncodes": {
      "compile": compile_res["returncode"],
      "direct": direct_res.get("returncode"),
      "rocprof": rocprof_res.get("returncode"),
    },
    "stdout_tail": {
      "compile": compile_res["stdout"][-4000:],
      "direct": direct_res.get("stdout", "")[-4000:],
      "rocprof": rocprof_res.get("stdout", "")[-6000:],
    },
    "gates": gates,
    "next": {
      "if_pass": "Run ATT or PC sampling against this HIP runner and join PCs to the OES-4 semantic map.",
      "if_blocked": "Use the compile/direct/rocprof stdout tails to classify whether the blocker is host compile, HIP runtime, or rocprof output parsing.",
    },
  }
  RESULT.parent.mkdir(parents=True, exist_ok=True)
  RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "runner": runner,
    "kernel_trace_rows": len(kernel_rows),
    "out": str(RESULT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
