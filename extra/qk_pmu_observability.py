#!/usr/bin/env python3
"""Primitive-local PMU observability probe.

Implements PMU-1..PMU-3 from docs/primitive-pmu-observability-scope-20260619.md:
  - inventory ROCm profiler tools and gfx1100 counters
  - run a HIP/rocBLAS control under rocprofv3 trace + PMC collection
  - run a tinygrad HCQ smoke under the same trace path and classify visibility

The raw profiler files stay under bench/qk-pmu-observability/raw/. The committed
artifact is the compact result.json summary.
"""
from __future__ import annotations

import argparse, contextlib, csv, json, os, pathlib, shutil, subprocess, textwrap, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-pmu-observability"
RAW = OUT / "raw"
ROCM_BINS = [pathlib.Path("/opt/rocm/bin"), pathlib.Path("/opt/rocm-7.2.4/bin")]
COUNTERS = [
  "SQ_INSTS_VALU", "SQ_INSTS_SALU", "SQ_INSTS_LDS", "SQ_INSTS_SMEM", "SQ_INSTS_TEX_LOAD",
  "SQ_WAIT_ANY", "SQ_WAVES", "L2CacheHit", "MemUnitBusy",
]

def _tool(name:str) -> str | None:
  found = shutil.which(name)
  if found: return found
  for root in ROCM_BINS:
    cand = root / name
    if cand.exists() and os.access(cand, os.X_OK): return str(cand)
  return None

def _run(cmd:list[str], *, cwd:pathlib.Path=ROOT, env:dict[str, str]|None=None, timeout:int=120,
         stdout:pathlib.Path|None=None, stderr:pathlib.Path|None=None) -> dict[str, Any]:
  started = time.time()
  out_ctx = stdout.open("w") if stdout else contextlib.nullcontext(subprocess.DEVNULL)
  err_ctx = stderr.open("w") if stderr else contextlib.nullcontext(subprocess.DEVNULL)
  with out_ctx as out, err_ctx as err:
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=out, stderr=err, timeout=timeout)
  return {"cmd": cmd, "returncode": p.returncode, "elapsed_s": round(time.time()-started, 3),
          "stdout": str(stdout.relative_to(ROOT)) if stdout and stdout.is_relative_to(ROOT) else str(stdout) if stdout else None,
          "stderr": str(stderr.relative_to(ROOT)) if stderr and stderr.is_relative_to(ROOT) else str(stderr) if stderr else None}

def _read_csv(path:pathlib.Path) -> list[dict[str, str]]:
  if not path.exists(): return []
  with path.open(newline="") as f:
    return list(csv.DictReader(f))

def _summarize_kernel_trace(path:pathlib.Path) -> dict[str, Any]:
  rows = _read_csv(path)
  names: dict[str, dict[str, Any]] = {}
  for row in rows:
    name = row.get("Kernel_Name", "")
    start, end = int(row.get("Start_Timestamp", "0") or 0), int(row.get("End_Timestamp", "0") or 0)
    dur = max(0, end-start)
    item = names.setdefault(name, {"calls": 0, "total_ns": 0, "max_ns": 0, "vgpr": row.get("VGPR_Count"),
                                  "sgpr": row.get("SGPR_Count"), "lds": row.get("LDS_Block_Size"),
                                  "grid": [row.get("Grid_Size_X"), row.get("Grid_Size_Y"), row.get("Grid_Size_Z")],
                                  "workgroup": [row.get("Workgroup_Size_X"), row.get("Workgroup_Size_Y"), row.get("Workgroup_Size_Z")]})
    item["calls"] += 1; item["total_ns"] += dur; item["max_ns"] = max(item["max_ns"], dur)
  top = sorted(({"kernel": k, **v, "avg_ns": (v["total_ns"]/v["calls"] if v["calls"] else 0)} for k,v in names.items()),
               key=lambda x: x["total_ns"], reverse=True)
  return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
          "exists": path.exists(), "dispatches": len(rows), "unique_kernels": len(names), "top_kernels": top[:8]}

def _summarize_pmc(path:pathlib.Path) -> dict[str, Any]:
  rows = _read_csv(path)
  by_counter: dict[str, dict[str, Any]] = {}
  nonzero = 0
  for row in rows:
    name = row.get("Counter_Name", "")
    try: val = float(row.get("Counter_Value", "0") or 0)
    except ValueError: val = 0.0
    if val != 0.0: nonzero += 1
    item = by_counter.setdefault(name, {"rows": 0, "nonzero_rows": 0, "sum": 0.0, "max": 0.0})
    item["rows"] += 1; item["sum"] += val; item["max"] = max(item["max"], val)
    if val != 0.0: item["nonzero_rows"] += 1
  return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
          "exists": path.exists(), "rows": len(rows), "nonzero_rows": nonzero, "counters": by_counter}

def inventory() -> dict[str, Any]:
  rocprofv3, avail = _tool("rocprofv3"), _tool("rocprofv3-avail")
  inv: dict[str, Any] = {
    "tools": {name: _tool(name) for name in ["rocprofv3", "rocprofv3-avail", "rocprof-compute", "rocprof-compute-viewer"]},
    "rocprofv3_version": None,
    "available_counter_count": 0,
    "available_counters_sample": [],
    "requested_counters": COUNTERS,
  }
  if rocprofv3:
    p = subprocess.run([rocprofv3, "--version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
    inv["rocprofv3_version"] = p.stdout.strip().splitlines()
  if avail:
    p = subprocess.run([avail, "list"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    words = [w for w in p.stdout.replace(":", " ").split() if w.isidentifier() or "_" in w]
    counters = sorted(set(w for w in words if w.startswith(("SQ_", "GL2C_", "TA_", "L2", "LDS", "Mem", "Wave", "GPU", "VALU", "SALU"))))
    inv["available_counter_count"] = len(counters)
    inv["available_counters_sample"] = counters[:80]
  return inv

def compile_hip_control(outbin:pathlib.Path) -> dict[str, Any]:
  cmd = ["g++", "-std=c++17", "-D__HIP_PLATFORM_AMD__=1", "-I/opt/rocm-7.2.4/include",
         "-L/opt/rocm-7.2.4/lib", "-Wl,-rpath,/opt/rocm-7.2.4/lib",
         "extra/qk_prefill_blas_ceiling.cpp", "-lamdhip64", "-lrocblas", "-lhipblaslt", "-o", str(outbin)]
  return _run(cmd, timeout=60, stderr=RAW / "hip_control_compile.stderr.txt")

def run_rocprof_trace(app_cmd:list[str], outdir:pathlib.Path, *, include_pmc:bool) -> dict[str, Any]:
  rocprofv3 = _tool("rocprofv3")
  assert rocprofv3 is not None
  outdir.mkdir(parents=True, exist_ok=True)
  cmd = [rocprofv3]
  if include_pmc:
    cmd += ["--pmc", *COUNTERS, "--kernel-include-regex", "Cijk|UserArgs|rocblas|hipblas|test_pmu|matmul"]
  else:
    cmd += ["--kernel-trace", "--hsa-trace", "--stats", "--summary"]
  cmd += ["-f", "csv", "json", "--output-directory", str(outdir), "--output-file", "trace", "--", *app_cmd]
  return _run(cmd, timeout=180, stdout=outdir / "app_stdout.txt", stderr=outdir / "rocprof_stderr.txt")

def hip_control() -> dict[str, Any]:
  bindir = RAW / "bin"; bindir.mkdir(parents=True, exist_ok=True)
  outbin = bindir / "qk_prefill_blas_ceiling_pmu"
  comp = compile_hip_control(outbin)
  res: dict[str, Any] = {"compile": comp, "trace": {}, "pmc": {}, "verdict": "KILL"}
  if comp["returncode"] != 0: return res
  trace_run = run_rocprof_trace([str(outbin)], RAW / "hip_control_trace", include_pmc=False)
  pmc_run = run_rocprof_trace([str(outbin)], RAW / "hip_control_pmc", include_pmc=True)
  trace = _summarize_kernel_trace(RAW / "hip_control_trace/trace_kernel_trace.csv")
  pmc = _summarize_pmc(RAW / "hip_control_pmc/trace_counter_collection.csv")
  res.update({"trace_run": trace_run, "pmc_run": pmc_run, "trace": trace, "pmc": pmc,
              "verdict": "PASS" if trace["dispatches"] > 0 and pmc["rows"] > 0 else "KILL"})
  return res

def tinygrad_hcq() -> dict[str, Any]:
  script = RAW / "tinygrad_hcq_smoke.py"
  script.write_text(textwrap.dedent("""
    from tinygrad import Tensor, Device, dtypes
    Tensor.manual_seed(0)
    a = Tensor.randn(512, 512, dtype=dtypes.half).realize()
    b = Tensor.randn(512, 512, dtype=dtypes.half).realize()
    c = (a @ b).realize()
    Device[Device.DEFAULT].synchronize()
    print(c.shape)
  """).lstrip())
  env = os.environ.copy(); env["DEV"] = "AMD"; env["PYTHONPATH"] = "."
  rocprofv3 = _tool("rocprofv3")
  assert rocprofv3 is not None
  outdir = RAW / "tinygrad_hcq_trace"; outdir.mkdir(parents=True, exist_ok=True)
  cmd = [rocprofv3, "--kernel-trace", "--hsa-trace", "--stats", "--summary", "-f", "csv", "json",
         "--output-directory", str(outdir), "--output-file", "trace", "--", ".venv/bin/python", str(script)]
  run = _run(cmd, env=env, timeout=120, stdout=outdir / "app_stdout.txt", stderr=outdir / "rocprof_stderr.txt")
  trace = _summarize_kernel_trace(outdir / "trace_kernel_trace.csv")
  verdict = "PASS_VISIBLE" if trace["dispatches"] > 0 else "VISIBILITY_GAP"
  return {"run": run, "trace": trace, "verdict": verdict,
          "classification": "rocprof_hcq_visible" if trace["dispatches"] > 0 else "rocprof_hcq_visibility_gap"}

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", type=pathlib.Path, default=OUT / "result.json")
  ap.add_argument("--skip-run", action="store_true", help="only inventory tools/counters")
  args = ap.parse_args()
  RAW.mkdir(parents=True, exist_ok=True)
  inv = inventory()
  result: dict[str, Any] = {"schema": "qk_pmu_observability_v1", "generated_at": int(time.time()),
                            "inventory": inv, "hip_control": None, "tinygrad_hcq": None, "verdict": "INVENTORY_ONLY"}
  if not args.skip_run:
    if not inv["tools"].get("rocprofv3"):
      result["verdict"] = "KILL_NO_ROCPROF"
    else:
      result["hip_control"] = hip_control()
      result["tinygrad_hcq"] = tinygrad_hcq()
      hip_ok = result["hip_control"]["verdict"] == "PASS"
      hcq_gap = result["tinygrad_hcq"]["verdict"] == "VISIBILITY_GAP"
      result["verdict"] = "REDIRECT_HCQ_NATIVE_ADAPTER" if hip_ok and hcq_gap else ("PASS_ROCPROF_VISIBLE" if hip_ok else "KILL")
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(args.out.relative_to(ROOT)), "verdict": result["verdict"],
                    "hip": result["hip_control"]["verdict"] if result["hip_control"] else None,
                    "hcq": result["tinygrad_hcq"]["verdict"] if result["tinygrad_hcq"] else None}, indent=2))
  return 0 if result["verdict"] not in {"KILL", "KILL_NO_ROCPROF"} else 2

if __name__ == "__main__":
  raise SystemExit(main())
