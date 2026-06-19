#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, shutil, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
WORKDIR = OUTDIR / "t1c_att_decoder_repair_work"
OUT = OUTDIR / "t1c_att_decoder_repair.json"
ROCM = pathlib.Path("/opt/rocm-7.2.4")

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#define CHECK(x) do { hipError_t e=(x); if(e!=hipSuccess){fprintf(stderr,"HIP %s\n",hipGetErrorString(e)); return 2;} } while(0)
__global__ void body_kernel(float *out, const float *a, const float *b, int n) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  float x = a[gid], y = b[gid];
  #pragma unroll 64
  for (int i = 0; i < 64; i++) { x = fmaf(x, 1.000113f, y); y = fmaf(y, 0.999887f, x); }
  out[gid] = x + y;
}
int main() {
  int n = 1 << 22;
  float *a=nullptr,*b=nullptr,*c=nullptr;
  CHECK(hipSetDevice(0));
  CHECK(hipMalloc(&a,n*sizeof(float))); CHECK(hipMalloc(&b,n*sizeof(float))); CHECK(hipMalloc(&c,n*sizeof(float)));
  CHECK(hipMemset(a,1,n*sizeof(float))); CHECK(hipMemset(b,2,n*sizeof(float)));
  dim3 block(256), grid((n + block.x - 1) / block.x);
  body_kernel<<<grid, block>>>(c, a, b, n); CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
  body_kernel<<<grid, block>>>(c, a, b, n); CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
  CHECK(hipFree(a)); CHECK(hipFree(b)); CHECK(hipFree(c));
  return 0;
}
"""

def run(cmd: list[str], *, cwd: pathlib.Path = ROOT, env: dict[str, str] | None = None, timeout: int = 180) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter()-t0, 3),
            "stdout_tail": cp.stdout.splitlines()[-80:], "stderr_tail": cp.stderr.splitlines()[-80:]}
  except subprocess.TimeoutExpired as e:
    return {"cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter()-t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-80:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-80:] if isinstance(e.stderr, str) else []}

def build_control() -> dict[str, Any]:
  src, exe = WORKDIR / "att_body_control.cpp", WORKDIR / "att_body_control"
  src.write_text(HIP_SRC)
  cmd = [shutil.which("hipcc") or "/usr/bin/hipcc", "--offload-arch=gfx1100", "-O3", str(src), "-o", str(exe)]
  res = run(cmd, cwd=WORKDIR)
  return {"source": str(src.relative_to(ROOT)), "exe": str(exe.relative_to(ROOT)), "build": res,
          "ok": res.get("returncode") == 0 and exe.exists()}

def inspect_packages() -> dict[str, Any]:
  debs = sorted(pathlib.Path("/tmp/rocprof_pkg").glob("*.deb"))
  rows = []
  for deb in debs:
    res = run(["dpkg-deb", "-c", str(deb)], timeout=60)
    hits = [ln for ln in res.get("stdout_tail", []) if "trace-decoder" in ln or "libatt" in ln]
    # stdout_tail may miss hits in long output; run a tighter shell grep for the exact package.
    grep = run(["bash", "-lc", f"dpkg-deb -c {str(deb)!r} | rg 'trace-decoder|libatt|rocprofiler-sdk.*so' || true"], timeout=60)
    rows.append({"deb": deb.name, "returncode": res.get("returncode"), "grep": grep.get("stdout_tail", [])})
  return {"download_dir": "/tmp/rocprof_pkg", "packages": rows}

def run_rocprof(exe_rel: str, variant: dict[str, str | None]) -> dict[str, Any]:
  outdir = WORKDIR / f"rocprof_{variant['name']}"
  shutil.rmtree(outdir, ignore_errors=True)
  outdir.mkdir(parents=True, exist_ok=True)
  libdir = WORKDIR / f"lib_{variant['name']}"
  shutil.rmtree(libdir, ignore_errors=True)
  libdir.mkdir(parents=True, exist_ok=True)
  target = variant.get("target")
  if target:
    (libdir / "librocprof-trace-decoder.so").symlink_to(pathlib.Path(target))
  cmd = [
    str(ROCM / "bin/rocprofv3"), "--att", "--kernel-trace",
    "--att-buffer-size", "67108864", "--att-shader-engine-mask", "1", "--att-target-cu", "1",
    "--att-simd-select", "1", "--att-serialize-all", "--att-library-path", str(libdir),
    "-d", str(outdir), "-o", "att_control", "-f", "json", "--", str(ROOT / exe_rel),
  ]
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{ROCM / 'lib'}:{ROCM / 'lib/rocprofiler'}:{env.get('LD_LIBRARY_PATH','')}"
  res = run(cmd, cwd=WORKDIR, env=env, timeout=240)
  files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in sorted(outdir.rglob("*")) if p.is_file()]
  return {"variant": variant, "command_result": res, "files": files,
          "ok": res.get("returncode") == 0 and bool(files),
          "has_payload": any(f["bytes"] > 0 for f in files)}

def main() -> int:
  shutil.rmtree(WORKDIR, ignore_errors=True)
  WORKDIR.mkdir(parents=True, exist_ok=True)
  OUTDIR.mkdir(parents=True, exist_ok=True)

  result: dict[str, Any] = {
    "date": "2026-06-19",
    "phase": "T1c_ATT_decoder_repair",
    "purpose": "Exhaust local repair candidates for rocprofv3 --att's missing librocprof-trace-decoder.so path.",
    "package_inventory": inspect_packages(),
    "decoder_symbol_expectation": [
      "rocprof_trace_decoder_parse_data",
      "rocprof_trace_decoder_get_info_string",
      "rocprof_trace_decoder_get_status_string",
    ],
  }
  control = build_control()
  result["hip_control"] = control
  variants = [
    {"name": "missing_decoder", "target": None},
    {"name": "sdk_so_alias", "target": str(ROCM / "lib/librocprofiler-sdk.so")},
    {"name": "legacy_att_plugin_alias", "target": str(ROCM / "lib/rocprofiler/libatt_plugin.so")},
  ]
  result["variants"] = [run_rocprof(control["exe"], v) for v in variants] if control["ok"] else []
  passing = [r["variant"]["name"] for r in result["variants"] if r.get("ok") and r.get("has_payload")]
  result["gate"] = {"required": "rocprofv3 --att produces files for HIP control without crashing", "passing_variants": passing}
  result["verdict"] = "ATT_DECODER_REPAIR_PASS" if passing else "ATT_DECODER_REPAIR_BLOCKED"
  result["decision"] = (
    "Use the passing decoder candidate for the external ATT oracle."
    if passing else
    "No installed package or local alias provides the required rocprof-trace-decoder ABI. External ATT remains a ROCm packaging/toolchain blocker."
  )
  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "passing": passing}, indent=2))
  return 0 if passing else 1

if __name__ == "__main__":
  raise SystemExit(main())
