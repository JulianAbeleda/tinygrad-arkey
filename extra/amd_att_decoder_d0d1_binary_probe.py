#!/usr/bin/env python3
from __future__ import annotations

import json, os, pathlib, re, shutil, subprocess, tarfile, time, urllib.request
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "bench/amd-scheduler-tooling-backend"
WORKDIR = OUTDIR / "att_decoder_binary_probe_work"
OUT = OUTDIR / "att_decoder_binary_probe.json"
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

REQUIRED_SYMBOLS = [
  "rocprof_trace_decoder_parse_data",
  "rocprof_trace_decoder_get_info_string",
  "rocprof_trace_decoder_get_status_string",
]

def run(cmd: list[str], *, cwd: pathlib.Path = ROOT, env: dict[str, str] | None = None, timeout: int = 180) -> dict[str, Any]:
  t0 = time.perf_counter()
  try:
    cp = subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    return {"cmd": cmd, "returncode": cp.returncode, "elapsed_s": round(time.perf_counter()-t0, 3),
            "stdout_tail": cp.stdout.splitlines()[-120:], "stderr_tail": cp.stderr.splitlines()[-120:]}
  except subprocess.TimeoutExpired as e:
    return {"cmd": cmd, "timeout": True, "elapsed_s": round(time.perf_counter()-t0, 3),
            "stdout_tail": (e.stdout or "").splitlines()[-120:] if isinstance(e.stdout, str) else [],
            "stderr_tail": (e.stderr or "").splitlines()[-120:] if isinstance(e.stderr, str) else []}

def version_key(tag: str) -> tuple[int, ...]:
  nums = [int(x) for x in re.findall(r"\d+", tag)]
  return tuple(nums or [0])

def github_releases() -> dict[str, Any]:
  url = "https://api.github.com/repos/ROCm/rocprof-trace-decoder/releases"
  req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "tinygrad-att-decoder-probe"})
  with urllib.request.urlopen(req, timeout=30) as resp:
    releases = json.loads(resp.read().decode())
  rows = []
  for rel in releases:
    assets = []
    for asset in rel.get("assets", []):
      name = asset.get("name", "")
      lname = name.lower()
      assets.append({"name": name, "size": asset.get("size"), "url": asset.get("browser_download_url"),
                     "linux_candidate": ("linux" in lname or "manylinux" in lname) and (lname.endswith(".tar.gz") or lname.endswith(".tgz"))})
    rows.append({"tag": rel.get("tag_name"), "name": rel.get("name"), "published_at": rel.get("published_at"),
                 "prerelease": rel.get("prerelease"), "assets": assets})
  candidates = []
  for rel in rows:
    for asset in rel["assets"]:
      if asset["linux_candidate"] and asset["url"]:
        candidates.append({"tag": rel["tag"], "asset": asset})
  candidates.sort(key=lambda x: version_key(x["tag"] or ""), reverse=True)
  return {"url": url, "release_count": len(rows), "releases": rows, "candidates": candidates}

def download_and_unpack(candidate: dict[str, Any], slot: str = "selected") -> dict[str, Any]:
  asset = candidate["asset"]
  dl_dir = WORKDIR / "downloads"
  install = WORKDIR / f"decoder_install_{slot}"
  dl_dir.mkdir(parents=True, exist_ok=True)
  shutil.rmtree(install, ignore_errors=True)
  install.mkdir(parents=True, exist_ok=True)
  dest = dl_dir / asset["name"]
  if not dest.exists(): urllib.request.urlretrieve(asset["url"], dest)
  unpack_error = None
  try:
    if tarfile.is_tarfile(dest):
      with tarfile.open(dest) as tf:
        tf.extractall(install)
    else:
      unpack_error = "asset is not a tar archive"
  except Exception as exc:
    unpack_error = repr(exc)
  libs = sorted(install.rglob("librocprof-trace-decoder.so*"))
  return {"candidate": candidate, "download": str(dest.relative_to(ROOT)), "bytes": dest.stat().st_size,
          "install_dir": str(install.relative_to(ROOT)), "unpack_error": unpack_error,
          "decoder_libs": [str(p.relative_to(ROOT)) for p in libs]}

def symbol_check(lib: pathlib.Path) -> dict[str, Any]:
  res = run(["nm", "-D", str(lib)], timeout=60)
  text = "\n".join(res.get("stdout_tail", []))
  # stdout_tail may not include all symbols. Use direct grep for required symbols.
  grep = run(["bash", "-lc", f"nm -D {str(lib)!r} | rg 'rocprof_trace_decoder_(parse_data|get_info_string|get_status_string)' || true"], timeout=60)
  gtext = "\n".join(grep.get("stdout_tail", []))
  found = {sym: (sym in gtext or sym in text) for sym in REQUIRED_SYMBOLS}
  return {"nm": res, "grep": grep, "found": found, "ok": all(found.values())}

def build_control() -> dict[str, Any]:
  src, exe = WORKDIR / "att_body_control.cpp", WORKDIR / "att_body_control"
  include_copy = WORKDIR / "rocm724_include"
  shutil.rmtree(include_copy, ignore_errors=True)
  include_copy.mkdir(parents=True, exist_ok=True)
  shutil.copytree(ROCM / "include/hip", include_copy / "hip")
  src.write_text(HIP_SRC)
  # Force the 7.2 compiler stack. The host also has Ubuntu HIP 5.7 headers/libs under /usr, and mixing them with
  # rocprofv3 7.2 crashes at hipSetDevice before any ATT payload is produced.
  cmd = [str(ROCM / "bin/hipcc"), "--offload-arch=gfx1100", "-O3", "-I", str(include_copy), str(src),
         "-L", str(ROCM / "lib"), "-Wl,-rpath," + str(ROCM / "lib"), "-o", str(exe)]
  env = os.environ.copy()
  env["ROCM_PATH"] = str(ROCM)
  env["HIP_PATH"] = str(ROCM)
  res = run(cmd, cwd=WORKDIR, env=env)
  return {"source": str(src.relative_to(ROOT)), "exe": str(exe.relative_to(ROOT)), "build": res,
          "include_copy": str(include_copy.relative_to(ROOT)),
          "ok": res.get("returncode") == 0 and exe.exists()}

def run_att(exe_rel: str, lib: pathlib.Path) -> dict[str, Any]:
  outdir = WORKDIR / "rocprof_att_binary_decoder"
  shutil.rmtree(outdir, ignore_errors=True)
  outdir.mkdir(parents=True, exist_ok=True)
  cmd = [
    str(ROCM / "bin/rocprofv3"), "--att", "--kernel-trace",
    "--att-buffer-size", "67108864", "--att-shader-engine-mask", "1", "--att-target-cu", "1",
    "--att-simd-select", "1", "--att-serialize-all", "--att-library-path", str(lib.parent),
    "-d", str(outdir), "-o", "att_control", "-f", "json", "--", str(ROOT / exe_rel),
  ]
  env = os.environ.copy()
  env["LD_LIBRARY_PATH"] = f"{lib.parent}:{ROCM / 'lib'}:{env.get('LD_LIBRARY_PATH','')}"
  res = run(cmd, cwd=WORKDIR, env=env, timeout=240)
  files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in sorted(outdir.rglob("*")) if p.is_file()]
  text_hits = []
  for f in files:
    p = ROOT / f["path"]
    if f["bytes"] < 2_000_000 and p.suffix.lower() in {".json", ".csv", ".txt", ".log"}:
      txt = p.read_text(errors="replace")
      if any(x in txt.lower() for x in ("thread", "trace", "wave", "body_kernel", "instruction")):
        text_hits.append({"path": f["path"], "preview": txt[:3000]})
  return {"command_result": res, "output_dir": str(outdir.relative_to(ROOT)), "files": files, "text_hits": text_hits[:10],
          "ok": res.get("returncode") == 0 and bool(files),
          "has_payload": any(f["bytes"] > 0 for f in files)}

def main() -> int:
  shutil.rmtree(WORKDIR, ignore_errors=True)
  WORKDIR.mkdir(parents=True, exist_ok=True)
  OUTDIR.mkdir(parents=True, exist_ok=True)
  result: dict[str, Any] = {"date": "2026-06-19", "phase": "D0_D1_ATT_decoder_binary_probe"}
  try:
    result["release_audit"] = github_releases()
  except Exception as exc:
    result["release_audit"] = {"error": repr(exc), "candidates": []}
  candidates = result["release_audit"].get("candidates", [])[:12]
  result["selected_candidate"] = candidates[0] if candidates else None
  control = build_control() if candidates else {"ok": False}
  result["hip_control"] = control
  attempts = []
  for idx, candidate in enumerate(candidates):
    install = download_and_unpack(candidate, slot=f"{idx}_{candidate.get('tag', 'unknown')}")
    libs = install.get("decoder_libs", [])
    lib = ROOT / libs[0] if libs else None
    symbols = symbol_check(lib) if lib else {"ok": False, "reason": "no decoder lib found after unpack"}
    att = run_att(control["exe"], lib) if lib and symbols.get("ok") and control.get("ok") else {"ok": False, "reason": "missing lib/symbols/control"}
    attempts.append({"candidate": candidate, "binary_install": install, "symbol_check": symbols, "rocprof_att": att,
                     "pass": bool(symbols.get("ok") and att.get("ok") and att.get("has_payload"))})
    if attempts[-1]["pass"]: break
  result["attempts"] = attempts
  passing = [a for a in attempts if a["pass"]]
  if passing:
    result["binary_install"] = passing[0]["binary_install"]
    result["symbol_check"] = passing[0]["symbol_check"]
    result["rocprof_att"] = passing[0]["rocprof_att"]
  elif attempts:
    result["binary_install"] = attempts[0]["binary_install"]
    result["symbol_check"] = attempts[0]["symbol_check"]
    result["rocprof_att"] = attempts[0]["rocprof_att"]
  else:
    result["binary_install"] = {"ok": False, "reason": "no release candidate"}
    result["symbol_check"] = {"ok": False}
    result["rocprof_att"] = {"ok": False}
  result["gate"] = {
    "required": "release decoder lib has required symbols and rocprofv3 --att produces payloads for HIP control",
    "symbols_ok": bool(result.get("symbol_check", {}).get("ok")),
    "att_ok": bool(result.get("rocprof_att", {}).get("ok") and result.get("rocprof_att", {}).get("has_payload")),
  }
  result["verdict"] = "ATT_DECODER_BINARY_PASS" if result["gate"]["symbols_ok"] and result["gate"]["att_ok"] else "ATT_DECODER_BINARY_FAIL"
  result["next"] = "Run full oracle capture and tinygrad diff." if result["verdict"].endswith("PASS") else "Proceed to D2 source build from ROCm/rocm-systems."
  OUT.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(OUT.relative_to(ROOT)), "verdict": result["verdict"], "gate": result["gate"], "next": result["next"]}, indent=2))
  return 0 if result["verdict"].endswith("PASS") else 1

if __name__ == "__main__":
  raise SystemExit(main())
