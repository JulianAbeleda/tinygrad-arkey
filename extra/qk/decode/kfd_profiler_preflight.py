#!/usr/bin/env python3
"""Read-only KFD/ROCm profiler preflight.  Never dispatches work or changes GPU state."""
import argparse, errno, fcntl, glob, json, os, platform, shutil, stat, struct, subprocess
from datetime import datetime, timezone

KFD_GET_VERSION = 0xC0084B01
KFD_PROFILER = 0xC0284B86
KFD_IOC_PROFILER_PMC, KFD_IOC_PROFILER_VERSION = 0, 2
CAP_SYS_ADMIN, CAP_PERFMON = 21, 38

def read(path):
  try:
    with open(path) as f: return f.read().strip()
  except OSError: return None

def node(path):
  try:
    s = os.stat(path)
    return {"path": path, "mode": f"{stat.S_IMODE(s.st_mode):04o}", "uid": s.st_uid, "gid": s.st_gid,
            "read_write_access": os.access(path, os.R_OK | os.W_OK)}
  except OSError as e: return {"path": path, "error": {"errno": e.errno, "name": errno.errorcode.get(e.errno)}}

def ioctl(fd, request, payload):
  try:
    fcntl.ioctl(fd, request, payload, True)
    return {"status": "ok", "payload_hex": payload.hex()}
  except OSError as e:
    return {"status": "error", "errno": e.errno, "errno_name": errno.errorcode.get(e.errno), "message": str(e)}

def cap_enabled(value, cap): return bool(int(value or "0", 16) & (1 << cap))

def tool(name, candidates=()):
  found = next((x for x in (shutil.which(name), *candidates) if x and os.path.isfile(x)), None)
  if not found: return {"name": name, "path": None, "version": None}
  result = subprocess.run([found, "--version"], capture_output=True, text=True)
  lines = (result.stdout or result.stderr).splitlines()
  return {"name": name, "path": found, "version": lines[0] if lines else None}

def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--output", help="write JSON to this path instead of stdout")
  p.add_argument("--probe-pmc-lock", action="store_true", help="attempt mutating PMC lock only after explicit operator approval")
  p.add_argument("--allow-state-change", action="store_true", help="required with --probe-pmc-lock")
  a = p.parse_args()
  if a.probe_pmc_lock and not a.allow_state_change: p.error("--probe-pmc-lock requires --allow-state-change")
  caps = {line.split(":", 1)[0]: line.split(":", 1)[1].strip() for line in (read("/proc/self/status") or "").splitlines() if line.startswith("Cap")}
  report = {"schema": "tinygrad.kfd_profiler_preflight.v1", "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "safety": {"gpu_dispatch": False, "gpu_reset": False, "sudo": False,
               "pmc_lock_probe": a.probe_pmc_lock, "pmc_lock_may_change_state": a.probe_pmc_lock},
    "system": {"kernel": platform.release(), "amdgpu_module_version": read("/sys/module/amdgpu/version"),
               "amdgpu_ppfeaturemask": read("/sys/module/amdgpu/parameters/ppfeaturemask"), "rocm_version": read("/opt/rocm/.info/version")},
    "devices": [node("/dev/kfd")] + [node(x) for x in sorted(glob.glob("/dev/dri/renderD*"))],
    "capabilities": {"effective_hex": caps.get("CapEff"), "cap_sys_admin": cap_enabled(caps.get("CapEff"), CAP_SYS_ADMIN),
                     "cap_perfmon": cap_enabled(caps.get("CapEff"), CAP_PERFMON), "perf_event_paranoid": read("/proc/sys/kernel/perf_event_paranoid")},
    "tools": [tool("llvm-readobj", ("/opt/rocm/llvm/bin/llvm-readobj",)), tool("rocprof", ("/opt/rocm/bin/rocprof",)),
              tool("rocprofv3", ("/opt/rocm/bin/rocprofv3",)), tool("rocm-smi", ("/opt/rocm/bin/rocm-smi",))],
    "profiler_ioctl": {"request": "AMDKFD_IOC_PROFILER", "request_hex": hex(KFD_PROFILER),
      "pmc_operation": {"op": KFD_IOC_PROFILER_PMC, "name": "KFD_IOC_PROFILER_PMC", "arguments": {"lock": 1, "perfcount_enable": 1}},
      "version_operation": {"op": KFD_IOC_PROFILER_VERSION, "name": "KFD_IOC_PROFILER_VERSION"}}}
  try:
    fd = os.open("/dev/kfd", os.O_RDWR)
    try:
      v = bytearray(8); report["kfd_ioctl_version"] = ioctl(fd, KFD_GET_VERSION, v)
      if report["kfd_ioctl_version"]["status"] == "ok": report["kfd_ioctl_version"].update(dict(zip(("major", "minor"), struct.unpack("II", v))))
      profiler_version = bytearray(40); struct.pack_into("I", profiler_version, 0, KFD_IOC_PROFILER_VERSION)
      report["profiler_ioctl"]["version_result"] = ioctl(fd, KFD_PROFILER, profiler_version)
      if a.probe_pmc_lock:
        pmc = bytearray(40); struct.pack_into("IIII", pmc, 0, KFD_IOC_PROFILER_PMC, 0, 1, 1)
        report["profiler_ioctl"]["pmc_lock_result"] = ioctl(fd, KFD_PROFILER, pmc)
      else:
        report["profiler_ioctl"]["pmc_lock_result"] = {"status": "not_attempted", "reason": "would change perfmon state; use explicit operator-approved flags"}
    finally: os.close(fd)
  except OSError as e: report["kfd_open"] = {"status": "error", "errno": e.errno, "errno_name": errno.errorcode.get(e.errno), "message": str(e)}
  text = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if a.output:
    with open(a.output, "w") as f: f.write(text)
  else: print(text, end="")

if __name__ == "__main__": main()
