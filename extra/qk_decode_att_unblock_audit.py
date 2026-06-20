#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_att_unblock_audit_result.json"
ROCPROF = Path("/opt/rocm/bin/rocprofv3")
SEARCH_ROOTS = [Path("/opt/rocm"), Path("/opt/rocm-7.2.4")]


def run(cmd: list[str], timeout: int = 30) -> dict[str, Any]:
  p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout}


def find_paths(patterns: list[str]) -> list[str]:
  out: list[str] = []
  for root in SEARCH_ROOTS:
    if not root.exists():
      continue
    for pattern in patterns:
      out.extend(str(p) for p in root.rglob(pattern) if p.exists())
  return sorted(set(out))


def apt_search(query: str) -> list[str]:
  if shutil.which("apt-cache") is None:
    return []
  res = run(["apt-cache", "search", query])
  return [line for line in res["stdout"].splitlines() if line.strip()]


def dpkg_list() -> list[str]:
  if shutil.which("dpkg") is None:
    return []
  res = run(["dpkg", "-l"])
  return [line for line in res["stdout"].splitlines() if any(x in line.lower() for x in ["rocprof", "rocprofiler", "rocm-core"])]


def main() -> int:
  rocprof_version = run([str(ROCPROF), "--version"]) if ROCPROF.exists() else {"returncode": None, "stdout": ""}
  decoder_headers = find_paths(["*trace_decoder*.h", "*trace*decoder*.hpp"])
  decoder_libs = find_paths(["*trace*decoder*.so", "*trace*decoder*.so.*", "*att*decoder*.so", "*att*decoder*.so.*"])
  candidate_packages = sorted(set(
    apt_search("rocprofiler-sdk") +
    apt_search("rocprofiler") +
    apt_search("thread trace") +
    apt_search("trace decoder")
  ))
  installed = dpkg_list()
  rpath_candidates = [line for line in candidate_packages if "rpath" in line.lower()]
  sdk_candidates = [line for line in candidate_packages if "rocprofiler-sdk" in line.lower()]
  known_att = load_att_result()

  gates = {
    "rocprofv3_exists": ROCPROF.exists(),
    "rocprofv3_version_ok": rocprof_version.get("returncode") == 0 and "rocm_version" in rocprof_version.get("stdout", ""),
    "decoder_headers_present": len(decoder_headers) > 0,
    "decoder_library_present": len(decoder_libs) > 0,
    "previous_att_blocker_recorded": known_att.get("verdict") == "BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING",
    "candidate_packages_visible": len(candidate_packages) > 0,
  }
  blocked = not gates["decoder_library_present"]
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ATT_UNBLOCK_AUDIT",
    "schema": "decode_att_unblock_audit_v1",
    "verdict": "BLOCKED_DECODE_ATT_DECODER_SO_MISSING" if blocked else "PASS_DECODE_ATT_DECODER_SO_PRESENT",
    "gate_pass": not blocked,
    "default_behavior_changed": False,
    "performance_claim": False,
    "rocprofv3": {
      "path": str(ROCPROF),
      "version_stdout": rocprof_version.get("stdout", ""),
    },
    "decoder_headers": decoder_headers,
    "decoder_libraries": decoder_libs,
    "installed_rocprofiler_packages": installed,
    "candidate_packages": candidate_packages[:120],
    "rpath_candidate_packages": rpath_candidates,
    "sdk_candidate_packages": sdk_candidates,
    "gates": gates,
    "next": {
      "if_blocked": [
        "install/provide a ROCm 7.2.4-compatible rocprof trace decoder shared library",
        "likely package family to inspect: rocprofiler-sdk rpath/7.2.4 variants or ROCm SDK build artifact containing the thread-trace decoder",
        "rerun extra/qk_decode_oracle_att_probe.py",
      ],
      "if_pass": [
        "rerun extra/qk_decode_oracle_att_probe.py",
        "join decoded PCs to oracle semantic stages",
      ],
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "decoder_headers": decoder_headers[:5],
    "decoder_libraries": decoder_libs[:5],
    "rpath_candidates": rpath_candidates[:8],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0


def load_att_result() -> dict[str, Any]:
  p = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_att_result.json"
  return json.loads(p.read_text()) if p.exists() else {}


if __name__ == "__main__":
  raise SystemExit(main())
