#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNNER = ROOT / "bench/qk-decode-primitive-transfer/oracle_hip_runner/q8_mmvq_gateup_runner"
OUTDIR = ROOT / "bench/qk-decode-primitive-transfer/oracle_hip_runner/rocprof_att"
RESULT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_att_result.json"
ROCPROF = pathlib.Path("/opt/rocm/bin/rocprofv3")
ROCM_ENV = {
  **os.environ,
  "LD_LIBRARY_PATH": "/opt/rocm/lib:" + os.environ.get("LD_LIBRARY_PATH", ""),
  "PATH": "/opt/rocm/bin:" + os.environ.get("PATH", ""),
}


def run(cmd: list[str], timeout: int = 180) -> dict[str, Any]:
  p = subprocess.run(cmd, cwd=ROOT, env=ROCM_ENV, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout}


def find_decoder_candidates() -> list[str]:
  roots = [pathlib.Path("/opt/rocm"), pathlib.Path("/opt/rocm-7.2.4")]
  out: list[str] = []
  for root in roots:
    if not root.exists():
      continue
    for path in root.rglob("*trace*decoder*"):
      if path.is_file():
        out.append(str(path))
  return sorted(set(out))


def list_outputs(path: pathlib.Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  return [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size} for p in sorted(path.rglob("*")) if p.is_file()]


def main() -> int:
  if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
  OUTDIR.mkdir(parents=True, exist_ok=True)

  decoder_candidates = find_decoder_candidates()
  cmd = [
    str(ROCPROF), "--att", "--kernel-trace", "-f", "csv", "-d", str(OUTDIR), "-o", "q8_gateup_att",
    "--att-buffer-size", "67108864", "--att-shader-engine-mask", "1", "--att-target-cu", "1",
    "--att-simd-select", "1", "--att-serialize-all",
    "--", str(RUNNER), "--warmups", "1", "--iters", "1",
  ]
  res = run(cmd) if RUNNER.exists() and ROCPROF.exists() else {"cmd": cmd, "returncode": None, "stdout": ""}
  outputs = list_outputs(OUTDIR)
  stdout = res.get("stdout", "")
  gates = {
    "runner_exists": RUNNER.exists(),
    "rocprofv3_exists": ROCPROF.exists(),
    "decoder_library_present": any(path.endswith(".so") or ".so." in path for path in decoder_candidates),
    "att_run_returned_zero": res.get("returncode") == 0,
    "att_outputs_present": bool(outputs),
    "stdout_mentions_decoder_missing": "rocprof-trace-decoder library path not found" in stdout,
  }
  blocked = not (gates["att_run_returned_zero"] and gates["att_outputs_present"])
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_ATT_OES5",
    "schema": "decode_oracle_att_probe_v1",
    "verdict": "BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING" if blocked else "PASS_DECODE_ORACLE_ATT_CAPTURED",
    "gate_pass": not blocked,
    "default_behavior_changed": False,
    "performance_claim": False,
    "decoder_candidates": decoder_candidates[:80],
    "outputs": outputs,
    "command": cmd,
    "returncode": res.get("returncode"),
    "stdout_tail": stdout[-6000:],
    "gates": gates,
    "next": {
      "if_blocked": "Install/provide a ROCm-compatible rocprof trace decoder library, then rerun this probe; until then use kernel-trace resource/timing plus native PMC as the coarse OES-5 fallback.",
      "if_pass": "Join ATT PCs to docs/decode-oracle-semantic-map-result-20260620.md stages.",
    },
  }
  RESULT.parent.mkdir(parents=True, exist_ok=True)
  RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": gates,
    "outputs": outputs[:10],
    "out": str(RESULT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

