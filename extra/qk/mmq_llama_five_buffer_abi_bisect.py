"""Fail-closed AMD ABI bisect for the full-grid five-buffer dispatch.

These probes intentionally keep the full-grid argument sizes and launch shape,
but replace the WMMA/LDS body with one scalar read (or no read) followed by a
write to the first 256 output elements.  Each probe is isolated in a child
process because an AMD MMU fault poisons the process/device queue.  The probes
do not validate llama arithmetic; they only answer whether the five-pointer
runtime ABI and an individual global buffer access can dispatch safely.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import get_runtime
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "tinygrad.mmq_llama_five_buffer_abi_bisect.v1"
CASES = ("output", "q4", "q8", "scale", "sum")
LOCAL = 256
OUTPUT_ELEMENTS = 128 * 128
Q4_WORDS = 128 * 36
Q8_VALUES = 2 * 128 * 128
METADATA = 2 * 128 * 4


def _params() -> tuple[UOp, UOp, UOp, UOp, UOp]:
  return (UOp.param(0, dtypes.float32.ptr(OUTPUT_ELEMENTS)),
          UOp.param(1, dtypes.uint32.ptr(Q4_WORDS)),
          UOp.param(2, dtypes.int8.ptr(Q8_VALUES)),
          UOp.param(3, dtypes.float32.ptr(METADATA)),
          UOp.param(4, dtypes.float32.ptr(METADATA)))


def build_bisect_sink(case: str) -> UOp:
  """Build one probe sink; no device or compiler state is touched here."""
  if case not in CASES: raise ValueError(f"unknown ABI bisect case: {case}")
  out, q4, q8, scales, sums = _params()
  lane = UOp.special(LOCAL, "lidx0")
  source = {"output": UOp.const(dtypes.float32, 1.25),
            "q4": q4[lane].cast(dtypes.float32),
            "q8": q8[lane].cast(dtypes.float32),
            "scale": scales[lane],
            "sum": sums[lane]}[case]
  store = out[lane].store(source)
  # Keep all five pointer parameters in ProgramInfo.globals while avoiding
  # reads from the four non-target buffers.  BARRIER accepts void operands and
  # emits only the argument loads; it is not a semantic dependency.
  sink = UOp(Ops.BARRIER, dtypes.void, (store, out, q4, q8, scales, sums))
  return UOp(Ops.SINK, dtypes.void, (sink,), KernelInfo(name=f"mmq_abi_bisect_{case}", opts_to_apply=()))


def _worker(case: str) -> dict[str, Any]:
  sink = build_bisect_sink(case)
  info = ProgramInfo.from_sink(sink)
  if info.globals != tuple(range(5)):
    return {"case": case, "passed": False, "blocker": "probe did not retain five-buffer ABI",
            "globals": list(info.globals)}
  program = to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  if program.arg.globals != tuple(range(5)):
    return {"case": case, "passed": False, "blocker": "PROGRAM reordered five-buffer ABI",
            "globals": list(program.arg.globals)}
  rng = np.random.default_rng(20260717)
  buffers = (Tensor.empty(OUTPUT_ELEMENTS, dtype=dtypes.float32, device="AMD").realize(),
             Tensor(rng.integers(0, 2**32, Q4_WORDS, dtype=np.uint32), device="AMD").realize(),
             Tensor(rng.integers(-128, 128, Q8_VALUES, dtype=np.int8), device="AMD").realize(),
             Tensor(rng.standard_normal(METADATA, dtype=np.float32), device="AMD").realize(),
             Tensor(rng.standard_normal(METADATA, dtype=np.float32), device="AMD").realize())
  raw = tuple(buf.uop.buffer.get_buf("AMD") for buf in buffers)
  runtime = get_runtime("AMD", program)
  runtime(*raw, global_size=program.arg.global_size, local_size=program.arg.local_size,
          vals=program.arg.vals({}), wait=True)
  got = buffers[0].numpy()
  return {"case": case, "passed": True, "globals": list(program.arg.globals),
          "global_size": list(program.arg.global_size), "local_size": list(program.arg.local_size),
          "sample": got[:4].tolist(), "read_slot": {"output": None, "q4": 1, "q8": 2, "scale": 3, "sum": 4}[case]}


def run_bisect(*, timeout_seconds: float = 120.0, python: str = sys.executable,
               env: dict[str, str] | None = None) -> dict[str, Any]:
  rows = []
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  for case in CASES:
    try:
      proc = subprocess.run([python, "-m", "extra.qk.mmq_llama_five_buffer_abi_bisect", "--worker", case],
                            cwd=ROOT, env=child_env, text=True, capture_output=True,
                            timeout=timeout_seconds, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
      rows.append({"case": case, "passed": False, "blocker": type(exc).__name__})
      continue
    try: row = json.loads(proc.stdout.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
      row = {"case": case, "passed": False, "blocker": "worker returned invalid JSON",
             "returncode": proc.returncode, "stderr": proc.stderr[-2000:]}
    rows.append(row)
  return {"protocol": PROTOCOL, "shape": [128, 128, 256], "cases": rows,
          "passed": all(row.get("passed", False) for row in rows)}


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", choices=CASES)
  args = parser.parse_args()
  if args.worker:
    try: row = _worker(args.worker)
    except BaseException as exc: row = {"case": args.worker, "passed": False,
                                        "blocker": type(exc).__name__, "error": str(exc)}
    print(json.dumps(row, sort_keys=True)); return 0 if row.get("passed") else 1
  print(json.dumps(run_bisect(), sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
