"""Fail-closed GPU validation harness for the generated five-buffer MMQ graph.

The full-grid graph is deliberately kept separate from the runtime route.  This
module is an evidence harness only: it builds one deterministic 128x128x256
case in a child interpreter, compiles the exact AMD PROGRAM, dispatches it once,
and compares the result with an independent NumPy DS4 reference.  A compiler
timeout or a missing PROGRAM is reported as a blocker; it is never interpreted
as a numerical pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


PROTOCOL = "tinygrad.mmq_llama_five_buffer_gpu_harness.v1"
PASS = "MMQ_LLAMA_FIVE_BUFFER_GPU_PASS"
BLOCKED = "MMQ_LLAMA_FIVE_BUFFER_GPU_BLOCKED"
ROOT = Path(__file__).resolve().parents[2]
SHAPE = (128, 128, 256)


def _blocked(reason: str, **evidence: Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "shape": list(SHAPE), "passed": False,
          "verdict": BLOCKED, "blocker": reason, "evidence": evidence}


def _random_q4_words(n: int, k: int, seed: int) -> np.ndarray:
  """Independent finite Q4_K bytes (metadata/scales are not emitter-derived)."""
  if k % 256: raise ValueError("Q4_K probe requires K divisible by 256")
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, size=(n, k // 256, 144), dtype=np.uint8)
  # Keep super-block scales finite and moderate while leaving all other bytes
  # random.  This mirrors the independent packed-byte tests without importing
  # a test fixture.
  raw[:, :, :4] = np.frombuffer(np.array([0.03125, 0.0078125], dtype="<f2").tobytes(), dtype=np.uint8)
  return np.ascontiguousarray(raw.reshape(-1).view(np.uint32))


def _bind_sink(sink, args):
  """Replace the five slot parameters in a generated sink with call placeholders."""
  from tinygrad.uop.ops import Ops
  params = sorted({u for u in sink.toposort() if u.op is Ops.PARAM}, key=lambda u: u.arg.slot)
  if [u.arg.slot for u in params] != list(range(5)):
    raise ValueError("full-grid PROGRAM must expose exactly ABI slots 0..4")
  if len(args) != 5: raise ValueError("full-grid call requires five buffers")
  return sink.substitute(dict(zip(params, args)), walk=True)


def _worker() -> dict[str, Any]:
  """Compile and dispatch the sole AMD case.  Called only in a child process."""
  from tinygrad import Tensor, dtypes
  from tinygrad.codegen import to_program
  from tinygrad.device import Device
  from tinygrad.engine.realize import runtime_cache
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.helpers import Target
  from tinygrad.uop.ops import Ops
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  from extra.qk.mmq_llama_five_buffer_full_kernel import build_llama_five_buffer_full_kernel, compile_llama_five_buffer_full_kernel
  from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec,
    q4k_q8_1_mmq_ds4_tile_reference, q8_1_mmq_ds4_quantize_reference,
    Q8_1_MMQ_DS4_LAYOUT, Q4KQ81MMQTileSpec)

  m, n, k = SHAPE
  words_np = _random_q4_words(n, k, 20260717)
  source_np = np.random.default_rng(20260718).standard_normal((m, k), dtype=np.float32)
  values_np, scales_np, sums_np = q8_1_mmq_ds4_quantize_reference(source_np)
  ds4 = Q81MMQDS4Activation(values_np, scales_np, sums_np,
    Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  ref_spec = Q4KQ81MMQTileSpec(role="five_buffer_gpu_probe", m=m, n=n, k=k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words_np.view(np.uint8), ds4, ref_spec)

  kernel = build_llama_five_buffer_full_kernel(m, n, k)
  compiled = compile_llama_five_buffer_full_kernel(kernel)
  if not compiled.emitted or compiled.program is None:
    return _blocked("full-grid compile did not produce PROGRAM", reported_blocker=compiled.blocker)
  program = compiled.program
  if program.op is not Ops.PROGRAM:
    return _blocked("compile result is not an AMD PROGRAM", op=str(program.op))
  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  if programs != [program]:
    return _blocked("expected exactly one PROGRAM", program_count=len(programs))

  # Materialize all buffers before dispatch.  The sink uses the exact five
  # parameter slots; runtime invocation avoids a second custom_kernel compile.
  device = Device["AMD"]
  out = Tensor.empty(m * n, dtype=dtypes.float32, device="AMD").realize()
  q4 = Tensor(words_np, device="AMD").realize()
  values = Tensor(values_np.reshape(-1), device="AMD").realize()
  scales = Tensor(scales_np.reshape(-1), device="AMD").realize()
  sums = Tensor(sums_np.reshape(-1), device="AMD").realize()
  args = (out.uop.buffer._buf, q4.uop.buffer._buf, values.uop.buffer._buf,
          scales.uop.buffer._buf, sums.uop.buffer._buf)
  runtime = runtime_cache.get((program.key, "AMD"))
  if runtime is None:
    return _blocked("AMD runtime cache has no compiled PROGRAM", program_key=program.key.hex())
  runtime(*args, global_size=program.arg.global_size, local_size=program.arg.local_size, wait=True)
  got = out.numpy().reshape(m, n)
  np.testing.assert_allclose(got, reference, rtol=3e-3, atol=3e-3)
  binary = next((u.arg for u in program.src if u.op is Ops.BINARY), None)
  source = next((u.arg for u in program.src if u.op is Ops.SOURCE), None)
  metadata = parse_amdgpu_metadata(binary) if isinstance(binary, bytes) else None
  return {"protocol": PROTOCOL, "shape": [m, n, k], "passed": True, "verdict": PASS,
          "blocker": None, "evidence": {
            "dispatch_performed": True, "full_output_compared": True,
            "global_size": list(program.arg.global_size), "local_size": list(program.arg.local_size),
            "source_sha256": hashlib.sha256(source.encode()).hexdigest() if isinstance(source, str) else None,
            "binary_sha256": hashlib.sha256(binary).hexdigest() if isinstance(binary, bytes) else None,
            "resources": metadata, "max_abs_error": float(np.max(np.abs(got-reference))),
            "mean_abs_error": float(np.mean(np.abs(got-reference))),
          }}


def run_amd_validation(*, timeout_seconds: float = 300.0,
                       python: str = sys.executable,
                       env: dict[str, str] | None = None) -> dict[str, Any]:
  """Run compilation/dispatch in an isolated child with a hard deadline."""
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  try:
    # ``__name__`` is ``__main__`` when this file is invoked with ``-m`` and
    # cannot be resolved by a child interpreter.  Use the importable module
    # path explicitly so the isolated worker can always start.
    proc = subprocess.run([python, "-m", "extra.qk.mmq_llama_five_buffer_gpu_harness", "--worker"], cwd=ROOT, env=child_env,
                          text=True, capture_output=True, timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired:
    return _blocked("AMD full-grid compile/dispatch timed out", timeout_seconds=timeout_seconds)
  except OSError as exc:
    return _blocked(f"AMD worker could not start: {exc}")
  try:
    row = json.loads(proc.stdout.splitlines()[-1])
  except (json.JSONDecodeError, IndexError) as exc:
    return _blocked("AMD worker failed" if proc.returncode else f"AMD worker returned invalid JSON: {exc}",
                    returncode=proc.returncode, stdout=proc.stdout[-4000:], stderr=proc.stderr[-2000:])
  # The worker intentionally exits nonzero for a structured BLOCKED verdict;
  # preserve that evidence instead of replacing it with a generic failure.
  return row


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  args = parser.parse_args()
  if not args.worker:
    print(json.dumps(run_amd_validation(), indent=2, sort_keys=True))
    return 0
  try: row = _worker()
  except BaseException as exc:
    row = _blocked("AMD worker exception", exception=type(exc).__name__, error=str(exc))
  print(json.dumps(row, sort_keys=True))
  return 0 if row.get("passed") else 1


if __name__ == "__main__":
  raise SystemExit(main())
