#!/usr/bin/env python3
"""Fail-closed validation for scalar and bounded-WMMA Q4_K x Q8_1 kernels.

The parent process is deliberately import-light and never opens an AMD device.
Use :func:`run_amd_validation` to run the small, deterministic canary in a
separate interpreter with a hard deadline.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROTOCOL = "tinygrad.q4k_q8_mmq_uop_validation.v2"
SCALAR_MODE, WMMA_MODE = "scalar", "wmma_16x16x256"
MODES = (SCALAR_MODE, WMMA_MODE)
CANDIDATE_KERNEL_NAMES = {SCALAR_MODE: "q4k_q8_mmq_uop_phase1",
                          WMMA_MODE: "q4k_q8_mmq_uop_wmma_16x16x256"}
CANDIDATE_KERNEL_NAME = CANDIDATE_KERNEL_NAMES[SCALAR_MODE]  # compatibility for scalar callers
PASS_VERDICT = "Q4K_Q8_MMQ_UOP_VALIDATION_PASS"
BLOCKED_VERDICT = "Q4K_Q8_MMQ_UOP_VALIDATION_BLOCKED"
ROOT = Path(__file__).resolve().parents[2]


def _blocked(reason:str, **evidence:Any) -> dict[str, Any]:
  return {"protocol": PROTOCOL, "passed": False, "verdict": BLOCKED_VERDICT,
          "blocker": reason, "evidence": evidence}


def independent_packed_byte_reference(words:np.ndarray, xq:np.ndarray, xscale:np.ndarray,
                                      *, m:int, n:int, k:int) -> np.ndarray:
  """Decode bytes directly, independently of tinygrad and the emitter UOps."""
  if k <= 0 or k % 256 or m <= 0 or n <= 0: raise ValueError("invalid Q4_K MMQ shape")
  raw = np.asarray(words, dtype=np.uint32).astype("<u4", copy=False).view(np.uint8)
  if raw.size != n * (k // 256) * 144: raise ValueError("packed Q4_K byte count does not match shape")
  q8 = np.asarray(xq, dtype=np.int8).reshape(m, k)
  xs = np.asarray(xscale, dtype=np.float32).reshape(m, k // 32)
  out = np.zeros((m, n), dtype=np.float32)
  blocks = raw.reshape(n, k // 256, 144)
  for col in range(n):
    for block in range(k // 256):
      b = blocks[col, block]
      d, dmin = np.frombuffer(b[:4].tobytes(), dtype="<f2").astype(np.float32)
      meta, payload = b[4:16], b[16:].reshape(4, 32)
      for group in range(8):
        if group < 4: scale, minimum = int(meta[group] & 63), int(meta[4+group] & 63)
        else:
          h = group - 4
          scale = int((meta[8+h] & 15) | ((meta[h] >> 6) << 4))
          minimum = int((meta[8+h] >> 4) | ((meta[4+h] >> 6) << 4))
        q = ((payload[group//2] >> (4 * (group & 1))) & 15).astype(np.float32)
        off, sg = block * 256 + group * 32, block * 8 + group
        for row in range(m):
          iv = q8[row, off:off+32].astype(np.float32)
          out[row, col] += xs[row, sg] * (d * scale * np.dot(q, iv) - dmin * minimum * iv.sum())
  return out


def classify_isa(source:str) -> dict[str, Any]:
  """Classify only exact emitted AMD evidence; absence never implies scalar."""
  text = source.lower()
  signed_mnemonic = "wmma_i32_16x16x16_iu8" in text
  # Renderer source uses two literal signed=true operands. ISA text carries
  # both signed input flags in the final control field (3).
  signed_flags = bool(re.search(r"wmma_i32_16x16x16_iu8[^\n]*(?:true[^\n]*true|,\s*3\s*\)?)", text))
  if signed_mnemonic:
    classification = "wmma"
  elif re.search(r"\bv_wmma_[a-z0-9_]+|\bv_mfma_[a-z0-9_]+", text):
    classification = "wmma"
  elif re.search(r"(?m)^\s*(?:s|v|global|buffer)_[a-z0-9_]+(?:\s|$)", text):
    classification = "scalar_direct"
  else: classification = "missing" if not text.strip() else "unknown"
  return {"classification": classification, "signed_integer_wmma": signed_mnemonic and signed_flags,
          "signed_integer_mnemonic": signed_mnemonic, "signed_input_flags": signed_flags}


def inspect_uop_graph(sink:Any) -> dict[str, Any]:
  """Return structural evidence without importing/initializing a device."""
  from tinygrad.uop.ops import Ops
  nodes = sink.toposort()
  from tinygrad.codegen.opt import OptOps
  stores = [u for u in nodes if u.op is Ops.STORE]
  wmmas = [u for u in nodes if u.op in (Ops.WMMA, Ops.SHAPED_WMMA)]
  indexed = len(stores) == 1 and bool(stores[0].src) and stores[0].src[0].op is Ops.INDEX
  computed = indexed and any(u.op in (Ops.SPECIAL, Ops.RANGE) for u in stores[0].src[0].toposort())
  name = getattr(getattr(sink, "arg", None), "name", None)
  opts = getattr(getattr(sink, "arg", None), "opts_to_apply", None) or ()
  tc_opts = [o for o in opts if getattr(o, "op", None) is OptOps.TC]
  classification = "generic_tc_candidate" if tc_opts and not wmmas else ("scalar_direct_uop" if not tc_opts and not wmmas else "route_local_wmma")
  return {"classification": classification, "route_local_wmma_count": len(wmmas),
          "tc_opt_count": len(tc_opts), "tc_opt": None if len(tc_opts) != 1 else
          {"axis": tc_opts[0].axis, "arg": list(tc_opts[0].arg) if isinstance(tc_opts[0].arg, tuple) else tc_opts[0].arg},
          "store_count": len(stores), "indexed_store": indexed, "computed_store_index": computed,
          "kernel_name": name}


def admit_evidence(row:Mapping[str, Any], *, mode:str|None=None) -> dict[str, Any]:
  """Apply the complete contract. Missing evidence fails closed."""
  if mode is None: mode = row.get("mode") if isinstance(row, Mapping) else None
  if mode not in MODES: return _blocked(f"unsupported or missing mode: {mode!r}")
  required = {"mode", "uop", "program_count", "kernel_count_delta", "kernel_name", "fallback_used",
              "numeric", "isa"}
  missing = sorted(required - row.keys())
  if missing: return _blocked("missing evidence: " + ", ".join(missing))
  uop, numeric, isa = row["uop"], row["numeric"], row["isa"]
  if row["mode"] != mode: return _blocked("requested mode and worker evidence mode differ", requested=mode, evidence_mode=row["mode"])
  scalar = mode == SCALAR_MODE
  checks = {
    "mode identity": row["mode"] == mode,
    "authored graph class": isinstance(uop, Mapping) and uop.get("classification") == ("scalar_direct_uop" if scalar else "generic_tc_candidate"),
    "no route-local WMMA": isinstance(uop, Mapping) and uop.get("route_local_wmma_count") == 0,
    "mode TC contract": isinstance(uop, Mapping) and (uop.get("tc_opt_count") == 0 if scalar else
      uop.get("tc_opt_count") == 1 and uop.get("tc_opt") == {"axis": 0, "arg": [-1, 2, 1]}),
    "one computed indexed STORE": isinstance(uop, Mapping) and uop.get("store_count") == 1 and uop.get("indexed_store") is True and uop.get("computed_store_index") is True,
    "exactly one PROGRAM": row["program_count"] == 1,
    "exactly one measured launch": row["kernel_count_delta"] == 1,
    "candidate kernel name": row["kernel_name"] == CANDIDATE_KERNEL_NAMES[mode] and uop.get("kernel_name") == CANDIDATE_KERNEL_NAMES[mode],
    "no fallback": row["fallback_used"] is False,
    "independent packed-byte numeric reference": isinstance(numeric, Mapping) and numeric.get("reference") == "independent_packed_byte" and numeric.get("allclose") is True,
    "mode-specific emitted ISA": isinstance(isa, Mapping) and isa.get("classification") == ("scalar_direct" if scalar else "wmma"),
    "signed integer WMMA": isinstance(isa, Mapping) and (isa.get("signed_integer_wmma") is False if scalar else isa.get("signed_integer_wmma") is True),
  }
  failed = [name for name, ok in checks.items() if not ok]
  if failed: return _blocked("; ".join(failed), checks=checks, raw=dict(row))
  return {"protocol": PROTOCOL, "passed": True, "verdict": PASS_VERDICT,
          "blocker": None, "evidence": dict(row)}


def run_amd_validation(*, mode:str, timeout_seconds:float=30.0, python:str=sys.executable,
                       env:Mapping[str, str]|None=None) -> dict[str, Any]:
  """Run the sole AMD launch in an isolated CLI worker with a hard timeout."""
  if mode not in MODES: return _blocked(f"unsupported mode: {mode!r}")
  if timeout_seconds <= 0: return _blocked("timeout_seconds must be positive")
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV": "AMD", "PYTHONPATH": str(ROOT) + os.pathsep + child_env.get("PYTHONPATH", "")})
  cmd = [python, "-m", "extra.qk.q4k_q8_mmq_uop_validation", "--worker", "--mode", mode]
  try:
    proc = subprocess.run(cmd, text=True, capture_output=True, env=child_env, cwd=ROOT,
                          timeout=timeout_seconds, check=False)
  except subprocess.TimeoutExpired: return _blocked("AMD worker timed out", timeout_seconds=timeout_seconds)
  except OSError as exc: return _blocked(f"AMD worker could not start: {exc}")
  if proc.returncode != 0: return _blocked("AMD worker failed", returncode=proc.returncode, stderr=proc.stderr[-2000:])
  try: row = json.loads(proc.stdout)
  except (json.JSONDecodeError, TypeError) as exc: return _blocked(f"AMD worker returned invalid JSON: {exc}", stdout=proc.stdout[-2000:])
  return admit_evidence(row, mode=mode)


def _fixture(m:int, n:int, k:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(20260715)
  raw = rng.integers(0, 256, size=n*(k//256)*144, dtype=np.uint8).reshape(n, k//256, 144)
  # Finite, moderate fp16 super-block scales; remaining metadata/payload stays random.
  for col in range(n):
    for block in range(k//256): raw[col, block, :4] = np.frombuffer(np.array([0.03125, 0.0078125], dtype="<f2").tobytes(), dtype=np.uint8)
  return raw.reshape(-1).view(np.uint32), rng.integers(-31, 32, size=(m, k), dtype=np.int8), rng.uniform(.01, .08, (m, k//32)).astype(np.float32)


def _amd_worker(mode:str) -> dict[str, Any]:
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import compile_linear, run_linear
  from tinygrad.helpers import GlobalCounters
  from tinygrad.uop.ops import Ops, UOp
  from extra.qk.q4k_q8_mmq_uop import (describe_q4k_q8_mmq_uop, describe_q4k_q8_mmq_wmma,
                                       emit_q4k_q8_mmq_uop, emit_q4k_q8_mmq_wmma)
  if mode == SCALAR_MODE:
    m, n, k = 2, 3, 256
    spec = describe_q4k_q8_mmq_uop(m, n, k); emitter = emit_q4k_q8_mmq_uop(spec)
  elif mode == WMMA_MODE:
    m, n, k = 16, 16, 256
    spec = describe_q4k_q8_mmq_wmma(); emitter = emit_q4k_q8_mmq_wmma(spec)
  else: raise ValueError(f"unsupported mode: {mode!r}")
  words, xq, xs = _fixture(m, n, k)
  graph = emitter(UOp.placeholder((m,n), dtypes.float32, 0), UOp.placeholder((n*36,), dtypes.uint32, 1),
                  UOp.placeholder((m*k,), dtypes.int8, 2), UOp.placeholder((m*8,), dtypes.float32, 3))
  uop = inspect_uop_graph(graph)
  # Materialize storage before authoring the candidate call. This keeps the
  # measured realization interval honest: no lazy allocation/copy calls are
  # hidden by subtracting them from GlobalCounters after the fact.
  out_storage = Tensor.empty(m,n,dtype=dtypes.float32,device="AMD").realize()
  words_storage = Tensor(words,device="AMD").realize()
  xq_storage = Tensor(xq.reshape(-1),device="AMD").realize()
  xs_storage = Tensor(xs.reshape(-1),device="AMD").realize()
  out = out_storage.custom_kernel(words_storage, xq_storage, xs_storage, fxn=emitter)[0]
  compiled = compile_linear(out.schedule_linear())
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
  sources = [str(next((x.arg for x in p.src if x.op is Ops.SOURCE), "")) for p in programs]
  names = [getattr(p.arg, "name", None) for p in programs]
  before = GlobalCounters.kernel_count
  run_linear(compiled)
  kernel_count_delta = GlobalCounters.kernel_count - before
  # Reading the already executed output must not be counted as launch evidence.
  got = out.numpy()
  ref = independent_packed_byte_reference(words, xq, xs, m=m, n=n, k=k)
  close = np.isclose(got, ref, rtol=3e-4, atol=3e-4)
  return {"mode": mode, "uop": uop, "program_count": len(programs), "kernel_count_delta": kernel_count_delta,
          "kernel_name": names[0] if len(names) == 1 else None, "fallback_used": False,
          "numeric": {"reference": "independent_packed_byte", "allclose": bool(close.all()), "rtol": 3e-4, "atol": 3e-4,
                      "max_abs": float(np.max(np.abs(got-ref))), "mismatch_count": int((~close).sum()),
                      "got_sample": got.reshape(-1)[:8].tolist(), "ref_sample": ref.reshape(-1)[:8].tolist()},
          "isa": classify_isa("\n".join(sources))}


def main(argv:Sequence[str]|None=None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--mode", choices=MODES, required=True)
  parser.add_argument("--timeout", type=float, default=30.0)
  args = parser.parse_args(argv)
  if args.worker:
    try: row = _amd_worker(args.mode)
    except BaseException as exc:  # worker boundary must always produce typed JSON
      row = _blocked(f"{type(exc).__name__}: {exc}")
    print(json.dumps(row, sort_keys=True, separators=(",", ":")))
    return 0 if row.get("passed") is not False else 2
  report = run_amd_validation(mode=args.mode, timeout_seconds=args.timeout)
  print(json.dumps(report, sort_keys=True, indent=2))
  return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
