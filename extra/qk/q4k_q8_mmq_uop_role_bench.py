#!/usr/bin/env python3
"""Fail-closed AMD execution/timing gate for loaded Qwen3-14B Q4_K roles.

The parent is AMD-import-free.  Roles execute serially in fresh subprocesses;
each worker compares the complete WMMA result with the scalar direct-UOp owner
using identical deterministic packed operands.  This is research evidence and
does not select or modify a model route.
"""
from __future__ import annotations

import argparse, json, os, platform, statistics, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from extra.qk.q4k_q8_mmq_uop_role_compile_gate import GGUF, ROLE_ORDER, RUNTIME_M, RoleShape, derive_role_shapes
from extra.qk.prefill_mmq_parity_gate import RTOL as REL_RMSE_THRESHOLD, _rel_rmse

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "tinygrad.q4k_q8_mmq_uop_role_bench.v1"
PASS = "Q4K_Q8_MMQ_UOP_ROLE_BENCH_PASS"
BLOCKED = "Q4K_Q8_MMQ_UOP_ROLE_BENCH_BLOCKED"
SEED = 20260715
STRICT_RTOL, STRICT_ATOL = 3e-4, 3e-4
SUBSET_M, SUBSET_N = 8, 8


def deterministic_fixture(shape:RoleShape, seed:int=SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Deterministic finite Q4_K bytes and already-packed Q8_1 values/scales."""
  rng = np.random.default_rng(seed + sum((i+1)*ord(c) for i,c in enumerate(shape.role)))
  raw = rng.integers(0, 256, size=(shape.n, shape.k//256, 144), dtype=np.uint8)
  raw[..., :4] = np.frombuffer(np.array([0.03125, 0.0078125], dtype="<f2").tobytes(), dtype=np.uint8)
  xq = rng.integers(-31, 32, size=(shape.m, shape.k), dtype=np.int8)
  xs = rng.uniform(0.01, 0.08, size=(shape.m, shape.k//32)).astype(np.float32)
  return raw.reshape(-1).view(np.uint32), xq, xs


def summarize_samples(wall_s:Sequence[float], device_s:Sequence[float], launches:Sequence[int], logical_ops:int) -> dict[str, Any]:
  if not wall_s or len(wall_s) != len(device_s) or len(wall_s) != len(launches): raise ValueError("inconsistent or empty samples")
  trusted = all(x > 0 for x in device_s) and all(x == 1 for x in launches)
  wm, dm = statistics.median(wall_s), statistics.median(device_s) if trusted else None
  return {"samples":len(wall_s), "wall_ms":[x*1e3 for x in wall_s], "device_ms":[x*1e3 for x in device_s],
          "wall_median_ms":wm*1e3, "device_median_ms":None if dm is None else dm*1e3,
          "measured_launch_counts":list(launches), "device_time_trustworthy":trusted,
          "logical_ops":logical_ops, "logical_wall_tflops":logical_ops/wm/1e12,
          "logical_device_tflops":None if dm is None else logical_ops/dm/1e12}


def validate_worker(shape:RoleShape, row:Mapping[str, Any], *, warmups:int, samples:int) -> tuple[bool, str|None]:
  if row.get("role") != shape.role or row.get("shape") != shape.to_json(): return False, f"{shape.role}: worker shape identity mismatch"
  target = row.get("target", {})
  if target.get("program_count") != 1 or target.get("program_name") != shape.kernel_name:
    return False, f"{shape.role}: target is not the one expected PROGRAM"
  if target.get("fallback_used") is not False: return False, f"{shape.role}: fallback evidence missing or true"
  timing = target.get("timing", {})
  if target.get("warmup_launch_count") != warmups or timing.get("samples") != samples:
    return False, f"{shape.role}: warmup/sample accounting mismatch"
  if timing.get("measured_launch_counts") != [1]*samples or timing.get("device_time_trustworthy") is not True:
    return False, f"{shape.role}: measured launch/device timing evidence failed"
  scalar, numeric = row.get("scalar_authority", {}), row.get("full_output_correctness", {})
  if scalar.get("program_count") != 1 or scalar.get("launch_count") != 1 or scalar.get("fallback_used") is not False:
    return False, f"{shape.role}: scalar direct-UOp authority did not execute exactly once"
  if (numeric.get("reference") != "scalar_direct_uop_same_operands_full_output" or
      numeric.get("rel_rmse_threshold") != REL_RMSE_THRESHOLD or numeric.get("rel_rmse_pass") is not True or
      numeric.get("finite") is not True or numeric.get("nan_count") != 0):
    return False, f"{shape.role}: full-output WMMA/scalar relative-RMSE correctness failed"
  subset = row.get("independent_subset_correctness", {})
  if (subset.get("reference") != "independent_packed_byte_reference" or subset.get("shape") != [SUBSET_M,SUBSET_N] or
      subset.get("rel_rmse_threshold") != REL_RMSE_THRESHOLD or subset.get("wmma_rel_rmse_pass") is not True or
      subset.get("scalar_rel_rmse_pass") is not True or subset.get("all_finite") is not True or subset.get("nan_count") != 0):
    return False, f"{shape.role}: independent packed-byte subset authority failed"
  prov = row.get("provenance", {})
  if prov.get("independent_random_byte_bounded_authority") != "extra.qk.q4k_q8_mmq_uop_validation:independent_packed_byte_reference":
    return False, f"{shape.role}: independent bounded authority provenance missing"
  return True, None


def _health(checker:Callable[..., Any]=subprocess.run) -> dict[str, Any]:
  cmd = ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showtemp"]
  try: proc = checker(cmd, text=True, capture_output=True, timeout=15, check=False)
  except (OSError, subprocess.TimeoutExpired) as exc: return {"healthy":False, "error":f"{type(exc).__name__}: {exc}"}
  text = (proc.stdout + "\n" + proc.stderr).strip()
  bad = any(x in text.lower() for x in ("gpu reset", "unresponsive", "xgmi error"))
  return {"healthy":proc.returncode == 0 and not bad, "returncode":proc.returncode, "output":text[-4000:]}


def _blocked(reason:str, **evidence:Any) -> dict[str, Any]:
  return {"protocol":PROTOCOL, "passed":False, "verdict":BLOCKED, "first_failure":reason, "evidence":evidence}


def run_gate(path:str|Path=GGUF, *, warmups:int=2, samples:int=5, timeout_seconds:float=600.0,
             python:str=sys.executable, env:Mapping[str,str]|None=None, runner:Callable[...,Any]=subprocess.run,
             health_checker:Callable[...,Any]=subprocess.run) -> dict[str, Any]:
  if warmups < 0 or samples <= 0 or timeout_seconds <= 0: return _blocked("invalid warmups, samples, or timeout")
  try: shapes = derive_role_shapes(path, runtime_m=RUNTIME_M)
  except Exception as exc: return _blocked(f"metadata: {type(exc).__name__}: {exc}")
  initial_health = _health(health_checker)
  if not initial_health["healthy"]: return _blocked("GPU health preflight failed", health=initial_health)
  child_env = dict(os.environ if env is None else env)
  child_env.update({"DEV":"AMD", "PYTHONPATH":str(ROOT)+os.pathsep+child_env.get("PYTHONPATH", "")})
  passed = []
  for shape in shapes:  # contractual order starts with attn_kv
    cmd = [python, "-m", "extra.qk.q4k_q8_mmq_uop_role_bench", "--worker", "--role", shape.role,
           "--m", str(shape.m), "--n", str(shape.n), "--k", str(shape.k), "--quant", shape.quant,
           "--warmups", str(warmups), "--samples", str(samples)]
    try: proc = runner(cmd, cwd=ROOT, env=child_env, text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired: return _blocked(f"{shape.role}: timed out after {timeout_seconds:g}s", passed_roles=passed)
    except OSError as exc: return _blocked(f"{shape.role}: worker could not start: {exc}", passed_roles=passed)
    if proc.returncode != 0:
      return _blocked(f"{shape.role}: worker failed with exit {proc.returncode}", stderr=proc.stderr[-4000:], stdout=proc.stdout[-4000:], passed_roles=passed)
    try: row = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
      return _blocked(f"{shape.role}: invalid worker JSON: {exc}", stdout=proc.stdout[-4000:], passed_roles=passed)
    ok, reason = validate_worker(shape, row, warmups=warmups, samples=samples)
    if not ok: return _blocked(reason or f"{shape.role}: validation failed", row=row, passed_roles=passed)
    passed.append(row)
    health = _health(health_checker)
    if not health["healthy"]: return _blocked(f"{shape.role}: GPU health postflight failed", health=health, passed_roles=passed)
  return {"protocol":PROTOCOL, "passed":True, "verdict":PASS, "first_failure":None,
          "evidence":{"gguf":str(path), "metadata_only":True, "full_model_loaded":False, "runtime_m":RUNTIME_M,
                      "role_order":list(ROLE_ORDER), "initial_health":initial_health, "roles":passed}}


def _compile(emitter, shape:RoleShape, words, xq, xs):
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import compile_linear
  from tinygrad.uop.ops import Ops
  storage = Tensor.empty(shape.m, shape.n, dtype=dtypes.float32, device="AMD").realize()
  out = storage.custom_kernel(words, xq, xs, fxn=emitter)[0]
  start = time.perf_counter(); linear = compile_linear(out.schedule_linear()); compile_ms = (time.perf_counter()-start)*1e3
  programs = [u for u in linear.toposort() if u.op is Ops.PROGRAM]
  return out, linear, programs, compile_ms


def _worker(shape:RoleShape, warmups:int, samples:int) -> dict[str, Any]:
  from tinygrad import Device, Tensor
  from tinygrad.engine.realize import run_linear
  from tinygrad.helpers import GlobalCounters
  from extra.qk.q4k_q8_mmq_uop import (describe_q4k_q8_mmq_uop, describe_q4k_q8_mmq_wmma,
    emit_q4k_q8_mmq_uop, emit_q4k_q8_mmq_wmma)
  from extra.qk.q4k_q8_mmq_uop_validation import independent_packed_byte_reference
  words_np, xq_np, xs_np = deterministic_fixture(shape)
  words, xq, xs = (Tensor(x.reshape(-1), device="AMD").realize() for x in (words_np, xq_np, xs_np))
  dev = Device["AMD"]; dev.synchronize()
  wout, wlin, wprograms, wcompile = _compile(emit_q4k_q8_mmq_wmma(
    describe_q4k_q8_mmq_wmma(m=shape.m,n=shape.n,k=shape.k)), shape, words, xq, xs)
  if len(wprograms) != 1: raise RuntimeError(f"WMMA target compiled {len(wprograms)} PROGRAMs")
  for _ in range(warmups): run_linear(wlin, wait=True)
  dev.synchronize()
  wall, device, launches = [], [], []
  for _ in range(samples):
    dev.synchronize(); bt,bk = GlobalCounters.time_sum_s,GlobalCounters.kernel_count; start=time.perf_counter()
    run_linear(wlin, wait=True); dev.synchronize(); wall.append(time.perf_counter()-start)
    device.append(GlobalCounters.time_sum_s-bt); launches.append(GlobalCounters.kernel_count-bk)
  timing = summarize_samples(wall, device, launches, 2*shape.m*shape.n*shape.k)
  sout, slin, sprograms, scompile = _compile(emit_q4k_q8_mmq_uop(
    describe_q4k_q8_mmq_uop(shape.m,shape.n,shape.k)), shape, words, xq, xs)
  if len(sprograms) != 1: raise RuntimeError(f"scalar authority compiled {len(sprograms)} PROGRAMs")
  dev.synchronize(); bk=GlobalCounters.kernel_count; start=time.perf_counter(); run_linear(slin, wait=True); dev.synchronize()
  scalar_wall_ms=(time.perf_counter()-start)*1e3; scalar_launches=GlobalCounters.kernel_count-bk
  got, ref = wout.numpy(), sout.numpy()
  close = np.isclose(got, ref, rtol=STRICT_RTOL, atol=STRICT_ATOL)
  full_rel = _rel_rmse(got, ref)
  # The first 8 output columns map to the first 8 complete packed Q4_K rows;
  # every selected output still contracts across the role's full K.
  subset_words = words_np.reshape(shape.n, shape.k//256, 36)[:SUBSET_N].reshape(-1)
  oracle = independent_packed_byte_reference(subset_words, xq_np[:SUBSET_M], xs_np[:SUBSET_M],
                                             m=SUBSET_M, n=SUBSET_N, k=shape.k)
  wsub, ssub = got[:SUBSET_M,:SUBSET_N], ref[:SUBSET_M,:SUBSET_N]
  wrel, srel = _rel_rmse(wsub, oracle), _rel_rmse(ssub, oracle)
  arrays = (got, ref, wsub, ssub, oracle)
  nan_count = sum(int(np.isnan(x).sum()) for x in arrays)
  return {"role":shape.role, "shape":shape.to_json(), "seed":SEED,
    "target":{"emitter":"direct_uop_wmma", "program_count":1, "program_name":wprograms[0].arg.name,
              "fallback_used":False, "compile_ms":wcompile, "warmups":warmups,
              "warmup_launch_count":warmups, "timing":timing},
    "scalar_authority":{"emitter":"proven_scalar_direct_uop", "program_count":1,
                        "program_name":sprograms[0].arg.name, "fallback_used":False,
                        "compile_ms":scompile, "launch_count":scalar_launches, "wall_ms":scalar_wall_ms},
    "full_output_correctness":{"reference":"scalar_direct_uop_same_operands_full_output", "elements":int(got.size),
      "rel_rmse":full_rel, "rel_rmse_threshold":REL_RMSE_THRESHOLD, "rel_rmse_pass":bool(full_rel < REL_RMSE_THRESHOLD),
      "finite":bool(np.isfinite(got).all() and np.isfinite(ref).all()), "nan_count":int(np.isnan(got).sum()+np.isnan(ref).sum()),
      "allclose":bool(close.all()), "rtol":STRICT_RTOL, "atol":STRICT_ATOL, "mismatch_count":int((~close).sum()),
      "max_abs":float(np.max(np.abs(got-ref))), "wmma_sample":got.reshape(-1)[:8].tolist(), "scalar_sample":ref.reshape(-1)[:8].tolist()},
    "independent_subset_correctness":{"reference":"independent_packed_byte_reference", "shape":[SUBSET_M,SUBSET_N],
      "full_k":shape.k, "wmma_rel_rmse":wrel, "scalar_rel_rmse":srel, "rel_rmse_threshold":REL_RMSE_THRESHOLD,
      "wmma_rel_rmse_pass":bool(wrel < REL_RMSE_THRESHOLD), "scalar_rel_rmse_pass":bool(srel < REL_RMSE_THRESHOLD),
      "all_finite":bool(all(np.isfinite(x).all() for x in arrays)), "nan_count":nan_count,
      "wmma_max_abs":float(np.max(np.abs(wsub-oracle))), "scalar_max_abs":float(np.max(np.abs(ssub-oracle))),
      "oracle_sample":oracle.reshape(-1)[:8].tolist()},
    "operands":{"q4_words":int(words_np.size), "q8_values":int(xq_np.size), "q8_scales":int(xs_np.size),
                "deterministic":True, "same_storage_for_both_emitters":True},
    "provenance":{"full_role_authority":"proven_scalar_direct_uop_same_operands",
      "independent_random_byte_bounded_authority":"extra.qk.q4k_q8_mmq_uop_validation:independent_packed_byte_reference",
      "independent_authority_scope":"bounded random-byte canary; retained independently, not substituted for full-role scalar comparison"}}


def main(argv:Sequence[str]|None=None) -> int:
  p=argparse.ArgumentParser(description=__doc__); p.add_argument("--gguf",default=str(GGUF)); p.add_argument("--timeout",type=float,default=600)
  p.add_argument("--warmups",type=int,default=2); p.add_argument("--samples",type=int,default=5); p.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
  p.add_argument("--role",choices=ROLE_ORDER); p.add_argument("--m",type=int); p.add_argument("--n",type=int); p.add_argument("--k",type=int); p.add_argument("--quant")
  args=p.parse_args(argv)
  if args.worker:
    try: row=_worker(RoleShape(args.role,args.m,args.n,args.k,args.quant),args.warmups,args.samples); code=0
    except BaseException as exc: row={"worker_error":f"{type(exc).__name__}: {exc}"}; code=2
    print(json.dumps(row,sort_keys=True,separators=(",",":"))); return code
  row=run_gate(args.gguf,warmups=args.warmups,samples=args.samples,timeout_seconds=args.timeout)
  row["platform"]=platform.platform(); print(json.dumps(row,sort_keys=True,indent=2)); return 0 if row["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
