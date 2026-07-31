#!/usr/bin/env python3
"""T7 Stage 1: does per-layer dequant-to-fp16 (materialize the whole packed Q4_K weight matrix
into an fp16 scratch buffer, THEN run the same fp16 generic-TC GEMM T4 measured) actually reach
the ~2590 GFLOPS effective projection, or does paying dequant bandwidth once still fall short?

This is a MEASUREMENT script, not a production change. No PACKED_WMMA_ROUTES row is touched, no
prefill route is modified. It reuses, verbatim, without re-deriving:
  - scratchpad/t1_generic_tc_dequant_probe.py (T1): M,N,K,TARGETS,_dense_gemm_ast,_find_mnk,
    _force_generic_tc -- the exact fp16 dense GEMM ceiling AST/opt-forcing T4 used.
  - scratchpad/t4_fused_generic_tc_execute.py (T4): the to_program(...)->get_runtime(...) raw
    dispatch technique (real compiled Metal binary, real Device["METAL"] buffers), and the
    warmup/timed-rep harness shape.
  - extra/llm_research/prefill/packed_wmma_correctness_canary.py:build_artifact -- the same
    independent numpy Q4_K decoder/artifact generator T4 used for its correctness reference.

The dequant step itself is NEW here (T4 never materialized a full weight matrix -- its fused
kernel decodes per-element inside the GEMM inner loop). It is built as an ordinary tinygrad
Tensor graph (bitwise/cast/reshape ops on the packed uint8 view), scheduled and compiled through
tinygrad's NORMAL scheduler (no forced TC opt -- there is no matmul in this graph, only decode),
and its correctness is independently verified below against the numpy full-matrix Q4_K decode,
which is itself cross-checked against build_artifact's reference (see the validation block).

Q4_K block layout (144 bytes / 256 elements per block), reused verbatim from
_decode_selected_q4 in packed_wmma_correctness_canary.py:
  bytes[0:2]   = d     (fp16 block scale)
  bytes[2:4]   = dmin  (fp16 block min)
  bytes[4:16]  = 12 packed 6-bit scale/min sub-block bytes
  bytes[16:144]= 128 bytes of 4-bit quants, 2 per byte (256 quants total)
The vectorized form used below is verified bit-identical to a numpy transliteration of
_decode_selected_q4, and that numpy transliteration is separately verified (max_abs_error==0)
against build_artifact's own reference array before this script is trusted for anything.
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/scratchpad")

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_runtime

import t1_generic_tc_dequant_probe as T1

from extra.llm_research.prefill.packed_wmma_correctness_canary import build_artifact

M, N, K = T1.M, T1.N, T1.K
assert (M, N, K) == (512, 12288, 4096)
DEVICE = "METAL"
KBLOCKS = K // 256
BLOCK_COUNT = N * KBLOCKS
GFLOP_TOTAL = 2 * M * N * K / 1e9


# ------------------------------------------------------------------ numpy reference decode -----
def _numpy_full_decode(raw: np.ndarray) -> np.ndarray:
  """Full-matrix vectorized transliteration of _decode_selected_q4 (packed_wmma_correctness_canary.py),
  decoding EVERY element instead of one selected k-position per row. raw: (BLOCK_COUNT, 144) uint8."""
  d = raw[:, 0:2].reshape(-1).view(np.float16).astype(np.float32)
  dmin = raw[:, 2:4].reshape(-1).view(np.float16).astype(np.float32)
  sb = raw[:, 4:16]
  scale_lo, minimum_lo = sb[:, 0:4] & 63, sb[:, 4:8] & 63
  high = sb[:, 8:12]
  scale_hi = (high & 15) | ((sb[:, 0:4] >> 6) << 4)
  minimum_hi = (high >> 4) | ((sb[:, 4:8] >> 6) << 4)
  scale = np.concatenate([scale_lo, scale_hi], axis=1).astype(np.float32).reshape(-1, 4, 2)
  minimum = np.concatenate([minimum_lo, minimum_hi], axis=1).astype(np.float32).reshape(-1, 4, 2)
  payload = raw[:, 16:144].reshape(-1, 4, 32)
  quant_even, quant_odd = (payload & 15).astype(np.float32), ((payload >> 4) & 15).astype(np.float32)
  value = np.empty((raw.shape[0], 4, 2, 32), dtype=np.float32)
  value[:, :, 0, :] = d[:, None, None]*scale[:, :, 0][:, :, None]*quant_even - dmin[:, None, None]*minimum[:, :, 0][:, :, None]
  value[:, :, 1, :] = d[:, None, None]*scale[:, :, 1][:, :, None]*quant_odd - dmin[:, None, None]*minimum[:, :, 1][:, :, None]
  return value.reshape(raw.shape[0], 256).astype(np.float16).reshape(N, KBLOCKS, 256).reshape(N, K)


def _validate_numpy_decode(raw: np.ndarray, activation: np.ndarray, reference: np.ndarray) -> float:
  """Cross-check the numpy full decode against build_artifact's own reference (independent origin:
  reference was built by _decode_selected_q4 at one k-position per row; full[:,k_positions] should
  match it exactly since decoding every position is a strict superset)."""
  full = _numpy_full_decode(raw)
  rows = np.arange(M, dtype=np.int64)
  k_positions = (rows * 251 + 17) % K
  coefficients = (rows % 4 + 1).astype(np.float32)
  # build_artifact itself rounds each row's product to fp16 before storing (see
  # packed_wmma_correctness_canary.py:128: `(decode(...).astype(np.float32) * coefficient).astype(np.float16)`)
  # -- replicate that exact rounding here so this is an EXACT-match check, not a fresh-rounding-error check.
  recon = (full[:, k_positions].T.astype(np.float32) * coefficients[:, None]).astype(np.float16)
  err = float(np.max(np.abs(recon.astype(np.float32) - reference.astype(np.float32))))
  print(f"[validate] numpy full-decode vs build_artifact reference (per-row recon, fp16-rounded): max_abs_error={err}")
  gemm_recon = activation.astype(np.float32) @ full.astype(np.float32).T
  gemm_err = float(np.max(np.abs(gemm_recon - reference.astype(np.float32))))
  print(f"[validate] numpy full-decode vs build_artifact reference (via fp32-matmul recon, informational only): max_abs_error={gemm_err}")
  return err


# ------------------------------------------------------------------ GPU dequant tensor graph ---
def _dequant_tensor(packed_t: Tensor) -> Tensor:
  """packed_t: persistent, already-realized (BLOCK_COUNT,144) uint8 Tensor on METAL. Builds a FRESH
  ordinary-op graph each call (matching the real per-layer-materialize cost: this is what a
  production callsite would re-emit every forward pass) and returns an UNREALIZED (N,K) fp16 Tensor.
  Caller is responsible for .realize() and for timing that realize."""
  raw = packed_t
  d16 = raw[:, 0].cast(dtypes.uint16) | (raw[:, 1].cast(dtypes.uint16) << 8)
  d = d16.bitcast(dtypes.float16).cast(dtypes.float32)
  dmin16 = raw[:, 2].cast(dtypes.uint16) | (raw[:, 3].cast(dtypes.uint16) << 8)
  dmin = dmin16.bitcast(dtypes.float16).cast(dtypes.float32)

  sb = raw[:, 4:16]
  scale_lo, minimum_lo = sb[:, 0:4] & 63, sb[:, 4:8] & 63
  high = sb[:, 8:12]
  scale_hi = (high & 15) | ((sb[:, 0:4] >> 6) << 4)
  minimum_hi = (high >> 4) | ((sb[:, 4:8] >> 6) << 4)
  scale = scale_lo.cat(scale_hi, dim=1).reshape(BLOCK_COUNT, 4, 2, 1).cast(dtypes.float32)
  minimum = minimum_lo.cat(minimum_hi, dim=1).reshape(BLOCK_COUNT, 4, 2, 1).cast(dtypes.float32)

  payload = raw[:, 16:144].reshape(BLOCK_COUNT, 4, 32)
  quant_even = (payload & 15).reshape(BLOCK_COUNT, 4, 1, 32).cast(dtypes.float32)
  quant_odd = ((payload >> 4) & 15).reshape(BLOCK_COUNT, 4, 1, 32).cast(dtypes.float32)
  quant = quant_even.cat(quant_odd, dim=2)

  d_b = d.reshape(BLOCK_COUNT, 1, 1, 1)
  dmin_b = dmin.reshape(BLOCK_COUNT, 1, 1, 1)
  value = d_b * scale * quant - dmin_b * minimum
  value = value.reshape(BLOCK_COUNT, 256).cast(dtypes.float16)
  full = value.reshape(N, KBLOCKS, 256).reshape(N, K).contiguous()
  return full


# ------------------------------------------------------------------ dense fp16 GEMM (same as T4/T1)
def _build_dense_gemm_prog():
  target_str, make_renderer = T1.TARGETS[DEVICE]
  renderer = make_renderer(Target.parse(target_str))
  ast = T1._force_generic_tc(T1._dense_gemm_ast(DEVICE))
  prog = to_program(ast, renderer)
  assert prog.op is Ops.PROGRAM
  binary = prog.src[4].arg
  assert isinstance(binary, bytes) and len(binary) > 0
  source = next((u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  print(f"[dense_gemm] compiled OK. wmma={source.count('__WMMA') if source else None} "
        f"sgma={source.count('simdgroup_multiply_accumulate') if source else None} "
        f"globals={prog.arg.globals} global_size={prog.arg.global_size} local_size={prog.arg.local_size}")
  dense_ast_unforced = T1._dense_gemm_ast(DEVICE)
  red, in0, in1, n_rng, k_rng = T1._find_mnk(dense_ast_unforced)
  a_param_slot = next(u.arg.slot for u in in0.toposort() if u.op is Ops.PARAM)
  b_param_slot = next(u.arg.slot for u in in1.toposort() if u.op is Ops.PARAM)
  outs = set(prog.arg.outs)
  assert len(outs) == 1
  out_slot = next(iter(outs))
  return prog, a_param_slot, b_param_slot, out_slot


def _metal_mem_bytes() -> int | None:
  """Live MTLBuffer footprint owned by tinygrad (ops_metal.py MetalAllocator.memory_stats: total,
  free -- used = total-free). Returns None if unavailable (non-Metal / API absence)."""
  try:
    stats = Device[DEVICE].allocator.memory_stats()
  except Exception:
    return None
  if stats is None: return None
  total, free = stats
  return total - free


def main():
  print("\n===== T7 STAGE 1: dequant-to-fp16 THEN generic-TC GEMM, whole-operation timing =====")
  artifact_path = "/tmp/t7_q4k_reference.npz"
  meta = build_artifact("Q4_K", artifact_path, shape=(M, N, K))
  print(f"reference artifact: {meta}")
  npz = np.load(artifact_path)
  activation, packed_raw, reference = npz["a"], npz["b"], npz["reference"]
  raw_np = np.ascontiguousarray(packed_raw).reshape(-1).view(np.uint8).reshape(BLOCK_COUNT, 144)
  assert raw_np.shape[0] == BLOCK_COUNT == N * KBLOCKS

  # -------- independent numpy validation of the decode formula, BEFORE trusting the GPU version --------
  numpy_validate_err = _validate_numpy_decode(raw_np, activation, reference)
  assert numpy_validate_err == 0.0, f"numpy full-decode formula does not match build_artifact reference: {numpy_validate_err}"

  # -------- compile the dense fp16 GEMM (generic TC path), same as T4's ceiling --------
  mem_before_all = _metal_mem_bytes()
  dense_prog, a_slot, b_slot, out_slot = _build_dense_gemm_prog()

  # -------- persistent (uploaded ONCE) buffers: packed weight + activation, matching how a real
  # per-layer materialize call would reuse already-resident model weights, not re-upload them --------
  packed_t = Tensor(raw_np, device=DEVICE).realize()
  act_buf = Buffer(DEVICE, activation.size, dtypes.half, initial_value=np.ascontiguousarray(activation).tobytes())
  Device[DEVICE].synchronize()
  mem_after_persistent = _metal_mem_bytes()

  # -------- correctness pass (round 0 of the timed measurement) --------
  def _one_pass():
    full_t = _dequant_tensor(packed_t)
    full_t = full_t.realize()
    dequant_buf: Buffer = full_t.uop.buffer
    out_buf = Buffer(DEVICE, M * N, dtypes.half)
    rt = get_runtime(DEVICE, dense_prog)
    order = list(dense_prog.arg.globals)
    slot_to_buf = {out_slot: out_buf, a_slot: act_buf, b_slot: dequant_buf}
    bufs = [slot_to_buf[s].get_buf(DEVICE) for s in order]
    et = rt(*bufs, global_size=dense_prog.arg.global_size, local_size=dense_prog.arg.local_size, vals=(), wait=True)
    return et, out_buf, dequant_buf

  print("\n--- warmup (3 reps) ---")
  for i in range(3):
    Device[DEVICE].synchronize()
    _one_pass()
    Device[DEVICE].synchronize()

  print("\n--- correctness check ---")
  Device[DEVICE].synchronize()
  _, out_buf, dequant_buf = _one_pass()
  Device[DEVICE].synchronize()
  mem_peak_candidate = _metal_mem_bytes()
  out_mv = out_buf.copyout(memoryview(bytearray(out_buf.nbytes)))
  out_np = np.frombuffer(out_mv, dtype=np.float16).copy().reshape(M, N).astype(np.float32)
  ref32 = reference.astype(np.float32)
  max_abs_error = float(np.max(np.abs(out_np - ref32)))
  mean_abs_error = float(np.mean(np.abs(out_np - ref32)))
  print(f"max_abs_error (dequant-then-GEMM vs build_artifact reference) = {max_abs_error}")
  print(f"mean_abs_error = {mean_abs_error}")

  # spot-check the dequant buffer itself against the numpy full decode (bit-for-bit expected)
  dq_mv = dequant_buf.copyout(memoryview(bytearray(dequant_buf.nbytes)))
  dq_np = np.frombuffer(dq_mv, dtype=np.float16).copy().reshape(N, K)
  full_ref = _numpy_full_decode(raw_np)
  dq_diff = float(np.max(np.abs(dq_np.astype(np.float32) - full_ref.astype(np.float32))))
  print(f"dequant-buffer max_abs_diff vs numpy full decode = {dq_diff}  "
        f"bit_identical={np.array_equal(dq_np.view(np.uint16), full_ref.view(np.uint16))}")

  # -------- timed reps: whole operation (dequant realize + GEMM dispatch), synced before/after --------
  print("\n--- timed reps (5) ---")
  times = []
  for i in range(5):
    Device[DEVICE].synchronize()
    t0 = time.perf_counter()
    et, _, _ = _one_pass()
    Device[DEVICE].synchronize()
    t1 = time.perf_counter()
    times.append({"rep": i, "host_wall_s": t1 - t0, "gpu_et_s": et})
    mem_peak_candidate = max(mem_peak_candidate, _metal_mem_bytes() or 0)

  gflops = [GFLOP_TOTAL / t["host_wall_s"] for t in times]
  print("per-rep (host wall ms, GFLOPS):", [(t["rep"], round(t["host_wall_s"]*1e3, 4), round(g, 2))
                                             for t, g in zip(times, gflops)])
  print(f"GFLOPS: min={min(gflops):.2f} max={max(gflops):.2f} mean={sum(gflops)/len(gflops):.2f} "
        f"spread={max(gflops)-min(gflops):.2f}")

  print(f"\nmetal mem_used (bytes): before_any_alloc={mem_before_all} after_persistent_upload={mem_after_persistent} "
        f"peak_during_timed_reps={mem_peak_candidate}")
  if mem_peak_candidate is not None:
    print(f"peak (MB) = {mem_peak_candidate/1e6:.2f}")

  mean_gflops = sum(gflops) / len(gflops)
  if mean_gflops >= 2000: verdict = "PASS (>= ~2000 GFLOPS) -- projection holds, proceed to stage 2"
  elif mean_gflops < 1500: verdict = "FAIL (< ~1500 GFLOPS) -- projection is wrong, STOP, do not proceed to stage 2"
  else: verdict = "AMBIGUOUS (1500-2000 GFLOPS) -- stop, report as ambiguous"
  print(f"\nDECISION: mean={mean_gflops:.2f} GFLOPS -> {verdict}")

  result = {
    "gflop_per_call": GFLOP_TOTAL,
    "numpy_validate_max_abs_error": numpy_validate_err,
    "times": times, "gflops": gflops,
    "gflops_min": min(gflops), "gflops_max": max(gflops), "gflops_mean": mean_gflops,
    "gflops_spread": max(gflops) - min(gflops),
    "max_abs_error": max_abs_error, "mean_abs_error": mean_abs_error,
    "dequant_buffer_vs_numpy_max_abs_diff": dq_diff,
    "mem_used_bytes": {"before_any_alloc": mem_before_all, "after_persistent_upload": mem_after_persistent,
                        "peak_during_timed_reps": mem_peak_candidate},
    "verdict": verdict,
  }
  with open("/tmp/t7_stage1_result.json", "w") as f:
    json.dump(result, f, indent=2)
  print("\nwrote /tmp/t7_stage1_result.json")


if __name__ == "__main__":
  main()
