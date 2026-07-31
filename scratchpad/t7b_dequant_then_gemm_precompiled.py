#!/usr/bin/env python3
"""T7b Stage 1b: same measurement as T7 (scratchpad/t7_dequant_then_gemm.py) -- dequant-to-fp16
THEN dense fp16 generic-TC GEMM at (512,12288,4096) -- with exactly ONE methodology change: the
dequant AST is precompiled ONCE (schedule_linear + compile_linear, the same LINEAR/CALL/PROGRAM
machinery tinygrad's own realize() uses internally) and then REPLAYED per call by walking the
already-compiled CALL list directly (pm_exec.rewrite per call), instead of building a fresh Tensor
op graph and calling .realize() on every rep. No Python graph rebuild happens inside the timed
region. Buffers (packed weight, dequant output, activation, GEMM output) are the same persistent
buffers every rep -- "rebind buffers per call" here means: fetch device buffer handles from the
already-allocated Buffer objects and dispatch, exactly the pattern T4/T7 already used for the GEMM
side, now applied to the dequant side too.

Nothing else changes: same shape, same GEMM AST/compile path, same build_artifact reference, same
warmup/timed-rep/3-run bracketing T7 used. This is still a MEASUREMENT script, not a production
change; no PACKED_WMMA_ROUTES row is touched, no prefill route is modified.

Mechanism verified against tinygrad internals (tinygrad/engine/realize.py) before writing this:
  - Tensor.schedule_linear() -> Tensor.linear_with_vars() calls transform_to_call() then
    _apply_map_to_tensors(), which resolves the tensor's buffer identity (full_t.uop.buffer becomes
    valid) WITHOUT executing anything yet.
  - compile_linear(linear, validate=False) walks the LINEAR once, converting each CALL's SINK into
    a compiled PROGRAM via to_program (this is the one-time compile cost -- codegen, Metal source
    render, MetalCompiler binary compile).
  - Replaying is: `for call in compiled.src: pm_exec.rewrite(call, ctx)` -- this is exactly what
    tinygrad's own `run_linear` does per call, and exactly what `time_call` (also in realize.py)
    does for a single CALL to get a real device et back. No graph_rewrite, no to_program, no new
    UOp construction happens on replay -- verified by inspecting compile_linear/run_linear source.
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
from tinygrad.engine.realize import get_runtime, compile_linear, ExecContext, pm_exec

import t1_generic_tc_dequant_probe as T1
# reuse T7's numpy reference / decode / validation / GEMM-AST-build helpers verbatim
import t7_dequant_then_gemm as T7

from extra.llm_research.prefill.packed_wmma_correctness_canary import build_artifact

M, N, K = T1.M, T1.N, T1.K
assert (M, N, K) == (512, 12288, 4096)
DEVICE = "METAL"
KBLOCKS = K // 256
BLOCK_COUNT = N * KBLOCKS
GFLOP_TOTAL = 2 * M * N * K / 1e9


def _metal_mem_bytes() -> int | None:
  try:
    stats = Device[DEVICE].allocator.memory_stats()
  except Exception:
    return None
  if stats is None: return None
  total, free = stats
  return total - free


def main():
  print("\n===== T7b STAGE 1b: precompiled dequant (compile once, replay per call) THEN generic-TC GEMM =====")
  artifact_path = "/tmp/t7_q4k_reference.npz"
  meta = build_artifact("Q4_K", artifact_path, shape=(M, N, K))
  print(f"reference artifact: {meta}")
  npz = np.load(artifact_path)
  activation, packed_raw, reference = npz["a"], npz["b"], npz["reference"]
  raw_np = np.ascontiguousarray(packed_raw).reshape(-1).view(np.uint8).reshape(BLOCK_COUNT, 144)
  assert raw_np.shape[0] == BLOCK_COUNT == N * KBLOCKS

  # -------- independent numpy validation of the decode formula, BEFORE trusting the GPU version --------
  numpy_validate_err = T7._validate_numpy_decode(raw_np, activation, reference)
  assert numpy_validate_err == 0.0, f"numpy full-decode formula does not match build_artifact reference: {numpy_validate_err}"

  # -------- compile the dense fp16 GEMM (generic TC path), same as T7's ceiling, unchanged --------
  mem_before_all = _metal_mem_bytes()
  dense_prog, a_slot, b_slot, out_slot = T7._build_dense_gemm_prog()

  # -------- persistent (uploaded ONCE) buffers: packed weight + activation --------
  packed_t = Tensor(raw_np, device=DEVICE).realize()
  act_buf = Buffer(DEVICE, activation.size, dtypes.half, initial_value=np.ascontiguousarray(activation).tobytes())
  Device[DEVICE].synchronize()
  mem_after_persistent = _metal_mem_bytes()

  # ============================================================================================
  # THE ONE METHODOLOGY CHANGE: build the dequant Tensor graph ONCE, schedule it to a LINEAR ONCE,
  # compile that LINEAR ONCE (to_program happens inside compile_linear) -- all OUTSIDE the timed
  # region. The timed region below only ever replays the already-compiled CALL list.
  # ============================================================================================
  full_t = T7._dequant_tensor(packed_t)
  assert not full_t.uop.has_buffer_identity()
  linear = full_t.schedule_linear()          # transform_to_call + buffer-identity resolution (one time)
  assert full_t.uop.has_buffer_identity()    # dequant_buf is now fixed for the rest of the run
  compiled_dequant = compile_linear(linear, validate=False)   # to_program happens here (one time)
  n_kernels = sum(1 for c in compiled_dequant.src if c.src and c.src[0].op is Ops.PROGRAM)
  print(f"[dequant] precompiled LINEAR: {len(compiled_dequant.src)} CALL node(s), {n_kernels} PROGRAM kernel(s)")
  dequant_buf: Buffer = full_t.uop.buffer
  exec_ctx = ExecContext(wait=True)

  def _dequant_replay() -> float:
    """Replay the precompiled dequant LINEAR. No graph rebuild, no to_program call. Returns summed
    device et (seconds) across whatever kernels the dequant graph compiled to."""
    total_et = 0.0
    for call in compiled_dequant.src:
      r = pm_exec.rewrite(call, exec_ctx)
      if isinstance(r, float): total_et += r
    return total_et

  # -------- correctness pass / one full dequant+GEMM pass --------
  def _one_pass():
    dq_et = _dequant_replay()
    out_buf = Buffer(DEVICE, M * N, dtypes.half)
    rt = get_runtime(DEVICE, dense_prog)
    order = list(dense_prog.arg.globals)
    slot_to_buf = {out_slot: out_buf, a_slot: act_buf, b_slot: dequant_buf}
    bufs = [slot_to_buf[s].get_buf(DEVICE) for s in order]
    gemm_et = rt(*bufs, global_size=dense_prog.arg.global_size, local_size=dense_prog.arg.local_size, vals=(), wait=True)
    return dq_et, gemm_et, out_buf, dequant_buf

  print("\n--- warmup (3 reps) ---")
  for i in range(3):
    Device[DEVICE].synchronize()
    _one_pass()
    Device[DEVICE].synchronize()

  print("\n--- correctness check ---")
  Device[DEVICE].synchronize()
  _, _, out_buf, _ = _one_pass()
  Device[DEVICE].synchronize()
  mem_peak_candidate = _metal_mem_bytes()
  out_mv = out_buf.copyout(memoryview(bytearray(out_buf.nbytes)))
  out_np = np.frombuffer(out_mv, dtype=np.float16).copy().reshape(M, N).astype(np.float32)
  ref32 = reference.astype(np.float32)
  max_abs_error = float(np.max(np.abs(out_np - ref32)))
  mean_abs_error = float(np.mean(np.abs(out_np - ref32)))
  print(f"max_abs_error (precompiled dequant-then-GEMM vs build_artifact reference) = {max_abs_error}")
  print(f"mean_abs_error = {mean_abs_error}")

  # spot-check the dequant buffer itself against the numpy full decode (bit-for-bit expected)
  dq_mv = dequant_buf.copyout(memoryview(bytearray(dequant_buf.nbytes)))
  dq_np = np.frombuffer(dq_mv, dtype=np.float16).copy().reshape(N, K)
  full_ref = T7._numpy_full_decode(raw_np)
  dq_diff = float(np.max(np.abs(dq_np.astype(np.float32) - full_ref.astype(np.float32))))
  print(f"dequant-buffer max_abs_diff vs numpy full decode = {dq_diff}  "
        f"bit_identical={np.array_equal(dq_np.view(np.uint16), full_ref.view(np.uint16))}")

  # -------- dequant-only device/host split, standalone (5 reps), to see whether the ~5ms of Python
  # graph-construction/scheduling overhead T7 measured actually went away under the replay technique --------
  print("\n--- dequant-only device/host split (5 reps, standalone) ---")
  dq_split = []
  for i in range(5):
    Device[DEVICE].synchronize()
    t0 = time.perf_counter()
    dq_et = _dequant_replay()
    Device[DEVICE].synchronize()
    t1 = time.perf_counter()
    dq_split.append({"rep": i, "host_wall_s": t1 - t0, "device_et_s": dq_et})
  print("dequant per-rep (host_ms, device_ms):", [(d["rep"], round(d["host_wall_s"]*1e3, 4), round(d["device_et_s"]*1e3, 4))
                                                    for d in dq_split])
  dq_host_mean = sum(d["host_wall_s"] for d in dq_split) / len(dq_split)
  dq_dev_mean = sum(d["device_et_s"] for d in dq_split) / len(dq_split)
  print(f"dequant device time mean = {dq_dev_mean*1e3:.4f} ms; dequant host time mean = {dq_host_mean*1e3:.4f} ms; "
        f"overhead (host-device) mean = {(dq_host_mean-dq_dev_mean)*1e3:.4f} ms")

  # -------- timed reps: whole operation (precompiled dequant replay + GEMM dispatch), synced before/after,
  # exactly the same bracketing T7 used for its whole-operation GFLOPS number --------
  print("\n--- timed reps (5) ---")
  times = []
  for i in range(5):
    Device[DEVICE].synchronize()
    t0 = time.perf_counter()
    dq_et, gemm_et, _, _ = _one_pass()
    Device[DEVICE].synchronize()
    t1 = time.perf_counter()
    times.append({"rep": i, "host_wall_s": t1 - t0, "dequant_device_et_s": dq_et, "gemm_device_et_s": gemm_et})
    mem_peak_candidate = max(mem_peak_candidate, _metal_mem_bytes() or 0)

  gflops = [GFLOP_TOTAL / t["host_wall_s"] for t in times]
  print("per-rep (host wall ms, GFLOPS, dequant_dev_ms, gemm_dev_ms):",
        [(t["rep"], round(t["host_wall_s"]*1e3, 4), round(g, 2), round(t["dequant_device_et_s"]*1e3, 4),
          round(t["gemm_device_et_s"]*1e3, 4)) for t, g in zip(times, gflops)])
  print(f"GFLOPS: min={min(gflops):.2f} max={max(gflops):.2f} mean={sum(gflops)/len(gflops):.2f} "
        f"spread={max(gflops)-min(gflops):.2f}")

  print(f"\nmetal mem_used (bytes): before_any_alloc={mem_before_all} after_persistent_upload={mem_after_persistent} "
        f"peak_during_timed_reps={mem_peak_candidate}")
  if mem_peak_candidate is not None:
    print(f"peak (MB) = {mem_peak_candidate/1e6:.2f}")

  mean_gflops = sum(gflops) / len(gflops)
  if mean_gflops >= 2000: verdict = "PASS (>= 2000 GFLOPS threshold, unchanged from T7)"
  elif mean_gflops < 1500: verdict = "FAIL (< 1500 GFLOPS) -- hypothesis does not clear its own bar even with orchestration cost removed"
  else: verdict = "AMBIGUOUS (1500-2000 GFLOPS) -- still below the 2000 threshold, hypothesis does not clear its own bar"
  print(f"\nDECISION: mean={mean_gflops:.2f} GFLOPS -> {verdict}")

  result = {
    "gflop_per_call": GFLOP_TOTAL,
    "numpy_validate_max_abs_error": numpy_validate_err,
    "n_kernels_dequant_compiled": n_kernels,
    "dequant_split_standalone": dq_split,
    "dequant_device_mean_ms": dq_dev_mean*1e3, "dequant_host_mean_ms": dq_host_mean*1e3,
    "times": times, "gflops": gflops,
    "gflops_min": min(gflops), "gflops_max": max(gflops), "gflops_mean": mean_gflops,
    "gflops_spread": max(gflops) - min(gflops),
    "max_abs_error": max_abs_error, "mean_abs_error": mean_abs_error,
    "dequant_buffer_vs_numpy_max_abs_diff": dq_diff,
    "mem_used_bytes": {"before_any_alloc": mem_before_all, "after_persistent_upload": mem_after_persistent,
                        "peak_during_timed_reps": mem_peak_candidate},
    "verdict": verdict,
  }
  with open("/tmp/t7b_stage1b_result.json", "w") as f:
    json.dump(result, f, indent=2)
  print("\nwrote /tmp/t7b_stage1b_result.json")


if __name__ == "__main__":
  main()
