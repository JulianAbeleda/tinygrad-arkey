#!/usr/bin/env python3
"""T4: EXECUTE (not just render) T1b Experiment A -- the fused packed-Q4_K -> Metal tensor-core
kernel reached through the GENERIC TC opt -- and check correctness against an independent numpy
reference, then (if correct) measure its GFLOPS against a plain fp16 GEMM ceiling at the same shape.

Reuses, verbatim, without re-deriving:
  - scratchpad/t1_generic_tc_dequant_probe.py: M,N,K,QUANT,TC_OPT,TARGETS,_dense_gemm_ast,
    _find_mnk,_force_generic_tc (imported as T1).
  - scratchpad/t1b_generic_tc_dequant_vectorized_probe.py: _experiment_a_naive_dodge,
    WIDTH_FOR_BACKEND (imported as T1B). This is the exact AST construction whose rendered
    source is the single 3-buffer r_64_1536_32_2_512_... kernel described in the task.
  - extra/llm_research/prefill/packed_wmma_correctness_canary.py: build_artifact -- the
    device-independent numpy Q4_K decoder -- to synthesize packed weights + sparse activation +
    an independent full-output reference, WITHOUT touching any of the precontract-path machinery
    (current_prefill_execution_adapter, guarded_execution, etc.) that this task is not about.

Execution technique: to_program(ast, renderer) (compile-only, as T1/T1b already use) actually runs
ALL the way through do_compile too (codegen/__init__.py's do_to_program chains do_linearize ->
do_estimates -> do_assemble/do_render -> do_compile), so the returned UOp is a real Ops.PROGRAM with
a real compiled Metal binary (MetalCompiler is an offline MTLCodeGenService call, device-independent
-- see tinygrad/runtime/ops_metal.py:MetalCompiler). tinygrad.engine.realize.get_runtime(device, prog)
then resolves Device[device].runtime(...) -- the REAL Device["METAL"] MTLDevice/queue -- bound to that
binary. Buffers are ordinary tinygrad.device.Buffer objects allocated on "METAL" via the real
allocator; dispatch order follows prog.arg.globals (sorted PARAM slots), not an assumed literal
data0/data1/data3 order, though it is verified to match.

Nothing here is a production file. No PACKED_WMMA_ROUTES row is added.
"""
from __future__ import annotations
import sys, time, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp/scratchpad")

import numpy as np

from tinygrad import Device, dtypes
from tinygrad.device import Buffer
from tinygrad.helpers import Target
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_runtime

import t1_generic_tc_dequant_probe as T1
import t1b_generic_tc_dequant_vectorized_probe as T1B

from extra.llm_research.prefill.packed_wmma_correctness_canary import build_artifact

M, N, K = T1.M, T1.N, T1.K
assert (M, N, K) == (512, 12288, 4096)
DEVICE = "METAL"
WIDTH = T1B.WIDTH_FOR_BACKEND[DEVICE]
SENTINEL = np.float16(12345.0)
SENTINEL_BITS = SENTINEL.view(np.uint16).item()


def _build_and_compile(ast_builder, label: str):
  """ast_builder() -> unforced AST. Forces generic TC (T1._force_generic_tc), renders + FULLY
  compiles (to_program runs do_compile too -- real MTLB binary), returns (prog, renderer)."""
  ast = ast_builder()
  ast_forced = T1._force_generic_tc(ast)
  target_str, make_renderer = T1.TARGETS[DEVICE]
  renderer = make_renderer(Target.parse(target_str))
  prog = to_program(ast_forced, renderer)
  assert prog.op is Ops.PROGRAM, f"{label}: to_program did not return a PROGRAM ({prog.op})"
  assert len(prog.src) >= 5 and prog.src[4].op is Ops.BINARY, f"{label}: PROGRAM has no compiled BINARY"
  binary = prog.src[4].arg
  assert isinstance(binary, bytes) and len(binary) > 0, f"{label}: empty compiled binary"
  source = next((u.arg for u in prog.src if u.op is Ops.SOURCE and isinstance(u.arg, str)), None)
  print(f"[{label}] compiled OK. binary_len={len(binary)} source_len={len(source) if source else None} "
        f"wmma={source.count('__WMMA') if source else None} "
        f"sgma={source.count('simdgroup_multiply_accumulate') if source else None} "
        f"globals={prog.arg.globals} global_size={prog.arg.global_size} local_size={prog.arg.local_size}")
  return prog


def _make_buffer(size: int, dtype, data_bytes: bytes) -> Buffer:
  buf = Buffer(DEVICE, size, dtype, initial_value=data_bytes)
  return buf


def _dispatch(prog, slot_to_buf: dict[int, Buffer], wait=True):
  order = list(prog.arg.globals)
  rt = get_runtime(DEVICE, prog)
  bufs = [slot_to_buf[s].get_buf(DEVICE) for s in order]
  et = rt(*bufs, global_size=prog.arg.global_size, local_size=prog.arg.local_size, vals=(), wait=wait)
  return et, order


def _readback_half(buf: Buffer, n_elems: int) -> np.ndarray:
  mv = buf.copyout(memoryview(bytearray(buf.nbytes)))
  return np.frombuffer(mv, dtype=np.float16).copy().reshape(-1)[:n_elems]


# ------------------------------------------------------------------ Part 1: correctness ------
def part1_correctness():
  print("\n===== PART 1: correctness (fused packed-Q4_K generic-TC kernel) =====")

  artifact_path = "/tmp/t4_q4k_reference.npz"
  meta = build_artifact("Q4_K", artifact_path, shape=(M, N, K))
  print(f"reference artifact: {meta}")
  npz = np.load(artifact_path)
  activation, packed_raw, reference = npz["a"], npz["b"], npz["reference"]
  print(f"activation.shape={activation.shape} dtype={activation.dtype}  "
        f"packed.shape={packed_raw.shape} dtype={packed_raw.dtype}  "
        f"reference.shape={reference.shape} dtype={reference.dtype}")

  assert activation.size == M * K
  assert reference.size == M * N
  packed_words = np.ascontiguousarray(packed_raw).reshape(-1)
  assert packed_words.dtype == np.uint32, packed_words.dtype
  print(f"packed word count = {packed_words.size} (expect 7077888)")

  def build_ast():
    return T1B._experiment_a_naive_dodge(DEVICE, WIDTH)

  prog = _build_and_compile(build_ast, "expA_q4k")

  packed_buf = _make_buffer(packed_words.size, dtypes.uint32, packed_words.tobytes())
  act_buf = _make_buffer(activation.size, dtypes.half, np.ascontiguousarray(activation).tobytes())

  # map slots -> buffers by role, not by assumed literal number: the packed source (uint dtype,
  # size 7077888) and the activation source (half dtype, size 2097152) are unambiguous by size/dtype
  # against prog.arg.ins; the remaining (unique) global slot not in ins/outs union with a STORE is
  # the output.
  slots = list(prog.arg.globals)
  ins, outs = set(prog.arg.ins), set(prog.arg.outs)
  print(f"prog.arg.globals={slots} ins={sorted(ins)} outs={sorted(outs)}")
  assert len(outs) == 1, f"expected exactly one output slot, got {outs}"
  out_slot = next(iter(outs))
  in_slots = [s for s in slots if s != out_slot]
  assert len(in_slots) == 2, f"expected exactly two input slots, got {in_slots}"

  rounds = []
  raw_rounds = []
  for r in range(3):
    sentinel_arr = np.full(M * N, SENTINEL, dtype=np.float16)
    out_buf = _make_buffer(M * N, dtypes.half, sentinel_arr.tobytes())
    # decide slot->buffer assignment for this round by dtype size match (uint32 packed vs half activation)
    slot_to_buf = {out_slot: out_buf}
    # packed buffer occupies 4 bytes/elem (uint32); activation 2 bytes/elem (half) -- assign by nbytes-per-elem
    # of the *declared* kernel param dtype, read directly off the rendered PARAM dtypes in the ast, not guessed.
    param_dtypes = {u.arg.slot: u.dtype for u in prog.src[0].toposort() if u.op is Ops.PARAM}
    for s in in_slots:
      pdt = param_dtypes[s]
      slot_to_buf[s] = packed_buf if pdt == dtypes.uint32 else act_buf
    Device[DEVICE].synchronize()
    et, order = _dispatch(prog, slot_to_buf)
    Device[DEVICE].synchronize()
    out_np = _readback_half(out_buf, M * N)
    raw_bits = np.frombuffer(out_np.tobytes(), dtype=np.uint16)
    written_mask = raw_bits != SENTINEL_BITS
    coverage = written_mask.mean()
    rounds.append({"round": r, "gpu_et_s": et, "coverage_fraction": float(coverage),
                   "written": int(written_mask.sum()), "total": int(written_mask.size)})
    raw_rounds.append(out_np.reshape(M, N).copy())
    print(f"round {r}: dispatch order(slots)={order} gpu_et={et} coverage={coverage:.6f} "
          f"({int(written_mask.sum())}/{written_mask.size})")

  ref32 = reference.astype(np.float32)
  out0_32 = raw_rounds[0].astype(np.float32)
  max_abs_error = float(np.max(np.abs(out0_32 - ref32)))
  mean_abs_error = float(np.mean(np.abs(out0_32 - ref32)))
  print(f"max_abs_error (round0 vs numpy reference) = {max_abs_error}")
  print(f"mean_abs_error (round0 vs numpy reference) = {mean_abs_error}")

  # determinism across rounds
  bit_identical = all(np.array_equal(raw_rounds[0].view(np.uint16), r.view(np.uint16)) for r in raw_rounds[1:])
  max_inter_round_diff = 0.0
  for i in range(len(raw_rounds)):
    for j in range(i + 1, len(raw_rounds)):
      d = float(np.max(np.abs(raw_rounds[i].astype(np.float32) - raw_rounds[j].astype(np.float32))))
      max_inter_round_diff = max(max_inter_round_diff, d)
  print(f"bit_identical_across_rounds = {bit_identical}  max_inter_round_diff = {max_inter_round_diff}")

  result = {"max_abs_error": max_abs_error, "mean_abs_error": mean_abs_error,
            "rounds": rounds, "bit_identical_across_rounds": bit_identical,
            "max_inter_round_diff": max_inter_round_diff,
            "n_written_round0": rounds[0]["written"], "n_total": rounds[0]["total"],
            "coverage_fraction_round0": rounds[0]["coverage_fraction"]}
  with open("/tmp/t4_part1_result.json", "w") as f:
    json.dump(result, f, indent=2)
  return result, prog


# ------------------------------------------------------------------ Part 2: performance -------
def _time_dispatch(prog, slot_to_buf, warmup, reps):
  times = []
  for i in range(warmup):
    Device[DEVICE].synchronize()
    _dispatch(prog, slot_to_buf)
    Device[DEVICE].synchronize()
  for i in range(reps):
    Device[DEVICE].synchronize()
    t0 = time.perf_counter()
    et, _ = _dispatch(prog, slot_to_buf)
    Device[DEVICE].synchronize()
    t1 = time.perf_counter()
    times.append({"rep": i, "host_wall_s": t1 - t0, "gpu_et_s": et})
  return times


def part2_performance(fused_prog):
  print("\n===== PART 2: performance (fused kernel vs fp16 GEMM ceiling) =====")
  gflop_total = 2 * M * N * K / 1e9
  print(f"2*M*N*K = {gflop_total} GFLOP per call")

  # ---- fused kernel timing: reuse the SAME compiled prog from part 1, same buffers (output sentinel
  # doesn't matter for timing; reuse a fresh out buffer once). ----
  npz = np.load("/tmp/t4_q4k_reference.npz")
  activation, packed_raw = npz["a"], npz["b"]
  packed_words = np.ascontiguousarray(packed_raw).reshape(-1)
  packed_buf = _make_buffer(packed_words.size, dtypes.uint32, packed_words.tobytes())
  act_buf = _make_buffer(activation.size, dtypes.half, np.ascontiguousarray(activation).tobytes())
  out_buf = _make_buffer(M * N, dtypes.half, np.zeros(M * N, dtype=np.float16).tobytes())
  slots = list(fused_prog.arg.globals)
  outs = set(fused_prog.arg.outs)
  out_slot = next(iter(outs))
  in_slots = [s for s in slots if s != out_slot]
  param_dtypes = {u.arg.slot: u.dtype for u in fused_prog.src[0].toposort() if u.op is Ops.PARAM}
  slot_to_buf = {out_slot: out_buf}
  for s in in_slots:
    slot_to_buf[s] = packed_buf if param_dtypes[s] == dtypes.uint32 else act_buf

  fused_times = _time_dispatch(fused_prog, slot_to_buf, warmup=3, reps=8)
  fused_gflops = [gflop_total / t["host_wall_s"] for t in fused_times]
  print("fused kernel per-rep (host wall):", [(t["rep"], round(t["host_wall_s"]*1e3,4), round(g,2))
                                               for t, g in zip(fused_times, fused_gflops)])
  print(f"fused GFLOPS: min={min(fused_gflops):.2f} max={max(fused_gflops):.2f} "
        f"mean={sum(fused_gflops)/len(fused_gflops):.2f} spread={max(fused_gflops)-min(fused_gflops):.2f}")

  # ---- fp16 dense GEMM ceiling: T1._dense_gemm_ast, generic TC, same M/N/K ----
  def build_dense_ast():
    return T1._dense_gemm_ast(DEVICE)
  dense_prog = _build_and_compile(build_dense_ast, "rung1_dense_fp16")

  rng = np.random.default_rng(0)
  a_np = rng.standard_normal((M, K), dtype=np.float32).astype(np.float16)
  b_np = rng.standard_normal((N, K), dtype=np.float32).astype(np.float16)
  a_buf = _make_buffer(M * K, dtypes.half, np.ascontiguousarray(a_np).tobytes())
  b_buf = _make_buffer(N * K, dtypes.half, np.ascontiguousarray(b_np).tobytes())
  out2_buf = _make_buffer(M * N, dtypes.half, np.zeros(M * N, dtype=np.float16).tobytes())

  dslots = list(dense_prog.arg.globals)
  douts = set(dense_prog.arg.outs)
  dout_slot = next(iter(douts))
  din_slots = [s for s in dslots if s != dout_slot]
  dparam_sizes = {u.arg.slot: u.dtype.itemsize * (u.arg.size if hasattr(u.arg, "size") else None) for u in []}
  # Distinguish A (M,K) vs B (N,K) input slots by which PARAM UOp has the shape matching the N range
  # dependency: use _find_mnk-style structural check instead of a numeric guess.
  dense_ast_unforced = T1._dense_gemm_ast(DEVICE)
  red, in0, in1, n_rng, k_rng = T1._find_mnk(dense_ast_unforced)
  a_param_slot = next(u.arg.slot for u in in0.toposort() if u.op is Ops.PARAM)
  b_param_slot = next(u.arg.slot for u in in1.toposort() if u.op is Ops.PARAM)
  assert set(din_slots) == {a_param_slot, b_param_slot}, (din_slots, a_param_slot, b_param_slot)
  d_slot_to_buf = {dout_slot: out2_buf, a_param_slot: a_buf, b_param_slot: b_buf}

  dense_times = _time_dispatch(dense_prog, d_slot_to_buf, warmup=3, reps=8)
  dense_gflops = [gflop_total / t["host_wall_s"] for t in dense_times]
  print("dense fp16 per-rep (host wall):", [(t["rep"], round(t["host_wall_s"]*1e3,4), round(g,2))
                                             for t, g in zip(dense_times, dense_gflops)])
  print(f"dense fp16 GFLOPS: min={min(dense_gflops):.2f} max={max(dense_gflops):.2f} "
        f"mean={sum(dense_gflops)/len(dense_gflops):.2f} spread={max(dense_gflops)-min(dense_gflops):.2f}")

  # correctness spot-check for the ceiling GEMM (harness sanity, not the gating check)
  out2_np = _readback_half(out2_buf, M * N).reshape(M, N).astype(np.float32)
  ref_dense = (a_np.astype(np.float32) @ b_np.astype(np.float32).T)
  dense_max_err = float(np.max(np.abs(out2_np - ref_dense)))
  print(f"[sanity] dense fp16 GEMM max_abs_error vs numpy fp32 reference = {dense_max_err}")

  ratio_mean = (sum(fused_gflops)/len(fused_gflops)) / (sum(dense_gflops)/len(dense_gflops))
  ratio_best = max(fused_gflops) / max(dense_gflops)
  print(f"\nfused/ceiling ratio: mean={ratio_mean:.4f}  best-vs-best={ratio_best:.4f}")

  result = {"gflop_per_call": gflop_total,
            "fused": {"times": fused_times, "gflops": fused_gflops},
            "dense_fp16_ceiling": {"times": dense_times, "gflops": dense_gflops, "sanity_max_abs_error": dense_max_err},
            "ratio_mean": ratio_mean, "ratio_best_vs_best": ratio_best}
  with open("/tmp/t4_part2_result.json", "w") as f:
    json.dump(result, f, indent=2)
  return result


def main():
  p1, fused_prog = part1_correctness()
  print("\n" + json.dumps(p1, indent=2))
  gate_ok = (p1["max_abs_error"] < 1e-2 and p1["coverage_fraction_round0"] > 0.999 and p1["bit_identical_across_rounds"])
  print(f"\nPART 1 GATE: {'PASS' if gate_ok else 'FAIL'}")
  if not gate_ok:
    print("STOPPING: Part 1 did not pass. Not proceeding to Part 2 performance measurement.")
    return
  p2 = part2_performance(fused_prog)
  print("\n" + json.dumps({k: v for k, v in p2.items() if k not in ("fused", "dense_fp16_ceiling")}, indent=2))


if __name__ == "__main__":
  main()
