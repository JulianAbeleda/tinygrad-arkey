#!/usr/bin/env python3
"""Synchronized research benchmark for the direct-UOp Q4_K x Q8_1 emitters.

This file deliberately bypasses route selection.  Every measured contraction
uses the same prepacked Q4_K weights and row-major Q8_1 values/scales.
"""
from __future__ import annotations

import argparse, json, platform, statistics, time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_SHAPES = ((16, 32, 256), (16, 32, 512), (16, 32, 5120))
CANONICAL_14B_ROLE_SHAPES = {
  "attn_kv": (512, 1024, 5120), "attn_qo": (512, 5120, 5120),
  "ffn_down": (512, 5120, 17408), "ffn_gate_up": (512, 17408, 5120),
}
SEED = 20260715
REL_RMSE_THRESHOLD = 6e-3
STRICT_RTOL, STRICT_ATOL = 3e-4, 3e-4
ORACLE_M, ORACLE_N = 8, 8


@dataclass(frozen=True)
class Shape:
  m: int
  n: int
  k: int

  def validate(self) -> None:
    if self.m <= 0 or self.n <= 0 or self.k <= 0: raise ValueError("shape dimensions must be positive")
    if self.m % 16 or self.n % 16 or self.k % 256: raise ValueError("benchmark shapes require M/N multiples of 16 and K multiple of 256")

  @property
  def logical_ops(self) -> int: return 2 * self.m * self.n * self.k

  def text(self) -> str: return f"{self.m}x{self.n}x{self.k}"


def parse_shape(value:str) -> Shape:
  try: shape = Shape(*(int(x) for x in value.lower().split("x")))
  except (TypeError, ValueError): raise argparse.ArgumentTypeError("shape must be MxNxK") from None
  try: shape.validate()
  except ValueError as exc: raise argparse.ArgumentTypeError(str(exc)) from None
  return shape


def deterministic_fixture(shape:Shape, seed:int=SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Canonical finite packed Q4_K and llama-reference-quantized Q8_1 operands."""
  words, xq, xscale, _, _ = prepare_fixture(shape, seed)
  return words, xq, xscale


def prepare_fixture(shape:Shape, seed:int=SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
  """Produce one shared row-major Q8 payload plus llama's original-fp DS4 sums.

  The fourth result cannot be reconstructed from the second and third: it is
  the pre-quantization group sum returned by the canonical llama DS4 reference.
  """
  shape.validate()
  rng = np.random.default_rng(seed + shape.m * 1000003 + shape.n * 1009 + shape.k)
  raw = rng.integers(0, 256, size=(shape.n, shape.k//256, 144), dtype=np.uint8)
  raw[..., :4] = np.frombuffer(np.array([0.03125, 0.0078125], dtype="<f2").tobytes(), dtype=np.uint8)
  source = rng.standard_normal((shape.m, shape.k), dtype=np.float32)
  from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference
  started = time.perf_counter()
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  wall_ms = (time.perf_counter() - started) * 1e3
  xq = values.reshape(shape.k//128, shape.m, 4, 32).transpose(1, 0, 2, 3).reshape(shape.m, shape.k)
  xscale = scales.transpose(1, 0, 2).reshape(shape.m, shape.k//32)
  original_sums = sums.transpose(1, 0, 2).reshape(shape.m, shape.k//32)
  validate_original_fp_sum_payload(xq, xscale, original_sums)
  prep = {"producer":"extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
          "sum_semantics":"llama_ds4_y_original_fp32_group_sum", "wall_ms":wall_ms,
          "device_ms":None, "program_count":0, "kernel_count":0}
  return raw.reshape(-1).view(np.uint32), xq, xscale, original_sums, prep


def summarize_samples(wall_s:Sequence[float], device_s:Sequence[float|None], kernels:Sequence[int], logical_ops:int) -> dict:
  if not wall_s or len(wall_s) != len(device_s) or len(wall_s) != len(kernels): raise ValueError("inconsistent or empty samples")
  wall_med = statistics.median(wall_s)
  trusted = all(x is not None and x > 0 for x in device_s) and all(x == 1 for x in kernels)
  dev_med = statistics.median(x for x in device_s if x is not None) if trusted else None
  return {
    "rounds": len(wall_s), "wall_ms": [x*1e3 for x in wall_s],
    "wall_median_ms": wall_med*1e3, "device_ms": [None if x is None else x*1e3 for x in device_s],
    "device_median_ms": None if dev_med is None else dev_med*1e3,
    "device_time_trustworthy": trusted, "kernel_counts": list(kernels),
    "logical_ops": logical_ops, "logical_wall_tops": logical_ops/wall_med/1e12,
    "logical_wall_tflops": logical_ops/wall_med/1e12,
    "logical_device_tops": None if dev_med is None else logical_ops/dev_med/1e12,
    "logical_device_tflops": None if dev_med is None else logical_ops/dev_med/1e12,
  }


def relative_performance(candidate:dict, baseline:dict) -> dict:
  """Positive percent means the explicit wide candidate is faster."""
  ctime, btime = candidate.get("device_median_ms"), baseline.get("device_median_ms")
  if ctime is None or btime is None or ctime <= 0 or btime <= 0:
    return {"device_speedup_x":None, "device_change_percent":None, "classification":"unavailable"}
  ratio = btime / ctime
  return {"device_speedup_x":ratio, "device_change_percent":(ratio-1)*100,
          "classification":"speedup" if ratio > 1 else "slowdown" if ratio < 1 else "tie"}


def relative_rmse(got:np.ndarray, reference:np.ndarray) -> float:
  got_f, ref_f = np.asarray(got, dtype=np.float32), np.asarray(reference, dtype=np.float32)
  if got_f.shape != ref_f.shape or got_f.size == 0: raise ValueError("relative RMSE requires equal nonempty shapes")
  return float(np.sqrt(np.mean((got_f-ref_f)**2)) / (np.sqrt(np.mean(ref_f**2)) + 1e-12))


def validate_original_fp_sum_payload(xq:np.ndarray, xscale:np.ndarray, original_sums:np.ndarray) -> dict:
  values, scales, sums = np.asarray(xq), np.asarray(xscale), np.asarray(original_sums)
  if values.ndim != 2 or scales.shape != sums.shape or scales.shape != (values.shape[0], values.shape[1]//32):
    raise ValueError("original-fp sum payload shapes do not match row-major Q8 values/scales")
  if not np.isfinite(scales).all() or not np.isfinite(sums).all(): raise RuntimeError("original-fp sum payload is non-finite")
  dequant = scales * values.reshape(values.shape[0], values.shape[1]//32, 32).astype(np.float32).sum(axis=2)
  if np.array_equal(sums, dequant):
    raise RuntimeError("sum payload has dequantized-Q8 semantics; llama original-fp sums required")
  return {"source":"canonical_llama_ds4_reference", "semantic_family":"llama_original_fp_sum",
          "not_dequantized_q8_sum":True, "max_abs_vs_dequantized_q8_sum":float(np.max(np.abs(sums-dequant)))}


def bounded_original_fp_ds4_oracle(shape:Shape, words:np.ndarray, xq:np.ndarray, xscale:np.ndarray,
                                   original_sums:np.ndarray, *, tile_m:int=ORACLE_M, tile_n:int=ORACLE_N) -> np.ndarray:
  """Independent NumPy DS4 authority for a bounded M/N tile spanning full K."""
  from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q8_1_MMQ_DS4_LAYOUT,
    describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference)
  tm, tn = min(tile_m, shape.m), min(tile_n, shape.n)
  q4 = np.asarray(words, dtype=np.uint32).reshape(shape.n, shape.k//256, 36)[:tn].reshape(-1)
  values = np.asarray(xq, dtype=np.int8)[:tm].reshape(tm, shape.k//128, 128).transpose(1, 0, 2).copy()
  scales = np.asarray(xscale, dtype=np.float32)[:tm].reshape(tm, shape.k//128, 4).transpose(1, 0, 2).copy()
  sums = np.asarray(original_sums, dtype=np.float32)[:tm].reshape(tm, shape.k//128, 4).transpose(1, 0, 2).copy()
  ds4 = Q81MMQDS4Activation(values, scales, sums, Q81MMQDS4ActivationSpec(m=tm, k=shape.k, m_tile=tm))
  spec = describe_q4k_q8_1_mmq_tile(role="uop_bench_original_fp_oracle", m=tm, n=tn, k=shape.k,
    m_tile=tm, n_tile=tn, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  return q4k_q8_1_mmq_ds4_tile_reference(q4.view(np.uint8), ds4, spec)


def original_fp_subset_authority(got:np.ndarray, oracle:np.ndarray, *, full_k:int) -> dict:
  tile = np.asarray(got)[:oracle.shape[0], :oracle.shape[1]]
  finite = bool(np.isfinite(tile).all() and np.isfinite(oracle).all())
  rel = relative_rmse(tile, oracle) if finite else float("inf")
  return {"reference":"independent_canonical_llama_ds4_numpy_subset", "semantic_family":"llama_original_fp_sum",
          "shape":list(oracle.shape), "full_k":full_k, "finite":finite,
          "nan_count":int(np.isnan(tile).sum()+np.isnan(oracle).sum()), "rel_rmse":rel,
          "rel_rmse_threshold":REL_RMSE_THRESHOLD, "rel_rmse_pass":bool(finite and rel < REL_RMSE_THRESHOLD),
          "max_abs":float(np.max(np.abs(tile-oracle))) if finite else None}


def _numeric_correctness(shape:Shape, outputs:dict[str,np.ndarray], words:np.ndarray, xq:np.ndarray,
                         xscale:np.ndarray, original_sums:np.ndarray) -> dict:
  scalar = outputs["scalar_direct_uop"]
  families = {"row_major_dequantized_q8_sum":{"members":[], "authority":"scalar_direct_uop_full_output"},
              "llama_original_fp_sum":{"members":["sum_original_fp_wmma"],
                                        "authority":"independent_canonical_llama_ds4_numpy_subset_full_k"}}
  rows = {}
  for name in ("scalar_direct_uop", "wmma_uop", "wide_wmma_uop"):
    if name not in outputs: continue
    value = outputs[name]; finite = bool(np.isfinite(value).all() and np.isfinite(scalar).all())
    close = np.isclose(value, scalar, rtol=STRICT_RTOL, atol=STRICT_ATOL)
    rel = relative_rmse(value, scalar) if finite else float("inf")
    rows[name] = {"semantic_family":"row_major_dequantized_q8_sum", "finite":finite,
      "nan_count":int(np.isnan(value).sum()+np.isnan(scalar).sum()), "rel_rmse_vs_scalar":rel,
      "rel_rmse_threshold":REL_RMSE_THRESHOLD, "rel_rmse_pass":bool(finite and rel < REL_RMSE_THRESHOLD),
      "strict_allclose_diagnostic":bool(close.all()), "strict_rtol":STRICT_RTOL, "strict_atol":STRICT_ATOL,
      "strict_mismatch_count":int((~close).sum()), "max_abs_vs_scalar":float(np.max(np.abs(value-scalar)))}
    families["row_major_dequantized_q8_sum"]["members"].append(name)
  oracle = bounded_original_fp_ds4_oracle(shape, words, xq, xscale, original_sums)
  rows["sum_original_fp_wmma"] = original_fp_subset_authority(outputs["sum_original_fp_wmma"], oracle, full_k=shape.k)
  rows["sum_original_fp_wmma"]["sum_source"] = "canonical_llama_ds4_reference"
  rows["sum_original_fp_wmma"]["payload_semantics"] = validate_original_fp_sum_payload(xq, xscale, original_sums)
  if "custom_ds4_same_algebra" in outputs:
    exact = bool(np.allclose(outputs["sum_original_fp_wmma"], outputs["custom_ds4_same_algebra"], rtol=3e-4, atol=3e-3))
    rows["sum_original_fp_wmma"]["bounded_ds4_exact_allclose"] = exact
    rows["custom_ds4_same_algebra"] = {"semantic_family":"llama_original_fp_sum", "bounded_exact_authority":exact}
    families["llama_original_fp_sum"]["members"].append("custom_ds4_same_algebra")
  return {"semantic_families":families, "paths":rows}


def _gate_correctness(numeric:dict) -> None:
  rows = numeric["paths"]
  for name in numeric["semantic_families"]["row_major_dequantized_q8_sum"]["members"]:
    if not rows[name]["finite"] or rows[name]["nan_count"] or not rows[name]["rel_rmse_pass"]:
      raise RuntimeError(f"row-major relative-RMSE correctness failed for {name}: {rows[name]}")
  original = rows["sum_original_fp_wmma"]
  if (not original["finite"] or original["nan_count"] or not original["rel_rmse_pass"] or
      original["payload_semantics"].get("not_dequantized_q8_sum") is not True):
    raise RuntimeError(f"original-fp DS4 subset correctness failed: {original}")
  if original.get("bounded_ds4_exact_allclose") is False:
    raise RuntimeError(f"bounded original-fp DS4 exact correctness failed: {original}")


def wide_execution_contract(by_name:dict[str, dict]) -> dict[str, dict[str, bool]]:
  contract = {name:{"one_program":row["compile"]["program_count"] == 1,
                    "one_launch_every_round":all(k == 1 for k in row["kernel_counts"])} for name,row in by_name.items()}
  if "wide_wmma_uop" in by_name:
    contract["wide_wmma_uop"].update(signed_i8_wmma=by_name["wide_wmma_uop"]["compile"]["signed_i8_wmma_programs"] == 1,
                                      exact_one_workgroup=by_name["wide_wmma_uop"]["compile"]["global_sizes"] == [[1,1,1]])
  contract["wmma_uop"].update(signed_i8_wmma=by_name["wmma_uop"]["compile"]["signed_i8_wmma_programs"] == 1,
                               valid_workgroups=all(x > 0 for x in by_name["wmma_uop"]["compile"]["global_sizes"][0]))
  if "wide_wmma_uop" in by_name:
    contract["wmma_uop"]["exact_two_workgroups"] = by_name["wmma_uop"]["compile"]["global_sizes"] == [[2,1,1]]
  return contract


def _compile_path(name:str, shape:Shape, emitter:Callable, *operands):
  from tinygrad import Tensor, dtypes
  from tinygrad.engine.realize import compile_linear
  out_storage = Tensor.empty(shape.m, shape.n, dtype=dtypes.float32, device=operands[0].device).realize()
  out = out_storage.custom_kernel(*operands, fxn=emitter)[0]
  started = time.perf_counter()
  compiled = compile_linear(out.schedule_linear())
  return {"name": name, "out": out, "linear": compiled, "compile_ms": (time.perf_counter()-started)*1e3,
          "compile_facts": _compile_facts(compiled)}


def _compile_facts(compiled) -> dict:
  from tinygrad.uop.ops import Ops
  from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
  programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
  sources = [str(next((x.arg for x in p.src if x.op is Ops.SOURCE), "")) for p in programs]
  resources = []
  for program in programs:
    binary = next((x.arg for x in program.src if x.op is Ops.BINARY), None)
    resources.append(None if binary is None else parse_amdgpu_metadata(binary))
  return {"program_count":len(programs), "kernel_names":[getattr(p.arg,"name",None) for p in programs],
          "signed_i8_wmma_programs":sum("wmma_i32_16x16x16_iu8" in source for source in sources),
          "global_sizes":[list(p.arg.global_size) for p in programs], "local_sizes":[list(p.arg.local_size) for p in programs],
          "resources":resources}


def _compile_ds4(shape:Shape, words, xq_np:np.ndarray, xs_np:np.ndarray, sums_np:np.ndarray):
  if (shape.m, shape.n, shape.k) != (16, 16, 256): return None
  from tinygrad import Tensor
  from extra.qk.mmq_ds4_logical_emitter import packed_ds4_candidate, emit_q4k_q8_mmq_ds4
  candidate = packed_ds4_candidate(shape.m, shape.n, shape.k, role="attn_kv")
  vals = xq_np.reshape(shape.m, shape.k//128, 4, 32).transpose(1, 0, 2, 3).reshape(-1)
  scales = xs_np.reshape(shape.m, shape.k//128, 4).transpose(1, 0, 2).reshape(-1)
  sums = sums_np.reshape(shape.m, shape.k//128, 4).transpose(1, 0, 2).reshape(-1)
  qv, qs, qsum = (Tensor(x, device=words.device).realize() for x in (vals, scales, sums))
  out = emit_q4k_q8_mmq_ds4(words, qv, qs, qsum, candidate)
  from tinygrad.engine.realize import compile_linear
  started = time.perf_counter(); linear = compile_linear(out.schedule_linear())
  return {"name": "custom_ds4_same_algebra", "out": out, "linear": linear,
          "compile_ms": (time.perf_counter()-started)*1e3, "compile_facts":_compile_facts(linear)}


def _measure(path:dict, shape:Shape, warmups:int, rounds:int) -> dict:
  from tinygrad import Device
  from tinygrad.engine.realize import run_linear
  from tinygrad.helpers import GlobalCounters
  dev = Device[path["out"].device]
  for _ in range(warmups): run_linear(path["linear"], wait=True)
  dev.synchronize()
  wall, device, kernels = [], [], []
  for _ in range(rounds):
    dev.synchronize(); before_t, before_k = GlobalCounters.time_sum_s, GlobalCounters.kernel_count
    start = time.perf_counter(); run_linear(path["linear"], wait=True); dev.synchronize(); wall.append(time.perf_counter()-start)
    dt, dk = GlobalCounters.time_sum_s-before_t, GlobalCounters.kernel_count-before_k
    device.append(dt if dt > 0 else None); kernels.append(dk)
  result = summarize_samples(wall, device, kernels, shape.logical_ops)
  result.update(name=path["name"], compile_ms=path["compile_ms"], compile=path["compile_facts"])
  return result


def benchmark_shape(shape:Shape, warmups:int=3, rounds:int=9) -> dict:
  from tinygrad import Device, Tensor
  from tinygrad.engine.realize import run_linear
  from extra.qk.q4k_q8_mmq_uop import (describe_q4k_q8_mmq_uop, describe_q4k_q8_mmq_wmma,
    describe_q4k_q8_mmq_wide_wmma, describe_q4k_q8_mmq_sum_original_fp_wmma, emit_q4k_q8_mmq_uop,
    emit_q4k_q8_mmq_wmma, emit_q4k_q8_mmq_wide_wmma, emit_q4k_q8_mmq_sum_original_fp_wmma)
  shape.validate(); words_np, xq_np, xs_np, sums_np, sum_prep = prepare_fixture(shape)
  words, xq, xs, sums = (Tensor(x.reshape(-1), device="AMD").realize() for x in (words_np, xq_np, xs_np, sums_np))
  Device["AMD"].synchronize()
  paths = [
    _compile_path("wmma_uop", shape, emit_q4k_q8_mmq_wmma(describe_q4k_q8_mmq_wmma(m=shape.m,n=shape.n,k=shape.k)), words, xq, xs),
    _compile_path("scalar_direct_uop", shape, emit_q4k_q8_mmq_uop(describe_q4k_q8_mmq_uop(shape.m,shape.n,shape.k)), words, xq, xs),
    _compile_path("sum_original_fp_wmma", shape, emit_q4k_q8_mmq_sum_original_fp_wmma(
      describe_q4k_q8_mmq_sum_original_fp_wmma(shape.m,shape.n,shape.k)), words, xq, xs, sums),
  ]
  if shape.m == 16 and shape.n == 32:
    paths.insert(0, _compile_path("wide_wmma_uop", shape, emit_q4k_q8_mmq_wide_wmma(
      describe_q4k_q8_mmq_wide_wmma(m=shape.m,n=shape.n,k=shape.k)), words, xq, xs))
  if (ds4 := _compile_ds4(shape, words, xq_np, xs_np, sums_np)) is not None: paths.append(ds4)
  # Correctness is a prerequisite, not a post-timing observation. Execute each compiled path once before reading
  # its realized output storage; compiling a custom kernel does not itself populate that storage.
  for path in paths: run_linear(path["linear"], wait=True)
  Device["AMD"].synchronize()
  outputs = {path["name"]: path["out"].numpy() for path in paths}
  numeric = _numeric_correctness(shape, outputs, words_np, xq_np, xs_np, sums_np)
  _gate_correctness(numeric)
  measured = [_measure(path, shape, warmups, rounds) for path in paths]
  by_name = {row["name"]:row for row in measured}
  contract = wide_execution_contract(by_name)
  if not all(all(checks.values()) for checks in contract.values()): raise RuntimeError(f"program/launch contract failed: {contract}")
  comparison = None
  if "wide_wmma_uop" in by_name:
    comparison = relative_performance(by_name["wide_wmma_uop"], by_name["wmma_uop"])
    comparison.update(baseline="wmma_uop", baseline_workgroups=by_name["wmma_uop"]["compile"]["global_sizes"][0][0],
                      candidate="wide_wmma_uop", candidate_workgroups=by_name["wide_wmma_uop"]["compile"]["global_sizes"][0][0])
  original_comparison = relative_performance(by_name["sum_original_fp_wmma"], by_name["wmma_uop"])
  original_comparison.update(baseline="wmma_uop", candidate="sum_original_fp_wmma",
    shared_payload="identical packed Q4 and row-major Q8 values/scales; only sum semantic differs")
  contraction = {row["name"]:{"wall_median_ms":row["wall_median_ms"], "device_median_ms":row["device_median_ms"],
    "program_count":row["compile"]["program_count"], "kernel_counts":row["kernel_counts"]} for row in measured}
  final_wmma = {row["name"]:{"signed_i8_wmma_programs":row["compile"]["signed_i8_wmma_programs"],
    "kernel_names":row["compile"]["kernel_names"]} for row in measured}
  return {"shape": {"m":shape.m,"n":shape.n,"k":shape.k}, "warmups":warmups, "results":measured, "numeric":numeric,
          "execution_contract":contract, "wide_vs_two_workgroup_default":comparison,
          "sum_original_fp_vs_current_qsum_wmma":original_comparison,
          "contraction_only":contraction, "final_wmma_evidence":final_wmma, "sum_preparation":sum_prep,
          "comparators": {"scalar_direct_uop":"same packed Q4_K/Q8_1 operands", "custom_ds4_same_algebra":
                          "available only at its proven bounded 16x16x256 shape",
                          "sum_original_fp_wmma":"same Q4/Q8 values/scales; canonical llama original-fp sums"}}


def main(argv:Sequence[str]|None=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shape", action="append", type=parse_shape, help="repeatable MxNxK (defaults to aligned research set)")
  parser.add_argument("--role", action="append", choices=tuple(CANONICAL_14B_ROLE_SHAPES),
                      help="canonical Qwen3-14B role shape; run one role per process for isolation")
  parser.add_argument("--warmups", type=int, default=3); parser.add_argument("--rounds", type=int, default=9)
  args = parser.parse_args(argv)
  if args.warmups < 0 or args.rounds <= 0: parser.error("warmups must be nonnegative and rounds positive")
  if args.shape and args.role: parser.error("--shape and --role are mutually exclusive")
  shapes = ([Shape(*CANONICAL_14B_ROLE_SHAPES[x]) for x in args.role] if args.role else
            args.shape or [Shape(*x) for x in DEFAULT_SHAPES])
  report = {"schema":"q4k_q8_mmq_uop_bench.v4", "research_only":True, "route_changes":False,
            "wide_scope":"explicit exact-shape 16x32 research candidate only; no role or route application",
            "activation_semantics":"shared packed Q8 values/scales; row-major dequantized-sum and llama original-fp-sum families gated separately",
            "device":"AMD", "platform":platform.platform(), "shapes":[]}
  for shape in shapes:
    print(f"benchmarking {shape.text()}...", flush=True); report["shapes"].append(benchmark_shape(shape,args.warmups,args.rounds))
  print(json.dumps(report, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
