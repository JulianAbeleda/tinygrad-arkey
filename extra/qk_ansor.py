#!/usr/bin/env python3
from __future__ import annotations

import argparse, collections, copy, json, os, pathlib, re, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import prod

from extra.q4_k_safety import assert_q4k_native_sweep_allowed
from extra.qk_layout import (
  GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES,
  Q6_K_BLOCK_ELEMS, format_name, model_shape_targets, packed_byte_range, q6_k_weight_bytes, quant_weight_bytes,
  read_metadata, role_from_name, tensor_shape,
)

GENERATOR_VERSION = 1
SUPPORTED_GENERATOR_VERSIONS = (0, 1)
RX7900XTX_MEM_GBS = 960.0
RX7900XTX_FP32_TFLOPS = 61.4
RX7900XTX_FP32_RIDGE_OPS_PER_BYTE = RX7900XTX_FP32_TFLOPS * 1e12 / (RX7900XTX_MEM_GBS * 1e9)
RUNTIME_SUPPORTED_FAMILIES = {GGML_Q4_K: "q4_k_packed_u32", GGML_Q6_K: "q6_k_packed_u16"}

Q4_SUMMARY_RE = re.compile(
  r"^(?P<tensor>\S+) (?P<shape>\S+) (?P<name>\S+): (?P<ms>[0-9.]+) ms .*?"
  r"q4_eff=(?P<wall_gbs>[0-9.]+) GB/s device_q4_eff=(?P<device_gbs>[0-9.]+ GB/s|n/a) "
  r"kernels=(?P<kernels>[0-9.]+)",
  re.MULTILINE,
)
Q4_GEMV_RE = re.compile(r"^primitive_gemv_correctness: PASS \S+ max_abs=([0-9.eE+-]+)", re.MULTILINE)
Q4_UNPACK_RE = re.compile(r"^primitive_unpack_correctness: PASS \S+ .* max_abs=([0-9.eE+-]+)", re.MULTILINE)
Q4_Q8_BENCH_RE = re.compile(
  r"^(?P<name>q4k_q8_1_(?:gemv|intdot|vdot|vdot_parallel)_partial): wall=(?P<wall_ms>[0-9.]+) ms \((?P<wall_gbs>[0-9.]+) Q4-GB/s\), "
  r"device=(?P<device_ms>[0-9.]+ ms \((?P<device_gbs>[0-9.]+) Q4-GB/s\)|n/a), kernels=(?P<kernels>[0-9.]+)",
  re.MULTILINE,
)
Q4_Q8_GEMV_RE = re.compile(r"^correctness: max_abs=([0-9.eE+-]+)", re.MULTILINE)
Q4_Q8_UNPACK_RE = re.compile(r"^unpack_correctness: .* max_abs=([0-9.eE+-]+)", re.MULTILINE)
Q4_Q8_PACK_RE = re.compile(r"^q8_1_pack_correctness: .* activation_max_abs=([0-9.eE+-]+)", re.MULTILINE)

Q6_HEADER_RE = re.compile(r"^tensor=(?P<tensor>\S+) full_shape=\((?P<shape>[^)]*)\).*?quant_bytes=(?P<bytes>\d+)", re.MULTILINE)
Q6_BENCH_RE = re.compile(
  r"^(?P<name>q6k_(?:fused_graph|gemv_primitive_partial)): .*?device=(?P<ms>[0-9.]+) ms "
  r"\((?P<gbs>[0-9.]+) quant-GB/s\)",
  re.MULTILINE,
)
Q6_GEMV_RE = re.compile(r"^correctness: max_abs=([0-9.eE+-]+)", re.MULTILINE)
Q6_UNPACK_RE = re.compile(r"^unpack_correctness: .* max_abs=([0-9.eE+-]+)", re.MULTILINE)

@dataclass(frozen=True)
class QuantGemvDescriptor:
  model: str
  tensor: str
  role: str
  ggml_type: int
  format: str
  rows: int
  cols: int
  block_elems: int
  block_bytes: int
  data_start: int
  tensor_offset: int
  byte_start: int
  packed_bytes: int
  dtype_activation: str
  dtype_output: str
  device: str
  arch: str|None

@dataclass(frozen=True)
class CandidateSpec:
  name: str
  family: str
  activation: str
  reduction: str
  parts: int
  opts: tuple[str, ...]
  requires: tuple[str, ...]

def estimate_candidate(desc:QuantGemvDescriptor, cand:CandidateSpec) -> dict:
  logical_ops = 2 * desc.rows * desc.cols
  activation_read_bytes = desc.cols * (2 if cand.activation == "fp16" else 1)
  q8_blocks = desc.cols // 32
  q8_scale_bytes = q8_blocks * 4
  q8_stage_bytes = 0
  if cand.activation.startswith("q8_1"):
    # Read fp16 activation, write/read int8 activation and fp32 q8_1 scales. Biased vdot also writes/reads packed uint32 lanes.
    q8_stage_bytes = desc.cols * 2 + desc.cols + q8_scale_bytes + desc.cols + q8_scale_bytes
    if "biased" in cand.activation: q8_stage_bytes += desc.cols
  partial_bytes = 0 if cand.parts <= 1 else desc.rows * cand.parts * 4 * 2
  output_bytes = desc.rows * 4
  min_global_bytes = desc.packed_bytes + activation_read_bytes + q8_stage_bytes + partial_bytes + output_bytes
  ops_per_min_byte = logical_ops / max(1, min_global_bytes)
  ops_per_quant_byte = logical_ops / max(1, desc.packed_bytes)
  packed_dot = "vdot" in cand.family or "amd_v_dot4_u32_u8" in cand.requires
  q8_staging = cand.activation.startswith("q8_1")
  generated_schedule = cand.family not in ("fused_graph", "q4_k_packed_u32", "q6_k_packed_u16")
  isolated_packed_dot = packed_dot and q8_staging and "semantic_layout_schedule_package" not in cand.requires
  stop_reason = None
  if isolated_packed_dot and ops_per_min_byte < RX7900XTX_FP32_RIDGE_OPS_PER_BYTE:
    stop_reason = ("isolated_packed_dot_below_compute_ridge: v1 roofline says this shape is memory/schedule-bound; "
                   "run only as an explicit experiment, not as the next default compiler task")
  return {
    "logical_ops": logical_ops,
    "packed_weight_bytes": desc.packed_bytes,
    "activation_read_bytes": activation_read_bytes,
    "q8_stage_bytes": q8_stage_bytes,
    "partial_reduction_bytes": partial_bytes,
    "output_bytes": output_bytes,
    "min_global_bytes": min_global_bytes,
    "ops_per_quant_byte": round(ops_per_quant_byte, 3),
    "ops_per_min_byte": round(ops_per_min_byte, 3),
    "fp32_ridge_ops_per_byte": round(RX7900XTX_FP32_RIDGE_OPS_PER_BYTE, 3),
    "peak_mem_gbs": RX7900XTX_MEM_GBS,
    "features": {
      "packed_weight": cand.family != "fused_graph",
      "q8_staging": q8_staging,
      "packed_dot": packed_dot,
      "generated_schedule": generated_schedule,
      "isolated_packed_dot": isolated_packed_dot,
    },
    "stop_reason": stop_reason,
  }

def _git_commit(repo:pathlib.Path) -> str:
  try:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True).strip()
  except Exception:
    return "unknown"

def _device_arch(device:str) -> str|None:
  try:
    from tinygrad import Device
    target = getattr(Device[device].renderer, "target", None)
    if target is None: return None
    return getattr(target, "arch", None) or str(target)
  except Exception:
    return None

def descriptor_from_info(model:pathlib.Path, meta:GGUFMetadata, info:GGUFInfo, device:str="AMD", arch:str|None=None) -> QuantGemvDescriptor:
  if info.typ not in (GGML_Q4_K, GGML_Q6_K): raise ValueError(f"{info.name} has unsupported ggml_type={info.typ}")
  shape = tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, cols = shape
  block_elems = Q4_K_BLOCK_ELEMS if info.typ == GGML_Q4_K else Q6_K_BLOCK_ELEMS
  block_bytes = Q4_K_BLOCK_BYTES if info.typ == GGML_Q4_K else Q6_K_BLOCK_BYTES
  if cols % block_elems != 0: raise ValueError(f"{info.name} K={cols} is not {format_name(info.typ)} block aligned")
  byte_start, packed_bytes = packed_byte_range(meta, info)
  return QuantGemvDescriptor(
    model=str(model.expanduser()), tensor=info.name, role=role_from_name(info.name), ggml_type=info.typ, format=format_name(info.typ),
    rows=rows, cols=cols, block_elems=block_elems, block_bytes=block_bytes, data_start=meta.data_start, tensor_offset=info.off,
    byte_start=byte_start, packed_bytes=packed_bytes, dtype_activation="fp16", dtype_output="fp32", device=device,
    arch=arch if arch is not None else _device_arch(device),
  )

def descriptor_to_json(desc:QuantGemvDescriptor) -> dict:
  return asdict(desc)

def descriptor_from_json(obj:dict) -> QuantGemvDescriptor:
  return QuantGemvDescriptor(**obj)

def candidate_to_json(cand:CandidateSpec) -> dict:
  d = asdict(cand)
  d["opts"] = list(cand.opts)
  d["requires"] = list(cand.requires)
  return d

def candidate_from_json(obj:dict) -> CandidateSpec:
  return CandidateSpec(obj["name"], obj["family"], obj["activation"], obj["reduction"], obj["parts"],
                       tuple(obj["opts"]), tuple(obj["requires"]))

def _require_alignment(desc:QuantGemvDescriptor, itemsize:int, family:str) -> None:
  if desc.byte_start % itemsize != 0 or desc.packed_bytes % itemsize != 0:
    raise ValueError(f"{family} requires uint{itemsize*8}-aligned packed storage for {desc.tensor}: "
                     f"byte_start={desc.byte_start} packed_bytes={desc.packed_bytes}")

def _q4_v1_default(desc:QuantGemvDescriptor) -> tuple[int, tuple[str, ...]]:
  # Shape-derived current v1 default: tall output/attention rows use direct reduction; ffn_down split-K keeps occupancy up.
  if desc.cols > desc.rows: return 4, ("LOCAL:0:32",)
  return 1, ("LOCAL:0:64",)

def generate_candidates(desc:QuantGemvDescriptor, level:int=0) -> list[CandidateSpec]:
  if level < 0: raise ValueError("--level must be >= 0")
  candidates = [CandidateSpec("fused_graph", "fused_graph", desc.dtype_activation, "generic_fused_reduce", 0, (), ("ggml_data_to_tensor",))]
  if desc.ggml_type == GGML_Q4_K:
    _require_alignment(desc, 4, "v1_q4_packed")
    parts, opts = _q4_v1_default(desc)
    candidates.append(CandidateSpec("v1_q4_packed", "q4_k_packed_u32", desc.dtype_activation, "split_k_partial", parts, opts,
                                    ("q4k_gemv_partial_kernel", "u32_packed_storage")))
    if level >= 1:
      for name, parts, opts in (
        ("q4_local16_p1", 1, ("LOCAL:0:16",)), ("q4_local32_p1", 1, ("LOCAL:0:32",)),
        ("q4_local64_p1", 1, ("LOCAL:0:64",)), ("q4_local32_p2", 2, ("LOCAL:0:32",)),
        ("q4_local32_p4", 4, ("LOCAL:0:32",)),
      ):
        candidates.append(CandidateSpec(name, "q4_k_packed_u32", desc.dtype_activation, "split_k_partial", parts, opts,
                                        ("q4k_gemv_partial_kernel", "u32_packed_storage")))
    if level >= 2:
      parts, opts = _q4_v1_default(desc)
      candidates.append(CandidateSpec("q8_1_q4_packed", "q4_k_q8_1_packed_u32", "q8_1", "split_k_partial", parts, opts,
                                      ("q8_1_pack", "q4k_q8_1_gemv_partial_kernel", "u32_packed_storage")))
      candidates.append(CandidateSpec("q8_1_q4_intdot", "q4_k_q8_1_intdot_u32", "q8_1", "split_k_intdot_partial", parts, opts,
                                      ("q8_1_pack", "q4k_q8_1_intdot_partial_kernel", "u32_packed_storage")))
      candidates.append(CandidateSpec("q8_1_q4_vdot", "q4_k_q8_1_vdot_u32", "q8_1_biased_u8", "direct_vdot_partial", 1, (),
                                      ("q8_1_pack", "q8_1_bias_u32", "amd_v_dot4_u32_u8", "q4k_q8_1_vdot_partial_kernel",
                                       "u32_packed_storage")))
      for name, vdot_parts, vdot_opts in (
        ("q8_1_q4_vdot_parallel_p1", 1, ("LOCAL:0:64",)),
        ("q8_1_q4_vdot_parallel_p2", 2, ("LOCAL:0:32",)),
        ("q8_1_q4_vdot_parallel_p4", 4, ("LOCAL:0:32",)),
      ):
        candidates.append(CandidateSpec(name, "q4_k_q8_1_vdot_parallel_u32", "q8_1_biased_u8",
                                        "split_k_vdot_parallel_partial", vdot_parts, vdot_opts,
                                        ("q8_1_pack", "q8_1_bias_u32", "amd_v_dot4_u32_u8",
                                         "q4k_q8_1_vdot_parallel_partial_kernel", "u32_packed_storage")))
  elif desc.ggml_type == GGML_Q6_K:
    _require_alignment(desc, 2, "v1_q6_packed")
    candidates.append(CandidateSpec("v1_q6_packed", "q6_k_packed_u16", desc.dtype_activation, "split_k_partial", 1, ("LOCAL:0:64",),
                                    ("q6k_gemv_partial_kernel", "u16_packed_storage")))
    if level >= 1:
      for name, parts, opts in (
        ("q6_local32_p1", 1, ("LOCAL:0:32",)), ("q6_local64_p1", 1, ("LOCAL:0:64",)),
        ("q6_local128_p1", 1, ("LOCAL:0:128",)), ("q6_local64_p2", 2, ("LOCAL:0:64",)),
      ):
        candidates.append(CandidateSpec(name, "q6_k_packed_u16", desc.dtype_activation, "split_k_partial", parts, opts,
                                        ("q6k_gemv_partial_kernel", "u16_packed_storage")))
  else:
    raise ValueError(f"unsupported descriptor type {desc.ggml_type}")
  if desc.ggml_type == GGML_Q6_K and level >= 2:
    candidates.append(CandidateSpec(f"q8_1_{desc.format.lower()}_sketch", "q8_1_packed_activation_sketch", "q8_1",
                                    "split_k_partial", max(1, candidates[-1].parts), candidates[-1].opts,
                                    ("q8_1_pack", "q8_1_x_quant_dot", "not_implemented")))
  return candidates

def _classify(rc:int, out:str, timeout:bool) -> str:
  if timeout: return "timeout"
  if rc == 0: return "pass"
  if "KernelOptError" in out: return "illegal-opt"
  if "CompileError" in out or "compile failed" in out: return "compile-fail"
  if "correctness failed" in out or "AssertionError" in out: return "wrong"
  return "error"

def _run_subprocess(cmd:list[str], repo:pathlib.Path, device:str, debug:int, timeout:float) -> tuple[int, str, bool, float]:
  env = {**os.environ, "DEV": device, "DEBUG": str(debug), "PYTHONPATH": "."}
  st = time.perf_counter()
  try:
    proc = subprocess.run(cmd, cwd=repo, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return proc.returncode, proc.stdout, False, time.perf_counter() - st
  except subprocess.TimeoutExpired as e:
    out = (e.stdout or "") + "\nTIMEOUT"
    return 124, out, True, time.perf_counter() - st

def _parse_q4(desc:QuantGemvDescriptor, candidate:CandidateSpec, out:str, status:str, elapsed_s:float, tail_lines:int) -> dict:
  rows = {m["name"]: m for m in Q4_SUMMARY_RE.finditer(out)}
  row_name = "decode_q4_k_plus_matmul" if candidate.name == "fused_graph" else "q4k_primitive_gemv"
  row = rows.get(row_name)
  device_gbs = None if row is None or row["device_gbs"] == "n/a" else float(row["device_gbs"].split()[0])
  quant_gbs = device_gbs if device_gbs is not None else (None if row is None else float(row["wall_gbs"]))
  device_ms = None if quant_gbs in (None, 0) else desc.packed_bytes / (quant_gbs * 1e9) * 1000
  return {
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "candidate": candidate.name,
    "family": candidate.family, "status": status, "elapsed_s": round(elapsed_s, 3), "device_ms": device_ms,
    "quant_gbs": quant_gbs, "wall_ms": None if row is None else float(row["ms"]),
    "wall_quant_gbs": None if row is None else float(row["wall_gbs"]), "kernels": None if row is None else float(row["kernels"]),
    "gemv_max_abs": float(m.group(1)) if (m:=Q4_GEMV_RE.search(out)) else None,
    "unpack_max_abs": float(m.group(1)) if (m:=Q4_UNPACK_RE.search(out)) else None,
    "parts": candidate.parts, "opts": list(candidate.opts), "requires": list(candidate.requires),
    "tail": "\n".join(out.strip().splitlines()[-tail_lines:]),
  }

def _parse_q6(desc:QuantGemvDescriptor, candidate:CandidateSpec, out:str, status:str, elapsed_s:float, tail_lines:int) -> dict:
  benches = {m["name"]: m for m in Q6_BENCH_RE.finditer(out)}
  row_name = "q6k_fused_graph" if candidate.name == "fused_graph" else "q6k_gemv_primitive_partial"
  row = benches.get(row_name)
  header = Q6_HEADER_RE.search(out)
  quant_gbs = None if row is None else float(row["gbs"])
  return {
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "candidate": candidate.name,
    "family": candidate.family, "status": status, "elapsed_s": round(elapsed_s, 3),
    "device_ms": None if row is None else float(row["ms"]), "quant_gbs": quant_gbs,
    "quant_bytes": None if header is None else int(header["bytes"]), "kernels": None,
    "gemv_max_abs": float(m.group(1)) if (m:=Q6_GEMV_RE.search(out)) else None,
    "unpack_max_abs": float(m.group(1)) if (m:=Q6_UNPACK_RE.search(out)) else None,
    "parts": candidate.parts, "opts": list(candidate.opts), "requires": list(candidate.requires),
    "tail": "\n".join(out.strip().splitlines()[-tail_lines:]),
  }

def _parse_q4_q8(desc:QuantGemvDescriptor, candidate:CandidateSpec, out:str, status:str, elapsed_s:float, tail_lines:int) -> dict:
  row = Q4_Q8_BENCH_RE.search(out)
  device_gbs = None if row is None or row["device_ms"] == "n/a" else float(row["device_gbs"])
  quant_gbs = device_gbs if device_gbs is not None else (None if row is None else float(row["wall_gbs"]))
  return {
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "candidate": candidate.name,
    "family": candidate.family, "status": status, "elapsed_s": round(elapsed_s, 3),
    "device_ms": None if row is None or row["device_ms"] == "n/a" else float(row["device_ms"].split()[0]),
    "quant_gbs": quant_gbs, "wall_ms": None if row is None else float(row["wall_ms"]),
    "wall_quant_gbs": None if row is None else float(row["wall_gbs"]),
    "kernels": None if row is None else float(row["kernels"]),
    "gemv_max_abs": float(m.group(1)) if (m:=Q4_Q8_GEMV_RE.search(out)) else None,
    "unpack_max_abs": float(m.group(1)) if (m:=Q4_Q8_UNPACK_RE.search(out)) else None,
    "activation_max_abs": float(m.group(1)) if (m:=Q4_Q8_PACK_RE.search(out)) else None,
    "parts": candidate.parts, "opts": list(candidate.opts), "requires": list(candidate.requires),
    "tail": "\n".join(out.strip().splitlines()[-tail_lines:]),
  }

def run_candidate(desc:QuantGemvDescriptor, candidate:CandidateSpec, repo:pathlib.Path, iters:int, debug:int, timeout:float,
                  seed:int, tail_lines:int) -> dict:
  if "not_implemented" in candidate.requires:
    return {
      "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "candidate": candidate.name,
      "family": candidate.family, "status": "not-implemented", "elapsed_s": 0.0, "device_ms": None, "quant_gbs": None,
      "gemv_max_abs": None, "unpack_max_abs": None, "parts": candidate.parts, "opts": list(candidate.opts),
      "requires": list(candidate.requires), "tail": "q8_1 generated sketch only; no kernel lowering exists yet",
    }
  if desc.ggml_type == GGML_Q4_K:
    if candidate.family in ("q4_k_q8_1_packed_u32", "q4_k_q8_1_intdot_u32", "q4_k_q8_1_vdot_u32",
                            "q4_k_q8_1_vdot_parallel_u32"):
      cmd = [sys.executable, "extra/q8_1_q4k_bench.py", desc.model, "--device", desc.device, "--tensor", desc.tensor,
             "--iters", str(iters), "--parts", str(candidate.parts), "--seed", str(seed)]
      if candidate.family == "q4_k_q8_1_intdot_u32": cmd += ["--kernel", "intdot"]
      if candidate.family == "q4_k_q8_1_vdot_u32": cmd += ["--kernel", "vdot"]
      if candidate.family == "q4_k_q8_1_vdot_parallel_u32": cmd += ["--kernel", "vdot_parallel"]
      for opt in candidate.opts: cmd += ["--opt", opt]
      rc, out, timeout_hit, elapsed_s = _run_subprocess(cmd, repo, desc.device, debug, timeout)
      return _parse_q4_q8(desc, candidate, out, _classify(rc, out, timeout_hit), elapsed_s, tail_lines)
    cmd = [sys.executable, "extra/q4_k_bench.py", desc.model, "--device", desc.device, "--tensor", desc.tensor,
           "--iters", str(iters), "--format", "text", "--activation", "random", "--seed", str(seed)]
    if candidate.name != "fused_graph":
      cmd += ["--primitive", "--primitive-mode", "partial", "--primitive-parts", str(candidate.parts), "--primitive-schedule", "none"]
      for opt in candidate.opts: cmd += ["--primitive-opt", opt]
    rc, out, timeout_hit, elapsed_s = _run_subprocess(cmd, repo, desc.device, debug, timeout)
    return _parse_q4(desc, candidate, out, _classify(rc, out, timeout_hit), elapsed_s, tail_lines)
  if desc.ggml_type == GGML_Q6_K:
    cmd = [sys.executable, "extra/q6_k_gemv_primitive.py", desc.model, "--device", desc.device, "--tensor", desc.tensor,
           "--iters", str(iters), "--parts", str(max(1, candidate.parts)), "--seed", str(seed)]
    for opt in (candidate.opts or ("LOCAL:0:64",)): cmd += ["--opt", opt]
    rc, out, timeout_hit, elapsed_s = _run_subprocess(cmd, repo, desc.device, debug, timeout)
    return _parse_q6(desc, candidate, out, _classify(rc, out, timeout_hit), elapsed_s, tail_lines)
  raise ValueError(f"unsupported descriptor type {desc.ggml_type}")

def select_winner(desc:QuantGemvDescriptor, candidates:list[CandidateSpec], results:list[dict], min_gain:float=0.0) -> dict:
  passed = [r for r in results if r["status"] == "pass" and r.get("quant_gbs") is not None]
  if not passed:
    return {"tensor": desc.tensor, "winner": None, "reason": "no passing timed candidates", "candidate": None}
  best = max(passed, key=lambda r: r["quant_gbs"])
  fused = next((r for r in passed if r["candidate"] == "fused_graph"), None)
  if fused is not None and best["candidate"] != "fused_graph" and best["quant_gbs"] < fused["quant_gbs"] * (1.0 + min_gain):
    best = fused
    reason = f"fused_graph within min_gain={min_gain:.3f}"
  else:
    reason = "quant_gbs best after correctness"
  cand = next(c for c in candidates if c.name == best["candidate"])
  return {
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "winner": best["candidate"],
    "reason": reason, "metric": "quant_gbs", "metric_value": best["quant_gbs"],
    "candidate": candidate_to_json(cand), "result": {k: v for k, v in best.items() if k != "tail"},
  }

def select_runtime_policy_winner(desc:QuantGemvDescriptor, candidates:list[CandidateSpec], results:list[dict], min_gain:float=0.0) -> dict:
  supported_family = RUNTIME_SUPPORTED_FAMILIES.get(desc.ggml_type)
  supported_names = {"fused_graph"} | {c.name for c in candidates if c.family == supported_family}
  supported_results = [r for r in results if r.get("candidate") in supported_names]
  winner = select_winner(desc, candidates, supported_results, min_gain)
  winner["policy_reason"] = "best runtime-supported generated candidate"
  if winner["winner"] is None: return winner
  best_overall = select_winner(desc, candidates, results, min_gain)
  if best_overall.get("winner") != winner.get("winner"):
    winner["research_winner"] = {
      "winner": best_overall.get("winner"),
      "metric_value": best_overall.get("metric_value"),
      "reason": "not runtime-supported by model.py generated policy integration",
    }
  return winner

def _shape_key_from_desc(desc:QuantGemvDescriptor) -> tuple[int, int, int]:
  return (desc.ggml_type, desc.rows, desc.cols)

def _shape_key_from_info(info:GGUFInfo) -> tuple[int, int, int]|None:
  if info.typ not in RUNTIME_SUPPORTED_FAMILIES or len(info.dims) != 2 or not info.name.endswith(".weight"): return None
  rows, cols = tensor_shape(info)
  return (info.typ, int(rows), int(cols))

def _runtime_storage_bytes(desc:QuantGemvDescriptor, winner:dict) -> int:
  cand = winner.get("candidate") or {}
  if winner.get("winner") == "fused_graph": return 0
  if cand.get("family") != RUNTIME_SUPPORTED_FAMILIES.get(desc.ggml_type): return 0
  return int(desc.packed_bytes)

def _result_device_ms(results:list[dict], candidate_name:str) -> float|None:
  for row in results:
    if row.get("candidate") == candidate_name and row.get("status") == "pass" and row.get("device_ms") is not None:
      return float(row["device_ms"])
  return None

def _benefit_ms(report:dict, winner:dict) -> float:
  if winner.get("winner") in (None, "fused_graph"): return 0.0
  fused_ms = _result_device_ms(report["results"], "fused_graph")
  winner_ms = (winner.get("result") or {}).get("device_ms")
  if fused_ms is None or winner_ms is None: return 0.0
  return max(0.0, fused_ms - float(winner_ms))

def _policy_entry_from_winner(desc:QuantGemvDescriptor, repo:pathlib.Path, winner:dict, scope:str,
                              storage_decision:str, storage_bytes:int, benefit_ms:float) -> dict:
  entry = copy.deepcopy(winner)
  entry.update({
    "key": cache_key(desc, repo), "descriptor": descriptor_to_json(desc), "scope": scope,
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols],
    "storage": {
      "decision": storage_decision,
      "persistent_bytes": int(storage_bytes),
      "benefit_ms": round(float(benefit_ms), 6),
      "benefit_ms_per_mb": round(float(benefit_ms) / max(storage_bytes / 1e6, 1e-9), 6) if storage_bytes else 0.0,
    },
  })
  return entry

def _fused_policy_entry(desc:QuantGemvDescriptor, repo:pathlib.Path, reason:str, capped_from:dict|None=None) -> dict:
  cand = CandidateSpec("fused_graph", "fused_graph", desc.dtype_activation, "generic_fused_reduce", 0, (), ("ggml_data_to_tensor",))
  entry = {
    "key": cache_key(desc, repo), "descriptor": descriptor_to_json(desc), "scope": "tensor",
    "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols],
    "winner": "fused_graph", "reason": reason, "metric": "memory_budget", "metric_value": 0.0,
    "candidate": candidate_to_json(cand), "policy_reason": reason,
    "storage": {"decision": reason, "persistent_bytes": 0, "benefit_ms": 0.0, "benefit_ms_per_mb": 0.0},
  }
  if capped_from is not None:
    entry["capped_from"] = {
      "winner": capped_from.get("winner"),
      "metric_value": capped_from.get("metric_value"),
      "candidate": (capped_from.get("candidate") or {}).get("name"),
      "storage_bytes": capped_from.get("storage", {}).get("persistent_bytes"),
      "benefit_ms": capped_from.get("storage", {}).get("benefit_ms"),
    }
  return entry

def build_policy_entries(model:pathlib.Path, repo:pathlib.Path, meta:GGUFMetadata, descriptor_reports:list[dict],
                         max_storage_bytes:int|None=None) -> tuple[list[dict], dict]:
  reports_by_shape = {
    _shape_key_from_desc(descriptor_from_json(report["descriptor"])): report for report in descriptor_reports
  }
  if max_storage_bytes is None:
    entries = []
    storage_by_format: collections.Counter[str] = collections.Counter()
    for report in descriptor_reports:
      desc = descriptor_from_json(report["descriptor"])
      winner = report["policy_winner"]
      storage_bytes, benefit = _runtime_storage_bytes(desc, winner), _benefit_ms(report, winner)
      entries.append(_policy_entry_from_winner(desc, repo, winner, "shape", "shape_policy", storage_bytes, benefit))
      storage_by_format[desc.format] += storage_bytes
    total = sum(e["storage"]["persistent_bytes"] for e in entries)
    return entries, {
      "mode": "uncapped_shape", "cap_bytes": None, "selected_bytes": total, "selected_entries": len(entries),
      "selected_primitive_entries": sum(1 for e in entries if e["winner"] != "fused_graph"),
      "by_format": dict(sorted(storage_by_format.items())), "note": "shape-scoped entries multiply at runtime by tensor count",
    }

  selected_entries: list[dict] = []
  primitive_items: list[dict] = []
  unsupported_infos = 0
  for info in meta.infos:
    key = _shape_key_from_info(info)
    if key is None: continue
    report = reports_by_shape.get(key)
    if report is None:
      unsupported_infos += 1
      continue
    desc = descriptor_from_info(model, meta, info, report["descriptor"].get("device", "AMD"), report["descriptor"].get("arch"))
    winner = report["policy_winner"]
    storage_bytes, benefit = _runtime_storage_bytes(desc, winner), _benefit_ms(report, winner)
    if storage_bytes <= 0 or winner.get("winner") == "fused_graph" or benefit <= 0:
      selected_entries.append(_fused_policy_entry(desc, repo, "memory_cap_fused_nonpositive_benefit", winner))
      continue
    entry = _policy_entry_from_winner(desc, repo, winner, "tensor", "memory_cap_candidate", storage_bytes, benefit)
    primitive_items.append(entry)

  primitive_items.sort(key=lambda e: (e["storage"]["benefit_ms_per_mb"], e["storage"]["benefit_ms"]), reverse=True)
  selected_bytes = 0
  capped_entries: list[dict] = []
  for entry in primitive_items:
    storage_bytes = int(entry["storage"]["persistent_bytes"])
    if selected_bytes + storage_bytes <= max_storage_bytes:
      entry["storage"]["decision"] = "memory_cap_selected"
      entry["policy_reason"] = "memory_cap_selected"
      selected_bytes += storage_bytes
      selected_entries.append(entry)
    else:
      desc = descriptor_from_json(entry["descriptor"])
      capped_entries.append(_fused_policy_entry(desc, repo, "memory_cap_fused_over_budget", entry))
  entries = selected_entries + capped_entries
  by_format: collections.Counter[str] = collections.Counter()
  by_decision: collections.Counter[str] = collections.Counter()
  by_role_selected: collections.Counter[str] = collections.Counter()
  for entry in entries:
    by_decision[entry["storage"]["decision"]] += 1
    if entry["winner"] != "fused_graph":
      by_format[entry["format"]] += int(entry["storage"]["persistent_bytes"])
      by_role_selected[entry["descriptor"].get("role", "unknown")] += 1
  return entries, {
    "mode": "tensor_memory_cap", "cap_bytes": int(max_storage_bytes), "selected_bytes": int(selected_bytes),
    "selected_entries": len(entries), "selected_primitive_entries": sum(1 for e in entries if e["winner"] != "fused_graph"),
    "capped_primitive_entries": len(capped_entries), "unsupported_tensor_infos": unsupported_infos,
    "by_format": dict(sorted(by_format.items())), "by_decision": dict(sorted(by_decision.items())),
    "selected_by_role": dict(sorted(by_role_selected.items())),
  }

def cache_key(desc:QuantGemvDescriptor, repo:pathlib.Path) -> dict:
  return {
    "device": desc.device, "arch": desc.arch, "ggml_type": desc.ggml_type, "format": desc.format,
    "shape": [desc.rows, desc.cols], "activation": desc.dtype_activation, "generator_version": GENERATOR_VERSION,
    "commit": _git_commit(repo),
  }

def make_policy_cache(model:pathlib.Path, repo:pathlib.Path, entries:list[dict], storage_policy:dict|None=None) -> dict:
  return {
    "kind": "qk_generated_policy", "generator_version": GENERATOR_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
    "model": str(model.expanduser()), "commit": _git_commit(repo), "entries": entries,
    "storage_policy": storage_policy or {},
  }

def validate_policy_cache(cache:dict, repo:pathlib.Path) -> None:
  if cache.get("kind") != "qk_generated_policy": raise ValueError("not a qk generated policy cache")
  if cache.get("generator_version") not in SUPPORTED_GENERATOR_VERSIONS:
    raise ValueError(f"stale generator version: cache={cache.get('generator_version')} current={GENERATOR_VERSION}")
  current_commit = _git_commit(repo)
  if cache.get("commit") != current_commit: raise ValueError(f"stale policy commit: cache={cache.get('commit')} current={current_commit}")

def select_infos(meta:GGUFMetadata, tensor_names:list[str]|None, max_shapes:int|None) -> list[GGUFInfo]:
  if tensor_names:
    out = []
    for name in tensor_names:
      matches = [x for x in meta.infos if x.name == name]
      if not matches: raise ValueError(f"tensor {name!r} not found")
      out.append(matches[0])
    return out
  targets = list(model_shape_targets(meta.infos, meta.kv, max_shapes, GGML_Q4_K))
  q6_seen: set[tuple[int, int]] = set()
  for info in meta.infos:
    if info.typ != GGML_Q6_K or len(info.dims) != 2: continue
    shape = tensor_shape(info)
    if shape in q6_seen: continue
    q6_seen.add(shape)
    targets.append(info)
    if max_shapes is not None and len(targets) >= max_shapes: break
  return targets[:max_shapes] if max_shapes is not None else targets

def run_search(args) -> dict:
  repo = args.repo.resolve()
  model = args.model.expanduser()
  meta = read_metadata(model)
  arch = _device_arch(args.device)
  descriptors = [descriptor_from_info(model, meta, info, args.device, arch) for info in select_infos(meta, args.tensor, args.max_shapes)]
  if args.descriptors_json:
    args.descriptors_json.write_text(json.dumps([descriptor_to_json(d) for d in descriptors], indent=2, sort_keys=True))
  if args.describe_only:
    return {"descriptors": [descriptor_to_json(d) for d in descriptors]}
  assert_q4k_native_sweep_allowed(args.device, "QK generated candidate runner")
  descriptor_reports = []
  only = set(args.only)
  for desc in descriptors:
    candidates = [c for c in generate_candidates(desc, args.level) if not only or c.name in only]
    print(f"=== {desc.tensor} {desc.format} {desc.rows}x{desc.cols} candidates={len(candidates)} ===", flush=True)
    results = []
    for cand in candidates:
      estimate = estimate_candidate(desc, cand)
      print(f"--- {cand.name} parts={cand.parts} opts={list(cand.opts)} ops/byte={estimate['ops_per_min_byte']} ---", flush=True)
      if args.skip_stopped and estimate["stop_reason"] is not None:
        res = {
          "tensor": desc.tensor, "format": desc.format, "shape": [desc.rows, desc.cols], "candidate": cand.name,
          "family": cand.family, "status": "skipped-stop", "elapsed_s": 0.0, "device_ms": None, "quant_gbs": None,
          "gemv_max_abs": None, "unpack_max_abs": None, "parts": cand.parts, "opts": list(cand.opts),
          "requires": list(cand.requires), "tail": estimate["stop_reason"],
        }
      else:
        res = run_candidate(desc, cand, repo, args.iters, args.debug, args.timeout, args.seed, args.tail_lines)
      res["estimate"] = estimate
      results.append(res)
      print(f"{res['status']} quant_gbs={res.get('quant_gbs')} gemv={res.get('gemv_max_abs')}", flush=True)
    winner = select_winner(desc, candidates, results, args.min_gain)
    policy_winner = select_runtime_policy_winner(desc, candidates, results, args.min_gain)
    descriptor_reports.append({"descriptor": descriptor_to_json(desc),
                               "candidates": [{**candidate_to_json(c), "estimate": estimate_candidate(desc, c)} for c in candidates],
                               "results": results, "winner": winner, "policy_winner": policy_winner})
  cap_bytes = None if args.policy_max_storage_mb is None else int(args.policy_max_storage_mb * 1024 * 1024)
  policy_entries, storage_policy = build_policy_entries(model, repo, meta, descriptor_reports, cap_bytes)
  policy = make_policy_cache(model, repo, policy_entries, storage_policy)
  report = {"generator_version": GENERATOR_VERSION, "model": str(model), "device": args.device, "arch": arch,
            "descriptors": descriptor_reports, "policy": policy, "storage_policy": storage_policy}
  if args.json: args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.policy_json: args.policy_json.write_text(json.dumps(policy, indent=2, sort_keys=True))
  return report

def main() -> None:
  parser = argparse.ArgumentParser(description="Ansor-direction generated candidate runner for packed Q4_K/Q6_K GEMV")
  parser.add_argument("--model", type=pathlib.Path, required=True)
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path.cwd())
  parser.add_argument("--tensor", action="append", help="exact tensor name; default emits representative Q4_K plus distinct Q6_K shapes")
  parser.add_argument("--max-shapes", type=int)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--level", type=int, default=0, help="0=fused+v1, 1=old sweep space, 2=q8_1 sketch")
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--debug", type=int, default=2)
  parser.add_argument("--timeout", type=float, default=120)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--min-gain", type=float, default=0.0)
  parser.add_argument("--only", action="append", default=[])
  parser.add_argument("--skip-stopped", action="store_true",
                      help="skip candidates rejected by semantic stop gates instead of timing them")
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--describe-only", action="store_true")
  parser.add_argument("--descriptors-json", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--policy-json", type=pathlib.Path)
  parser.add_argument("--policy-max-storage-mb", type=float,
                      help="emit a tensor-scoped policy capped to this much persistent primitive storage")
  args = parser.parse_args()
  report = run_search(args)
  if args.describe_only and not args.json:
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
  main()
