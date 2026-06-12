#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, re, subprocess, sys, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import prod

from extra.q4_k_safety import assert_q4k_native_sweep_allowed
from extra.qk_layout import (
  GGML_Q4_K, GGML_Q6_K, GGUFInfo, GGUFMetadata, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q6_K_BLOCK_BYTES,
  Q6_K_BLOCK_ELEMS, format_name, model_shape_targets, packed_byte_range, q6_k_weight_bytes, quant_weight_bytes,
  read_metadata, role_from_name, tensor_shape,
)

GENERATOR_VERSION = 0

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

def cache_key(desc:QuantGemvDescriptor, repo:pathlib.Path) -> dict:
  return {
    "device": desc.device, "arch": desc.arch, "ggml_type": desc.ggml_type, "format": desc.format,
    "shape": [desc.rows, desc.cols], "activation": desc.dtype_activation, "generator_version": GENERATOR_VERSION,
    "commit": _git_commit(repo),
  }

def make_policy_cache(model:pathlib.Path, repo:pathlib.Path, entries:list[dict]) -> dict:
  return {
    "kind": "qk_generated_policy", "generator_version": GENERATOR_VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
    "model": str(model.expanduser()), "commit": _git_commit(repo), "entries": entries,
  }

def validate_policy_cache(cache:dict, repo:pathlib.Path) -> None:
  if cache.get("kind") != "qk_generated_policy": raise ValueError("not a qk generated policy cache")
  if cache.get("generator_version") != GENERATOR_VERSION:
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
  policy_entries, descriptor_reports = [], []
  only = set(args.only)
  for desc in descriptors:
    candidates = [c for c in generate_candidates(desc, args.level) if not only or c.name in only]
    print(f"=== {desc.tensor} {desc.format} {desc.rows}x{desc.cols} candidates={len(candidates)} ===", flush=True)
    results = []
    for cand in candidates:
      print(f"--- {cand.name} parts={cand.parts} opts={list(cand.opts)} ---", flush=True)
      res = run_candidate(desc, cand, repo, args.iters, args.debug, args.timeout, args.seed, args.tail_lines)
      results.append(res)
      print(f"{res['status']} quant_gbs={res.get('quant_gbs')} gemv={res.get('gemv_max_abs')}", flush=True)
    winner = select_winner(desc, candidates, results, args.min_gain)
    policy_entries.append({"key": cache_key(desc, repo), "descriptor": descriptor_to_json(desc), **winner})
    descriptor_reports.append({"descriptor": descriptor_to_json(desc), "candidates": [candidate_to_json(c) for c in candidates],
                               "results": results, "winner": winner})
  policy = make_policy_cache(model, repo, policy_entries)
  report = {"generator_version": GENERATOR_VERSION, "model": str(model), "device": args.device, "arch": arch,
            "descriptors": descriptor_reports, "policy": policy}
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
  parser.add_argument("--tail-lines", type=int, default=8)
  parser.add_argument("--describe-only", action="store_true")
  parser.add_argument("--descriptors-json", type=pathlib.Path)
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--policy-json", type=pathlib.Path)
  args = parser.parse_args()
  report = run_search(args)
  if args.describe_only and not args.json:
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__":
  main()
