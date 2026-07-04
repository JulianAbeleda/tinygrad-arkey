"""Emit BoltBeam timing evidence for tinygrad prefill.

This is a trace producer, not a benchmark authority replacement. It uses the same synced prefill methodology for
whole-step tok/s, then runs one PROFILE pass to attribute the measured wall time across kernels/roles.

  DEV=AMD PREFILL_V2=1 PREFILL_CHUNKED=1 PREFILL_GRAPH_GEMM=1 PROFILE=1 PYTHONPATH=. \
    python3 extra/qk/prefill_boltbeam_trace.py --model /path/model.gguf --context 512 --out timing_trace.json
"""
from __future__ import annotations

import argparse, itertools, json, os, pathlib, re, sys, time
from typing import Any

os.environ.setdefault("PREFILL_V2", "1")
os.environ.setdefault("PREFILL_CHUNKED", "1")
os.environ.setdefault("PREFILL_GRAPH_GEMM", "1")
os.environ.setdefault("PROFILE", "1")
if "--hw-trace" in sys.argv:
  os.environ.setdefault("PMC", "1")

from tinygrad import Tensor, Device
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent, prod
from tinygrad.llm.qk_primitives import Q4KPrimitiveLinear, Q6KPrimitiveLinear
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL

SCHEMA = "boltbeam.timing_trace.v1"
HW_SCHEMA = "boltbeam.hw_trace.v1"
_GEMM_RE = re.compile(r"prefill_(?:graph|gen_sched)_gemm_(\d+)_(\d+)_(\d+)")
_DIRECT_PACKED_RE = re.compile(r"prefill_(q4k|q6k)(?:_q8_1)?(?:_sdot4|_mmq)?_direct_packed(?:_load)?(?:_direct_out|_reduce_out)?_gemm_(\d+)_(\d+)_(\d+)_(\d+)")
_GENERATED_PACKED_TILE_RE = re.compile(r"prefill_(q4_k|q6_k)_generated_tile(?:_([a-z0-9_]+))?_(\d+)_(\d+)_(\d+)")
_ROLE_BY_LINEAR = {
  "ffn_gate": "ffn_gate_up",
  "ffn_up": "ffn_gate_up",
  "ffn_gate_shexp": "ffn_gate_up",
  "ffn_up_shexp": "ffn_gate_up",
  "ffn_down": "ffn_down",
  "ffn_down_shexp": "ffn_down",
  "attn_q": "attn_qo",
  "attn_output": "attn_qo",
  "attn_k": "attn_kv",
  "attn_v": "attn_kv",
}


def _event_name(e:ProfileRangeEvent) -> str:
  return getattr(e.name, "name", None) or str(e.name)


def _event_us(e:ProfileRangeEvent) -> float:
  return float((e.en or e.st) - e.st)


def _model_id(model_path:str) -> str:
  return pathlib.Path(model_path).stem or "model"


def _quant_and_bytes(lin:Any) -> tuple[str | None, float]:
  if isinstance(lin, Q4KPrimitiveLinear):
    return "Q4_K", float(lin.q4k_storage.source_bytes)
  if isinstance(lin, Q6KPrimitiveLinear):
    return "Q6_K", float(lin.q6k_storage.source_bytes)
  weight = getattr(lin, "weight", None)
  if weight is None or not getattr(weight, "shape", None):
    return None, 0.0
  itemsize = getattr(getattr(weight, "dtype", None), "itemsize", 2) or 2
  return str(getattr(weight, "dtype", "unknown")).replace("dtypes.", "").upper(), float(prod(weight.shape) * itemsize)


def _linear_dims(lin:Any) -> tuple[int | None, int | None]:
  out_f = getattr(lin, "out_features", None)
  in_f = getattr(lin, "in_features", None)
  weight = getattr(lin, "weight", None)
  if (out_f is None or in_f is None) and weight is not None and getattr(weight, "shape", None) and len(weight.shape) >= 2:
    out_f, in_f = weight.shape[0], weight.shape[1]
  return (int(out_f), int(in_f)) if isinstance(out_f, int) and isinstance(in_f, int) else (None, None)


def _role_inventory(model:Any, chunk:int) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[tuple[str, str | None, tuple[int, ...]], float]]:
  by_shape: dict[tuple[int, int], dict[str, Any]] = {}
  bytes_by_role: dict[tuple[str, str | None, tuple[int, ...]], float] = {}
  for block in getattr(model, "blk", []):
    for name, role in _ROLE_BY_LINEAR.items():
      lin = getattr(block, name, None)
      if lin is None:
        continue
      out_f, in_f = _linear_dims(lin)
      if out_f is None or in_f is None:
        continue
      quant, nbytes = _quant_and_bytes(lin)
      shape = (chunk, out_f, in_f)
      key = (role, quant, shape)
      bytes_by_role[key] = bytes_by_role.get(key, 0.0) + nbytes
      by_shape[(out_f, in_f)] = {"role": role, "quant": quant}
  return by_shape, bytes_by_role


def _classify_kernel(name:str, role_by_shape:dict[tuple[int, int], dict[str, Any]]) -> dict[str, Any]:
  low = name.lower()
  m = _GEMM_RE.search(name)
  if m:
    mb, n, k = (int(x) for x in m.groups())
    info = dict(role_by_shape.get((n, k), {}))
    info.setdefault("role", "quantized_matmul")
    info.setdefault("quant", None)
    info["shape"] = [mb, n, k]
    info["kind"] = "gemm"
    return info
  m = _DIRECT_PACKED_RE.search(name)
  if m:
    quant_tag, n, k, mb, _parts = m.groups()
    mb, n, k = int(mb), int(n), int(k)
    info = dict(role_by_shape.get((n, k), {}))
    info.setdefault("role", "quantized_matmul")
    info["quant"] = "Q4_K" if quant_tag == "q4k" else "Q6_K"
    info["shape"] = [mb, n, k]
    info["kind"] = "gemm"
    return info
  m = _GENERATED_PACKED_TILE_RE.search(name)
  if m:
    quant_tag, role_tag, mb, n, k = m.groups()
    mb, n, k = int(mb), int(n), int(k)
    info = dict(role_by_shape.get((n, k), {}))
    if role_tag: info["role"] = role_tag
    info.setdefault("role", "quantized_matmul")
    info["quant"] = "Q4_K" if quant_tag == "q4_k" else "Q6_K"
    info["shape"] = [mb, n, k]
    info["kind"] = "gemm"
    return info
  if "flash" in low or "attn" in low:
    return {"kind": "attention", "role": "attention", "quant": None, "shape": []}
  if "rope" in low:
    return {"kind": "rope", "role": "rope", "quant": None, "shape": []}
  if "norm" in low or low.startswith("r_"):
    return {"kind": "norm", "role": "norm", "quant": None, "shape": []}
  if "silu" in low or "gelu" in low or "mul" in low:
    return {"kind": "activation", "role": "ffn_activation", "quant": None, "shape": []}
  if "copy" in low or "cast" in low or low.startswith("d_"):
    return {"kind": "copy", "role": "copy_cast", "quant": None, "shape": []}
  if low.startswith("e_"):
    return {"kind": "elementwise", "role": "elementwise", "quant": None, "shape": []}
  return {"kind": "elementwise", "role": "unknown", "quant": None, "shape": []}


def _profile_events() -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for e in Compiled.profile_events:
    if not isinstance(e, ProfileRangeEvent) or e.en is None:
      continue
    name = _event_name(e)
    if name.startswith("TracingKey"):
      continue
    row = out.setdefault(name, {"raw_wall_us": 0.0, "calls": 0})
    row["raw_wall_us"] += _event_us(e)
    row["calls"] += 1
  return out


def _pmc_stats(e:Any) -> dict[str, tuple[int, int, int]]:
  stats: dict[str, tuple[int, int, int]] = {}
  view, ptr = memoryview(e.blob).cast("Q"), 0
  for s in e.sched:
    total, max_val, cnt = 0, 0, 0
    for _sample in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa)):
      for _ in range(s.wgp):
        val = int(view[ptr])
        ptr += 1
        total += val
        max_val = max(max_val, val)
        cnt += 1
    stats[s.name] = (total, max_val, cnt)
  return stats


def _pct(num:float, den:float) -> float:
  return 100.0 * num / den if den else 0.0


def _normalize_pmc_stats(stats:dict[str, tuple[int, int, int]]) -> dict[str, float]:
  def total(name:str) -> float: return float(stats.get(name, (0, 0, 0))[0])
  def maxv(name:str) -> float: return float(stats.get(name, (0, 0, 0))[1])
  def count(name:str) -> float: return float(stats.get(name, (0, 0, 0))[2])
  out: dict[str, float] = {}
  hits, misses = total("GL2C_HIT") + total("TCC_HIT"), total("GL2C_MISS") + total("TCC_MISS")
  if hits or misses: out["l2_hit_pct"] = _pct(hits, hits + misses)
  fetch_bytes = total("GL2C_EA_RDREQ_32B") * 32 + total("GL2C_EA_RDREQ_64B") * 64 + \
                total("GL2C_EA_RDREQ_96B") * 96 + total("GL2C_EA_RDREQ_128B") * 128
  write_bytes = total("GL2C_EA_WRREQ_64B") * 64 + total("GL2C_MC_WRREQ") * 32
  if fetch_bytes: out["fetch_kb"] = fetch_bytes / 1024.0
  if write_bytes: out["write_kb"] = write_bytes / 1024.0
  lds_active = total("SQC_LDS_IDX_ACTIVE") + total("SQ_LDS_IDX_ACTIVE")
  lds_conflict = total("SQC_LDS_BANK_CONFLICT") + total("SQ_LDS_BANK_CONFLICT")
  if lds_active or lds_conflict: out["lds_conflict_pct"] = _pct(lds_conflict, max(lds_active, 1.0))
  gui = maxv("GRBM_GUI_ACTIVE")
  if gui:
    valu = total("SQ_INSTS_VALU")
    valu_cnt = count("SQ_INSTS_VALU") or 1.0
    busy = total("SQ_BUSY_CYCLES")
    busy_cnt = count("SQ_BUSY_CYCLES") or 1.0
    vmem_cycles = total("SQ_INST_CYCLES_VMEM")
    vmem_cnt = count("SQ_INST_CYCLES_VMEM") or 1.0
    out["valu_busy_pct"] = _pct(valu / valu_cnt, gui * 4.0)
    out["occupancy_pct"] = min(100.0, _pct(busy / busy_cnt, gui))
    if vmem_cycles: out["memory_busy_pct"] = min(100.0, _pct(vmem_cycles / vmem_cnt, gui))
  return out


def _pmc_by_program() -> dict[str, dict[str, float]]:
  programs: dict[int, str] = {}
  counters: dict[int, dict[str, float]] = {}
  for e in Compiled.profile_events:
    if type(e).__name__ == "ProfileProgramEvent" and getattr(e, "tag", None) is not None:
      programs[int(e.tag)] = _event_name(e)
    elif type(e).__name__ == "ProfilePMCEvent":
      counters[int(e.kern)] = _normalize_pmc_stats(_pmc_stats(e))
  return {programs[k]: v for k, v in counters.items() if k in programs and v}


def _time_prefill(model:Any, chunk:Tensor, start_pos:int, temp:Tensor, repeats:int) -> float:
  for _ in range(4):
    model(chunk, start_pos, temp).realize()
  Device["AMD"].synchronize()
  samples = []
  for _ in range(3):
    Device["AMD"].synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
      model(chunk, start_pos, temp).realize()
    Device["AMD"].synchronize()
    samples.append((time.perf_counter() - t0) / repeats * 1e6)
  return min(samples)


def _profile_prefill(model:Any, chunk:Tensor, start_pos:int, temp:Tensor) -> dict[str, dict[str, Any]]:
  model(chunk, start_pos, temp).realize()
  Device["AMD"].synchronize()
  program_events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfileProgramEvent"]
  Compiled.profile_events = program_events
  model(chunk, start_pos, temp).realize()
  Device["AMD"].synchronize()
  out = _profile_events()
  for name, counters in _pmc_by_program().items():
    if name in out:
      out[name]["counters"] = counters
      continue
    for row_name in out:
      if name in row_name or row_name in name:
        out[row_name]["counters"] = counters
        break
  return out


def _build_rows(context:int, wall_us:float, per_kernel:dict[str, dict[str, Any]],
                role_by_shape:dict[tuple[int, int], dict[str, Any]],
                bytes_by_role:dict[tuple[str, str | None, tuple[int, ...]], float],
                *, chunk_start:int) -> list[dict[str, Any]]:
  raw_total = sum(float(v["raw_wall_us"]) for v in per_kernel.values()) or 1.0
  rows = []
  role_rows: dict[tuple[str, str | None, tuple[int, ...]], list[dict[str, Any]]] = {}
  for name, raw in sorted(per_kernel.items(), key=lambda kv: -float(kv[1]["raw_wall_us"])):
    info = _classify_kernel(name, role_by_shape)
    scaled_us = wall_us * float(raw["raw_wall_us"]) / raw_total
    row = {
      "scope": "kernel",
      "context": context,
      "kernel": name,
      "kind": info["kind"],
      "role": info["role"],
      "quant": info.get("quant"),
      "shape": info.get("shape") or [],
      "wall_us": scaled_us,
      "raw_wall_us": float(raw["raw_wall_us"]),
      "calls": int(raw["calls"]),
      "time_source": "profile_scaled_to_synced_wall",
      "chunk_start": chunk_start,
    }
    if raw.get("counters"):
      row["counters"] = raw["counters"]
    key = (row["role"], row.get("quant"), tuple(row["shape"]))
    role_rows.setdefault(key, []).append(row)
    rows.append(row)
  for key, nbytes in bytes_by_role.items():
    matches = role_rows.get(key, [])
    denom = sum(float(r["wall_us"]) for r in matches)
    for row in matches:
      row["phys_bytes"] = nbytes * float(row["wall_us"]) / denom if denom else 0.0
      row["gbs"] = row["phys_bytes"] / max(float(row["wall_us"]), 1e-9) / 1e3
  for row in rows:
    row.setdefault("phys_bytes", 0.0)
    row.setdefault("gbs", 0.0)
  return rows


def _merge_kernel_rows(rows:list[dict[str, Any]]) -> list[dict[str, Any]]:
  merged: dict[tuple[Any, ...], dict[str, Any]] = {}
  for row in rows:
    key = (row["kernel"], row["kind"], row.get("role"), row.get("quant"), tuple(row.get("shape") or ()))
    if key not in merged:
      out = {k: v for k, v in row.items() if k != "chunk_start"}
      out["chunk_starts"] = [row["chunk_start"]]
      merged[key] = out
      continue
    out = merged[key]
    out["wall_us"] = float(out.get("wall_us") or 0.0) + float(row.get("wall_us") or 0.0)
    out["raw_wall_us"] = float(out.get("raw_wall_us") or 0.0) + float(row.get("raw_wall_us") or 0.0)
    out["calls"] = int(out.get("calls") or 0) + int(row.get("calls") or 0)
    out["phys_bytes"] = float(out.get("phys_bytes") or 0.0) + float(row.get("phys_bytes") or 0.0)
    out["gbs"] = float(out["phys_bytes"]) / max(float(out["wall_us"]), 1e-9) / 1e3
    out["chunk_starts"].append(row["chunk_start"])
  return sorted(merged.values(), key=lambda r: -float(r.get("wall_us") or 0.0))


def _whole_row(args:argparse.Namespace, wall_us:float, total_bytes:float, launch_count:int) -> dict[str, Any]:
  return {
    "scope": "whole_step",
    "context": args.context,
    "wall_us": wall_us,
    "tok_s": args.context / wall_us * 1e6,
    "total_bytes": total_bytes,
    "launch_count": launch_count,
    "time_source": "synced_min_of_bursts" if args.mode in ("timing", "full") else "profile_event_sum",
    "bytes_source": "tinygrad_qk_primitive_source_bytes",
  }


def build_trace(args:argparse.Namespace) -> dict[str, Any]:
  if args.hw_trace:
    os.environ["PMC"] = "1"
  dev = Device["AMD"]
  model, _tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  for block in getattr(model, "blk", []):
    block._use_flash, block._prefill_v2 = True, True
  temp = Tensor([0.0])
  chunk = Tensor([[(i * 7) % 1000 for i in range(args.chunk)]], dtype="int32").contiguous()
  role_by_shape, bytes_by_role = _role_inventory(model, args.chunk)
  chunk_timings = []
  chunk_rows = []
  profile_raw_total_us = 0.0
  for start_pos in range(0, args.context, args.chunk):
    if args.mode in ("timing", "full"):
      wall_us = _time_prefill(model, chunk, start_pos, temp, args.repeats)
      chunk_timings.append({"start_pos": start_pos, "wall_us": wall_us, "tok_s": args.chunk / wall_us * 1e6})
    if args.mode in ("profile", "full"):
      per_kernel = _profile_prefill(model, chunk, start_pos, temp)
      profile_wall_us = float(sum(v["raw_wall_us"] for v in per_kernel.values()))
      profile_raw_total_us += profile_wall_us
      if args.mode == "profile":
        wall_us = profile_wall_us
        chunk_timings.append({"start_pos": start_pos, "wall_us": wall_us, "tok_s": args.chunk / wall_us * 1e6})
      chunk_rows.extend(_build_rows(args.context, wall_us, per_kernel, role_by_shape, bytes_by_role, chunk_start=start_pos))
  dev.synchronize()
  kernel_rows = _merge_kernel_rows(chunk_rows)
  wall_us = sum(float(r["wall_us"]) for r in chunk_timings)
  total_bytes = sum(bytes_by_role.values()) * len(chunk_timings)
  whole = _whole_row(args, wall_us, total_bytes, sum(int(r.get("calls", 1)) for r in kernel_rows))
  return {
    "schema": HW_SCHEMA if args.hw_trace else SCHEMA,
    **({"source_schema": SCHEMA, "trace_source": "tinygrad_internal_pmc", "counter_vocab": [
      "occupancy_pct", "memory_busy_pct", "memory_stall_pct", "l2_hit_pct", "fetch_kb", "write_kb",
      "valu_busy_pct", "lds_conflict_pct", "lds_stall_pct", "mfma_util_pct",
    ]} if args.hw_trace else {}),
    "model_id": args.model_id or _model_id(args.model),
    "target_id": args.target_id,
    "workload": "prefill",
    "provider_id": "tinygrad/profile",
    "timing_source": "synced_wall_plus_profile_attribution",
    "peak_gbs": args.peak_gbs,
    "contexts": [args.context],
    "metadata": {
      "model_path": args.model,
      "chunk": args.chunk,
      "repeats": args.repeats,
      "mode": args.mode,
      "chunk_timings": chunk_timings,
      "route_flags": {k: os.environ.get(k) for k in ("PREFILL_V2", "PREFILL_CHUNKED", "PREFILL_GRAPH_GEMM",
                                                     "PREFILL_GENERATED_SCHEDULE", "PREFILL_ROUTE",
                                                     "PREFILL_QK_DIRECT", "PREFILL_DIRECT_QUANTS",
                                                     "PREFILL_DIRECT_TENSORS", "PREFILL_DIRECT_SKIP_TENSORS",
                                                     "PREFILL_Q4K_PACKED_LOAD", "PREFILL_Q6K_PACKED_LOAD",
                                                     "PREFILL_DIRECT_B_UPCAST", "PREFILL_DIRECT_OUT",
                                                     "PREFILL_Q4K_Q8", "PROFILE", "PMC", "PMC_COUNTERS")},
      "profile_raw_total_us": profile_raw_total_us,
      "role_inventory_count": len(bytes_by_role),
    },
    "notes": [
      "Whole-step timing is synced wall-clock min-of-bursts through the real model call path.",
      "Kernel rows are PROFILE events scaled to the measured whole-step wall time.",
      "Physical bytes are model weight source bytes assigned to matching prefill GEMM role rows.",
    ],
    "rows": [whole] + kernel_rows,
  }


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF model path")
  ap.add_argument("--model-id", default=None, help="BoltBeam model id override")
  ap.add_argument("--target-id", default="amd_gfx1100", help="BoltBeam target id")
  ap.add_argument("--context", type=int, default=512, help="prefill context tokens represented by the trace")
  ap.add_argument("--chunk", type=int, default=512, help="concrete prefill chunk size")
  ap.add_argument("--mode", choices=["timing", "profile", "full"], default="full",
                  help="timing=whole-step only, profile=kernel attribution only, full=timing plus profile attribution")
  ap.add_argument("--max-context", type=int, default=4608, help="model max context to allocate")
  ap.add_argument("-K", "--repeats", type=int, default=4, help="repeats per timing burst")
  ap.add_argument("--seed", type=int, default=20260617)
  ap.add_argument("--peak-gbs", type=float, default=960.0)
  ap.add_argument("--hw-trace", action="store_true", help="emit boltbeam.hw_trace.v1 with tinygrad internal PMC counters")
  ap.add_argument("--out", default=None, help="write trace JSON here instead of stdout")
  args = ap.parse_args()
  if args.context % args.chunk != 0:
    raise SystemExit("--context must be a multiple of --chunk for v1 scaled traces")
  trace = build_trace(args)
  text = json.dumps(trace, indent=2) + "\n"
  if args.out:
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(text)
  else:
    print(text, end="")


if __name__ == "__main__":
  main()
