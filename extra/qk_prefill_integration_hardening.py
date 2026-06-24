#!/usr/bin/env python3
"""Prefill long-context integration hardening benchmark (full-lattice, non-search).

Run a fixed whole-prefill schedule across contexts and emit full attribution:
- chunk-level synced timings for every start_pos (no interpolation)
- full runtime decomposition snapshots (wall / gpu-only / host-sync / launch)
- per-role attribution timeseries per ctx/start_pos
- split buckets for kv / attention / copy-materialization

Example:
  DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python \
    extra/qk_prefill_integration_hardening.py \
      --out bench/qk-prefill-long-context-integration-hardening-20260624
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import time

from tinygrad import Context, Device, Tensor, TinyJit
from tinygrad.device import Compiled
from tinygrad.llm.model import PREFILL_GRAPH_GEMM, PREFILL_TENSILE_GEMM

from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL

os.environ.setdefault("PREFILL_V2", "1")

ANSI = re.compile(r"\x1b\[[0-9;]*m")
GRAPH_PATTERN = re.compile(r"prefill_graph_gemm_512_(\d+)_(\d+)")


def _git(path: Path, *args: str) -> str:
  try:
    return subprocess.check_output(["git", *args], text=True, cwd=str(path)).strip()
  except Exception:
    return "unknown"


def _dirty_tree(path: Path) -> bool:
  try:
    return bool(subprocess.check_output(["git", "status", "--short"], cwd=str(path), text=True).strip())
  except Exception:
    return False


def _rocm_vram_json() -> dict:
  try:
    raw = subprocess.check_output(["rocm-smi", "--showmeminfo", "vram", "--json"], text=True)
    return json.loads(raw)
  except Exception as e:
    return {"error": str(e)}


def _coerce_vram_bytes(value: object) -> int | None:
  if value is None:
    return None
  if isinstance(value, int):
    return int(value)
  if isinstance(value, float):
    return int(value)
  if isinstance(value, str):
    s = value.strip()
    if not s:
      return None
    # examples: "34359738368", "34,359,738,368"
    digits = re.sub(r"[^0-9]", "", s)
    return int(digits) if digits else None
  return None


def _extract_vram_stats(data: dict) -> tuple[int | None, int | None]:
  if not isinstance(data, dict):
    return None, None
  card = data.get("card0") if isinstance(data.get("card0"), dict) else None
  if card is None:
    if len(data) == 1:
      only = next(iter(data.values()))
      if isinstance(only, dict):
        card = only
  if not isinstance(card, dict):
    return None, None
  used = _coerce_vram_bytes(card.get("VRAM Total Used Memory (B)"))
  total = _coerce_vram_bytes(card.get("VRAM Total Memory (B)"))
  return used, total


def _median(xs: list[float]) -> float:
  return statistics.median(xs) if xs else 0.0


def _pct_change(a: float, b: float) -> float:
  if a == 0:
    return 0.0
  return (b - a) / a * 100.0


def _route_tag() -> str:
  return f"route_graph_gemm_{'on' if PREFILL_GRAPH_GEMM else 'off'}_tensile_{'on' if PREFILL_TENSILE_GEMM else 'off'}"


def _route_key() -> str:
  if PREFILL_TENSILE_GEMM and not PREFILL_GRAPH_GEMM:
    return "tensile"
  return "graph_gemm"


def _shape_from_name(name: str) -> tuple[int, int] | None:
  m = GRAPH_PATTERN.search(name)
  if not m:
    return None
  return int(m.group(1)), int(m.group(2))


def _role_from_name(name: str) -> tuple[str, str, str]:
  nkl = _shape_from_name(name)
  lower = ANSI.sub("", name.lower())

  if nkl is not None:
    n, k = nkl
    shape = f"512x{n}x{k}"
    if n == 12288 and k == 4096:
      return "ffn_gate_up", "graph_gemm", shape
    if n == 4096 and k == 12288:
      return "ffn_down", "graph_gemm", shape
    if n == 4096 and k == 4096:
      return "qo_proj", "graph_gemm", shape
    if n == 1024 and k == 4096:
      return "kv_proj", "graph_gemm", shape
    if k == 4096 and n > 50000:
      return "lm_head", "graph_gemm", shape
    return f"gemm_{n}_{k}", "graph_gemm", shape

  # non-GEMM classes
  if "attn" in lower and "qk" in lower:
    return "attention_qk", "non_gemm", "mixed"
  if "attn" in lower and "pv" in lower:
    return "attention_pv", "non_gemm", "mixed"
  if "qk" in lower:
    return "attention_qk", "non_gemm", "mixed"
  if "pv" in lower:
    return "attention_pv", "non_gemm", "mixed"
  if any(x in lower for x in ["copy", "contiguous", "material", "pack", "unpack", "transpose"]):
    return "copy_materialization", "non_gemm", "mixed"
  return "other", "non_gemm", "mixed"


def measure_chunk_wall(f, dev, chunk, start_pos: int, temp, repeats: int = 5, inner: int = 8, warmup: int = 4):
  for _ in range(warmup):
    f(chunk, start_pos, temp).realize()
  dev.synchronize()

  wall_samples = []
  launch_samples = []
  sync_samples = []
  for _ in range(repeats):
    dev.synchronize()
    t0 = time.perf_counter()
    for _ in range(inner):
      f(chunk, start_pos, temp).realize()
    t1 = time.perf_counter()
    dev.synchronize()
    t2 = time.perf_counter()
    wall_samples.append((t2 - t0) / inner * 1e3)
    launch_samples.append((t1 - t0) / inner * 1e3)
    sync_samples.append((t2 - t1) / inner * 1e3)

  return {
    "median_ms": _median(wall_samples),
    "samples_ms": wall_samples,
    "launch_overhead_ms": _median(launch_samples),
    "host_sync_ms": _median(sync_samples),
    "sync_calls": repeats,
  }


def measure_chunk_profile(f, dev, chunk, start_pos: int, temp, repeats: int = 3):
  repeats = max(1, repeats)
  rows = []

  with Context(PROFILE=1):
    try:
      Compiled.profile_events.clear()
    except Exception:
      pass

    # one warmup pass inside PROFILE
    for _ in range(2):
      f(chunk, start_pos, temp).realize()
    dev.synchronize()
    try:
      dev._at_profile_finalize()
    except Exception:
      pass

    for _ in range(repeats):
      base = len(Compiled.profile_events)
      f(chunk, start_pos, temp).realize()
      dev.synchronize()
      try:
        dev._at_profile_finalize()
      except Exception:
        pass

      agg = collections.defaultdict(float)
      for e in Compiled.profile_events[base:]:
        if type(e).__name__ != "ProfileGraphEvent":
          continue
        sigs = [float(s) for s in e.sigs]
        for ent in e.ents:
          agg[ANSI.sub("", str(ent.name))] += sigs[ent.en_id] - sigs[ent.st_id]

      rows.append(dict(agg))

      # clear old events each run to bound memory for long ctx
      try:
        del Compiled.profile_events[:base]
      except Exception:
        pass

  if not rows:
    return {"roles": {}, "profile_ms": 0.0}

  role_to_samples = collections.defaultdict(list)
  role_to_shape = {}
  for r in rows:
    for name, us in r.items():
      role, route, shape = _role_from_name(name)
      role_to_samples[role].append((us, route))
      role_to_shape.setdefault(role, shape)

  role_rows = {}
  for role, vals in role_to_samples.items():
    ms = _median([v[0] for v in vals]) / 1000.0
    route = vals[0][1] if vals else "non_gemm"
    role_rows[role] = {"ms": ms, "route": route, "shape": role_to_shape.get(role, "mixed"), "calls": len(vals)}

  total_ms = sum(v["ms"] for v in role_rows.values())
  return {"roles": role_rows, "profile_ms": total_ms}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("out")
  parser.add_argument("--contexts", default="512,1024,2048,4096,8192")
  parser.add_argument("--chunk", type=int, default=512)
  parser.add_argument("--repeats", type=int, default=5)
  parser.add_argument("--inner", type=int, default=8)
  parser.add_argument("--warmup", type=int, default=4)
  parser.add_argument("--profile-repeats", type=int, default=2)
  parser.add_argument("--seed", type=int, default=20260617)
  args = parser.parse_args()

  contexts = [int(x) for x in args.contexts.split(",") if x.strip()]
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  env = {
    "DEV": os.environ.get("DEV"),
    "JIT": os.environ.get("JIT"),
    "PREFILL_V2": os.environ.get("PREFILL_V2"),
    "PREFILL_GRAPH_GEMM": os.environ.get("PREFILL_GRAPH_GEMM"),
    "PREFILL_TENSILE_GEMM": os.environ.get("PREFILL_TENSILE_GEMM"),
    "PREFILL_GEMM_8WAVE": os.environ.get("PREFILL_GEMM_8WAVE"),
    "PREFILL_GEMM_DBUF": os.environ.get("PREFILL_GEMM_DBUF"),
    "PREFILL_GEMM_PLRA": os.environ.get("PREFILL_GEMM_PLRA"),
    "PREFILL_GEMM_PLRAB": os.environ.get("PREFILL_GEMM_PLRAB"),
    "PREFILL_CONCRETE_KV": os.environ.get("PREFILL_CONCRETE_KV"),
    "PREFILL_SERVER_PROFILE": os.environ.get("PREFILL_SERVER_PROFILE"),
  }

  model, _ = load_model_and_tokenizer(DEFAULT_MODEL, 4608, seed=args.seed)
  for b in model.blk:
    b._use_flash, b._prefill_v2 = True, True

  dev = Device[os.environ.get("DEV", "AMD")]
  chunk = Tensor([[(i * 7) % 1000 for i in range(args.chunk)]], dtype="int32").contiguous()
  temp = Tensor([0.0])

  forward = TinyJit(model.forward)
  forward_profile = TinyJit(model.forward)
  route_name = _route_tag()
  route_key = _route_key()

  root = Path(__file__).resolve().parents[1]
  authority = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "phase": "PREFILL_LONGCTX_INTEGRATION_HARDENING",
    "audit": "prefill long-context integration hardening",
    "command": (
      "DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python "
      f"extra/qk_prefill_integration_hardening.py --out {out} --contexts {','.join(map(str, contexts))}"
    ),
    "contexts": contexts,
    "chunk_tokens": args.chunk,
    "repeat": args.repeats,
    "inner_runs": args.inner,
    "warmup": args.warmup,
    "profile_repeats": args.profile_repeats,
    "model": str(DEFAULT_MODEL),
    "branch": _git(root, "branch", "--show-current"),
    "commit": _git(root, "rev-parse", "HEAD"),
    "dirty": _dirty_tree(root),
    "machine": {"name": "RX 7900 XTX", "arch": "gfx1100", "device": "AMD"},
    "env": env,
  }
  authority["artifacts"] = [
    {"file": "whole_prefill_by_ctx_raw.json", "type": "json"},
    {"file": "whole_prefill_chunk_series.json", "type": "json"},
    {"file": "single_chunk_vs_whole_prefill.json", "type": "json"},
    {"file": "runtime_overlap_by_ctx.json", "type": "json"},
    {"file": "per_role_time_tax_timeseries_by_ctx.json", "type": "json"},
    {"file": "route_coverage_by_ctx_and_role.json", "type": "json"},
    {"file": "kv_attention_split_timeseries.json", "type": "json"},
    {"file": "memory_pressure_watch.json", "type": "json"},
    {"file": "decision.json", "type": "json"},
  ]

  raw_by_ctx = {route_key: {}}
  chunk_rows = []
  single_vs_whole = []
  runtime_overlap_rows = []
  per_role_rows = []
  memory_rows = []
  kv_attention_rows = []
  role_totals_by_ctx: dict[int, list[dict]] = collections.defaultdict(list)

  for ctx in contexts:
    before_mem = _rocm_vram_json()
    start_positions = list(range(0, ctx, args.chunk))

    ctx_chunks = []
    profile_ms_sum = 0.0
    launch_sum = 0.0
    sync_sum = 0.0
    wall_sum = 0.0

    for sp in start_positions:
      wall_row = measure_chunk_wall(forward, dev, chunk, sp, temp, repeats=args.repeats, inner=args.inner, warmup=args.warmup)
      profile = measure_chunk_profile(forward_profile, dev, chunk, sp, temp, repeats=args.profile_repeats)

      roles_map = profile["roles"]
      profile_ms = profile["profile_ms"]
      profile_ms_sum += profile_ms
      launch_sum += max(0.0, wall_row["median_ms"] - profile_ms - wall_row["host_sync_ms"])
      sync_sum += wall_row["host_sync_ms"]
      wall_sum += wall_row["median_ms"]

      chunk_row = {
        "start_pos": sp,
        "median_ms": wall_row["median_ms"],
        "samples_ms": wall_row["samples_ms"],
        "raw": True,
        "launch_overhead_ms_raw": wall_row["launch_overhead_ms"],
        "launch_overhead_ms": max(0.0, wall_row["median_ms"] - profile_ms - wall_row["host_sync_ms"]),
        "host_sync_ms": wall_row["host_sync_ms"],
        "sync_calls": wall_row["sync_calls"],
        "gpu_only_ms": profile_ms,
      }
      ctx_chunks.append(chunk_row)
      chunk_rows.append({"route": route_key, "ctx": ctx, "start_pos": sp, **chunk_row})

      bucket = {
        "kv_proj_ms": 0.0,
        "attention_qk_ms": 0.0,
        "attention_pv_ms": 0.0,
        "copy_materialization_ms": 0.0,
        "other_ms": 0.0,
      }

      for role, r in roles_map.items():
        ms = r["ms"]
        role_row = {
          "ctx": ctx,
          "start_pos": sp,
          "role": role,
          "shape": r["shape"],
          "calls": r["calls"],
          "ms": ms,
          "share": 0.0,
          "route": r["route"],
          "authority": "TIMESERIES",
          "grows_with_ctx": False,
          "actionable": True,
        }
        if role == "kv_proj":
          bucket["kv_proj_ms"] += ms
        elif role == "attention_qk":
          bucket["attention_qk_ms"] += ms
        elif role == "attention_pv":
          bucket["attention_pv_ms"] += ms
        elif role == "copy_materialization":
          bucket["copy_materialization_ms"] += ms
        else:
          bucket["other_ms"] += ms
        role_row["actionable"] = role != "other"
        per_role_rows.append(role_row)
        role_totals_by_ctx[ctx].append(role_row)

      bucket["total_ms"] = bucket["kv_proj_ms"] + bucket["attention_qk_ms"] + bucket["attention_pv_ms"] + bucket["copy_materialization_ms"] + bucket["other_ms"]
      kv_attention_rows.append({"ctx": ctx, "start_pos": sp, **bucket})

    total_ms = sum(r["median_ms"] for r in ctx_chunks)
    whole = {
      "tok_s": int(round(1000.0 * ctx / total_ms)) if total_ms else None,
      "tok_s_raw": 1000.0 * ctx / total_ms if total_ms else None,
      "ms_per_token": total_ms / ctx if ctx else None,
      "total_ms": total_ms,
      "repeats": args.repeats,
      "inner_runs": args.inner,
      "sampled_chunks": len(ctx_chunks),
      "extrapolated_chunks": 0,
    }

    if ctx_chunks:
      first = ctx_chunks[0]
      whole_tok = 1000.0 * ctx / total_ms if total_ms else None
      single_tok = 1000.0 * args.chunk / first["median_ms"] if first["median_ms"] else None
      ratio = (single_tok / whole_tok) if (single_tok is not None and whole_tok is not None) else None
      single_vs_whole.append({
        "route": route_key,
        "ctx": ctx,
        "start_pos": 0,
        "single_chunk_tok_s": single_tok,
        "whole_prefill_tok_s": whole["tok_s"],
        "ratio_single_to_whole": ratio,
        "single_chunk_ms": first["median_ms"],
        "whole_prefill_ms": total_ms,
        "samples": first["samples_ms"],
        "repeated_chunks": True,
      })

    raw_by_ctx[route_key][str(ctx)] = {
      "lane": f"whole_prefill_synced {route_name}",
      "start_positions": start_positions,
      "chunk_series": ctx_chunks,
      "whole_prefill": whole,
    }

    runtime_overlap_rows.append({
      "ctx": ctx,
      "wall_ms": wall_sum,
      "gpu_only_ms": profile_ms_sum,
      "host_sync_ms": sync_sum,
      "launch_overhead_ms": launch_sum,
      "sync_calls": args.repeats * len(start_positions),
      "chunks": len(start_positions),
      "wall_repeats": args.repeats,
      "repeat": args.repeats,
    })

    after_mem = _rocm_vram_json()
    before_used, before_total = _extract_vram_stats(before_mem)
    after_used, after_total = _extract_vram_stats(after_mem)
    mem_used_delta = None if before_used is None or after_used is None else after_used - before_used
    memory_rows.append({
      "ctx": ctx,
      "before_used_bytes": before_used,
      "before_total_bytes": before_total,
      "after_used_bytes": after_used,
      "after_total_bytes": after_total,
      "before_bytes": before_used,
      "after_bytes": after_used,
      "delta_bytes": mem_used_delta,
      "profile_events_count": len(getattr(Compiled, "profile_events", [])),
    })

  # per role shares and growth flags
  role_total_ctx: dict[tuple[int, int], float] = {}
  for r in per_role_rows:
    role_total_ctx[(r["ctx"], r["start_pos"])] = role_total_ctx.get((r["ctx"], r["start_pos"]), 0.0) + r["ms"]
  share_rows = []
  for r in per_role_rows:
    denom = role_total_ctx.get((r["ctx"], r["start_pos"]), 0.0)
    r["share"] = (r["ms"] / denom) if denom > 0 else 0.0
    share_rows.append(r)

  # grows_with_ctx: compare median ms across ctx for each role (start_pos-agnostic)
  role_ctx = collections.defaultdict(list)
  for r in share_rows:
    role_ctx[r["role"]].append((r["ctx"], r["ms"], r["start_pos"]))

  grows = {}
  for role, vals in role_ctx.items():
    per_ctx = collections.defaultdict(list)
    for ctx, ms, _ in vals:
      per_ctx[ctx].append(ms)
    by_ctx = sorted((ctx, _median(ms_list)) for ctx, ms_list in per_ctx.items())
    if len(by_ctx) >= 2:
      first_ctx, first_val = by_ctx[0]
      last_ctx, last_val = by_ctx[-1]
      grows[role] = abs(_pct_change(first_val, last_val)) > 5.0
    else:
      grows[role] = False

  for r in share_rows:
    r["grows_with_ctx"] = bool(grows.get(r["role"], False))

  # route coverage by context/role
  route_rows = []
  for ctx, entries in sorted(role_totals_by_ctx.items()):
    role_map = {}
    for e in entries:
      rec = role_map.setdefault(e["role"], {"role": e["role"], "route": e["route"], "ms": 0.0})
      rec["ms"] += e["ms"]
    route_rows.append({
      "ctx": ctx,
      "roles": [{"role": rec["role"], "route": rec["route"], "actionable": rec["ms"] > 0.0} for rec in sorted(role_map.values(), key=lambda x: x["role"])],
    })

  decision_label = "PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED"
  if runtime_overlap_rows:
    c = runtime_overlap_rows[-1]
    if c["wall_ms"] > 0:
      gap_ratio = max(0.0, c["wall_ms"] - c["gpu_only_ms"]) / c["wall_ms"]
      if gap_ratio > 0.45:
        decision_label = "PREFILL_LONGCTX_INTEGRATION_HARDENING_HOSTSYNC_BOUND"
      elif c["launch_overhead_ms"] / c["wall_ms"] > 0.20:
        decision_label = "PREFILL_LONGCTX_INTEGRATION_HARDENING_DISPATCH_BOUND"

  if decision_label == "PREFILL_LONGCTX_INTEGRATION_HARDENING_NO_GROWTH_CONFIRMED":
    # attention/copy bound if final-ctx split buckets are clearly non-zero
    kf_rows = [r for r in kv_attention_rows if r["ctx"] == contexts[-1]]
    if kf_rows:
      max_attention = max(max(r.get("attention_qk_ms", 0.0), r.get("attention_pv_ms", 0.0)) for r in kf_rows)
      max_copy = max(r.get("copy_materialization_ms", 0.0) for r in kf_rows)
      if max_attention > 0.0 or max_copy > 0.0:
        decision_label = "PREFILL_LONGCTX_INTEGRATION_HARDENING_ATTENTION_COPY_BOUND"

  decision = {
    "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "phase": "PREFILL_LONGCTX_INTEGRATION_HARDENING_DECISION",
    "label": decision_label,
    "rationale": {
      "route": route_name,
      "contexts": contexts,
      "ctx8192_wall_ms": runtime_overlap_rows[-1]["wall_ms"] if runtime_overlap_rows else None,
      "ctx8192_gpu_only_ms": runtime_overlap_rows[-1]["gpu_only_ms"] if runtime_overlap_rows else None,
      "ctx8192_host_sync_ms": runtime_overlap_rows[-1]["host_sync_ms"] if runtime_overlap_rows else None,
      "ctx8192_launch_overhead_ms": runtime_overlap_rows[-1]["launch_overhead_ms"] if runtime_overlap_rows else None,
    },
    "next_step": "NONSEARCH_INTEGRATION_FIX_SCOPE",
    "action": {
      "primary": decision_label,
      "requires": "runtime_decomposition_probe" if decision_label.endswith("BOUND") else "confirm_integration_fix",
    },
  }

  (out / "whole_prefill_by_ctx_raw.json").write_text(json.dumps(raw_by_ctx, indent=2))
  (out / "whole_prefill_chunk_series.json").write_text(json.dumps(chunk_rows, indent=2))
  (out / "single_chunk_vs_whole_prefill.json").write_text(json.dumps(single_vs_whole, indent=2))
  (out / "runtime_overlap_by_ctx.json").write_text(json.dumps(runtime_overlap_rows, indent=2))
  (out / "per_role_time_tax_timeseries_by_ctx.json").write_text(json.dumps({"count": len(share_rows), "rows": share_rows}, indent=2))
  (out / "route_coverage_by_ctx_and_role.json").write_text(json.dumps(route_rows, indent=2))
  (out / "kv_attention_split_timeseries.json").write_text(json.dumps(kv_attention_rows, indent=2))
  (out / "memory_pressure_watch.json").write_text(json.dumps(memory_rows, indent=2))
  (out / "authority.json").write_text(json.dumps(authority, indent=2))
  (out / "decision.json").write_text(json.dumps(decision, indent=2))

  print(json.dumps({"ok": True, "out": str(out), "route": route_key, "label": decision_label}, sort_keys=True))


if __name__ == "__main__":
  main()
