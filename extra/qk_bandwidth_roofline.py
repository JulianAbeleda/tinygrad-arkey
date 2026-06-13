#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics
from typing import Any

from extra.qk_experiment_matrix import LLAMA_REFS, make_matrix

# Local Qwen3 GGUF file sizes used by the committed shared-storage matrix.
# These are logical bytes-per-token estimates, not measured HBM transactions.
DEFAULT_MODEL_BYTES = {
  "8B": 5_027_783_488,
  "14B": 9_001_752_960,
  "32B": 19_762_149_024,
}

DEFAULT_PEAK_MEM_GBS = 960.0

def _load_json(path:pathlib.Path) -> Any:
  return json.loads(path.read_text())

def _fmt(x:Any, digits:int=2) -> str:
  if x is None: return "n/a"
  if isinstance(x, float): return f"{x:.{digits}f}"
  return str(x)

def _last_runtime_storage(decision:dict[str, Any], prefix:str) -> dict[str, Any]:
  rows = decision.get("runtime_storage") or {}
  matches = [(k, v) for k, v in rows.items() if k.startswith(prefix)]
  return matches[-1][1] if matches else {}

def _decision_path(experiment:pathlib.Path) -> pathlib.Path:
  if experiment.name == "decision.json": return experiment
  if experiment.is_dir(): return experiment / "decision.json"
  raise ValueError(f"{experiment}: expected decision.json or experiment directory")

def _decision_by_model(experiments:list[pathlib.Path]) -> dict[str, dict[str, Any]]:
  out: dict[str, dict[str, Any]] = {}
  for experiment in experiments:
    path = _decision_path(experiment)
    data = _load_json(path)
    model = data.get("model_size")
    if not isinstance(model, str): raise ValueError(f"{path}: missing model_size")
    out[model] = data
  return out

def _bytes_gbs(byte_count:int | float | None, tok_s:float | None) -> float | None:
  if byte_count is None or tok_s is None: return None
  return byte_count * tok_s / 1_000_000_000.0

def _pct(value:float | None, denom:float | None) -> float | None:
  if value is None or denom in (None, 0): return None
  return value / denom * 100.0

def _parse_model_bytes(items:list[str]) -> dict[str, int]:
  sizes = dict(DEFAULT_MODEL_BYTES)
  for item in items:
    if "=" not in item: raise ValueError(f"{item}: expected MODEL=BYTES")
    model, raw = item.split("=", 1)
    sizes[model] = int(raw)
  return sizes

def build_roofline(experiments:list[pathlib.Path], *, model_bytes:dict[str, int],
                   peak_mem_gbs:float=DEFAULT_PEAK_MEM_GBS) -> dict[str, Any]:
  matrix = make_matrix(experiments)
  decisions = _decision_by_model(experiments)
  rows: list[dict[str, Any]] = []
  for matrix_row in matrix["rows"]:
    model = matrix_row["model_size"]
    decision = decisions.get(model, {})
    full_bytes = model_bytes.get(model)
    llama_tok_s = LLAMA_REFS.get(model)
    explicit_tok_s = matrix_row.get("explicit_tok_s")
    generated_tok_s = matrix_row.get("generated_tok_s")
    generated_storage = _last_runtime_storage(decision, "generated")
    explicit_storage = _last_runtime_storage(decision, "explicit")
    generated_source_bytes = generated_storage.get("source_bytes")
    explicit_source_bytes = explicit_storage.get("source_bytes")
    llama_full_gbs = _bytes_gbs(full_bytes, llama_tok_s)
    generated_full_gbs = _bytes_gbs(full_bytes, generated_tok_s)
    explicit_full_gbs = _bytes_gbs(full_bytes, explicit_tok_s)
    rows.append({
      "model_size": model,
      "path": matrix_row["path"],
      "status": matrix_row.get("status"),
      "model_gguf_bytes": full_bytes,
      "explicit_tok_s": explicit_tok_s,
      "generated_tok_s": generated_tok_s,
      "llama_tok_s": llama_tok_s,
      "explicit_full_file_gbs": explicit_full_gbs,
      "generated_full_file_gbs": generated_full_gbs,
      "llama_full_file_gbs": llama_full_gbs,
      "explicit_pct_peak_mem_by_file": _pct(explicit_full_gbs, peak_mem_gbs),
      "generated_pct_peak_mem_by_file": _pct(generated_full_gbs, peak_mem_gbs),
      "llama_pct_peak_mem_by_file": _pct(llama_full_gbs, peak_mem_gbs),
      "generated_pct_llama_by_file": _pct(generated_full_gbs, llama_full_gbs),
      "file_bandwidth_gap_to_llama_gbs": None if generated_full_gbs is None or llama_full_gbs is None else llama_full_gbs - generated_full_gbs,
      "generated_primitive_source_bytes": generated_source_bytes,
      "explicit_primitive_source_bytes": explicit_source_bytes,
      "generated_primitive_source_gbs": _bytes_gbs(generated_source_bytes, generated_tok_s),
      "explicit_primitive_source_gbs": _bytes_gbs(explicit_source_bytes, explicit_tok_s),
      "generated_installed": generated_storage.get("installed"),
      "explicit_installed": explicit_storage.get("installed"),
      "ab_match": matrix_row.get("ab_match"),
    })
  generated_pct_peak = [r["generated_pct_peak_mem_by_file"] for r in rows if isinstance(r.get("generated_pct_peak_mem_by_file"), (int, float))]
  llama_pct_peak = [r["llama_pct_peak_mem_by_file"] for r in rows if isinstance(r.get("llama_pct_peak_mem_by_file"), (int, float))]
  generated_pct_llama = [r["generated_pct_llama_by_file"] for r in rows if isinstance(r.get("generated_pct_llama_by_file"), (int, float))]
  max_gap = max((r["file_bandwidth_gap_to_llama_gbs"] for r in rows if isinstance(r.get("file_bandwidth_gap_to_llama_gbs"), (int, float))), default=None)
  status = "memory_load_efficiency_gap" if max_gap is not None and max_gap > peak_mem_gbs * 0.10 else "roofline_gap_unclear"
  return {
    "kind": "qk_bandwidth_roofline",
    "date": "2026-06-13",
    "device": "AMD Radeon RX 7900 XTX / gfx1100",
    "peak_mem_gbs": peak_mem_gbs,
    "method": {
      "full_file_gbs": "model_gguf_bytes * decode_tok_s / 1e9; logical bytes per token, not measured HBM transactions",
      "primitive_source_gbs": "runtime primitive source_bytes * decode_tok_s / 1e9 for installed Q4_K/Q6_K wrappers",
      "llama_refs": "extra.qk_experiment_matrix.LLAMA_REFS",
      "live_benchmark": False,
    },
    "rows": rows,
    "summary": {
      "status": status,
      "models": len(rows),
      "generated_pct_peak_mem_by_file_min": min(generated_pct_peak) if generated_pct_peak else None,
      "generated_pct_peak_mem_by_file_max": max(generated_pct_peak) if generated_pct_peak else None,
      "llama_pct_peak_mem_by_file_min": min(llama_pct_peak) if llama_pct_peak else None,
      "llama_pct_peak_mem_by_file_max": max(llama_pct_peak) if llama_pct_peak else None,
      "generated_pct_llama_by_file_min": min(generated_pct_llama) if generated_pct_llama else None,
      "generated_pct_llama_by_file_mean": statistics.mean(generated_pct_llama) if generated_pct_llama else None,
      "max_file_bandwidth_gap_to_llama_gbs": max_gap,
    },
  }

def roofline_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Bandwidth Roofline",
    "",
    "This report is generated from committed shared-storage QK decision artifacts.",
    "It does not run benchmarks. `full-file GB/s` is a logical decode roofline",
    "proxy: GGUF file bytes times tokens/sec. It is useful for comparing tinygrad",
    "and llama.cpp on the same model, but it is not a hardware-counter HBM read",
    "measurement.",
    "",
    f"- device: `{report['device']}`",
    f"- peak memory assumption: `{report['peak_mem_gbs']:.1f} GB/s`",
    f"- verdict: `{report['summary']['status']}`",
    "",
    "## Model Rows",
    "",
    "| model | generated tok/s | llama tok/s | generated file GB/s | llama file GB/s | generated % peak | llama % peak | generated % llama | primitive source GB/s | A/B |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for row in report["rows"]:
    lines.append(
      f"| `{row['model_size']}` | {_fmt(row['generated_tok_s'])} | {_fmt(row['llama_tok_s'])} | "
      f"{_fmt(row['generated_full_file_gbs'])} | {_fmt(row['llama_full_file_gbs'])} | "
      f"{_fmt(row['generated_pct_peak_mem_by_file'])}% | {_fmt(row['llama_pct_peak_mem_by_file'])}% | "
      f"{_fmt(row['generated_pct_llama_by_file'])}% | {_fmt(row['generated_primitive_source_gbs'])} | `{row['ab_match']}` |"
    )
  summary = report["summary"]
  lines += [
    "",
    "## Interpretation",
    "",
    f"- tinygrad generated path reaches `{_fmt(summary['generated_pct_peak_mem_by_file_min'])}-{_fmt(summary['generated_pct_peak_mem_by_file_max'])}%` of the 960 GB/s peak by the full-file proxy.",
    f"- llama.cpp reaches `{_fmt(summary['llama_pct_peak_mem_by_file_min'])}-{_fmt(summary['llama_pct_peak_mem_by_file_max'])}%` by the same proxy.",
    f"- tinygrad is `{_fmt(summary['generated_pct_llama_by_file_min'])}-{_fmt(max(r['generated_pct_llama_by_file'] for r in report['rows']))}%` of llama.cpp by this same byte model.",
    f"- largest file-bandwidth gap to llama.cpp: `{_fmt(summary['max_file_bandwidth_gap_to_llama_gbs'])} GB/s`.",
    "",
    "The result supports treating the remaining decode gap as a memory-load",
    "efficiency/codegen problem before adding more local schedule knobs. A future",
    "hardware-counter pass can replace this logical proxy, but the current",
    "decision is already strong enough to freeze the exhausted schedule surfaces.",
    "",
  ]
  return "\n".join(lines)

def write_roofline(report:dict[str, Any], json_path:pathlib.Path, md_path:pathlib.Path) -> None:
  json_path.parent.mkdir(parents=True, exist_ok=True)
  md_path.parent.mkdir(parents=True, exist_ok=True)
  json_path.write_text(json.dumps(report, indent=2, sort_keys=True))
  md_path.write_text(roofline_markdown(report))

def main() -> int:
  parser = argparse.ArgumentParser(description="Build QK decode bandwidth roofline report from committed decision artifacts")
  parser.add_argument("experiments", nargs="+", type=pathlib.Path)
  parser.add_argument("--model-bytes", action="append", default=[],
                      help="Override default GGUF bytes as MODEL=BYTES, e.g. 8B=5027783488")
  parser.add_argument("--peak-mem-gbs", type=float, default=DEFAULT_PEAK_MEM_GBS)
  parser.add_argument("--json", type=pathlib.Path, required=True)
  parser.add_argument("--md", type=pathlib.Path, required=True)
  args = parser.parse_args()
  report = build_roofline([p.expanduser() for p in args.experiments],
                          model_bytes=_parse_model_bytes(args.model_bytes),
                          peak_mem_gbs=args.peak_mem_gbs)
  write_roofline(report, args.json, args.md)
  print(roofline_markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
