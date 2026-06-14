#!/usr/bin/env python3
from __future__ import annotations

import argparse, copy, json, pathlib, re
from collections import Counter
from typing import Any

from extra import qk_flywheel_dataset as v0
from extra import qk_flywheel_dataset_v1 as v1
from extra.qk_flywheel_coverage_plan import write_coverage_plan
from extra.qk_flywheel_feature_audit import run_feature_audit
from extra.qk_flywheel_feature_enrich import enrich_row

DEFAULT_ROOT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614")
DEFAULT_BASE = DEFAULT_ROOT / "kernel-triage-v1-featured/examples.jsonl"
DEFAULT_TARGETED_OUT = DEFAULT_ROOT / "targeted-outcomes-v1"
DEFAULT_PLUS_OUT = DEFAULT_ROOT / "kernel-triage-v1-featured-plus"
DEFAULT_AUDIT_OUT = DEFAULT_ROOT / "triage-feature-audit-v1-featured-plus"
DEFAULT_COVERAGE_OUT = DEFAULT_ROOT / "triage-coverage-plan-v1-plus"

TARGETED_FAMILY = "targeted_outcomes_v1"
TARGETED_FAMILY_ORDER = 11
SEMANTIC_CODEGEN_FAMILIES = (
  ("semantic-codegen-v3", "semantic_codegen_v3"),
  ("semantic-codegen-v4", "semantic_codegen_v4"),
)


def _read_json(path:pathlib.Path) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if not isinstance(data, dict): raise ValueError(f"{path}: expected JSON object")
  return data

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    try:
      row = json.loads(raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    rows.append(row)
  return rows

def _jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def _portable(repo:pathlib.Path, path:pathlib.Path) -> str:
  try:
    return str(path.resolve().relative_to(repo.resolve()))
  except ValueError:
    return str(path)

def _repo_path(repo:pathlib.Path, value:str) -> pathlib.Path:
  path = pathlib.Path(value)
  return path if path.is_absolute() else repo / path

def _source(repo:pathlib.Path, value:str) -> str:
  path = _repo_path(repo, value)
  if not path.exists(): raise FileNotFoundError(path)
  return _portable(repo, path)

def _row(*, row_id:str, row_kind:str, model:str, tensor:str, role:str, fmt:str,
         mechanism:str, prediction_stage:str, pre_result_context:dict[str, Any],
         label:str, reason:str, retry:bool, evidence:dict[str, Any], source_files:list[str]) -> dict[str, Any]:
  if label not in v0.LABELS: raise ValueError(f"{row_id}: unknown label {label}")
  if reason not in v0.REASONS: raise ValueError(f"{row_id}: unknown reason {reason}")
  return {
    "id": row_id,
    "candidate_id": row_id.split(":", 1)[-1],
    "row_kind": row_kind,
    "family": TARGETED_FAMILY,
    "family_order": TARGETED_FAMILY_ORDER,
    "model": model,
    "tensor": tensor,
    "role": role,
    "format": fmt,
    "mechanism": mechanism,
    "prediction_stage": prediction_stage,
    "pre_result_context": pre_result_context,
    "label": label,
    "reason": reason,
    "retry": retry,
    "evidence": evidence,
    "source_files": source_files,
    "split": "train",
  }


def _semantic_codegen_mechanism(schedule:dict[str, Any], candidate_id:str) -> str:
  return v0._mechanism_from_id(candidate_id, "semantic_codegen", schedule)


def _canonical_model(value:str) -> str:
  text = str(value or "").lower().replace("qwen3-", "").replace("qwen-", "")
  if "14b" in text:
    return "Qwen3-14B-Q4_K_M"
  if "8b" in text:
    return "Qwen3-8B-Q4_K_M"
  if text in ("14", "14b", "14-b"):
    return "Qwen3-14B-Q4_K_M"
  if text in ("8", "8b", "8-b"):
    return "Qwen3-8B-Q4_K_M"
  return str(value)


def _mechanism_from_schedule(schedule_name:str, row_id:str) -> str:
  text = f"{schedule_name} {row_id}".lower().replace("-", "_")
  if "row_upcast" in text:
    return "row_upcast"
  if "reduce_unroll" in text:
    return "reduce_unroll"
  if "two_dim_local" in text:
    return "two_dim_local"
  if "direct_out" in text:
    return "direct_output"
  return "unknown"


def _tensor_from_row_id(value:str) -> str:
  text = re.sub(r"[.]", "-", str(value or "").lower())
  match = re.search(r"\bblk[-_](\d+)[-_]([a-z0-9_-]+?)-weight\b", text)
  if not match:
    return "unknown"
  block, role = match.groups()
  return f"blk.{block}.{role.replace('-', '_')}.weight"

def _mode(rows:list[dict[str, Any]], mode:str) -> dict[str, Any]:
  for row in rows:
    if row.get("mode") == mode: return row
  raise ValueError(f"missing mode {mode!r}")

def _memory_access_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  vector_probe = _source(repo, "bench/qk-memory-access-20260613/vector-probe.json")
  load_width = _source(repo, "bench/qk-memory-access-20260613/load-width/report.json")
  audit = _source(repo, "bench/qk-memory-access-20260613/audit.json")
  probe = _read_json(repo / vector_probe)
  audit_data = _read_json(repo / audit)
  rows = []
  for mode, mechanism, retry in (
    ("uop_vec_request", "vector_load", True),
    ("custom_uint4", "wide_load_only", False),
  ):
    item = _mode(probe.get("rows", []), mode)
    rows.append(_row(
      row_id=f"{TARGETED_FAMILY}:memory_access:{mode}",
      row_kind="diagnostic",
      model="unknown",
      tensor="unknown",
      role="unknown",
      fmt="unknown",
      mechanism=mechanism,
      prediction_stage="after_static_before_microbench",
      pre_result_context={
        "mode": mode,
        "load_width": {"inferred": "vector_u32x4", "report": "vector_u32x4"},
        "source_ok": bool(item.get("copy_exact")),
        "wide_loads": bool(item.get("copy_exact")),
        "full_decode_ready": False,
        "shape": {"k": probe.get("n_words", 0), "parts": 1},
      },
      label="diagnostic_only",
      reason="diagnostic_only",
      retry=retry,
      evidence={
        "copy_exact": item.get("copy_exact"),
        "mode": mode,
        "audit_status": audit_data.get("decision", {}).get("status"),
        "run_family_c_v1_now": audit_data.get("decision", {}).get("run_family_c_v1_now"),
      },
      source_files=[vector_probe, load_width, audit],
    ))
  rows.append(_row(
    row_id=f"{TARGETED_FAMILY}:memory_access:family_c_v1_source_supported",
    row_kind="diagnostic",
    model="unknown",
    tensor="unknown",
    role="unknown",
    fmt="unknown",
    mechanism="vector_load",
    prediction_stage="after_static_before_microbench",
    pre_result_context={
      "mode": "uop_vec_request",
      "load_width": {"inferred": "vector_u32x4", "report": "vector_u32x4"},
      "source_ok": bool(audit_data.get("decision", {}).get("run_family_c_v1_now")),
      "wide_loads": bool((audit_data.get("probe_load_width_summary") or {}).get("has_vector_load_evidence")),
      "full_decode_ready": False,
      "shape": {"k": probe.get("n_words", 0), "parts": 1},
    },
    label="diagnostic_only",
    reason="diagnostic_only",
    retry=True,
    evidence={
      "decision": audit_data.get("decision", {}).get("status"),
      "next_required_change": audit_data.get("decision", {}).get("next_required_change"),
      "run_family_c_v1_now": audit_data.get("decision", {}).get("run_family_c_v1_now"),
    },
    source_files=[audit, vector_probe, load_width],
  ))
  return rows

def _packed_tile_consumption_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  probe_path = _source(repo, "bench/qk-packed-tile-consumption-20260613/probe.json")
  load_width = _source(repo, "bench/qk-packed-tile-consumption-20260613/load-width/report.json")
  probe = _read_json(repo / probe_path)
  tile = ((probe.get("packed_qk") or {}).get("tile") or {})
  shape = tile.get("shape") if isinstance(tile.get("shape"), list) else []
  rows = []
  for mode in ("uop_lane_gep", "uop_vector_arith"):
    item = _mode(probe.get("rows", []), mode)
    rows.append(_row(
      row_id=f"{TARGETED_FAMILY}:packed_tile_consumption:{mode}",
      row_kind="candidate",
      model="Qwen3-8B-Q4_K_M",
      tensor=str(tile.get("tensor") or "blk.0.ffn_gate.weight"),
      role=str(tile.get("role") or "ffn_gate"),
      fmt="Q4_K",
      mechanism="vector_load",
      prediction_stage="after_static_before_microbench",
      pre_result_context={
        "mode": mode,
        "load_tile": (probe.get("packed_qk") or {}).get("required_load_tile"),
        "shape": {"rows": int(shape[0]) if len(shape) > 0 else 0, "k": int(shape[1]) if len(shape) > 1 else 0, "parts": 1},
        "source_ok": False,
        "wide_loads": True,
        "full_decode_ready": False,
      },
      label="construction_blocked",
      reason="construction_blocked",
      retry=False,
      evidence={
        "mode": mode,
        "status": item.get("status"),
        "error_type": item.get("error_type"),
        "error_message": item.get("error_message"),
      },
      source_files=[probe_path],
    ))
  custom = _mode(probe.get("rows", []), "custom_q4_dot")
  rows.append(_row(
    row_id=f"{TARGETED_FAMILY}:packed_tile_consumption:custom_q4_dot",
    row_kind="diagnostic",
    model="Qwen3-8B-Q4_K_M",
    tensor=str(tile.get("tensor") or "blk.0.ffn_gate.weight"),
    role=str(tile.get("role") or "ffn_gate"),
    fmt="Q4_K",
    mechanism="qk_block_dot",
    prediction_stage="after_static_before_microbench",
    pre_result_context={
      "mode": "packed_tile_custom_q4_dot",
      "load_tile": (probe.get("packed_qk") or {}).get("required_load_tile"),
      "shape": {"rows": int(shape[0]) if len(shape) > 0 else 0, "k": int(shape[1]) if len(shape) > 1 else 0, "parts": 1},
      "source_ok": bool(custom.get("source_contains_uint4_load")),
      "wide_loads": bool(custom.get("source_contains_uint4_load")),
      "full_decode_ready": False,
    },
    label="diagnostic_only",
    reason="diagnostic_only",
    retry=True,
    evidence={
      "mode": "custom_q4_dot",
      "status": custom.get("status"),
      "exact": custom.get("exact"),
      "decision": (probe.get("summary") or {}).get("decision"),
      "next_path": (probe.get("summary") or {}).get("next_path"),
    },
    source_files=[probe_path, load_width],
  ))
  return rows

def _packed_tile_closeout_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  diag_path = _source(repo, "bench/qk-packed-tile-research-closeout-20260613/diagnostic.json")
  load_width = _source(repo, "bench/qk-packed-tile-research-closeout-20260613/source/load-width-report.json")
  diag = _read_json(repo / diag_path)
  shape = diag.get("shape") if isinstance(diag.get("shape"), dict) else {}
  summary = diag.get("summary") if isinstance(diag.get("summary"), dict) else {}
  return [_row(
    row_id=f"{TARGETED_FAMILY}:packed_tile_closeout:tile_custom_disasm",
    row_kind="diagnostic",
    model="Qwen3-8B-Q4_K_M",
    tensor=str(shape.get("tensor") or "blk.0.ffn_gate.weight"),
    role="ffn_gate",
    fmt="Q4_K",
    mechanism="wide_load_only",
    prediction_stage="after_compile_before_microbench",
    pre_result_context={
      "mode": "tile_custom_partial",
      "reference": "v1_partial",
      "shape": {"rows": int(shape.get("rows") or 0), "k": int(shape.get("k") or 0), "parts": int(shape.get("parts") or 0)},
      "source_ok": bool(summary.get("tile_has_wider_loads")),
      "wide_loads": bool(summary.get("tile_has_wider_loads")),
      "full_decode_ready": False,
    },
    label="diagnostic_only",
    reason="diagnostic_only",
    retry=False,
    evidence={
      "decision": summary.get("decision"),
      "tile_has_wider_loads": summary.get("tile_has_wider_loads"),
      "tile_loses_parallelism": summary.get("tile_loses_parallelism"),
      "tile_larger_body": summary.get("tile_larger_body"),
      "next_allowed_path": summary.get("next_allowed_path"),
    },
      source_files=[diag_path, load_width],
  )]


def _qk_block_dot_microbench_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  microbench = _source(repo, "bench/qk-block-dot-microbench-20260613/microbench.json")
  data = _read_json(repo / microbench)
  compile_gate = repo / "bench/qk-block-dot-compile-gate-20260613/compile-gate.json"
  rows = []
  rows.append(_row(
    row_id=f"{TARGETED_FAMILY}:qk_block_dot:compile_gate",
    row_kind="diagnostic",
    model="Qwen3-8B-Q4_K_M",
    tensor=str((data.get("config") or {}).get("tensor") or "blk.0.ffn_gate.weight"),
    role="ffn_gate",
    fmt="Q4_K",
    mechanism="qk_block_dot",
    prediction_stage="after_compile_before_microbench",
    pre_result_context={
      "mode": "compile_gate",
      "shape": (data.get("config") or {}).get("shape"),
      "parts": (data.get("config") or {}).get("parts") or 0,
      "wide_loads": True,
      "source_ok": True,
      "full_decode_ready": False,
    },
    label="diagnostic_only",
    reason="diagnostic_only",
    retry=True,
    evidence={
      "decision": data.get("summary", {}).get("decision"),
      "next_allowed_gate": data.get("summary", {}).get("next_allowed_gate"),
      "source_compile_gate": "bench/qk-block-dot-compile-gate-20260613/compile-gate.json",
    },
    source_files=[microbench, _portable(repo, compile_gate)] if compile_gate.exists() else [microbench],
  ))

  cmp = data.get("comparison", {})
  comparison_gain = cmp.get("gain_pct")
  label, reason, retry = v0._label_reason_retry("reject", gain=comparison_gain / 100.0 if comparison_gain is not None else None)
  rows.append(_row(
    row_id=f"{TARGETED_FAMILY}:qk_block_dot_microbench:qk_block_dot",
    row_kind="candidate",
    model="Qwen3-8B-Q4_K_M",
    tensor=str((data.get("config") or {}).get("tensor") or "blk.0.ffn_gate.weight"),
    role="ffn_gate",
    fmt="Q4_K",
    mechanism="qk_block_dot",
    prediction_stage="after_compile_before_microbench",
    pre_result_context={
      "mode": "qk_block_dot",
      "reference": "v1_partial",
      "comparison": cmp,
      "shape": (data.get("config") or {}).get("shape"),
      "parts": (data.get("config") or {}).get("parts") or 0,
      "wide_loads": True,
      "full_decode_ready": False,
    },
    label=label,
    reason=reason,
    retry=retry,
    evidence={
      "decision": data.get("summary", {}).get("decision"),
      "gain_pct": comparison_gain,
      "correctness_ok": data.get("summary", {}).get("correctness_ok"),
      "next_allowed_gate": data.get("summary", {}).get("next_allowed_gate"),
      "run_full_decode": data.get("summary", {}).get("run_full_decode"),
    },
    source_files=[microbench],
  ))

  return rows


def _semantic_codegen_microbench_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  base = repo / "bench/qk-ansor-transition-20260612"
  for family_dir, _family_name in SEMANTIC_CODEGEN_FAMILIES:
    for model_dir in ("8b", "14b"):
      microbench = base / family_dir / model_dir / "microbench.json"
      if not microbench.exists():
        continue
      data = _read_json(microbench)
      model = _canonical_model(model_dir)
      source = _portable(repo, microbench)
      for item in data.get("rows", []):
        candidate_id = str(item.get("id") or "")
        schedule = (item.get("schedule") or {}) if isinstance(item.get("schedule"), dict) else {}
        status = item.get("status")
        gain = item.get("gain")
        label, reason, retry = v0._label_reason_retry(status, gain=gain)
        rows.append(_row(
          row_id=f"{TARGETED_FAMILY}:semantic_codegen:{model_dir}:{candidate_id}",
          row_kind="candidate",
          model=model,
          tensor=str(item.get("tensor") or "unknown"),
          role=str(item.get("role") or "unknown"),
          fmt=str(item.get("format") or "Q4_K"),
          mechanism=_semantic_codegen_mechanism(schedule, candidate_id),
          prediction_stage="after_static_before_microbench",
          pre_result_context={
            "mode": (schedule.get("codegen_mode") or schedule.get("name") or "semantic_codegen"),
            "candidate_id": candidate_id,
            "model_dir": model_dir,
            "family": family_dir,
            "source_candidates": f"bench/qk-shared-storage-20260612/{model_dir}/policy.json",
            "schedule": schedule,
          },
          label=label,
          reason=reason,
          retry=retry,
          evidence={
            "status": status,
            "gain": gain,
            "reasons": item.get("reasons", []),
            "quant_gbs": item.get("quant_gbs"),
            "full_decode_supported": item.get("full_decode_supported"),
            "candidate_status": (item.get("candidate") or {}).get("status"),
          },
          source_files=[source],
        ))
  return rows


def _qk_threeway_load_microbench_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  microbench = _source(repo, "bench/qk-threeway-load-microbench-20260613/microbench.json")
  data = _read_json(repo / microbench)
  rows = []
  for tensor_row in data.get("tensors", []) or []:
    tensor = str(tensor_row.get("tensor") or "unknown")
    tensor_slug = re.sub(r"[^a-z0-9]+", "-", tensor.lower()).strip("-")
    gains = tensor_row.get("gains_pct", {}) or {}
    for mode_row in tensor_row.get("modes", []) or []:
      mode = str(mode_row.get("mode") or "unknown")
      if mode == "v1_partial":
        rows.append(_row(
          row_id=f"{TARGETED_FAMILY}:threeway_load_microbench:{tensor_slug}:{mode}",
          row_kind="diagnostic",
          model="Qwen3-8B-Q4_K_M",
          tensor=tensor,
          role="ffn_gate",
          fmt="Q4_K",
          mechanism="wide_load_only",
          prediction_stage="after_compile_before_microbench",
          pre_result_context={
            "mode": mode,
            "reference": "qk-block-dot",
            "decision": tensor_row.get("decision"),
            "shape": (data.get("config") or {}).get("shape"),
            "runs": (data.get("config") or {}).get("runs") or 0,
            "wide_loads": True,
            "full_decode_ready": False,
          },
          label="diagnostic_only",
          reason="diagnostic_only",
          retry=False,
          evidence={"status": "pass", "decision": tensor_row.get("decision"), "reason": tensor_row.get("reason")},
          source_files=[microbench],
        ))
        continue

      gain_key = "vector_load_vs_v1" if mode == "vector_load" else "tile_custom_vs_v1"
      gain_pct = gains.get(gain_key)
      status = "reject" if mode in ("vector_load", "tile_custom") else "reject"
      label, reason, retry = v0._label_reason_retry(status, gain=gain_pct / 100.0 if gain_pct is not None else None)
      rows.append(_row(
        row_id=f"{TARGETED_FAMILY}:threeway_load_microbench:{tensor_slug}:{mode}",
        row_kind="candidate",
        model="Qwen3-8B-Q4_K_M",
        tensor=tensor,
        role="ffn_gate",
        fmt="Q4_K",
        mechanism="wide_load_only" if mode == "vector_load" else "tile_custom",
        prediction_stage="after_compile_before_microbench",
        pre_result_context={
          "mode": mode,
          "reference": "v1_partial",
          "meaningful_gain_pct": tensor_row.get("meaningful_gain_pct"),
          "shape": (data.get("config") or {}).get("shape") or {},
          "runs": (data.get("config") or {}).get("runs") or 0,
          "wide_loads": True,
          "full_decode_ready": False,
        },
        label=label,
        reason=reason,
        retry=retry,
        evidence={
          "status": status,
          "gain_pct": gain_pct,
          "device_q4_gbs": mode_row.get("device_q4_gbs"),
          "decision": tensor_row.get("decision"),
          "reason": tensor_row.get("reason"),
        },
        source_files=[microbench],
      ))
  return rows


def _semantic_schedule_microbench_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  base = repo / "bench/qk-ansor-transition-20260612/semantic-schedules"
  rows = []
  for model_dir in ("8b", "14b"):
    microbench = base / model_dir / "microbench.json"
    if not microbench.exists():
      continue
    source = _portable(repo, microbench)
    data = _read_json(microbench)
    model = _canonical_model(model_dir)
    for item in data.get("rows", []):
      status = item.get("status")
      gain = item.get("gain")
      label, reason, retry = v0._label_reason_retry(status, gain=gain)
      rows.append(_row(
        row_id=f"{TARGETED_FAMILY}:semantic_schedule_microbench:{model_dir}:{item['id']}",
        row_kind="candidate",
        model=model,
        tensor=str(item.get("tensor") or "unknown"),
        role=str(item.get("role") or "unknown"),
        fmt=str(item.get("format") or "unknown"),
        mechanism=_mechanism_from_schedule((item.get("schedule") or {}).get("name"), item.get("id", "")),
        prediction_stage="after_static_before_microbench",
        pre_result_context={
          "mode": item.get("id"),
          "candidate_id": item.get("id"),
          "schedule": item.get("schedule", {}),
          "policy": item.get("policy"),
          "full_decode_supported": item.get("full_decode_supported"),
          "microbench_source": source,
        },
        label=label,
        reason=reason,
        retry=retry,
        evidence={
          "status": status,
          "gain": gain,
          "reasons": item.get("reasons", []),
          "candidate_status": (item.get("candidate") or {}).get("status"),
        },
        source_files=[source],
      ))
  return rows


def _semantic_schedule_raw_accept_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  verdict_path = _source(repo, "bench/qk-ansor-transition-20260612/semantic-schedules/verdict.json")
  verdict = _read_json(repo / verdict_path)
  # Prefer per-model microbench rows for richer context when available.
  microbench_maps: dict[str, dict[str, Any]] = {}
  base = repo / "bench/qk-ansor-transition-20260612/semantic-schedules"
  for model_dir in ("8b", "14b"):
    path = base / model_dir / "microbench.json"
    if not path.exists():
      continue
    data = _read_json(path)
    by_id = {}
    for row in data.get("rows", []):
      rid = str(row.get("id") or "")
      if rid:
        by_id[rid] = row
    microbench_maps[model_dir] = by_id

  rows = []
  for model_row in verdict.get("models", []):
    model_name = str(model_row.get("model", "")).lower()
    model_dir = "14b" if "14" in model_name else "8b"
    model = _canonical_model(model_name)
    model_source = _portable(repo, base / model_dir / "microbench.json")
    for item in model_row.get("microbench_accepts", []) or []:
      candidate_id = str(item.get("id") or "")
      source_row = microbench_maps.get(model_dir, {}).get(candidate_id, {})
      schedule = source_row.get("schedule", {})
      status = item.get("status")
      gain = item.get("gain")
      label, reason, retry = v0._label_reason_retry(status, gain=gain)
      rows.append(_row(
        row_id=f"{TARGETED_FAMILY}:semantic_schedule_raw_accept:{model_dir}:{candidate_id}",
        row_kind="candidate",
        model=model,
        tensor=str((source_row.get("tensor") or item.get("tensor") or _tensor_from_row_id(candidate_id))),
        role=str(item.get("role") or source_row.get("role") or "unknown"),
        fmt=str(item.get("format") or source_row.get("format") or "unknown"),
        mechanism=_mechanism_from_schedule((schedule or {}).get("name"), candidate_id),
        prediction_stage="after_static_before_microbench",
        pre_result_context={
          "candidate_id": candidate_id,
          "schedule": schedule,
          "verdict_source": verdict_path,
          "microbench_source": model_source,
          "full_decode_supported": item.get("full_decode_supported", source_row.get("full_decode_supported")),
        },
        label=label,
        reason=reason,
        retry=retry,
        evidence={
          "status": status,
          "gain": gain,
          "full_accept": True,
          "source_model": model_name,
          "candidate_id": candidate_id,
        },
        source_files=[verdict_path, model_source],
      ))
  return rows

# --- Phase 3G coverage-closure batch -------------------------------------------------
# These ingest the dated Phase 3G source artifacts that close the residual mechanism
# and prediction-stage coverage gaps without touching the family-split holdout. They are
# real candidate outcomes on additional dominant Q4_K tensors, not synthetic labels.
PHASE3G_ROOT = "bench/amd-decode-flywheel-proof-20260614"


def _phase3g_microbench_pass(status:str) -> bool:
  return str(status or "").replace("-", "_").lower() in ("raw_accept", "accept", "accepted", "pass", "passed")


def _phase3g_packed_load_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  microbench_path = repo / PHASE3G_ROOT / "phase3g-packed-load/8b/microbench.json"
  if not microbench_path.exists():
    return []
  microbench = _portable(repo, microbench_path)
  load_width_path = repo / PHASE3G_ROOT / "phase3g-packed-load/8b/load-width/report.json"
  load_width = _portable(repo, load_width_path) if load_width_path.exists() else None
  load_width_data = _read_json(load_width_path) if load_width_path.exists() else {"tensors": []}
  load_width_by_tensor = {str(item.get("tensor")): item for item in load_width_data.get("tensors", [])}
  data = _read_json(microbench_path)
  rows = []
  for item in data.get("rows", []):
    candidate_id = str(item.get("id") or "")
    schedule = item.get("schedule") if isinstance(item.get("schedule"), dict) else {}
    status = item.get("status")
    gain = item.get("gain")
    tensor = str(item.get("tensor") or "unknown")
    lw = load_width_by_tensor.get(tensor, {})
    packed = lw.get("packed_load") if isinstance(lw.get("packed_load"), dict) else {}
    # A raw_accept packed-load candidate has cleared the microbench and is awaiting full-decode
    # confirmation, so it genuinely sits at the after_microbench_before_full_decode stage. Everything
    # else (tie, construction/correctness blocked) is recorded at the pre-microbench stage.
    prediction_stage = "after_microbench_before_full_decode" if _phase3g_microbench_pass(status) else "after_static_before_microbench"
    label, reason, retry = v0._label_reason_retry(status, gain=gain)
    source_files = [microbench] + ([load_width] if load_width else [])
    rows.append(_row(
      row_id=f"{TARGETED_FAMILY}:phase3g_packed_load:8b:{candidate_id}",
      row_kind="candidate",
      model="Qwen3-8B-Q4_K_M",
      tensor=tensor,
      role=str(item.get("role") or v0._role_from_text(tensor)),
      fmt=str(item.get("format") or "Q4_K"),
      mechanism=_semantic_codegen_mechanism(schedule, candidate_id),
      prediction_stage=prediction_stage,
      pre_result_context={
        "mode": "packed_load",
        "candidate_id": candidate_id,
        "schedule": schedule,
        "load_width": {
          "global_load_b128": packed.get("global_load_b128"),
          "has_wide_packed_load": lw.get("has_wide_packed_load"),
          "report": load_width,
        },
        "source_ok": bool(lw.get("has_wide_packed_load")),
        "wide_loads": bool(lw.get("has_wide_packed_load")),
        "full_decode_ready": False,
      },
      label=label,
      reason=reason,
      retry=retry,
      evidence={
        "status": status,
        "gain": gain,
        "reasons": item.get("reasons", []),
        "candidate_quant_gbs": (item.get("candidate") or {}).get("quant_gbs"),
        "current_quant_gbs": (item.get("current") or {}).get("quant_gbs"),
        "gemv_max_abs": (item.get("candidate") or {}).get("gemv_max_abs"),
        "full_decode_supported": item.get("full_decode_supported"),
        "global_load_b128": packed.get("global_load_b128"),
        "load_width_correctness": packed.get("correctness"),
      },
      source_files=source_files,
    ))
  return rows


def _phase3g_block_dot_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  artifacts = (
    ("phase3g-blk0-ffn-up", "qk-block-dot-microbench-blk0-ffn-up-20260614", "qk-block-dot-compile-gate-blk0-ffn-up-20260614"),
    ("phase3g-blk0-attn-q", "qk-block-dot-microbench-blk0-attn-q-20260614", "qk-block-dot-compile-gate-blk0-attn-q-20260614"),
  )
  rows = []
  for slug, microbench_dir, compile_dir in artifacts:
    microbench_path = repo / "bench" / microbench_dir / "microbench.json"
    if not microbench_path.exists():
      continue
    microbench = _portable(repo, microbench_path)
    compile_gate_path = repo / "bench" / compile_dir / "compile-gate.json"
    data = _read_json(microbench_path)
    config = data.get("config") or {}
    tensor = str(config.get("tensor") or "unknown")
    cmp = data.get("comparison", {})
    gain_pct = cmp.get("gain_pct")
    label, reason, retry = v0._label_reason_retry("reject", gain=gain_pct / 100.0 if gain_pct is not None else None)
    source_files = [microbench] + ([_portable(repo, compile_gate_path)] if compile_gate_path.exists() else [])
    rows.append(_row(
      row_id=f"{TARGETED_FAMILY}:phase3g_block_dot:{slug}",
      row_kind="candidate",
      model="Qwen3-8B-Q4_K_M",
      tensor=tensor,
      role=v0._role_from_text(tensor),
      fmt="Q4_K",
      mechanism="qk_block_dot",
      prediction_stage="after_compile_before_microbench",
      pre_result_context={
        "mode": "qk_block_dot",
        "reference": "v1_partial",
        "comparison": cmp,
        "shape": config.get("shape"),
        "parts": config.get("parts") or 0,
        "wide_loads": True,
        "full_decode_ready": False,
      },
      label=label,
      reason=reason,
      retry=retry,
      evidence={
        "decision": data.get("summary", {}).get("decision"),
        "gain_pct": gain_pct,
        "correctness_ok": data.get("summary", {}).get("correctness_ok"),
        "next_allowed_gate": data.get("summary", {}).get("next_allowed_gate"),
        "run_full_decode": data.get("summary", {}).get("run_full_decode"),
        "source_compile_gate": _portable(repo, compile_gate_path) if compile_gate_path.exists() else None,
      },
      source_files=source_files,
    ))
  return rows


def _phase3g_threeway_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  microbench_path = repo / "bench/qk-threeway-load-microbench-blk0-ffn-up-20260614/microbench.json"
  if not microbench_path.exists():
    return []
  microbench = _portable(repo, microbench_path)
  data = _read_json(microbench_path)
  rows = []
  # Only the newly covered dominant tensor (ffn_up) is added as branch-bounding diagnostic
  # evidence; the co-run ffn_gate tensor already has coverage in the dated 20260613 batch.
  for tensor_row in data.get("tensors", []) or []:
    tensor = str(tensor_row.get("tensor") or "unknown")
    if tensor == "blk.0.ffn_gate.weight":
      continue
    tensor_slug = re.sub(r"[^a-z0-9]+", "-", tensor.lower()).strip("-")
    v1 = next((m for m in tensor_row.get("modes", []) or [] if m.get("mode") == "v1_partial"), None)
    if v1 is None:
      continue
    rows.append(_row(
      row_id=f"{TARGETED_FAMILY}:phase3g_threeway_load:{tensor_slug}:v1_partial",
      row_kind="diagnostic",
      model="Qwen3-8B-Q4_K_M",
      tensor=tensor,
      role=v0._role_from_text(tensor),
      fmt="Q4_K",
      mechanism="wide_load_only",
      prediction_stage="after_compile_before_microbench",
      pre_result_context={
        "mode": "v1_partial",
        "reference": "qk-block-dot",
        "decision": tensor_row.get("decision"),
        "shape": (data.get("config") or {}).get("shape"),
        "runs": (data.get("config") or {}).get("runs") or 0,
        "wide_loads": True,
        "full_decode_ready": False,
      },
      label="diagnostic_only",
      reason="diagnostic_only",
      retry=False,
      evidence={"status": "pass", "decision": tensor_row.get("decision"), "reason": tensor_row.get("reason")},
      source_files=[microbench],
    ))
  return rows


def build_targeted_rows(repo:pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  repo = repo.resolve()
  raw_rows = _memory_access_rows(repo)
  raw_rows += _packed_tile_consumption_rows(repo)
  raw_rows += _packed_tile_closeout_rows(repo)
  raw_rows += _qk_block_dot_microbench_rows(repo)
  raw_rows += _qk_threeway_load_microbench_rows(repo)
  raw_rows += _semantic_codegen_microbench_rows(repo)
  raw_rows += _semantic_schedule_microbench_rows(repo)
  raw_rows += _semantic_schedule_raw_accept_rows(repo)
  raw_rows += _phase3g_packed_load_rows(repo)
  raw_rows += _phase3g_block_dot_rows(repo)
  raw_rows += _phase3g_threeway_rows(repo)
  excluded = []
  semantic_contract = repo / "bench/qk-packed-semantic-op-20260613/semantic-op-contract.json"
  if semantic_contract.exists():
    contract = _read_json(semantic_contract)
    excluded.append({
      "source_file": _portable(repo, semantic_contract),
      "rows": len(contract.get("contract_rows", [])) if isinstance(contract.get("contract_rows"), list) else 0,
      "reason": "design_only_no_runtime_lowering; recorded in plan but not used as train labels",
    })
  normalized = [v1.normalize_row(row) for row in raw_rows]
  enriched = [enrich_row(row, repo) for row in normalized]
  for row in enriched:
    if row.get("split") != "train": raise ValueError(f"{row['id']}: targeted outcome must be train split")
    if row.get("family") != TARGETED_FAMILY: raise ValueError(f"{row['id']}: bad targeted family")
  seen = set()
  for row in enriched:
    if row["id"] in seen: raise ValueError(f"duplicate targeted row id {row['id']}")
    seen.add(row["id"])
  return sorted(enriched, key=lambda row: row["id"]), excluded

def _targeted_summary(rows:list[dict[str, Any]], excluded:list[dict[str, Any]], *, base_count:int, plus_count:int) -> dict[str, Any]:
  real_rows = [row for row in rows if (row.get("feature_extraction") or {}).get("real_uop_available")]
  return {
    "kind": "qk_flywheel_targeted_outcomes_v1",
    "phase": "Phase 3F",
    "conclusion": "partial_real_outcome_batch_cost_model_rerun_still_gated",
    "rows": len(rows),
    "base_rows": base_count,
    "plus_rows": plus_count,
    "split": "train",
    "family": TARGETED_FAMILY,
    "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
    "mechanisms": dict(sorted(Counter(row["mechanism"] for row in rows).items())),
    "prediction_stages": dict(sorted(Counter(row["prediction_stage"] for row in rows).items())),
    "source_files": dict(Counter(src for row in rows for src in row.get("source_files", [])).most_common()),
    "real_feature_rows": len(real_rows),
    "excluded_sources": excluded,
    "rules": [
      "uses only committed real probe/compile/source diagnostic artifacts",
      "does not duplicate any existing v1-featured row id",
      "does not move family-split holdout rows into train",
      "does not use design-only contracts as train labels",
      "does not authorize a Phase 3B cost-model rerun as a decision point unless the plus audit clears the coverage gate",
    ],
  }

def _plus_summary(rows:list[dict[str, Any]], targeted:list[dict[str, Any]], prompts:list[dict[str, Any]], targeted_summary:dict[str, Any]) -> dict[str, Any]:
  integrity = v1.validate_examples(rows)
  train = [row for row in rows if row["split"] == "train"]
  holdout = [row for row in rows if row["split"] == "holdout"]
  real_rows = [row for row in rows if (row.get("feature_extraction") or {}).get("real_uop_available")]
  real_by_split = Counter(row["split"] for row in real_rows)
  real_by_mech = Counter(row["mechanism"] for row in real_rows)
  return {
    "kind": "qk_flywheel_kernel_triage_dataset_v1_featured_plus",
    "phase": "Phase 3F",
    "rows": len(rows),
    "prompts": len(prompts),
    "split_policy": "family_split_v0_preserved_plus_post_phase3e_train_batch",
    "train_rows": len(train),
    "holdout_rows": len(holdout),
    "feature_schema": "candidate_outcome_v1_featured",
    "targeted_rows_added": len(targeted),
    "targeted_family": TARGETED_FAMILY,
    "targeted_summary": {
      "labels": targeted_summary["labels"],
      "mechanisms": targeted_summary["mechanisms"],
      "excluded_sources": targeted_summary["excluded_sources"],
    },
    "integrity": integrity,
    "real_feature_coverage": {
      "uop_available_rows": len(real_rows),
      "uop_available_train_rows": real_by_split.get("train", 0),
      "uop_available_holdout_rows": real_by_split.get("holdout", 0),
      "uop_available_by_mechanism": dict(sorted(real_by_mech.items())),
    },
    "files": {
      "examples": "examples.jsonl",
      "prompts": "prompts.jsonl",
      "train_prompts": "prompts-train.jsonl",
      "holdout_prompts": "prompts-holdout.jsonl",
      "summary": "summary.json",
      "readme": "README.md",
    },
  }

def _targeted_readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Targeted Outcomes v1",
    "",
    "This Phase 3F artifact converts unused committed real probe/source",
    "diagnostics into a small post-Phase-3E train batch. It is deliberately",
    "partial: no holdout row is moved into train, no synthetic outcome is added,",
    "and design-only contracts remain excluded.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- targeted train rows: `{summary['rows']}`",
    f"- base rows: `{summary['base_rows']}`",
    f"- plus rows: `{summary['plus_rows']}`",
    f"- real feature rows: `{summary['real_feature_rows']}`",
    "",
    "## Mechanisms",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mechanism, count in summary["mechanisms"].items():
    lines.append(f"| `{mechanism}` | {count} |")
  lines += ["", "## Labels", "", "| label | rows |", "|---|---:|"]
  for label, count in summary["labels"].items():
    lines.append(f"| `{label}` | {count} |")
  lines += ["", "## Excluded Sources", "", "| source | rows | reason |", "|---|---:|---|"]
  for item in summary["excluded_sources"]:
    lines.append(f"| `{item['source_file']}` | {item['rows']} | {item['reason']} |")
  lines += ["", "## Rules", ""]
  for rule in summary["rules"]:
    lines.append(f"- {rule}")
  lines.append("")
  return "\n".join(lines)

def _plus_readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Kernel Triage Dataset v1 Featured Plus",
    "",
    "This Phase 3F dataset appends the targeted-outcomes train batch to the",
    "Phase 3E featured dataset while preserving the original family-split",
    "holdout. It is an intermediate coverage artifact, not a cost-model win.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- targeted rows added: `{summary['targeted_rows_added']}`",
    f"- split policy: `{summary['split_policy']}`",
    f"- real UOp/source rows: `{summary['real_feature_coverage']['uop_available_rows']}`",
    "",
    "## Targeted Mechanisms",
    "",
    "| mechanism | rows |",
    "|---|---:|",
  ]
  for mechanism, count in summary["targeted_summary"]["mechanisms"].items():
    lines.append(f"| `{mechanism}` | {count} |")
  lines += ["", "## Targeted Labels", "", "| label | rows |", "|---|---:|"]
  for label, count in summary["targeted_summary"]["labels"].items():
    lines.append(f"| `{label}` | {count} |")
  lines.append("")
  return "\n".join(lines)

def write_phase3f(repo:pathlib.Path, base:pathlib.Path, targeted_out:pathlib.Path, plus_out:pathlib.Path,
                  audit_out:pathlib.Path|None=None, coverage_out:pathlib.Path|None=None) -> dict[str, Any]:
  repo = repo.resolve()
  base_rows = _read_jsonl(base)
  targeted, excluded = build_targeted_rows(repo)
  base_ids = {row["id"] for row in base_rows}
  duplicates = sorted(base_ids & {row["id"] for row in targeted})
  if duplicates: raise ValueError(f"targeted rows duplicate base ids: {duplicates[:5]}")

  plus_rows = copy.deepcopy(base_rows) + targeted
  prompts = [v1._prompt(row) for row in plus_rows]
  targeted_summary = _targeted_summary(targeted, excluded, base_count=len(base_rows), plus_count=len(plus_rows))
  plus_summary = _plus_summary(plus_rows, targeted, prompts, targeted_summary)

  targeted_out.mkdir(parents=True, exist_ok=True)
  _jsonl(targeted_out / "examples.jsonl", targeted)
  (targeted_out / "summary.json").write_text(json.dumps(targeted_summary, indent=2, sort_keys=True) + "\n")
  (targeted_out / "README.md").write_text(_targeted_readme(targeted_summary))

  plus_out.mkdir(parents=True, exist_ok=True)
  _jsonl(plus_out / "examples.jsonl", plus_rows)
  _jsonl(plus_out / "prompts.jsonl", prompts)
  _jsonl(plus_out / "prompts-train.jsonl", [row for row in prompts if row["split"] == "train"])
  _jsonl(plus_out / "prompts-holdout.jsonl", [row for row in prompts if row["split"] == "holdout"])
  (plus_out / "summary.json").write_text(json.dumps(plus_summary, indent=2, sort_keys=True) + "\n")
  (plus_out / "README.md").write_text(_plus_readme(plus_summary))

  audit = None
  coverage = None
  if audit_out is not None:
    audit = run_feature_audit(plus_out / "examples.jsonl", audit_out)
  if coverage_out is not None:
    if audit_out is None: raise ValueError("--coverage-out requires --audit-out")
    coverage = write_coverage_plan(audit_out / "summary.json", coverage_out)

  return {
    "kind": "qk_flywheel_phase3f_write_result",
    "targeted": targeted_summary,
    "plus": plus_summary,
    "audit": {
      "path": str(audit_out) if audit_out else None,
      "conclusion": audit.get("conclusion") if audit else None,
      "unseen_holdout_value_total": ((audit.get("coverage") or {}).get("categorical") or {}).get("unseen_holdout_value_total") if audit else None,
      "weak_row_count": ((audit.get("row_quality") or {}).get("weak_row_count")) if audit else None,
    },
    "coverage": {
      "path": str(coverage_out) if coverage_out else None,
      "rerun_phase3b_allowed": coverage.get("rerun_phase3b_allowed") if coverage else None,
      "minimum_new_mechanism_rows": coverage.get("minimum_new_mechanism_rows") if coverage else None,
      "minimum_new_label_rows": coverage.get("minimum_new_label_rows") if coverage else None,
    },
  }

def main() -> int:
  parser = argparse.ArgumentParser(description="Build Phase 3F targeted real-outcome batch for QK flywheel triage")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--base", type=pathlib.Path, default=DEFAULT_BASE)
  parser.add_argument("--targeted-out", type=pathlib.Path, default=DEFAULT_TARGETED_OUT)
  parser.add_argument("--plus-out", type=pathlib.Path, default=DEFAULT_PLUS_OUT)
  parser.add_argument("--audit-out", type=pathlib.Path, default=DEFAULT_AUDIT_OUT)
  parser.add_argument("--coverage-out", type=pathlib.Path, default=DEFAULT_COVERAGE_OUT)
  args = parser.parse_args()
  result = write_phase3f(args.repo, args.base, args.targeted_out, args.plus_out, args.audit_out, args.coverage_out)
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
