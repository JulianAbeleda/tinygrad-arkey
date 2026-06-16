#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re
from collections import Counter, defaultdict
from typing import Any

LABELS = ("accept", "reject", "tie", "raw_accept_unconfirmed", "needs_rerun", "construction_blocked", "diagnostic_only")
REASONS = (
  "static_gate_fail", "construction_blocked", "correctness_fail", "microbench_regression",
  "microbench_tie", "full_decode_regression", "confirmation_failed", "insufficient_gain",
  "memory_pressure", "unsupported_runtime_scope", "diagnostic_only", "accepted_runtime_path",
  "needs_rerun",
)
MECHANISMS = (
  "parts_local_policy", "direct_output", "row_grouping", "packed_word_lane_unroll",
  "vector_load", "tile_custom", "qk_block_dot", "wide_load_only", "shared_storage",
  "storage_cap", "semantic_descriptor_replay", "unknown",
)
ROLES = ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_output", "unknown")
FORMATS = ("Q4_K", "Q6_K", "q8_1", "unknown")
HOLDOUT_FAMILIES = {"semantic_schedule_v0", "semantic_codegen_v3", "semantic_codegen_v4", "qk_block_dot", "threeway_load"}
FAMILY_ORDER = {
  "accepted_runtime": 0, "loop_v0_policy": 1, "semantic_schedule_v0": 2, "semantic_codegen_v1": 3,
  "semantic_codegen_v2": 4, "semantic_codegen_v3": 5, "semantic_codegen_v4": 6,
  "packed_tile_lowering": 7, "packed_tile_analysis": 8, "qk_block_dot": 9, "threeway_load": 10,
}

def _load_json(path:pathlib.Path) -> dict[str, Any]:
  try:
    data = json.loads(path.read_text())
  except json.JSONDecodeError as exc:
    raise ValueError(f"{path}: invalid JSON: {exc}") from exc
  if not isinstance(data, dict): raise ValueError(f"{path}: expected JSON object")
  return data

def _jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  with path.open("w") as f:
    for row in rows: f.write(json.dumps(row, sort_keys=True) + "\n")

def _slug(value:str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"

def _role_from_text(*values:Any) -> str:
  text = " ".join(str(v) for v in values if v is not None).lower()
  for role in ROLES:
    if role != "unknown" and role.replace("_", "-") in text.replace("_", "-"): return role
  return "unknown"

def _format(value:Any) -> str:
  if value in FORMATS: return str(value)
  return "unknown"

def _model(value:Any) -> str:
  text = str(value or "unknown")
  if text in ("8B", "14B", "32B"): return f"Qwen3-{text}-Q4_K_M"
  if "8B" in text or "8b" in text: return "Qwen3-8B-Q4_K_M"
  if "14B" in text or "14b" in text: return "Qwen3-14B-Q4_K_M"
  if "32B" in text or "32b" in text: return "Qwen3-32B-Q4_K_M"
  return text

def _mechanism_from_id(row_id:str, family:str, schedule:dict[str, Any]|None=None) -> str:
  sched_name = str((schedule or {}).get("name", "") or (schedule or {}).get("codegen_mode", ""))
  text = f"{family} {row_id} {sched_name}".lower()
  if "row-group" in text: return "row_grouping"
  if "direct-out" in text or "direct_out" in text: return "direct_output"
  if "vector-load" in text or "vector_load" in text: return "vector_load"
  if "packed-load" in text or "packed_word" in text: return "packed_word_lane_unroll"
  if "tile-custom" in text or "tile_custom" in text: return "tile_custom"
  if "qk-block-dot" in text or "qk_block_dot" in text: return "qk_block_dot"
  if "local" in text or "parts" in text or "-p" in text: return "parts_local_policy"
  return "unknown"

def _label_reason_retry(status:Any, *, gain:float|None=None, stage:str="microbench", raw_accept_confirmation_failed:bool=False) -> tuple[str, str, bool]:
  text = str(status or "").replace("-", "_").lower()
  if raw_accept_confirmation_failed: return "tie", "confirmation_failed", False
  if text in ("accept", "accepted", "pass", "passed") and stage == "runtime": return "accept", "accepted_runtime_path", False
  if text in ("accept", "accepted", "raw_accept"): return "raw_accept_unconfirmed", "insufficient_gain", True
  if text in ("needs_rerun", "needs-rerun"): return "needs_rerun", "needs_rerun", True
  if text in ("tie", "tied"): return "tie", "microbench_tie" if stage != "full_decode" else "confirmation_failed", False
  if text in ("invalid", "error", "compile_fail", "compile-fail", "illegal_opt", "illegal-opt", "construction_blocked"):
    return "construction_blocked", "construction_blocked", False
  if text in ("diagnostic", "diagnostic_only"): return "diagnostic_only", "diagnostic_only", False
  if text in ("reject", "rejected", "fail", "failed"):
    if stage == "full_decode": return "reject", "full_decode_regression", False
    if gain is not None and gain > 0: return "reject", "insufficient_gain", False
    return "reject", "microbench_regression", False
  if stage == "diagnostic": return "diagnostic_only", "diagnostic_only", False
  return "reject", "microbench_regression", False

def _split_for(family:str) -> str:
  return "holdout" if family in HOLDOUT_FAMILIES else "train"

def _context(row:dict[str, Any]) -> dict[str, Any]:
  return {
    "candidate_id": row["candidate_id"],
    "row_kind": row["row_kind"],
    "family": row["family"],
    "model": row["model"],
    "tensor": row["tensor"],
    "role": row["role"],
    "format": row["format"],
    "mechanism": row["mechanism"],
    "prediction_stage": row["prediction_stage"],
    "candidate_context": row["pre_result_context"],
  }

def assemble_row(*, row_id:str, row_kind:str, family:str, family_order:int, model:str, tensor:str,
                 role:str, fmt:str, mechanism:str, prediction_stage:str, pre_result_context:dict[str, Any],
                 label:str, reason:str, retry:bool, evidence:dict[str, Any], source_files:list[str],
                 split:str) -> dict[str, Any]:
  """Build a triage row dict from already-normalized fields.

  Single source of truth for the example row shape. Callers own field
  normalization/validation policy (it differs across sources — e.g. the v0
  builders clamp mechanism/role/format to the known sets, while the targeted
  builders preserve v1-only mechanisms that v1 normalization resolves later),
  so this helper only assembles the canonical key layout and must not mutate
  values.
  """
  if label not in LABELS: raise ValueError(f"{row_id}: unknown label {label}")
  if reason not in REASONS: raise ValueError(f"{row_id}: unknown reason {reason}")
  return {
    "id": row_id,
    "candidate_id": row_id.split(":", 1)[-1],
    "row_kind": row_kind,
    "family": family,
    "family_order": family_order,
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
    "split": split,
  }

def _make_row(*, row_id:str, row_kind:str, family:str, model:str, tensor:str, role:str, fmt:str,
              mechanism:str, prediction_stage:str, pre_result_context:dict[str, Any],
              label:str, reason:str, retry:bool, evidence:dict[str, Any], source_files:list[str]) -> dict[str, Any]:
  if mechanism not in MECHANISMS: mechanism = "unknown"
  if role not in ROLES: role = "unknown"
  if fmt not in FORMATS: fmt = "unknown"
  return assemble_row(
    row_id=row_id, row_kind=row_kind, family=family, family_order=FAMILY_ORDER.get(family, 999),
    model=model, tensor=tensor or "unknown", role=role, fmt=fmt, mechanism=mechanism,
    prediction_stage=prediction_stage, pre_result_context=pre_result_context, label=label,
    reason=reason, retry=retry, evidence=evidence, source_files=source_files, split=_split_for(family),
  )

def _prompt(row:dict[str, Any]) -> dict[str, Any]:
  context = _context(row)
  question = (
    "/no_think\n"
    "Return only compact JSON with exactly these keys: label, reason, retry. "
    "Do not include analysis, markdown, or extra text. "
    f"Allowed labels: {', '.join(LABELS)}. Allowed reasons: {', '.join(REASONS)}. "
    "Decide from the pre-result kernel candidate context. Context: "
    + json.dumps(context, sort_keys=True, separators=(",", ":"))
  )
  return {
    "id": row["id"],
    "prompt": question,
    "expected_json": {"label": row["label"], "reason": row["reason"], "retry": row["retry"]},
    "tags": ["qk_flywheel", "kernel_triage", row["split"], row["family"], row["mechanism"], row["label"]],
    "max_tokens": 64,
    "split": row["split"],
    "family": row["family"],
    "mechanism": row["mechanism"],
  }

def _confirmation_map(repo:pathlib.Path) -> dict[tuple[str, str], dict[str, Any]]:
  path = repo / "bench/qk-ansor-transition-20260612/benchmarks/verdict.json"
  if not path.exists(): return {}
  out: dict[tuple[str, str], dict[str, Any]] = {}
  data = _load_json(path)
  for model_row in data.get("rows", []):
    model = str(model_row.get("model"))
    for candidate_id, confirmation in (model_row.get("confirmations") or {}).items():
      out[(model, candidate_id)] = confirmation
  return out

def _loop_matrix_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  confirmations = _confirmation_map(repo)
  for path in sorted((repo / "bench/qk-ansor-transition-20260612/benchmarks").glob("*/matrix.json")):
    data = _load_json(path)
    model_short = str(data.get("model", path.parent.name.upper()))
    for item in data.get("rows", []):
      changes = item.get("changes") or []
      change = changes[0] if changes else {}
      row_id = f"loop_v0_policy:{_slug(model_short)}:{item['id']}"
      confirmation = confirmations.get((model_short, item["id"]))
      label, reason, retry = _label_reason_retry(item.get("status"), gain=item.get("gain"), raw_accept_confirmation_failed=confirmation is not None)
      rows.append(_make_row(
        row_id=row_id, row_kind="candidate", family="loop_v0_policy", model=_model(model_short),
        tensor=str(change.get("tensor") or "unknown"), role=_role_from_text(change.get("role"), item.get("id")),
        fmt=_format(change.get("format")), mechanism="parts_local_policy", prediction_stage="after_policy_before_full_decode",
        pre_result_context={"change": change, "source_loop": data.get("source_loop"), "candidate_stable": item.get("candidate_stable"), "reference_stable": item.get("reference_stable")},
        label=label, reason=reason, retry=retry,
        evidence={"status": item.get("status"), "gain": item.get("gain"), "reasons": item.get("reasons", []), "confirmation": confirmation},
        source_files=[str(path.relative_to(repo))],
      ))
  return rows

def _microbench_rows(repo:pathlib.Path, base_glob:str, family:str) -> list[dict[str, Any]]:
  rows = []
  for path in sorted(repo.glob(base_glob)):
    data = _load_json(path)
    model = _model(data.get("model") or path.parent.name.upper())
    for item in data.get("rows", []):
      schedule = item.get("schedule") or {}
      row_id = f"{family}:{_slug(model)}:{item['id']}"
      gain = item.get("gain")
      label, reason, retry = _label_reason_retry(item.get("status"), gain=gain)
      rows.append(_make_row(
        row_id=row_id, row_kind="candidate", family=family, model=model,
        tensor=str(item.get("tensor") or "unknown"), role=_role_from_text(item.get("role"), item.get("id")),
        fmt=_format(item.get("format")), mechanism=_mechanism_from_id(item["id"], family, schedule),
        prediction_stage="after_static_before_microbench",
        pre_result_context={"schedule": schedule, "full_decode_supported": item.get("full_decode_supported"), "policy": item.get("policy")},
        label=label, reason=reason, retry=retry,
        evidence={"status": item.get("status"), "gain": gain, "reasons": item.get("reasons", []), "candidate_status": (item.get("candidate") or {}).get("status")},
        source_files=[str(path.relative_to(repo))],
      ))
  return rows

def _semantic_verdict_rows(repo:pathlib.Path, rel:str, family:str) -> list[dict[str, Any]]:
  path = repo / rel
  if not path.exists(): return []
  data = _load_json(path)
  rows = []
  for model_row in data.get("models", []):
    model = _model(model_row.get("model"))
    for item in model_row.get("rows", []):
      row_id = f"{family}:{_slug(model)}:{item['id']}"
      gain = item.get("gain")
      label, reason, retry = _label_reason_retry(item.get("status"), gain=gain)
      rows.append(_make_row(
        row_id=row_id, row_kind="candidate", family=family, model=model,
        tensor=_tensor_from_id(item.get("id")), role=_role_from_text(item.get("id")),
        fmt="Q4_K", mechanism=_mechanism_from_id(str(item.get("id")), family),
        prediction_stage="after_static_before_microbench",
        pre_result_context={"full_decode_supported": item.get("full_decode_supported"), "candidate_id": item.get("id")},
        label=label, reason=reason, retry=retry,
        evidence={"status": item.get("status"), "gain": gain, "reasons": item.get("reasons", []), "candidate_gbs": item.get("candidate_gbs"), "current_gbs": item.get("current_gbs")},
        source_files=[rel],
      ))
  return rows

def _semantic_full_decode_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  rel = "bench/qk-ansor-transition-20260612/semantic-schedules/verdict.json"
  path = repo / rel
  if not path.exists(): return []
  data = _load_json(path)
  rows = []
  for model_row in data.get("models", []):
    model = _model(model_row.get("model"))
    for item in model_row.get("full_decode_decisions", []):
      row_id = f"semantic_schedule_v0_full_decode:{_slug(model)}:{item['candidate']}"
      label, reason, retry = _label_reason_retry(item.get("status"), gain=item.get("gain"), stage="full_decode")
      rows.append(_make_row(
        row_id=row_id, row_kind="candidate", family="semantic_schedule_v0", model=model,
        tensor=_tensor_from_id(item.get("candidate")), role=_role_from_text(item.get("candidate")),
        fmt="Q4_K", mechanism=_mechanism_from_id(str(item.get("candidate")), "semantic_schedule_v0"),
        prediction_stage="after_microbench_before_full_decode",
        pre_result_context={"candidate": item.get("candidate"), "policy": item.get("policy"), "full_decode_ready": True},
        label=label, reason=reason, retry=retry,
        evidence={"status": item.get("status"), "gain": item.get("gain"), "reasons": item.get("reasons", []), "ab_match": item.get("ab_match")},
        source_files=[rel],
      ))
  return rows

def _tensor_from_id(value:Any) -> str:
  text = str(value or "")
  m = re.search(r"blk-(\d+)-([a-z-]+)-weight", text)
  if not m: return "unknown"
  return f"blk.{m.group(1)}.{m.group(2).replace('-', '_')}.weight"

def _accepted_runtime_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  roots = [
    ("accepted_runtime", "shared_storage", "bench/qk-shared-storage-20260612/*/decision.json"),
    ("accepted_runtime", "parts_local_policy", "bench/qk-harness-20260612/*/decision.json"),
    ("accepted_runtime", "parts_local_policy", "bench/qk-policy-pipeline-20260612/*/decision.json"),
    ("accepted_runtime", "storage_cap", "bench/qk-policy-cap-20260612/*/decision.json"),
  ]
  rows = []
  for family, mechanism, glob_pat in roots:
    for path in sorted(repo.glob(glob_pat)):
      data = _load_json(path)
      status = data.get("status") or (data.get("stages", {}).get("decide", {}).get("metadata", {}) or {}).get("status")
      if status != "accept" and data.get("gain", 0) <= 0: continue
      row_id = f"accepted_runtime:{_slug(mechanism)}:{_slug(data.get('model_size') or path.parent.name)}:{_slug(str(path.parent))}"
      rows.append(_make_row(
        row_id=row_id, row_kind="baseline", family=family, model=_model(data.get("model_size") or data.get("model")),
        tensor="all", role="unknown", fmt="unknown", mechanism=mechanism,
        prediction_stage="after_full_decode",
        pre_result_context={"model_size": data.get("model_size"), "policy": data.get("policy"), "reference_mode": data.get("reference_mode"), "ab_match": data.get("ab_match")},
        label="accept", reason="accepted_runtime_path", retry=False,
        evidence={"gain": data.get("gain"), "reasons": data.get("reasons", []), "generated": data.get("generated"), "explicit": data.get("explicit")},
        source_files=[str(path.relative_to(repo))],
      ))
  return rows

def _packed_tile_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  verdict = repo / "bench/qk-packed-tile-lowering-20260613/verdict.json"
  if verdict.exists():
    data = _load_json(verdict)
    req = float(data.get("promotion_gate", {}).get("required_microbench_gain_pct", 10.0))
    for item in data.get("microbench", []):
      gain = item.get("gain_pct")
      label, reason, retry = ("reject", "insufficient_gain", False) if gain is not None and gain < req else ("raw_accept_unconfirmed", "insufficient_gain", True)
      rows.append(_make_row(
        row_id=f"packed_tile_lowering:{_slug(str(item.get('tensor')))}", row_kind="candidate", family="packed_tile_lowering",
        model="Qwen3-8B-Q4_K_M", tensor=str(item.get("tensor") or "unknown"), role=_role_from_text(item.get("tensor")),
        fmt="Q4_K", mechanism="tile_custom", prediction_stage="after_compile_before_microbench",
        pre_result_context={"mode": item.get("mode"), "parts": item.get("parts"), "load_width": data.get("load_width"), "required_gain_pct": req},
        label=label, reason=reason, retry=retry, evidence=item, source_files=[str(verdict.relative_to(repo))],
      ))
  analysis = repo / "bench/qk-packed-tile-lowering-analysis-20260613/analysis.json"
  if analysis.exists():
    data = _load_json(analysis)
    for item in data.get("comparisons", []):
      gain = item.get("gain_pct")
      label, reason, retry = ("reject", "insufficient_gain", False) if gain is not None and gain > 0 else _label_reason_retry("reject", gain=gain)
      if gain is not None and abs(gain) <= 3.0: label, reason, retry = "tie", "microbench_tie", False
      rows.append(_make_row(
        row_id=f"packed_tile_analysis:{_slug(str(item.get('tensor')))}", row_kind="candidate", family="packed_tile_analysis",
        model="Qwen3-8B-Q4_K_M", tensor=str(item.get("tensor") or "unknown"), role=_role_from_text(item.get("tensor")),
        fmt="Q4_K", mechanism="tile_custom", prediction_stage="after_compile_before_microbench",
        pre_result_context={"mode": "tile_custom", "reference": "v1_partial", "runs": data.get("config", {}).get("runs")},
        label=label, reason=reason, retry=retry, evidence=item, source_files=[str(analysis.relative_to(repo))],
      ))
  return rows

def _block_dot_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  compile_gate = repo / "bench/qk-block-dot-compile-gate-20260613/compile-gate.json"
  if compile_gate.exists():
    data = _load_json(compile_gate)
    rows.append(_make_row(
      row_id="qk_block_dot:compile_gate:8b-ffn-gate", row_kind="diagnostic", family="qk_block_dot",
      model="Qwen3-8B-Q4_K_M", tensor=data.get("shape", {}).get("tensor", "unknown"), role="ffn_gate",
      fmt="Q4_K", mechanism="qk_block_dot", prediction_stage="after_compile_before_microbench",
      pre_result_context={"shape": data.get("shape"), "source_ok": data.get("summary", {}).get("source_ok"), "wide_loads": data.get("summary", {}).get("wide_loads")},
      label="diagnostic_only", reason="diagnostic_only", retry=True,
      evidence={"decision": data.get("summary", {}).get("decision"), "run_microbench": data.get("summary", {}).get("run_microbench")},
      source_files=[str(compile_gate.relative_to(repo))],
    ))
  microbench = repo / "bench/qk-block-dot-microbench-20260613/microbench.json"
  if microbench.exists():
    data = _load_json(microbench)
    cmp = data.get("comparison", {})
    label, reason, retry = _label_reason_retry("reject", gain=(cmp.get("gain_pct") or 0) / 100.0)
    rows.append(_make_row(
      row_id="qk_block_dot:microbench:8b-ffn-gate", row_kind="candidate", family="qk_block_dot",
      model="Qwen3-8B-Q4_K_M", tensor=data.get("config", {}).get("tensor", "blk.0.ffn_gate.weight"),
      role="ffn_gate", fmt="Q4_K", mechanism="qk_block_dot", prediction_stage="after_compile_before_microbench",
      pre_result_context={"shape": data.get("config", {}).get("shape"), "parts": data.get("config", {}).get("parts"), "opts": data.get("config", {}).get("opts")},
      label=label, reason=reason, retry=retry,
      evidence={"decision": data.get("summary", {}).get("decision"), "gain_pct": cmp.get("gain_pct"), "correctness_ok": data.get("summary", {}).get("correctness_ok")},
      source_files=[str(microbench.relative_to(repo))],
    ))
  return rows

def _threeway_rows(repo:pathlib.Path) -> list[dict[str, Any]]:
  path = repo / "bench/qk-threeway-load-microbench-20260613/microbench.json"
  if not path.exists(): return []
  data = _load_json(path)
  rows = []
  for tensor_row in data.get("tensors", []):
    tensor = str(tensor_row.get("tensor") or "unknown")
    gains = tensor_row.get("gains_pct", {})
    for mode, gain_key, mechanism in (("vector_load", "vector_load_vs_v1", "vector_load"), ("tile_custom", "tile_custom_vs_v1", "tile_custom")):
      gain_pct = gains.get(gain_key)
      label, reason, retry = _label_reason_retry("reject", gain=(gain_pct or 0) / 100.0)
      rows.append(_make_row(
        row_id=f"threeway_load:{_slug(tensor)}:{mode}", row_kind="candidate", family="threeway_load",
        model="Qwen3-8B-Q4_K_M", tensor=tensor, role=_role_from_text(tensor), fmt="Q4_K",
        mechanism=mechanism if mode != "vector_load" else "wide_load_only",
        prediction_stage="after_compile_before_microbench",
        pre_result_context={"mode": mode, "reference": "v1_partial", "meaningful_gain_pct": tensor_row.get("meaningful_gain_pct")},
        label=label, reason=reason, retry=retry,
        evidence={"gain_pct": gain_pct, "decision": tensor_row.get("decision"), "reason": tensor_row.get("reason")},
        source_files=[str(path.relative_to(repo))],
      ))
  return rows

def build_examples(repo:pathlib.Path) -> list[dict[str, Any]]:
  repo = repo.resolve()
  rows = []
  rows += _accepted_runtime_rows(repo)
  rows += _loop_matrix_rows(repo)
  rows += _microbench_rows(repo, "bench/qk-ansor-transition-20260612/semantic-schedules/*/microbench.json", "semantic_schedule_v0")
  rows += _semantic_full_decode_rows(repo)
  rows += _microbench_rows(repo, "bench/qk-ansor-transition-20260612/semantic-codegen-v1/*/microbench.json", "semantic_codegen_v1")
  rows += _semantic_verdict_rows(repo, "bench/qk-ansor-transition-20260612/semantic-codegen-v2/verdict.json", "semantic_codegen_v2")
  rows += _semantic_verdict_rows(repo, "bench/qk-ansor-transition-20260612/semantic-codegen-v3/verdict.json", "semantic_codegen_v3")
  rows += _semantic_verdict_rows(repo, "bench/qk-ansor-transition-20260612/semantic-codegen-v4/verdict.json", "semantic_codegen_v4")
  rows += _packed_tile_rows(repo)
  rows += _block_dot_rows(repo)
  rows += _threeway_rows(repo)
  rows = sorted(rows, key=lambda row: (row["family_order"], row["family"], row["model"], row["id"]))
  seen: set[str] = set()
  for row in rows:
    if row["id"] in seen: raise ValueError(f"duplicate row id {row['id']}")
    seen.add(row["id"])
  return rows

def validate_examples(rows:list[dict[str, Any]]) -> dict[str, Any]:
  if not rows: raise ValueError("no examples")
  errors = []
  for row in rows:
    for key in ("id", "row_kind", "family", "model", "tensor", "role", "format", "mechanism", "prediction_stage", "pre_result_context", "label", "reason", "retry", "evidence", "source_files", "split"):
      if key not in row: errors.append(f"{row.get('id', '<missing>')}: missing {key}")
    if row.get("label") not in LABELS: errors.append(f"{row.get('id')}: bad label {row.get('label')!r}")
    if row.get("reason") not in REASONS: errors.append(f"{row.get('id')}: bad reason {row.get('reason')!r}")
    if not isinstance(row.get("pre_result_context"), dict): errors.append(f"{row.get('id')}: pre_result_context must be object")
    if not isinstance(row.get("evidence"), dict): errors.append(f"{row.get('id')}: evidence must be object")
  if errors: raise ValueError("; ".join(errors[:8]))
  return {
    "rows": len(rows),
    "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
    "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
    "reasons": dict(sorted(Counter(row["reason"] for row in rows).items())),
    "mechanisms": dict(sorted(Counter(row["mechanism"] for row in rows).items())),
    "families": dict(sorted(Counter(row["family"] for row in rows).items())),
    "source_files": len({src for row in rows for src in row["source_files"]}),
    "complete_numeric_evidence_rows": sum(any(k in row["evidence"] and row["evidence"][k] is not None for k in ("gain", "gain_pct")) for row in rows),
  }

def _summary(rows:list[dict[str, Any]], prompts:list[dict[str, Any]]) -> dict[str, Any]:
  integrity = validate_examples(rows)
  prompt_ids = {row["id"] for row in prompts}
  row_ids = {row["id"] for row in rows}
  if prompt_ids != row_ids: raise ValueError("prompt/example id mismatch")
  train_rows = [row for row in rows if row["split"] == "train"]
  holdout_rows = [row for row in rows if row["split"] == "holdout"]
  warnings = []
  if len(rows) < 30: warnings.append("dataset has fewer than 30 examples")
  if len(holdout_rows) < 8: warnings.append("holdout has fewer than 8 examples")
  if len(set(row["label"] for row in holdout_rows)) < 2: warnings.append("holdout has fewer than 2 label classes")
  if not any(row["label"] in {"accept", "raw_accept_unconfirmed", "needs_rerun"} for row in holdout_rows):
    warnings.append("holdout has no useful labels for ranking metrics")
  return {
    "kind": "qk_flywheel_kernel_triage_dataset",
    "rows": len(rows),
    "prompts": len(prompts),
    "split_policy": "family_split_v0",
    "train_rows": len(train_rows),
    "holdout_rows": len(holdout_rows),
    "integrity": integrity,
    "warnings": warnings,
    "files": {
      "examples": "examples.jsonl",
      "prompts": "prompts.jsonl",
      "train_prompts": "prompts-train.jsonl",
      "holdout_prompts": "prompts-holdout.jsonl",
      "summary": "summary.json",
      "readme": "README.md",
    },
  }

def _readme(summary:dict[str, Any]) -> str:
  lines = [
    "# AMD Decode Flywheel Kernel Triage Dataset",
    "",
    "This Phase 1 artifact converts existing QK/kernel experiment artifacts into",
    "structured candidate-outcome examples. Prompt rows expose pre-result context;",
    "example rows retain hidden labels, reasons, and evidence for deterministic",
    "Phase 2 triage evaluation.",
    "",
    f"- rows: `{summary['rows']}`",
    f"- train rows: `{summary['train_rows']}`",
    f"- holdout rows: `{summary['holdout_rows']}`",
    f"- split policy: `{summary['split_policy']}`",
    "- prompt contract: `/no_think`, strict compact JSON, `max_tokens=64`",
    "",
    "## Labels",
    "",
    "| label | rows |",
    "|---|---:|",
  ]
  for label, count in summary["integrity"]["labels"].items(): lines.append(f"| `{label}` | {count} |")
  lines += ["", "## Mechanisms", "", "| mechanism | rows |", "|---|---:|"]
  for mech, count in summary["integrity"]["mechanisms"].items(): lines.append(f"| `{mech}` | {count} |")
  if summary["warnings"]:
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in summary["warnings"]]
  lines.append("")
  return "\n".join(lines)

def write_dataset(repo:pathlib.Path, out:pathlib.Path) -> dict[str, Any]:
  rows = build_examples(repo)
  prompts = [_prompt(row) for row in rows]
  summary = _summary(rows, prompts)
  out.mkdir(parents=True, exist_ok=True)
  _jsonl(out / "examples.jsonl", rows)
  _jsonl(out / "prompts.jsonl", prompts)
  _jsonl(out / "prompts-train.jsonl", [row for row in prompts if row["split"] == "train"])
  _jsonl(out / "prompts-holdout.jsonl", [row for row in prompts if row["split"] == "holdout"])
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
  (out / "README.md").write_text(_readme(summary))
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Build AMD decode flywheel kernel-history triage dataset")
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--out", type=pathlib.Path, required=True)
  args = parser.parse_args()
  summary = write_dataset(args.repo, args.out)
  print(json.dumps(summary, indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
