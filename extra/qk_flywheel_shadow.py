#!/usr/bin/env python3
"""Phase 4 v0 live shadow mode for the AMD decode flywheel.

This is a blind, static-stage shadow test: it trains the audited leak-free cost
model on the full plus corpus, predicts/ranks a batch of fresh candidates on
untouched dominant Q4_K tensors BEFORE any GPU run, freezes those predictions,
then (after the deterministic generators run) scores the model against the same
baselines on the fresh batch. The model never steers the work; it only predicts.

The leak-free feature path, the XGBoost classifier/ranker, and the scorer are
reused verbatim from extra.qk_flywheel_cost_model / extra.qk_flywheel_triage_eval
so the shadow predictor shares one audited feature pipeline with the cost model.
"""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, pathlib, re, subprocess
from collections import Counter, defaultdict
from typing import Any

from extra import qk_flywheel_dataset as v0
from extra import qk_flywheel_dataset_v1 as v1
from extra.qk_flywheel_dataset import _label_reason_retry
from extra.qk_flywheel_targeted_outcomes import _mechanism_from_schedule
from extra.qk_flywheel_feature_enrich import enrich_row
from extra.qk_flywheel_cost_model import (
  FORBIDDEN_FEATURE_SOURCES, FeatureVectorizer, _label_policy, _prediction, _xgboost_predictions, extract_feature_map,
)
from extra.qk_flywheel_triage_eval import LABEL_SCORE, build_baseline_predictions, score_predictions
from extra.qk_layout import GGML_Q4_K, pick_tensor, read_metadata, tensor_shape

DEFAULT_ROOT = pathlib.Path("bench/amd-decode-flywheel-proof-20260614")
DEFAULT_CORPUS = DEFAULT_ROOT / "kernel-triage-v1-featured-plus/examples.jsonl"
DEFAULT_OUT = DEFAULT_ROOT / "shadow-v0"
DEFAULT_MODEL = pathlib.Path("~/models/Qwen3-8B-Q4_K_M.gguf")
SEED = 20260614

SHADOW_FAMILY = "shadow_v0_fresh"
SHADOW_FAMILY_ORDER = 12
SHADOW_MODEL = "Qwen3-8B-Q4_K_M"
USEFUL_LABELS = {"accept", "raw_accept_unconfirmed", "needs_rerun"}
DEAD_LABELS = {"reject", "construction_blocked", "diagnostic_only"}
# Curated leakage tokens that mark a target/result field if they appear in a feature
# NAME, mirroring extra.qk_flywheel_cost_model.run_cost_model's own audit. The raw
# FORBIDDEN_FEATURE_SOURCES field list is for documentation only; substring-matching it
# false-positives on legit feature values (e.g. reduction_mode=split_k_partial -> "split").
LEAKAGE_TOKENS = ("label", "reason", "retry", "evidence", "gain", "status", "candidate_gbs", "current_gbs")

# Fresh candidate batch: untouched dominant Q4_K tensors, same mechanism families as
# the corpus. None of these tensors appears in the training corpus (instance-level
# generalization). Chosen for label diversity so the shadow score is not single-label.
FRESH_SPECS: list[dict[str, Any]] = [
  {"mechanism": "packed_word_lane_unroll", "mode": "packed_load", "tensor": "blk.4.ffn_gate.weight", "role": "ffn_gate", "opts": ["LOCAL:0:64"], "parts": 1, "reference": "v1_partial"},
  {"mechanism": "packed_word_lane_unroll", "mode": "packed_load", "tensor": "blk.5.ffn_gate.weight", "role": "ffn_gate", "opts": ["LOCAL:0:64"], "parts": 1, "reference": "v1_partial"},
  {"mechanism": "packed_word_lane_unroll", "mode": "packed_load", "tensor": "blk.6.ffn_gate.weight", "role": "ffn_gate", "opts": ["LOCAL:0:64"], "parts": 1, "reference": "v1_partial"},
  {"mechanism": "qk_block_dot", "mode": "qk_block_dot", "tensor": "blk.0.attn_output.weight", "role": "attn_output", "opts": ["LOCAL:0:32"], "parts": 1, "reference": "v1_partial"},
  {"mechanism": "qk_block_dot", "mode": "qk_block_dot", "tensor": "blk.1.ffn_up.weight", "role": "ffn_up", "opts": ["LOCAL:0:32"], "parts": 1, "reference": "v1_partial"},
  {"mechanism": "wide_load_only", "mode": "v1_partial", "tensor": "blk.0.attn_output.weight", "role": "attn_output", "opts": ["LOCAL:0:32"], "parts": 1, "reference": "qk-block-dot"},
]

# ----- io helpers -------------------------------------------------------------

def _read_jsonl(path:pathlib.Path) -> list[dict[str, Any]]:
  rows = []
  for lineno, raw in enumerate(path.read_text().splitlines(), 1):
    if not raw.strip(): continue
    row = json.loads(raw)
    if not isinstance(row, dict): raise ValueError(f"{path}:{lineno}: expected JSON object")
    rows.append(row)
  return rows

def _jsonl_bytes(rows:list[dict[str, Any]]) -> bytes:
  return ("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)).encode()

def _write_jsonl(path:pathlib.Path, rows:list[dict[str, Any]]) -> None:
  path.write_bytes(_jsonl_bytes(rows))

from extra.llm_eval_common import read_json_object as _read_json

def _sha256(data:bytes) -> str:
  return hashlib.sha256(data).hexdigest()

def _git_commit(repo:pathlib.Path) -> str:
  try:
    out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return out.stdout.strip() if out.returncode == 0 else "unknown"
  except Exception:
    return "unknown"

# ----- fresh candidate construction (static, no GPU) --------------------------

def _tensor_slug(tensor:str) -> str:
  return re.sub(r"[^a-z0-9]+", "-", str(tensor).lower()).strip("-")

def fresh_id(spec:dict[str, Any]) -> str:
  return f"shadow_v0:{spec['mechanism']}:8b:{_tensor_slug(spec['tensor'])}"

def _static_context(spec:dict[str, Any], shape:tuple[int, int]) -> dict[str, Any]:
  # Static stage only: no compile/source/microbench-derived fields (wide_loads,
  # source_ok, comparison) exist before running, so the prediction is genuinely blind.
  return {
    "mode": spec["mode"],
    "reference": spec["reference"],
    "shape": {"rows": int(shape[0]), "k": int(shape[1]), "parts": int(spec["parts"])},
    "parts": int(spec["parts"]),
    "opts": list(spec.get("opts") or []),
    "full_decode_ready": False,
  }

def _raw_shadow_row(spec:dict[str, Any], shape:tuple[int, int], *, label:str, reason:str,
                    retry:bool, evidence:dict[str, Any], source_files:list[str], row_kind:str) -> dict[str, Any]:
  row_id = fresh_id(spec)
  return {
    "id": row_id,
    "candidate_id": row_id.split(":", 1)[-1],
    "row_kind": row_kind,
    "family": SHADOW_FAMILY,
    "family_order": SHADOW_FAMILY_ORDER,
    "model": SHADOW_MODEL,
    "tensor": spec["tensor"],
    "role": spec["role"],
    "format": "Q4_K",
    "mechanism": spec["mechanism"],
    "prediction_stage": "after_static_before_microbench",
    "pre_result_context": _static_context(spec, shape),
    "label": label,
    "reason": reason,
    "retry": retry,
    "evidence": evidence,
    "source_files": source_files,
  }

def _tensor_shapes(model_path:pathlib.Path) -> dict[str, tuple[int, int]]:
  meta = read_metadata(model_path.expanduser())
  shapes = {}
  for spec in FRESH_SPECS:
    if spec["tensor"] in shapes: continue
    info = pick_tensor(meta.infos, spec["tensor"])
    if info.typ != GGML_Q4_K: raise ValueError(f"{spec['tensor']} is ggml_type={info.typ}, expected Q4_K")
    shape = tensor_shape(info)
    if len(shape) != 2: raise ValueError(f"{spec['tensor']} is not a matrix: {shape}")
    shapes[spec["tensor"]] = (int(shape[0]), int(shape[1]))
  return shapes

def build_fresh_candidates(repo:pathlib.Path, model_path:pathlib.Path=DEFAULT_MODEL) -> list[dict[str, Any]]:
  """Unlabeled fresh candidate rows built from static GGUF metadata only (no GPU).

  The placeholder label/reason/retry are never read as features (extract_feature_map
  ignores them and candidate_record.outcome) and never used for prediction
  (the model reads labels only from the training corpus); they exist only so the
  row passes the shared normalize/enrich pipeline identically to corpus rows."""
  repo = repo.resolve()
  shapes = _tensor_shapes(model_path)
  rows = []
  seen = set()
  for spec in FRESH_SPECS:
    raw = _raw_shadow_row(
      spec, shapes[spec["tensor"]],
      label="diagnostic_only", reason="diagnostic_only", retry=False,
      evidence={"outcome_known": False, "note": "frozen blind static-stage shadow candidate"},
      source_files=[], row_kind="candidate",
    )
    normalized = v1.normalize_row(raw)
    enriched = enrich_row(normalized, repo)
    enriched["outcome_known"] = False
    if enriched["id"] in seen: raise ValueError(f"duplicate shadow candidate id {enriched['id']}")
    seen.add(enriched["id"])
    rows.append(enriched)
  return rows

# ----- freeze (train on corpus, predict on fresh, before any GPU run) ---------

def _freeze(repo:pathlib.Path, corpus_path:pathlib.Path, out:pathlib.Path, fresh:list[dict[str, Any]], *,
            kind:str, phase:str, note:str, include_excluded_sources:bool) -> dict[str, Any]:
  """Shared train-on-corpus / predict-on-fresh / freeze-write skeleton.

  The v0 (`freeze_predictions`) and staged (`freeze_staged`) freezes are the same
  pipeline -- fit the leak-free vectorizer on the corpus, train+predict xgboost on
  the fresh batch, hash everything, write candidates/predictions/freeze. They
  differ only in the candidate builder (passed in as `fresh`), the freeze
  `kind`/`phase`/`note`, and whether the leakage audit lists the documented
  `excluded_feature_sources`. _xgboost_predictions trains on the corpus (reads
  labels only from the corpus) and touches the fresh rows only for their id."""
  corpus = _read_jsonl(corpus_path)
  corpus_maps = [extract_feature_map(row) for row in corpus]
  fresh_maps = [extract_feature_map(row) for row in fresh]
  vec = FeatureVectorizer().fit(corpus_maps)
  preds, xgb_meta = _xgboost_predictions(corpus, fresh, vec.transform(corpus_maps), vec.transform(fresh_maps), SEED)

  forbidden = [name for name in vec.names if any(tok in name for tok in LEAKAGE_TOKENS)]
  leakage_audit:dict[str, Any] = {"forbidden_tokens_in_feature_names": forbidden, "leak_free": not forbidden}
  if include_excluded_sources:
    leakage_audit = {"excluded_feature_sources": list(FORBIDDEN_FEATURE_SOURCES), **leakage_audit}
  candidates_bytes, predictions_bytes = _jsonl_bytes(fresh), _jsonl_bytes(preds)
  freeze = {
    "kind": kind,
    "phase": phase,
    "seed": SEED,
    "corpus_path": str(corpus_path),
    "corpus_rows": len(corpus),
    "candidate_rows": len(fresh),
    "candidate_ids": [row["id"] for row in fresh],
    "corpus_sha256": _sha256(corpus_path.read_bytes()),
    "candidates_sha256": _sha256(candidates_bytes),
    "predictions_sha256": _sha256(predictions_bytes),
    "feature_vocab_sha256": _sha256(json.dumps(vec.names, sort_keys=True).encode()),
    "feature_count": len(vec.names),
    "xgboost_meta": xgb_meta,
    "git_commit_at_freeze": _git_commit(repo),
    "leakage_audit": leakage_audit,
    "note": note,
  }
  out.mkdir(parents=True, exist_ok=True)
  out.joinpath("candidates.jsonl").write_bytes(candidates_bytes)
  out.joinpath("predictions.jsonl").write_bytes(predictions_bytes)
  out.joinpath("freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
  return freeze

def freeze_predictions(repo:pathlib.Path, corpus_path:pathlib.Path=DEFAULT_CORPUS,
                       out:pathlib.Path=DEFAULT_OUT, model_path:pathlib.Path=DEFAULT_MODEL) -> dict[str, Any]:
  repo = repo.resolve()
  return _freeze(
    repo, corpus_path, out, build_fresh_candidates(repo, model_path),
    kind="qk_flywheel_shadow_v0_freeze", phase="Phase 4",
    note="Frozen before any fresh GPU run. Commit this file and predictions.jsonl before producing outcomes.jsonl.",
    include_excluded_sources=True,
  )

# ----- run the deterministic generators on the fresh batch (GPU) --------------

def _run(cmd:list[str], *, cwd:pathlib.Path, extra_env:dict[str, str], timeout:float) -> None:
  env = {**os.environ, "PYTHONPATH": ".", **extra_env}
  proc = subprocess.run(cmd, cwd=cwd, env=env, timeout=timeout)
  if proc.returncode != 0: raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")

def _build_packed_descriptor(repo:pathlib.Path, out_json:pathlib.Path) -> None:
  src = _read_json(repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json")
  blk0 = next(r for r in src["descriptors"] if r.get("tensor") == "blk.0.ffn_gate.weight")
  data_start = blk0["layout"]["data_start"]
  meta = read_metadata(DEFAULT_MODEL.expanduser())
  descs = []
  for spec in FRESH_SPECS:
    if spec["mechanism"] != "packed_word_lane_unroll": continue
    info = pick_tensor(meta.infos, spec["tensor"])
    row = copy.deepcopy(blk0)
    row["tensor"] = spec["tensor"]
    row["layout"]["tensor_offset"] = int(info.off)
    row["layout"]["byte_start"] = int(data_start + info.off)
    descs.append(row)
  out = {
    "kind": "qk_semantic_descriptor_set", "generator_version": 1, "schema_version": 1,
    "model": src["model"], "model_size": src["model_size"], "source_policy": src.get("source_policy"),
    "phase": "Phase 4 shadow-v0",
    "note": "Shadow-v0 fresh packed-load descriptor set on untouched dominant Q4_K ffn_gate tensors.",
    "descriptors": descs,
    "summary": {"entries": len(descs), "by_role": {"ffn_gate": len(descs)}, "by_format": {"Q4_K": len(descs)}},
  }
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")

def _packed_artifact(out:pathlib.Path) -> pathlib.Path:
  return out / "runs/packed-load/8b"

def _block_dot_artifact(out:pathlib.Path, tensor:str) -> pathlib.Path:
  return out / f"runs/block-dot-{_tensor_slug(tensor)}"

def _threeway_artifact(out:pathlib.Path, tensor:str) -> pathlib.Path:
  return out / f"runs/threeway-{_tensor_slug(tensor)}"

def run_outcomes(repo:pathlib.Path, out:pathlib.Path=DEFAULT_OUT, model_path:pathlib.Path=DEFAULT_MODEL,
                 device:str="AMD") -> None:
  repo = repo.resolve()
  py = ".venv/bin/python"
  model = str(model_path)
  # packed-load: fresh descriptor set -> v3 candidates/static-gate -> schedule_bench microbench
  packed = _packed_artifact(out)
  if not (packed / "microbench.json").exists():
    raise FileNotFoundError(
      "qk_flywheel_shadow packed-load microbench JSON is absent; qk_semantic_codegen_v3.py was removed and this replay path is no longer runnable."
    )
  # block-dot replay path removed: qk_block_dot_compile_gate.py / qk_block_dot_microbench.py were
  # deleted (Ops.QK_BLOCK_DOT gone from core; microbench rejected, -30% to -90%). The Phase-4
  # shadow is concluded, so this generator is no longer runnable past the packed-load gate above.
  # three-way load diagnostic
  for spec in FRESH_SPECS:
    if spec["mechanism"] != "wide_load_only": continue
    art = _threeway_artifact(out, spec["tensor"])
    if (art / "microbench.json").exists(): continue
    _run([py, "extra/qk_threeway_load_microbench.py", "--tensor", spec["tensor"],
          "--runs", "3", "--iters", "3", "--artifact", str(art), "--model", model, "--device", device],
         cwd=repo, extra_env={"DEV": device}, timeout=900)

# ----- outcomes (labeled rows sharing the candidate ids) ----------------------

def build_fresh_outcomes(repo:pathlib.Path, out:pathlib.Path=DEFAULT_OUT, model_path:pathlib.Path=DEFAULT_MODEL) -> list[dict[str, Any]]:
  repo = repo.resolve()
  shapes = _tensor_shapes(model_path)
  packed = _read_json(_packed_artifact(out) / "microbench.json") if (_packed_artifact(out) / "microbench.json").exists() else {"rows": []}
  packed_by_tensor = {str(r.get("tensor")): r for r in packed.get("rows", [])}
  rows = []
  for spec in FRESH_SPECS:
    mech, tensor = spec["mechanism"], spec["tensor"]
    if mech == "packed_word_lane_unroll":
      item = packed_by_tensor.get(tensor)
      if item is None: raise FileNotFoundError(f"no packed-load outcome for {tensor}; run run_outcomes first")
      status, gain = item.get("status"), item.get("gain")
      label, reason, retry = _label_reason_retry(status, gain=gain)
      evidence = {"status": status, "gain": gain, "candidate_quant_gbs": (item.get("candidate") or {}).get("quant_gbs"),
                  "current_quant_gbs": (item.get("current") or {}).get("quant_gbs"), "reasons": item.get("reasons", [])}
      source = str((_packed_artifact(out) / "microbench.json").relative_to(repo)) if (_packed_artifact(out) / "microbench.json").is_relative_to(repo) else str(_packed_artifact(out) / "microbench.json")
    elif mech == "qk_block_dot":
      art = _block_dot_artifact(out, tensor)
      path, blocked = art / "microbench.json", art / "shadow-outcome.json"
      if path.exists():
        data = _read_json(path)
        gain_pct = (data.get("comparison") or {}).get("gain_pct")
        label, reason, retry = _label_reason_retry("reject", gain=gain_pct / 100.0 if gain_pct is not None else None)
        evidence = {"decision": (data.get("summary") or {}).get("decision"), "gain_pct": gain_pct,
                    "correctness_ok": (data.get("summary") or {}).get("correctness_ok")}
        source = str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
      elif blocked.exists():
        marker = _read_json(blocked)
        label, reason, retry = _label_reason_retry("construction_blocked")
        evidence = {"status": "construction_blocked", "reason": marker.get("reason"), "decision": "qk_block_dot_correctness_failed"}
        source = str(blocked.relative_to(repo)) if blocked.is_relative_to(repo) else str(blocked)
      else:
        raise FileNotFoundError(f"no block-dot outcome for {tensor}; run run_outcomes first")
    elif mech == "wide_load_only":
      path = _threeway_artifact(out, tensor) / "microbench.json"
      if not path.exists(): raise FileNotFoundError(f"no three-way outcome for {tensor}; run run_outcomes first")
      data = _read_json(path)
      trow = next((t for t in data.get("tensors", []) if t.get("tensor") == tensor), {})
      label, reason, retry = "diagnostic_only", "diagnostic_only", False
      evidence = {"status": "pass", "decision": trow.get("decision"), "reason": trow.get("reason")}
      source = str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)
    else:
      raise ValueError(f"unknown mechanism {mech}")
    raw = _raw_shadow_row(spec, shapes[tensor], label=label, reason=reason, retry=retry,
                          evidence=evidence, source_files=[source],
                          row_kind="diagnostic" if mech == "wide_load_only" else "candidate")
    row = enrich_row(v1.normalize_row(raw), repo)
    row["split"] = "holdout"
    row["outcome_known"] = True
    rows.append(row)
  return rows

# ----- scoring ----------------------------------------------------------------

def _first_live_position(ordered_labels:list[str]) -> int | None:
  for idx, label in enumerate(ordered_labels):
    if label in USEFUL_LABELS: return idx
  return None

def dead_branch_metric(holdout:list[dict[str, Any]], predictions:list[dict[str, Any]], *, top_k:int=3) -> dict[str, Any]:
  by_id = {row["id"]: row for row in holdout}
  pred_by_id = {p["id"]: p for p in predictions}
  ordered = sorted(holdout, key=lambda row: (-float(pred_by_id.get(row["id"], {}).get("score", 0.0)), row["id"]))
  ordered_labels = [row["label"] for row in ordered]
  first_live = _first_live_position(ordered_labels)
  return {
    "experiments_to_first_live": first_live,
    "live_candidates": sum(1 for row in holdout if row["label"] in USEFUL_LABELS),
    "dead_in_top_k": sum(1 for row in ordered[:top_k] if row["label"] in DEAD_LABELS),
    "top_k": top_k,
    "order": [row["id"] for row in ordered],
  }

def _shadow_gate(model:dict[str, Any], prior:dict[str, Any], model_dead:dict[str, Any], heuristic_dead:dict[str, Any]) -> dict[str, Any]:
  mr, pr = model.get("ranking", {}), prior.get("ranking", {})
  beats_f1 = model.get("macro_f1", 0.0) > prior.get("macro_f1", 0.0)
  rank_improved = (
    (mr.get("ndcg") is not None and pr.get("ndcg") is not None and mr["ndcg"] > pr["ndcg"]) or
    (mr.get("precision_at_3") is not None and pr.get("precision_at_3") is not None and mr["precision_at_3"] > pr["precision_at_3"]) or
    (mr.get("precision_at_1") is not None and pr.get("precision_at_1") is not None and mr["precision_at_1"] > pr["precision_at_1"])
  )
  fp_ok = model.get("false_positive_accept_rate", 1.0) <= 0.05
  m_live, h_live = model_dead.get("experiments_to_first_live"), heuristic_dead.get("experiments_to_first_live")
  fewer_dead = (
    (m_live is not None and h_live is not None and m_live <= h_live and model_dead["dead_in_top_k"] <= heuristic_dead["dead_in_top_k"])
    or (model_dead.get("live_candidates", 0) > 0 and m_live == 0)
  )
  reasons = []
  if not beats_f1: reasons.append("macro_f1 not above mechanism_prior")
  if not rank_improved: reasons.append("ranking not above mechanism_prior")
  if not fp_ok: reasons.append("false_positive_accept_rate > 0.05")
  if model_dead.get("live_candidates", 0) == 0: reasons.append("no live candidate in fresh batch (ranking undefined)")
  elif not fewer_dead: reasons.append("did not reduce dead branches vs simple_family_heuristic")
  met = beats_f1 and rank_improved and fp_ok and model_dead.get("live_candidates", 0) > 0 and fewer_dead
  return {"shadow_gate_met": met, "blockers": reasons}

def score_shadow(repo:pathlib.Path, out:pathlib.Path=DEFAULT_OUT, corpus_path:pathlib.Path=DEFAULT_CORPUS,
                 model_path:pathlib.Path=DEFAULT_MODEL) -> dict[str, Any]:
  repo = repo.resolve()
  corpus = _read_jsonl(corpus_path)
  train_rows = [{**row, "split": "train"} for row in corpus]  # shadow trains on the full corpus
  holdout_rows = build_fresh_outcomes(repo, out, model_path)
  examples = train_rows + holdout_rows
  model_preds = _read_jsonl(out / "predictions.jsonl")

  baselines = build_baseline_predictions(examples, seed=SEED)
  baseline_metrics = {name: score_predictions(examples, preds) for name, preds in baselines.items()}
  model_metrics = score_predictions(examples, model_preds)

  dead = {"xgboost": dead_branch_metric(holdout_rows, model_preds)}
  for name, preds in baselines.items():
    dead[name] = dead_branch_metric(holdout_rows, preds)

  prior = baseline_metrics.get("mechanism_prior", {})
  heuristic_dead = dead.get("simple_family_heuristic", {})
  gate = _shadow_gate(model_metrics, prior, dead["xgboost"], heuristic_dead)

  if gate["shadow_gate_met"]:
    conclusion = "shadow_gate_met_model_beats_priors_on_fresh_batch"
  elif dead["xgboost"].get("live_candidates", 0) == 0:
    conclusion = "shadow_inconclusive_no_live_candidate_in_fresh_batch_model_underperforms_prior"
  else:
    conclusion = "shadow_gate_not_met_model_underperforms_prior"

  summary = {
    "kind": "qk_flywheel_shadow_v0_score",
    "phase": "Phase 4",
    "conclusion": conclusion,
    "seed": SEED,
    "corpus_rows": len(corpus),
    "fresh_rows": len(holdout_rows),
    "fresh_label_distribution": dict(sorted(Counter(r["label"] for r in holdout_rows).items())),
    "model": "xgboost",
    "model_metrics": {"xgboost": model_metrics},
    "baseline_metrics": baseline_metrics,
    "dead_branch": dead,
    "gate": gate,
    "freeze": _read_json(out / "freeze.json") if (out / "freeze.json").exists() else None,
  }
  _write_jsonl(out / "outcomes.jsonl", holdout_rows)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  (out / "README.md").write_text(_readme(summary))
  return summary

def _readme(summary:dict[str, Any]) -> str:
  def fmt(v:Any) -> str: return "n/a" if v is None else f"{v:.3f}"
  m = summary["model_metrics"]["xgboost"]
  rows = {"xgboost": m}
  for name in ("mechanism_prior", "simple_family_heuristic", "reject_all"):
    if name in summary["baseline_metrics"]: rows[name] = summary["baseline_metrics"][name]
  lines = [
    "# AMD Decode Flywheel Phase 4 Shadow v0",
    "",
    "Blind static-stage shadow: the cost model predicted/ranked a fresh batch of",
    "candidates on untouched dominant Q4_K tensors BEFORE any GPU run (frozen in",
    "`predictions.jsonl` / `freeze.json`), then was scored against the same",
    "baselines after the deterministic generators produced outcomes.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- shadow gate met: `{summary['gate']['shadow_gate_met']}`",
    f"- fresh rows: `{summary['fresh_rows']}`",
    f"- fresh label distribution: `{summary['fresh_label_distribution']}`",
    "",
    "## Metrics (fresh batch)",
    "",
    "| model | accuracy | macro-F1 | false accept | p@1 | p@3 | ndcg | first-live |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
  ]
  for name, row in rows.items():
    rk = row["ranking"]
    fl = summary["dead_branch"].get(name, {}).get("experiments_to_first_live")
    lines.append(f"| `{name}` | {fmt(row['accuracy'])} | {fmt(row['macro_f1'])} | {fmt(row['false_positive_accept_rate'])} | "
                 f"{fmt(rk['precision_at_1'])} | {fmt(rk['precision_at_3'])} | {fmt(rk['ndcg'])} | {fl if fl is not None else 'n/a'} |")
  lines += ["", "## Gate", "", f"- met: `{summary['gate']['shadow_gate_met']}`"]
  for blocker in summary["gate"]["blockers"]: lines.append(f"- blocker: {blocker}")
  lines += [
    "",
    "## Interpretation",
    "",
    "v0 is blind static-stage, instance-level generalization (new tensors, same",
    "mechanism families). A failed gate is a real result under the Phase 4 stop",
    "rule: the model stays documentation-only and does not steer execution order.",
    "",
  ]
  return "\n".join(lines)

# ==============================================================================
# Phase 4.1: Cost-Aware Staged Shadow
#
# v0 showed the cost model does not beat mechanism_prior on labels, and the corpus
# explains why (live outcomes cluster in semantic-schedule families; memory-access
# probes never win; identical-shape intra-family wins are weight-determined and
# unobservable). Phase 4.1 reframes to the actual flywheel value -- wasted GPU
# reduction -- on a live-bearing batch of semantic-schedule candidates that span
# roles/shapes/opts (observable variation the model could exploit over a
# mechanism-only prior).
#
# A pre-microbench gate is VALID only if it keeps 100% of the truly-live
# candidates. Its value is the number of expensive microbench experiments it can
# SAFELY skip: rank candidates by the gate score, skip everything scored below the
# lowest-scored live candidate. More dead candidates below the live floor = more
# experiments saved at full recall. This uses the continuous rank score and is
# immune to the model's conservative label bias.
# ==============================================================================

STAGED_OUT = DEFAULT_ROOT / "shadow-staged"
# Fresh, untouched, eligible Q4_K dominant-role tensors (role in ffn_gate/attn_q,
# parts=1) -> each yields 4 schedule mechanisms (direct_output, row_upcast,
# reduce_unroll, two_dim_local). Two shapes (ffn_gate 12288x4096, attn_q 4096x4096).
STAGED_SCHEDULE_TENSORS = ("blk.7.ffn_gate.weight", "blk.8.ffn_gate.weight", "blk.1.attn_q.weight", "blk.2.attn_q.weight")

# Phase 4.2: bigger multi-block batch. Centered on fresh attn_q (the live region:
# row_upcast 75% live, direct_output 42% live) for >=5 live across two patterns,
# with fresh ffn_gate blocks as dead controls. None appear in the corpus.
STAGED_V2_OUT = DEFAULT_ROOT / "shadow-staged-v2"
STAGED_V2_TENSORS = tuple([f"blk.{i}.attn_q.weight" for i in range(3, 10)] + [f"blk.{i}.ffn_gate.weight" for i in range(9, 12)])

# Phase 4.3 replication: K independent frozen batches, each an attn_q live control
# (row_upcast/direct_output win there) plus a larger surprise-prone ffn_gate block
# (row_upcast/direct_output there have ~0 historical live but can win, as 4.2 found).
# Fresh blocks not used in the corpus or prior batches.
def _staged_batch(attn_q_blocks:range, ffn_gate_blocks:range) -> tuple[str, ...]:
  return tuple([f"blk.{i}.attn_q.weight" for i in attn_q_blocks] + [f"blk.{i}.ffn_gate.weight" for i in ffn_gate_blocks])

STAGED_BATCHES: dict[str, tuple[pathlib.Path, tuple[str, ...]]] = {
  "v3": (DEFAULT_ROOT / "shadow-staged-v3", _staged_batch(range(10, 13), range(12, 17))),
  "v4": (DEFAULT_ROOT / "shadow-staged-v4", _staged_batch(range(13, 16), range(17, 22))),
  "v5": (DEFAULT_ROOT / "shadow-staged-v5", _staged_batch(range(16, 19), range(22, 27))),
}
STAGED_POOL_OUT = DEFAULT_ROOT / "shadow-staged-pool"
RECALL_LEVELS = (1.0, 0.95, 0.90)


def _staged_descriptor_path(out:pathlib.Path) -> pathlib.Path:
  return out / "runs/schedule/8b-descriptors.json"

def _staged_schedule_dir(out:pathlib.Path) -> pathlib.Path:
  return out / "runs/schedule/8b"

def build_staged_descriptor(repo:pathlib.Path, out:pathlib.Path=STAGED_OUT, tensors:tuple[str, ...]=STAGED_SCHEDULE_TENSORS) -> pathlib.Path:
  src = _read_json(repo / "bench/qk-ansor-transition-20260612/descriptors/8b.json")
  by_role = {r["role"]: r for r in src["descriptors"]}
  data_start = by_role["ffn_gate"]["layout"]["data_start"]
  meta = read_metadata(DEFAULT_MODEL.expanduser())
  descs = []
  for tensor in tensors:
    role = v0._role_from_text(tensor)
    if role not in by_role: raise ValueError(f"no descriptor template for role {role!r} ({tensor})")
    info = pick_tensor(meta.infos, tensor)
    row = copy.deepcopy(by_role[role])
    row["tensor"] = tensor
    row["layout"]["tensor_offset"] = int(info.off)
    row["layout"]["byte_start"] = int(data_start + info.off)
    descs.append(row)
  doc = {
    "kind": "qk_semantic_descriptor_set", "generator_version": 1, "schema_version": 1,
    "model": src["model"], "model_size": src["model_size"], "source_policy": src.get("source_policy"),
    "phase": "Phase 4.1 shadow-staged",
    "note": "Fresh dominant Q4_K tensors (ffn_gate, attn_q) for live-bearing semantic-schedule candidates.",
    "descriptors": descs, "summary": {"entries": len(descs)},
  }
  path = _staged_descriptor_path(out)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
  return path

def _generate_staged_candidates(repo:pathlib.Path, out:pathlib.Path, tensors:tuple[str, ...]=STAGED_SCHEDULE_TENSORS) -> dict[str, Any]:
  # Candidate generation is GPU-free; only the microbench step needs the device.
  sched = _staged_schedule_dir(out)
  cand_path = sched / "candidates.json"
  if not cand_path.exists():
    desc = build_staged_descriptor(repo, out, tensors)
    _run([".venv/bin/python", "extra/qk_semantic_schedule.py", "--descriptor", str(desc),
          "--json", str(cand_path), "--gate-json", str(sched / "static-gate.json")],
         cwd=repo, extra_env={}, timeout=300)
  return _read_json(cand_path)

def _staged_candidate_rows(repo:pathlib.Path, out:pathlib.Path, tensors:tuple[str, ...]=STAGED_SCHEDULE_TENSORS) -> list[dict[str, Any]]:
  repo = repo.resolve()
  data = _generate_staged_candidates(repo, out, tensors)
  meta = read_metadata(DEFAULT_MODEL.expanduser())
  rows = []
  for cand in data.get("candidates", []):
    if cand.get("id") == "current": continue
    change = (cand.get("changes") or [{}])[0]
    spec = cand.get("schedule_spec") or change.get("schedule_spec") or {}
    tensor = str(change.get("tensor") or "unknown")
    role = str(change.get("role") or v0._role_from_text(tensor))
    shape = tensor_shape(pick_tensor(meta.infos, tensor)) if tensor != "unknown" else (0, 0)
    mechanism = _mechanism_from_schedule(spec.get("name"), cand["id"])
    raw = {
      "id": f"shadow_staged:8b:{cand['id']}",
      "candidate_id": cand["id"],
      "row_kind": "candidate",
      "family": SHADOW_FAMILY, "family_order": SHADOW_FAMILY_ORDER,
      "model": SHADOW_MODEL, "tensor": tensor, "role": role, "format": "Q4_K",
      "mechanism": mechanism,
      "prediction_stage": "after_static_before_microbench",
      "pre_result_context": {
        "mode": spec.get("name"), "candidate_id": cand["id"], "schedule": spec,
        "shape": {"rows": int(shape[0]), "k": int(shape[1]), "parts": int(spec.get("parts") or 1)},
        "opts": list(spec.get("opts") or []), "parts": int(spec.get("parts") or 1),
        "full_decode_supported": spec.get("full_decode_supported"),
      },
      "label": "diagnostic_only", "reason": "diagnostic_only", "retry": False,
      "evidence": {"outcome_known": False}, "source_files": [],
    }
    row = enrich_row(v1.normalize_row(raw), repo)
    row["outcome_known"] = False
    rows.append(row)
  return rows

def freeze_staged(repo:pathlib.Path, corpus_path:pathlib.Path=DEFAULT_CORPUS, out:pathlib.Path=STAGED_OUT,
                  tensors:tuple[str, ...]=STAGED_SCHEDULE_TENSORS) -> dict[str, Any]:
  repo = repo.resolve()
  return _freeze(
    repo, corpus_path, out, _staged_candidate_rows(repo, out, tensors),
    kind="qk_flywheel_shadow_staged_freeze", phase="Phase 4.1",
    note="Frozen keep/skip rank scores before any microbench. Each candidate's score gates the expensive microbench.",
    include_excluded_sources=False,
  )

def run_staged(repo:pathlib.Path, out:pathlib.Path=STAGED_OUT, device:str="AMD") -> None:
  repo = repo.resolve()
  sched = _staged_schedule_dir(out)
  if (sched / "microbench.json").exists(): return
  _generate_staged_candidates(repo, out)  # ensure candidates/static-gate exist
  _run([".venv/bin/python", "extra/qk_semantic_schedule_bench.py", "--model", "8b",
        "--candidates", str(sched / "candidates.json"), "--static-gate", str(sched / "static-gate.json"),
        "--out", str(sched / "microbench-runs"), "--json", str(sched / "microbench.json"),
        "--md", str(sched / "microbench.md"), "--device", device, "--iters", "3"],
       cwd=repo, extra_env={"DEV": device}, timeout=1200)

def build_staged_outcomes(repo:pathlib.Path, out:pathlib.Path=STAGED_OUT) -> list[dict[str, Any]]:
  repo = repo.resolve()
  micro = _read_json(_staged_schedule_dir(out) / "microbench.json")
  by_cand = {str(r.get("id")): r for r in micro.get("rows", [])}
  fresh = _staged_candidate_rows(repo, out)
  rows = []
  for cand_row in fresh:
    cid = cand_row["candidate_id"]
    item = by_cand.get(cid)
    if item is None: raise FileNotFoundError(f"no microbench outcome for {cid}; run run_staged first")
    status, gain = item.get("status"), item.get("gain")
    label, reason, retry = _label_reason_retry(status, gain=gain)
    micro_seconds = float((item.get("current") or {}).get("elapsed_s") or 0.0) + float((item.get("candidate") or {}).get("elapsed_s") or 0.0)
    raw = copy.deepcopy(cand_row)
    raw["label"], raw["reason"], raw["retry"] = label, reason, retry
    raw["evidence"] = {"status": status, "gain": gain, "reasons": item.get("reasons", [])}
    raw["split"] = "holdout"
    raw["outcome_known"] = True
    raw["microbench_experiments"] = 1
    raw["microbench_seconds"] = round(micro_seconds, 3)
    rows.append(raw)
  return rows

def _majority_label(counter:Counter) -> str:
  return sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0] if counter else "reject"

def _always_construction_blocked_cells(train:list[dict[str, Any]]) -> set[tuple[Any, Any]]:
  # (role, mechanism) cells whose training rows are 100% construction_blocked: known-broken
  # schedule classes a deterministic gate can skip outright (the fair classifier baseline).
  cell = defaultdict(set)
  for r in train:
    cell[(r.get("role"), r.get("mechanism"))].add(r["label"])
  return {k for k, labels in cell.items() if labels == {"construction_blocked"}}

def _role_mechanism_prior(train:list[dict[str, Any]], holdout:list[dict[str, Any]]) -> list[dict[str, Any]]:
  # Stronger deterministic baseline than mechanism_prior: predict the majority label
  # among train rows matching BOTH role and mechanism, falling back to mechanism-only
  # (then global) when the (role, mechanism) cell is empty. This is the cheap lookup
  # the learned model must strictly beat to be worth shipping.
  cell, mech, allc = defaultdict(Counter), defaultdict(Counter), Counter()
  for r in train:
    cell[(r.get("role"), r.get("mechanism"))][r["label"]] += 1
    mech[r.get("mechanism")][r["label"]] += 1
    allc[r["label"]] += 1
  global_majority = _majority_label(allc)
  preds = []
  for r in holdout:
    key = (r.get("role"), r.get("mechanism"))
    if cell[key]: label = _majority_label(cell[key])
    elif mech[r.get("mechanism")]: label = _majority_label(mech[r.get("mechanism")])
    else: label = global_majority
    preds.append({"id": r["id"], "baseline": "role_mechanism_prior", "label": label, "score": LABEL_SCORE.get(label, 0.0)})
  return preds

def _safe_skips(scored:list[tuple[float, bool]]) -> dict[str, Any]:
  # scored: list of (gate_score, is_live). A gate keeping 100% live can safely skip
  # every candidate scored strictly below the lowest-scored live candidate.
  live_scores = [s for s, live in scored if live]
  total = len(scored)
  if not live_scores:
    return {"safe_skips": total, "live_floor_score": None, "live_candidates": 0, "valid": True}
  floor = min(live_scores)
  safe = sum(1 for s, _ in scored if s < floor)
  return {"safe_skips": safe, "live_floor_score": round(float(floor), 6), "live_candidates": len(live_scores), "valid": True}

def _savings_at_recall(scored:list[tuple[float, bool]], recall:float) -> dict[str, Any]:
  # Relax the all-or-nothing 100%-recall floor: to retain `recall` of live candidates
  # we may drop the lowest-scored live ones. The floor becomes the lowest live score we
  # must keep; everything below it is skipped. This defangs the single-surprise-winner
  # brittleness of the 100% point -- one mis-ranked winner can be among the allowed misses.
  live = sorted(s for s, live in scored if live)
  n_live, total = len(live), len(scored)
  if n_live == 0:
    return {"recall": recall, "saved": total, "missed_live": 0, "actual_recall": 1.0}
  allowed_miss = max(0, n_live - math.ceil(recall * n_live))
  floor = math.inf if allowed_miss >= n_live else live[allowed_miss]
  saved = sum(1 for s, _ in scored if s < floor)
  missed = sum(1 for s in live if s < floor)
  return {"recall": recall, "saved": saved, "missed_live": missed, "actual_recall": round((n_live - missed) / n_live, 4)}

def score_staged(repo:pathlib.Path, out:pathlib.Path=STAGED_OUT, corpus_path:pathlib.Path=DEFAULT_CORPUS) -> dict[str, Any]:
  repo = repo.resolve()
  corpus = [{**row, "split": "train"} for row in _read_jsonl(corpus_path)]
  outcomes = build_staged_outcomes(repo, out)
  examples = corpus + outcomes
  model_preds = _read_jsonl(out / "predictions.jsonl")
  baselines = build_baseline_predictions(examples, seed=SEED)

  live_by_id = {row["id"]: (row["label"] in USEFUL_LABELS) for row in outcomes}
  total_experiments = len(outcomes)
  total_live = sum(live_by_id.values())

  outcome_by_id = {row["id"]: row for row in outcomes}
  def gate_for(preds:list[dict[str, Any]]) -> dict[str, Any]:
    score_by_id = {p["id"]: float(p.get("score", 0.0)) for p in preds}
    scored = [(score_by_id.get(rid, 0.0), live_by_id[rid]) for rid in live_by_id]
    res = _safe_skips(scored)
    res["experiments_run"] = total_experiments - res["safe_skips"]
    res["experiments_saved_vs_run_all"] = res["safe_skips"]
    res["live_recall"] = 1.0  # by construction: never skips a candidate below the live floor that is live
    res["recall_curve"] = {f"{lvl:.2f}": _savings_at_recall(scored, lvl) for lvl in RECALL_LEVELS}
    # The safe-skip metric is hostage to the worst-ranked true winner: the live candidate
    # with the minimum gate score sets the floor and caps all savings. Record it so a single
    # surprise winner driving (or collapsing) a gate's savings is visible, not hidden.
    live_ids = [rid for rid in live_by_id if live_by_id[rid]]
    if live_ids:
      floor_id = min(live_ids, key=lambda rid: score_by_id.get(rid, 0.0))
      fo = outcome_by_id.get(floor_id, {})
      res["floor_setter"] = {"id": floor_id, "role": fo.get("role"), "mechanism": fo.get("mechanism"),
                             "score": round(score_by_id.get(floor_id, 0.0), 6)}
    return res

  pred_sets = {"xgboost": model_preds, "role_mechanism_prior": _role_mechanism_prior(corpus, outcomes)}
  for name, preds in baselines.items():
    pred_sets[name] = preds
  gates = {name: gate_for(preds) for name, preds in pred_sets.items()}
  gates["run_all"] = {"safe_skips": 0, "experiments_run": total_experiments, "experiments_saved_vs_run_all": 0,
                      "live_recall": 1.0, "live_candidates": total_live, "valid": True}

  # Per (role x mechanism) cell: how many dead candidates each gate safely skips, so a
  # single dominant pattern cannot hide a miss on another winning combo.
  cells = sorted({(r["role"], r["mechanism"]) for r in outcomes})
  per_cell = {}
  for role, mech in cells:
    members = [r for r in outcomes if r["role"] == role and r["mechanism"] == mech]
    live = sum(1 for r in members if live_by_id[r["id"]])
    entry = {"n": len(members), "live": live, "dead": len(members) - live}
    for name in ("xgboost", "role_mechanism_prior", "mechanism_prior"):
      floor = gates[name].get("live_floor_score")
      score_by_id = {p["id"]: float(p.get("score", 0.0)) for p in pred_sets[name]}
      entry[f"{name}_dead_skipped"] = sum(1 for r in members if not live_by_id[r["id"]] and floor is not None and score_by_id.get(r["id"], 0.0) < floor)
    per_cell[f"{role} x {mech}"] = entry

  model_saved = gates["xgboost"]["safe_skips"]
  rolemech_saved = gates["role_mechanism_prior"]["safe_skips"]
  mech_saved = gates.get("mechanism_prior", {}).get("safe_skips", 0)
  live_patterns = sum(1 for c in per_cell.values() if c["live"] > 0)
  if total_live < 5 or live_patterns < 2:
    conclusion = "inconclusive_insufficient_live_candidates_or_patterns"
  elif model_saved > rolemech_saved and model_saved > 0:
    conclusion = "cost_model_strictly_beats_role_mechanism_prior_ship_the_model"
  elif rolemech_saved > 0 and model_saved <= rolemech_saved:
    conclusion = "role_mechanism_prior_suffices_ship_the_lookup_model_documentation_only"
  elif mech_saved > 0:
    conclusion = "only_mechanism_prior_reduces_experiments"
  else:
    conclusion = "no_safe_pre_result_savings_run_all_is_the_only_full_recall_strategy"

  summary = {
    "kind": "qk_flywheel_shadow_staged_score", "phase": "Phase 4.2", "conclusion": conclusion, "seed": SEED,
    "corpus_rows": len(corpus), "fresh_rows": total_experiments,
    "fresh_label_distribution": dict(sorted(Counter(r["label"] for r in outcomes).items())),
    "live_candidates": total_live, "live_patterns": live_patterns,
    "metric": "max microbench experiments safely skipped at 100% live-recall (skip below lowest-scored live candidate)",
    "gates": gates,
    "per_cell": per_cell,
    "ablation": {
      "ladder": ["run_all", "mechanism_prior", "role_mechanism_prior", "xgboost"],
      "model_safe_skips": model_saved, "role_mechanism_prior_safe_skips": rolemech_saved,
      "mechanism_prior_safe_skips": mech_saved,
      "model_beats_role_mechanism_prior": model_saved > rolemech_saved,
      "simplest_sufficient_gate": ("xgboost" if model_saved > rolemech_saved and model_saved > 0 else
                                   "role_mechanism_prior" if rolemech_saved > 0 else
                                   "mechanism_prior" if mech_saved > 0 else "run_all"),
    },
    "freeze": _read_json(out / "freeze.json") if (out / "freeze.json").exists() else None,
  }
  _write_jsonl(out / "outcomes.jsonl", outcomes)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  (out / "README.md").write_text(_staged_readme(summary))
  return summary

def _staged_readme(summary:dict[str, Any]) -> str:
  g = summary["gates"]
  ab = summary.get("ablation", {})
  lines = [
    "# AMD Decode Flywheel Phase 4.2 Cost-Aware Staged Shadow (Minimal-Gate Ablation)",
    "",
    "Wasted-GPU reduction in shadow. A pre-microbench gate is valid only if it keeps",
    "100% of live candidates; its value is how many microbench experiments it can",
    "safely skip (skip everything scored below the lowest-scored live candidate).",
    "Predictions were frozen before any microbench. The ladder finds the SIMPLEST",
    "deterministic gate that captures the signal; the model ships only if it strictly",
    "beats the role x mechanism lookup.",
    "",
    f"- conclusion: `{summary['conclusion']}`",
    f"- fresh candidates: `{summary['fresh_rows']}` | live: `{summary['live_candidates']}` | live patterns: `{summary.get('live_patterns')}`",
    f"- label distribution: `{summary['fresh_label_distribution']}`",
    f"- simplest sufficient gate: `{ab.get('simplest_sufficient_gate')}` | model beats role x mechanism prior: `{ab.get('model_beats_role_mechanism_prior')}`",
    "",
    "## Experiments saved at 100% live-recall (gate ladder)",
    "",
    "| gate | experiments run | saved vs run-all | live recall |",
    "|---|---:|---:|---:|",
  ]
  for name in ("run_all", "mechanism_prior", "role_mechanism_prior", "simple_family_heuristic", "xgboost"):
    if name not in g: continue
    row = g[name]
    lines.append(f"| `{name}` | {row['experiments_run']} | {row['experiments_saved_vs_run_all']} | {row['live_recall']:.2f} |")
  lines += ["", "## Per (role x mechanism) cell (n / live / dead-skipped by model | role_mech | mech)", "",
            "| cell | n | live | model | role_mech | mech |", "|---|---:|---:|---:|---:|---:|"]
  for cell, e in summary.get("per_cell", {}).items():
    lines.append(f"| `{cell}` | {e['n']} | {e['live']} | {e.get('xgboost_dead_skipped',0)} | "
                 f"{e.get('role_mechanism_prior_dead_skipped',0)} | {e.get('mechanism_prior_dead_skipped',0)} |")
  lines += [
    "", "## Interpretation", "",
    "If the role x mechanism lookup matches the model, ship the lookup and keep the",
    "model documentation-only -- a cheap deterministic gate reducing wasted GPU is a",
    "flywheel win. The model is only worth shipping if it strictly beats the lookup,",
    "i.e. it captures signal beyond the (role x mechanism) cell.",
    "",
  ]
  return "\n".join(lines)

def pool_batches(repo:pathlib.Path, batches:tuple[str, ...]=("v3", "v4", "v5"), out:pathlib.Path=STAGED_POOL_OUT,
                 corpus_path:pathlib.Path=DEFAULT_CORPUS) -> dict[str, Any]:
  """Phase 4.3 replication: pool (gate_score, is_live) across independent batches and
  compare the recall-vs-savings curve. The pooled curve at 95% recall is the robust
  test the brittle single-batch 100% point is not -- one surprise winner can be among
  the allowed misses instead of zeroing a gate."""
  repo = repo.resolve()
  corpus = [{**row, "split": "train"} for row in _read_jsonl(corpus_path)]
  gates = ("xgboost", "role_mechanism_prior", "mechanism_prior")
  pooled = {g: [] for g in gates}
  per_batch = []
  all_outcomes = []
  for name in batches:
    bout = STAGED_BATCHES[name][0]
    if not (bout / "summary.json").exists(): raise FileNotFoundError(f"batch {name} not scored; run score-batch --batch {name}")
    outs = _read_jsonl(bout / "outcomes.jsonl")
    all_outcomes += outs
    live_by_id = {r["id"]: r["label"] in USEFUL_LABELS for r in outs}
    examples = corpus + [{**r, "split": "holdout"} for r in outs]
    base = build_baseline_predictions(examples, seed=SEED)
    score_maps = {
      "xgboost": {p["id"]: float(p.get("score", 0.0)) for p in _read_jsonl(bout / "predictions.jsonl")},
      "role_mechanism_prior": {p["id"]: float(p.get("score", 0.0)) for p in _role_mechanism_prior(corpus, outs)},
      "mechanism_prior": {p["id"]: float(p.get("score", 0.0)) for p in base.get("mechanism_prior", [])},
    }
    entry = {"batch": name, "n": len(outs), "live": sum(live_by_id.values())}
    scored_by_gate = {}
    for g in gates:
      scored = [(score_maps[g].get(rid, 0.0), live_by_id[rid]) for rid in live_by_id]
      scored_by_gate[g] = scored
      pooled[g] += scored
      entry[f"{g}_safe_skips"] = _safe_skips(scored)["safe_skips"]
      entry[f"{g}_saved_95"] = _savings_at_recall(scored, 0.95)["saved"]
    entry["model_beats_lookup_100"] = entry["xgboost_safe_skips"] > entry["role_mechanism_prior_safe_skips"]
    entry["model_beats_lookup_95"] = entry["xgboost_saved_95"] > entry["role_mechanism_prior_saved_95"]
    per_batch.append(entry)

  pooled_curves = {g: {f"{lvl:.2f}": _savings_at_recall(scored, lvl) for lvl in RECALL_LEVELS} for g, scored in pooled.items()}
  total_live = sum(1 for _, live in pooled["xgboost"] if live)
  total_n = len(pooled["xgboost"])

  # Fair deterministic baseline: skip the schedule classes that are 100% construction_blocked
  # in training (known-broken: reduce_unroll / two_dim_local cells). This is the natural rule
  # for a classification gate, and -- unlike the score-floor metric -- it is not penalized by a
  # discrete gate tying a surprise winner with the dead mass. It exposes whether the model's
  # safe-skip "win" over role_mechanism_prior is real or a floor-collapse artifact.
  cb_cells = _always_construction_blocked_cells(corpus)
  class_skipped = [o for o in all_outcomes if (o.get("role"), o.get("mechanism")) in cb_cells]
  class_missed = sum(1 for o in class_skipped if o["label"] in USEFUL_LABELS)
  class_skip = {"saved": len(class_skipped), "missed_live": class_missed,
                "recall": round(1.0 - class_missed / total_live, 4) if total_live else 1.0,
                "skipped_cells": sorted(f"{r} x {m}" for r, m in cb_cells)}

  model_100 = pooled_curves["xgboost"]["1.00"]["saved"]
  wins_100 = sum(1 for b in per_batch if b["model_beats_lookup_100"])
  # The honest bar is the best FULL-RECALL deterministic gate, not the floor-penalized lookup.
  det_full_recall_saved = class_skip["saved"] if class_missed == 0 else 0
  model_beats_determinism = model_100 > det_full_recall_saved
  if total_live < 10:
    conclusion, gate_source = "inconclusive_insufficient_pooled_live_candidates", "undecided"
  elif model_beats_determinism:
    conclusion, gate_source = "model_beats_deterministic_class_skip_model_earns_phase5", "xgboost"
  else:
    conclusion = "deterministic_class_skip_matches_model_ship_the_lookup_model_adds_no_value"
    gate_source = "construction_blocked_class_skip"

  summary = {
    "kind": "qk_flywheel_shadow_replication_pool", "phase": "Phase 4.3", "conclusion": conclusion, "seed": SEED,
    "batches": list(batches), "pooled_candidates": total_n, "pooled_live": total_live,
    "per_batch": per_batch,
    "pooled_recall_curve": pooled_curves,
    "deterministic_class_skip": class_skip,
    "decision": {
      "model_safe_skips_at_100": model_100,
      "deterministic_class_skip_saved": class_skip["saved"], "deterministic_class_skip_recall": class_skip["recall"],
      "model_beats_deterministic_class_skip": model_beats_determinism,
      "batches_model_beats_floor_lookup_at_100": wins_100, "of_batches": len(per_batch),
      "phase5_gate_source": gate_source,
    },
    "caveat": ("The model's safe-skip advantage over role_mechanism_prior is a floor-collapse artifact: the "
               "score-floor metric penalizes a discrete gate that ties a surprise winner with the dead mass. A "
               "deterministic class-skip gate (skip known-broken schedule classes) is the fair baseline; compare "
               "the model against it, not against the floor-penalized lookup."),
  }
  out.mkdir(parents=True, exist_ok=True)
  (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
  return summary

def main() -> int:
  parser = argparse.ArgumentParser(description="Phase 4 / 4.1 live shadow mode for the AMD decode flywheel")
  parser.add_argument("step", choices=("freeze", "run", "score", "all",
                                        "freeze-staged", "run-staged", "score-staged", "all-staged",
                                        "freeze-staged-v2", "run-staged-v2", "score-staged-v2", "all-staged-v2",
                                        "freeze-batch", "run-batch", "score-batch", "pool-batches"))
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
  parser.add_argument("--out", type=pathlib.Path, default=None)
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--batch", choices=tuple(STAGED_BATCHES), default=None, help="staged batch id for *-batch steps")
  args = parser.parse_args()
  v0_out = args.out or DEFAULT_OUT
  staged_out = args.out or STAGED_OUT
  if args.step in ("freeze", "all"):
    print(json.dumps(freeze_predictions(args.repo, args.corpus, v0_out, args.model), indent=2, sort_keys=True))
  if args.step in ("run", "all"):
    run_outcomes(args.repo, v0_out, args.model, args.device)
  if args.step in ("score", "all"):
    print(json.dumps(score_shadow(args.repo, v0_out, args.corpus, args.model), indent=2, sort_keys=True))
  if args.step in ("freeze-staged", "all-staged"):
    print(json.dumps(freeze_staged(args.repo, args.corpus, staged_out), indent=2, sort_keys=True))
  if args.step in ("run-staged", "all-staged"):
    run_staged(args.repo, staged_out, args.device)
  if args.step in ("score-staged", "all-staged"):
    print(json.dumps(score_staged(args.repo, staged_out, args.corpus), indent=2, sort_keys=True))
  v2_out = args.out or STAGED_V2_OUT
  if args.step in ("freeze-staged-v2", "all-staged-v2"):
    print(json.dumps(freeze_staged(args.repo, args.corpus, v2_out, STAGED_V2_TENSORS), indent=2, sort_keys=True))
  if args.step in ("run-staged-v2", "all-staged-v2"):
    run_staged(args.repo, v2_out, args.device)
  if args.step in ("score-staged-v2", "all-staged-v2"):
    print(json.dumps(score_staged(args.repo, v2_out, args.corpus), indent=2, sort_keys=True))
  if args.step in ("freeze-batch", "run-batch", "score-batch"):
    if not args.batch: parser.error(f"{args.step} requires --batch {{{','.join(STAGED_BATCHES)}}}")
    b_out, b_tensors = STAGED_BATCHES[args.batch]
    if args.step == "freeze-batch":
      print(json.dumps(freeze_staged(args.repo, args.corpus, b_out, b_tensors), indent=2, sort_keys=True))
    elif args.step == "run-batch":
      run_staged(args.repo, b_out, args.device)
    else:
      print(json.dumps(score_staged(args.repo, b_out, args.corpus), indent=2, sort_keys=True))
  if args.step == "pool-batches":
    print(json.dumps(pool_batches(args.repo, corpus_path=args.corpus), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
