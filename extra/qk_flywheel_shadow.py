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

import argparse, copy, hashlib, json, os, pathlib, re, subprocess
from collections import Counter
from typing import Any

from extra import qk_flywheel_dataset_v1 as v1
from extra.qk_flywheel_dataset import _label_reason_retry
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

def _read_json(path:pathlib.Path) -> dict[str, Any]:
  data = json.loads(path.read_text())
  if not isinstance(data, dict): raise ValueError(f"{path}: expected JSON object")
  return data

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

def freeze_predictions(repo:pathlib.Path, corpus_path:pathlib.Path=DEFAULT_CORPUS,
                       out:pathlib.Path=DEFAULT_OUT, model_path:pathlib.Path=DEFAULT_MODEL) -> dict[str, Any]:
  repo = repo.resolve()
  corpus = _read_jsonl(corpus_path)
  fresh = build_fresh_candidates(repo, model_path)
  corpus_maps = [extract_feature_map(row) for row in corpus]
  fresh_maps = [extract_feature_map(row) for row in fresh]
  vec = FeatureVectorizer().fit(corpus_maps)
  x_corpus, x_fresh = vec.transform(corpus_maps), vec.transform(fresh_maps)
  # _xgboost_predictions trains on the corpus (reads labels only from corpus) and
  # predicts on the fresh rows, which it touches only for their id.
  preds, xgb_meta = _xgboost_predictions(corpus, fresh, x_corpus, x_fresh, SEED)

  forbidden_in_features = [name for name in vec.names if any(tok in name for tok in LEAKAGE_TOKENS)]
  candidates_bytes, predictions_bytes = _jsonl_bytes(fresh), _jsonl_bytes(preds)
  freeze = {
    "kind": "qk_flywheel_shadow_v0_freeze",
    "phase": "Phase 4",
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
    "leakage_audit": {
      "excluded_feature_sources": list(FORBIDDEN_FEATURE_SOURCES),
      "forbidden_tokens_in_feature_names": forbidden_in_features,
      "leak_free": not forbidden_in_features,
    },
    "note": "Frozen before any fresh GPU run. Commit this file and predictions.jsonl before producing outcomes.jsonl.",
  }
  out.mkdir(parents=True, exist_ok=True)
  out.joinpath("candidates.jsonl").write_bytes(candidates_bytes)
  out.joinpath("predictions.jsonl").write_bytes(predictions_bytes)
  out.joinpath("freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
  return freeze

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
    desc = out / "runs/packed-load/8b-descriptors.json"
    _build_packed_descriptor(repo, desc)
    _run([py, "extra/qk_semantic_codegen_v3.py", "--descriptor", str(desc),
          "--json", str(packed / "candidates.json"), "--gate-json", str(packed / "static-gate.json")],
         cwd=repo, extra_env={"DEV": device}, timeout=300)
    _run([py, "extra/qk_semantic_schedule_bench.py", "--model", "8b",
          "--candidates", str(packed / "candidates.json"), "--static-gate", str(packed / "static-gate.json"),
          "--out", str(packed / "microbench-runs"), "--json", str(packed / "microbench.json"),
          "--md", str(packed / "microbench.md")],
         cwd=repo, extra_env={"DEV": device}, timeout=900)
  # block-dot: compile gate + microbench per tensor. A candidate whose correctness gate
  # fails (a real per-tensor fp16 outcome) is recorded as construction_blocked, not a crash.
  for spec in FRESH_SPECS:
    if spec["mechanism"] != "qk_block_dot": continue
    art = _block_dot_artifact(out, spec["tensor"])
    if (art / "microbench.json").exists() or (art / "shadow-outcome.json").exists(): continue
    try:
      _run([py, "extra/qk_block_dot_compile_gate.py", "--tensor", spec["tensor"],
            "--artifact", str(art) + "-compile-gate", "--model", model, "--device", device],
           cwd=repo, extra_env={"DEV": device}, timeout=480)
      _run([py, "extra/qk_block_dot_microbench.py", "--tensor", spec["tensor"],
            "--artifact", str(art), "--model", model, "--device", device],
           cwd=repo, extra_env={"DEV": device, "DEBUG": "2"}, timeout=600)
    except RuntimeError as exc:
      art.mkdir(parents=True, exist_ok=True)
      (art / "shadow-outcome.json").write_text(json.dumps(
        {"status": "construction_blocked", "reason": "correctness_failed", "tensor": spec["tensor"], "error": str(exc)},
        indent=2, sort_keys=True) + "\n")
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

def main() -> int:
  parser = argparse.ArgumentParser(description="Phase 4 v0 live shadow mode for the AMD decode flywheel")
  parser.add_argument("step", choices=("freeze", "run", "score", "all"))
  parser.add_argument("--repo", type=pathlib.Path, default=pathlib.Path("."))
  parser.add_argument("--corpus", type=pathlib.Path, default=DEFAULT_CORPUS)
  parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
  parser.add_argument("--model", type=pathlib.Path, default=DEFAULT_MODEL)
  parser.add_argument("--device", default="AMD")
  args = parser.parse_args()
  if args.step in ("freeze", "all"):
    print(json.dumps(freeze_predictions(args.repo, args.corpus, args.out, args.model), indent=2, sort_keys=True))
  if args.step in ("run", "all"):
    run_outcomes(args.repo, args.out, args.model, args.device)
  if args.step in ("score", "all"):
    print(json.dumps(score_shadow(args.repo, args.out, args.corpus, args.model), indent=2, sort_keys=True))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
