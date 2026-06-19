#!/usr/bin/env python3
"""Primitive-local observability ledger.

Read-only by default: ingest existing bench/docs artifacts and emit a unified primitive ledger.

This is the PLO-1..PLO-6 scaffold from
docs/primitive-local-observability-search-scope-20260619.md:
  PLO-1 ledger collector
  PLO-2 schema/evidence validator
  PLO-3 runner-wrapper registry (no hardware execution by default)
  PLO-4 deterministic failure classifier
  PLO-5 search-memory/candidate DB
  PLO-6 optional trace/counter plugin inventory

Run:
  PYTHONPATH=. .venv/bin/python extra/qk_primitive_ledger.py
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, os, pathlib, shutil, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
DOCS = ROOT / "docs"
OUTDIR = BENCH / "qk-primitive-observability"
HARDWARE = "RX 7900 XTX / gfx1100"
BACKEND = "AMD"

PRIMITIVES = {"mmvq_decode", "prefill_tensile", "prefill_wmma", "attention_kv", "runtime_boundary", "spec_decode"}
PHASES = {"decode", "prefill", "long_context", "graph_integration", "spec_verify"}
GATES = {"PASS", "REDIRECT", "KILL", "DEFERRED", "CLOSED", "SHIPPED", "REFUTED", "OPEN", "UNKNOWN"}

def _now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()

def _git_commit() -> str:
  try:
    sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10).returncode != 0
    return (sha + ("-dirty" if dirty else "")) or "unknown"
  except Exception:
    return "unknown"

def _rel(path:pathlib.Path) -> str:
  try: return str(path.relative_to(ROOT))
  except ValueError: return str(path)

def _read_json(path:pathlib.Path) -> dict[str, Any] | None:
  try:
    data = json.loads(path.read_text())
    return data if isinstance(data, dict) else None
  except Exception:
    return None

def _hash_obj(obj:Any) -> str:
  return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

def _base_obs(*, primitive:str, phase:str, role:str, shape:dict[str, Any] | None, candidate:dict[str, Any],
              correctness:dict[str, Any], timing:dict[str, Any] | None, metadata:dict[str, Any] | None,
              runtime:dict[str, Any] | None, gate:dict[str, Any], provenance:list[str],
              evidence_levels:list[int], bottleneck_inference:str, notes:str="") -> dict[str, Any]:
  obs = {
    "schema": "primitive_observation_v1",
    "id": "",
    "timestamp": _now(),
    "commit": _git_commit(),
    "hardware": HARDWARE,
    "backend": BACKEND,
    "primitive": primitive,
    "phase": phase,
    "role": role,
    "shape": shape or {},
    "candidate": candidate,
    "correctness": correctness,
    "timing": timing or {},
    "metadata": metadata or {},
    "runtime": runtime or {},
    "evidence_levels": sorted(set(int(x) for x in evidence_levels)),
    "bottleneck_inference": bottleneck_inference,
    "gate": gate,
    "provenance": provenance,
    "notes": notes,
  }
  obs["id"] = f"{primitive}:{phase}:{role}:{_hash_obj({'shape': obs['shape'], 'candidate': candidate, 'prov': provenance})}"
  return obs

def _gate(verdict:str, reason:str, passed:bool|None=None) -> dict[str, Any]:
  return {"verdict": verdict, "passed": (passed if passed is not None else verdict in {"PASS", "SHIPPED"}),
          "reason": reason}

def _obs_tpe_shape_matrix(path:pathlib.Path) -> list[dict[str, Any]]:
  data = _read_json(path)
  if not data or data.get("schema") != "qk_tensile_shape_matrix_v1": return []
  out = []
  for row in data.get("rows", []):
    if not isinstance(row, dict): continue
    role = str(row.get("role", "unknown"))
    out.append(_base_obs(
      primitive="prefill_tensile", phase="prefill", role=role,
      shape={"M": row.get("m"), "N": row.get("n"), "K": row.get("k"), "T": 512},
      candidate={"id": f"tensile:{role}", "parent_id": None, "legal_knobs": {"backend": "rocBLAS/Tensile"},
                 "source_hash": _hash_obj(row.get("kernel_symbol", "")), "kernel_symbol_short": str(row.get("kernel_symbol", ""))[:80],
                 "streamk": bool(row.get("streamk"))},
      correctness={"oracle": "tinygrad fp16 matmul", "pass": bool(row.get("correct")), "stable": bool(row.get("stable")),
                   "rel_err": row.get("rel_err"), "tolerance": 2e-2},
      timing={"median_ms": row.get("median_ms"), "best_ms": row.get("best_ms"), "median_tflops": row.get("median_tflops"),
              "best_tflops": row.get("best_tflops"), "ref_tflops": row.get("ref_tflops"),
              "tinygrad_tflops": row.get("tinygrad_tflops"), "speedup_vs_tinygrad": row.get("speedup_vs_tinygrad"),
              "pct_of_ref": row.get("pct_of_ref")},
      metadata={"kernarg_size": row.get("kernarg_size"), "global_size": row.get("global_size"),
                "local_size": row.get("local_size"), "workspace": row.get("workspace")},
      runtime={"no_layout_copies": data.get("gates", {}).get("no_layout_copies"), "no_workspace": data.get("gates", {}).get("no_workspace")},
      gate=_gate("PASS" if bool(row.get("correct")) and bool(row.get("stable")) else "KILL",
                 "TPE-5 role row correct/stable; weighted model handled at session level"),
      provenance=[_rel(path), "docs/prefill-tensile-tpe5-shape-matrix-result-20260619.md"],
      evidence_levels=[0, 1, 2, 3], bottleneck_inference="occupancy_or_issue",
      notes="TPE-5 extracted Tensile role row"))
  out.append(_base_obs(
    primitive="prefill_tensile", phase="prefill", role="weighted_shape_matrix",
    shape={"T": 512},
    candidate={"id": "tensile:tpe5:all_roles", "parent_id": "tensile:tpe4:ffn_gate_up",
               "legal_knobs": {"roles": ["ffn_gate_up", "ffn_down", "attn_q_o"]}, "source_hash": _hash_obj(data.get("rows", []))},
    correctness={"oracle": "per-role tinygrad fp16 oracles", "pass": data.get("gates", {}).get("all_correct"),
                 "stable": data.get("gates", {}).get("all_stable")},
    timing={"full_pp_speedup": data.get("full_pp_speedup"), "weighted_model": data.get("weighted_model")},
    metadata={"rows": len(data.get("rows", []))}, runtime=data.get("gates", {}),
    gate=_gate(str(data.get("verdict", "UNKNOWN")), "TPE-5 weighted pp model and role gates", data.get("verdict") == "PASS"),
    provenance=[_rel(path), "docs/prefill-tensile-tpe5-shape-matrix-result-20260619.md"],
    evidence_levels=[0, 1, 2, 3], bottleneck_inference="unknown",
    notes="TPE-5 PASS predicts ~1.40x full warm pp512 if routed"))
  return out

def _obs_hcq_perf(path:pathlib.Path) -> list[dict[str, Any]]:
  data = _read_json(path)
  if not data or data.get("schema") != "qk_tensile_hcq_perf_v1": return []
  return [_base_obs(
    primitive="prefill_tensile", phase="prefill", role=str(data.get("role", "ffn_gate/up")).replace("/", "_"),
    shape=data.get("shape"),
    candidate={"id": "tensile:tpe4:ffn_gate_up", "parent_id": None, "legal_knobs": {"backend": "rocBLAS/Tensile"},
               "source_hash": _hash_obj(data.get("kernel_symbol", "")), "kernel_symbol_short": data.get("kernel_symbol_short")},
    correctness=data.get("correctness", {}),
    timing={"median_ms": data.get("timed", {}).get("median_ms"), "median_tflops": data.get("median_tflops"),
            "ratios": data.get("ratios"), "references": data.get("references")},
    metadata={"kd_offset": data.get("kd_offset"), "launch": data.get("launch")},
    runtime={"process_libraries": data.get("process_libraries"), "gates": data.get("gates")},
    gate=_gate(str(data.get("verdict", "UNKNOWN")), "TPE-4 HCQ perf parity", data.get("verdict") == "PASS"),
    provenance=[_rel(path), "docs/prefill-tensile-tpe4-perf-result-20260619.md"],
    evidence_levels=[0, 1, 2, 3], bottleneck_inference="unknown")]

def _obs_block_transfer(path:pathlib.Path) -> list[dict[str, Any]]:
  data = _read_json(path)
  if not data or data.get("schema") != "qk_tensile_block_transfer_v1": return []
  timing = data.get("timing", {})
  return [_base_obs(
    primitive="runtime_boundary", phase="graph_integration", role="ffn_block",
    shape={"T": data.get("T"), "hidden": data.get("hidden"), "ffn": data.get("ffn")},
    candidate={"id": "tensile:tpe6:ffn_block_naive_per_op", "parent_id": "tensile:tpe5:all_roles",
               "legal_knobs": {"routing": "naive_per_op_hcq", "graph": False}, "source_hash": _hash_obj(data.get("routing", {}))},
    correctness={**data.get("correctness", {}), "oracle": "tinygrad fp16 FFN block"},
    timing=timing,
    metadata={"matmul_gflop": data.get("matmul_gflop")},
    runtime={**data.get("routing", {}), "no_hip_runtime": data.get("no_hip_runtime"), "no_weight_copies": data.get("no_weight_copies")},
    gate=_gate(str(data.get("verdict", "UNKNOWN")), data.get("verdict_note", "TPE-6 block-transfer gate"),
               data.get("verdict") == "PASS"),
    provenance=[_rel(path), "docs/prefill-tensile-tpe6-block-transfer-result-20260619.md"],
    evidence_levels=[0, 1, 3], bottleneck_inference="graph_boundary",
    notes="GPU matmul speed transfers; naive per-op routing redirects to graph integration")]

def _obs_rebindable_node(path:pathlib.Path) -> list[dict[str, Any]]:
  data = _read_json(path)
  if not data or data.get("schema") != "qk_tensile_rebindable_node_v1": return []
  gates = data.get("gates", {})
  rel_errs = [row.get("rel_err") for row in data.get("bindings", []) if isinstance(row, dict) and isinstance(row.get("rel_err"), (int, float))]
  return [_base_obs(
    primitive="runtime_boundary", phase="graph_integration", role="tensile_rebindable_node",
    shape={"T": 512, "hidden": 4096, "ffn": 12288},
    candidate={"id": "tensile:tpe7a:rebindable_node", "parent_id": "tensile:tpe6:ffn_block_naive_per_op",
               "legal_knobs": {"routing": "graph_style_fill_kernargs", "program_built_once": data.get("program_built_once")},
               "source_hash": _hash_obj({"bindings": data.get("bindings"), "gates": gates})},
    correctness={"oracle": "tinygrad fp16 matmul", "pass": data.get("verdict") == "PASS",
                 "stable": gates.get("replay_stable"), "rel_err_max": max(rel_errs) if rel_errs else None,
                 "replay_rel_err": data.get("replay_rel_err"), "tolerance": 2e-2},
    timing={},
    metadata={"bindings": len(data.get("bindings", [])), "program_built_once": data.get("program_built_once")},
    runtime={"one_node_many_buffers": gates.get("one_node_many_buffers"),
             "distinct_bindings_ok": data.get("distinct_bindings_ok"), "graph_protocol": "fill_kernargs_rebind"},
    gate=_gate(str(data.get("verdict", "UNKNOWN")),
               "TPE-7a proves one extracted Tensile node can be rebound to current buffers; next gate is in-model capture",
               data.get("verdict") == "PASS"),
    provenance=[_rel(path), "docs/prefill-tensile-tpe7-inmodel-route-scope-20260619.md"],
    evidence_levels=[0, 3], bottleneck_inference="graph_boundary",
    notes="Correctness-only graph-protocol keystone; no model route or timing gate yet")]

def _obs_static_docs() -> list[dict[str, Any]]:
  specs: list[dict[str, Any]] = [
    {
      "primitive": "mmvq_decode", "phase": "decode", "role": "q8_sidechannel_ffn_gate_up",
      "doc": "docs/q8-mmvq-lifecycle-deep-result-20260619.md", "verdict": "DEFERRED",
      "reason": "Q8L-2 expressibility wall: fused per-row -> per-32 multi-output producer needs LDS-reduction codegen",
      "bottleneck": "pack_lifecycle", "levels": [0, 1, 2], "candidate": "q8_sidechannel:fused_rmsnorm_apply",
    },
    {
      "primitive": "prefill_wmma", "phase": "prefill", "role": "ffn_gate_up",
      "doc": "docs/prefill-own-wmma-kernel-result-20260619.md", "verdict": "KILL",
      "reason": "POWN-1 bounded pure tinygrad WMMA sweep best 42.0 TFLOPS; no config reaches 62 TFLOPS gate",
      "bottleneck": "occupancy_or_issue", "levels": [0, 1, 2], "candidate": "pown1:bounded_wmma_sweep",
    },
    {
      "primitive": "spec_decode", "phase": "spec_verify", "role": "verify_forward",
      "doc": "docs/qk-spec-verify-component-breakdown-20260618.md", "verdict": "CLOSED",
      "reason": "Spec verify T-scaling distributed across attention + Q4_K + Q6_K; no single kernel clears gate",
      "bottleneck": "unknown", "levels": [1, 3], "candidate": "spec_verify:single_kernel_shortcut",
    },
    {
      "primitive": "attention_kv", "phase": "prefill", "role": "flash_prefill_reuse_free",
      "doc": "docs/amd-decode-prefill-v2-increment2-phase5-correction-20260617.md", "verdict": "REFUTED",
      "reason": "Reuse-free flash-prefill kernel correct but far slower; real primitive needs LDS/register locality",
      "bottleneck": "bandwidth", "levels": [0, 1], "candidate": "flash_prefill:reuse_free",
    },
  ]
  out = []
  for spec in specs:
    doc = ROOT / spec["doc"]
    out.append(_base_obs(
      primitive=spec["primitive"], phase=spec["phase"], role=spec["role"], shape={},
      candidate={"id": spec["candidate"], "parent_id": None, "legal_knobs": {}, "source_hash": _hash_obj(spec)},
      correctness={"oracle": "see provenance", "pass": spec["verdict"] not in {"KILL", "REFUTED"}},
      timing={}, metadata={}, runtime={},
      gate=_gate(spec["verdict"], spec["reason"], spec["verdict"] in {"PASS", "SHIPPED"}),
      provenance=[_rel(doc)], evidence_levels=spec["levels"], bottleneck_inference=spec["bottleneck"]))
  return out

def collect_observations() -> list[dict[str, Any]]:
  observations: list[dict[str, Any]] = []
  observations += _obs_tpe_shape_matrix(BENCH / "qk-tensile-extraction/shape_matrix.json")
  observations += _obs_hcq_perf(BENCH / "qk-tensile-extraction/hcq_perf.json")
  observations += _obs_block_transfer(BENCH / "qk-tensile-extraction/block_transfer.json")
  observations += _obs_rebindable_node(BENCH / "qk-tensile-extraction/rebindable_node.json")
  observations += _obs_static_docs()
  return observations

def validate_observation(obs:dict[str, Any]) -> list[str]:
  errs: list[str] = []
  for key in ("schema", "id", "commit", "hardware", "backend", "primitive", "phase", "role", "candidate", "correctness", "gate", "provenance"):
    if key not in obs: errs.append(f"missing {key}")
  if obs.get("schema") != "primitive_observation_v1": errs.append("bad schema")
  if obs.get("primitive") not in PRIMITIVES: errs.append(f"bad primitive {obs.get('primitive')}")
  if obs.get("phase") not in PHASES: errs.append(f"bad phase {obs.get('phase')}")
  if obs.get("gate", {}).get("verdict") not in GATES: errs.append(f"bad verdict {obs.get('gate', {}).get('verdict')}")
  levels = obs.get("evidence_levels")
  if not isinstance(levels, list) or not levels: errs.append("missing evidence levels")
  elif any(not isinstance(x, int) or x < 0 or x > 4 for x in levels): errs.append(f"bad evidence levels {levels}")
  if obs.get("gate", {}).get("verdict") in {"PASS", "SHIPPED"} and not obs.get("correctness", {}).get("pass", False):
    errs.append("passing verdict without correctness pass")
  if obs.get("bottleneck_inference") != "unknown" and max(levels or [0]) < 1:
    errs.append("bottleneck inference without timing evidence")
  if "PMU" in str(obs.get("notes", "")) and max(levels or [0]) < 4:
    errs.append("PMU claim without level-4 evidence")
  return errs

def classify(obs:dict[str, Any]) -> dict[str, Any]:
  verdict = obs.get("gate", {}).get("verdict", "UNKNOWN")
  bottleneck = obs.get("bottleneck_inference", "unknown")
  primitive = obs.get("primitive")
  role = obs.get("role")
  action = "review"
  if verdict in {"KILL", "REFUTED", "CLOSED"}: action = "do_not_reopen_without_new_evidence"
  elif verdict == "DEFERRED": action = "wait_for_named_capability"
  elif verdict == "REDIRECT": action = "build_redirect_target"
  elif verdict == "PASS": action = "eligible_for_next_gate"
  if primitive == "runtime_boundary" and verdict == "REDIRECT": action = "graph_integration_next"
  if primitive == "prefill_tensile" and role == "weighted_shape_matrix" and verdict == "PASS": action = "route_only_after_graph_gate_and_policy"
  return {"class": bottleneck, "next_action": action, "verdict": verdict}

def build_ledger(observations:list[dict[str, Any]]) -> dict[str, Any]:
  latest: dict[str, dict[str, Any]] = {}
  for obs in observations:
    key = f"{obs['primitive']}:{obs['phase']}:{obs['role']}"
    latest[key] = obs
  return {
    "schema": "primitive_ledger_v1",
    "generated_at": _now(),
    "commit": _git_commit(),
    "observations": len(observations),
    "latest": {k: {"id": v["id"], "verdict": v["gate"]["verdict"], "next_action": classify(v)["next_action"],
                   "provenance": v["provenance"]} for k, v in sorted(latest.items())},
  }

def build_search_sessions(observations:list[dict[str, Any]]) -> list[dict[str, Any]]:
  sessions = [
    {
      "schema": "primitive_search_session_v1",
      "id": "session:tpe5_shape_matrix_replay",
      "primitive_target": "prefill_tensile",
      "candidate_generator": "extra/qk_tensile_shape_matrix.py",
      "candidate_count": sum(1 for o in observations if o["primitive"] == "prefill_tensile" and o["role"] not in {"weighted_shape_matrix"}),
      "budget": "replay_existing_artifacts",
      "ranking_policy": "correct/stable, then median_tflops and weighted pp model",
      "accepted_frontier": [o["id"] for o in observations if o["primitive"] == "prefill_tensile" and o["gate"]["verdict"] == "PASS"],
      "refuted_candidate_classes": [],
      "artifact_paths": ["bench/qk-tensile-extraction/shape_matrix.json", "bench/qk-tensile-extraction/hcq_perf.json"],
    },
    {
      "schema": "primitive_search_session_v1",
      "id": "session:tpe6_runtime_boundary_replay",
      "primitive_target": "runtime_boundary",
      "candidate_generator": "extra/qk_tensile_block_transfer.py",
      "candidate_count": 1,
      "budget": "replay_existing_artifact",
      "ranking_policy": "correctness, GPU speedup, end-to-end routing overhead",
      "accepted_frontier": [],
      "refuted_candidate_classes": ["naive_per_op_host_sync"],
      "artifact_paths": ["bench/qk-tensile-extraction/block_transfer.json"],
    },
    {
      "schema": "primitive_search_session_v1",
      "id": "session:tpe7a_rebindable_node_replay",
      "primitive_target": "runtime_boundary",
      "candidate_generator": "extra/qk_tensile_rebindable_node.py",
      "candidate_count": sum(1 for o in observations if o["candidate"].get("id") == "tensile:tpe7a:rebindable_node"),
      "budget": "replay_existing_artifact",
      "ranking_policy": "correctness across distinct buffer bindings, then replay stability",
      "accepted_frontier": [o["id"] for o in observations if o["candidate"].get("id") == "tensile:tpe7a:rebindable_node" and o["gate"]["verdict"] == "PASS"],
      "refuted_candidate_classes": [],
      "artifact_paths": ["bench/qk-tensile-extraction/rebindable_node.json"],
    },
  ]
  return sessions

def build_search_memory(observations:list[dict[str, Any]]) -> dict[str, Any]:
  candidates = []
  for obs in observations:
    candidates.append({
      "candidate_id": obs["candidate"].get("id"),
      "observation_id": obs["id"],
      "primitive": obs["primitive"],
      "role": obs["role"],
      "parent_id": obs["candidate"].get("parent_id"),
      "verdict": obs["gate"]["verdict"],
      "bottleneck": obs["bottleneck_inference"],
      "next_action": classify(obs)["next_action"],
      "source_hash": obs["candidate"].get("source_hash"),
    })
  return {"schema": "primitive_search_memory_v1", "generated_at": _now(), "candidate_count": len(candidates),
          "candidates": candidates}

def build_runner_registry() -> dict[str, Any]:
  return {
    "schema": "primitive_runner_registry_v1",
    "mode": "replay_only_no_hardware_execution",
    "runners": [
      {"primitive": "mmvq_decode", "runner": "extra/q4_k_profile_report.py / extra/qk_gap_profile.py", "status": "available_adapter"},
      {"primitive": "prefill_tensile", "runner": "extra/qk_tensile_shape_matrix.py / extra/qk_tensile_hcq_perf.py", "status": "available_adapter"},
      {"primitive": "prefill_wmma", "runner": "extra/qk_prefill_wmma_sweep.py", "status": "legacy_artifact_adapter"},
      {"primitive": "attention_kv", "runner": "future_attention_probe", "status": "scoped_not_built"},
      {"primitive": "runtime_boundary", "runner": "extra/qk_tensile_block_transfer.py / extra/qk_tensile_rebindable_node.py", "status": "available_adapter"},
    ],
  }

def build_smoke_checks(sessions:list[dict[str, Any]], registry:dict[str, Any]) -> dict[str, Any]:
  checks = []
  for sess in sessions:
    artifact_checks = []
    for raw in sess.get("artifact_paths", []):
      path = ROOT / raw
      artifact_checks.append({"path": raw, "exists": path.exists()})
    checks.append({
      "session_id": sess["id"],
      "mode": "replay_only_no_hardware_execution",
      "candidate_count": sess.get("candidate_count"),
      "artifacts": artifact_checks,
      "pass": all(item["exists"] for item in artifact_checks),
    })
  runner_checks = []
  for item in registry["runners"]:
    runner = str(item["runner"]).split(" / ")[0]
    path = ROOT / runner
    runner_checks.append({"primitive": item["primitive"], "runner": runner, "exists": path.exists()})
  return {
    "schema": "primitive_runner_smoke_v1",
    "mode": "replay_only_no_hardware_execution",
    "sessions": checks,
    "runner_registry": runner_checks,
    "pass": all(item["pass"] for item in checks),
  }

def build_trace_plugins() -> dict[str, Any]:
  def _find_tool(name:str) -> str | None:
    found = shutil.which(name)
    if found: return found
    for root in (pathlib.Path("/opt/rocm/bin"), pathlib.Path("/opt/rocm-7.2.4/bin")):
      cand = root / name
      if cand.exists() and os.access(cand, os.X_OK): return str(cand)
    return None
  tools = {name: _find_tool(name) for name in ("rocprofv3", "rocprof-compute", "rocprof-sys", "rocprof-compute-viewer")}
  sqtt_examples = sorted(_rel(p) for p in (ROOT / "extra/sqtt/examples").glob("gfx1100/*.pkl")) if (ROOT / "extra/sqtt/examples/gfx1100").exists() else []
  rocprof_traces = sorted(_rel(p) for p in (BENCH / "llama-residual-exhaustion-20260619").glob("**/trace_results.json")) if (BENCH / "llama-residual-exhaustion-20260619").exists() else []
  pmu_probe = _read_json(BENCH / "qk-pmu-observability/result.json")
  hcq_attr = _read_json(BENCH / "qk-hcq-attribution/result.json")
  return {
    "schema": "primitive_trace_plugin_inventory_v1",
    "mode": "inventory_only_no_trace_collection",
    "tools": tools,
    "tinygrad_sqtt_examples": sqtt_examples[:20],
    "rocprof_trace_artifacts": rocprof_traces[:20],
    "pmu_probe": {
      "path": "bench/qk-pmu-observability/result.json",
      "present": pmu_probe is not None,
      "verdict": pmu_probe.get("verdict") if pmu_probe else None,
      "hip_control": pmu_probe.get("hip_control", {}).get("verdict") if pmu_probe else None,
      "tinygrad_hcq": pmu_probe.get("tinygrad_hcq", {}).get("verdict") if pmu_probe else None,
      "hcq_classification": pmu_probe.get("tinygrad_hcq", {}).get("classification") if pmu_probe else None,
    },
    "hcq_attribution": {
      "path": "bench/qk-hcq-attribution/result.json",
      "present": hcq_attr is not None,
      "classification": hcq_attr.get("classification") if hcq_attr else None,
      "program_count": hcq_attr.get("summary", {}).get("program_count") if hcq_attr else None,
      "graph_count": hcq_attr.get("summary", {}).get("graph_count") if hcq_attr else None,
      "graph_replay_count": hcq_attr.get("summary", {}).get("graph_replay_count") if hcq_attr else None,
    },
    "evidence_level": 4 if any(tools.values()) else 3,
    "note": "Trace/counter plugins are optional. This inventory does not run rocprof or require HIP runtime.",
  }

def write_outputs(observations:list[dict[str, Any]], outdir:pathlib.Path) -> dict[str, pathlib.Path]:
  outdir.mkdir(parents=True, exist_ok=True)
  validations = [{"id": obs["id"], "errors": validate_observation(obs)} for obs in observations]
  classified = [{**obs, "classification": classify(obs)} for obs in observations]
  ledger = build_ledger(observations)
  sessions = build_search_sessions(observations)
  memory = build_search_memory(observations)
  registry = build_runner_registry()
  smoke = build_smoke_checks(sessions, registry)
  traces = build_trace_plugins()

  paths = {
    "ledger_jsonl": outdir / "ledger.jsonl",
    "ledger_json": outdir / "ledger.json",
    "validation_json": outdir / "validation.json",
    "search_sessions_json": outdir / "search_sessions.json",
    "search_memory_json": outdir / "search_memory.json",
    "runner_registry_json": outdir / "runner_registry.json",
    "runner_smoke_json": outdir / "runner_smoke.json",
    "trace_plugins_json": outdir / "trace_plugins.json",
    "summary_md": outdir / "summary.md",
  }
  paths["ledger_jsonl"].write_text("".join(json.dumps(o, sort_keys=True) + "\n" for o in classified))
  paths["ledger_json"].write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
  paths["validation_json"].write_text(json.dumps({"schema": "primitive_observation_validation_v1", "pass": all(not v["errors"] for v in validations),
                                                  "rows": validations}, indent=2, sort_keys=True) + "\n")
  paths["search_sessions_json"].write_text(json.dumps({"schema": "primitive_search_sessions_v1", "sessions": sessions}, indent=2, sort_keys=True) + "\n")
  paths["search_memory_json"].write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n")
  paths["runner_registry_json"].write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
  paths["runner_smoke_json"].write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n")
  paths["trace_plugins_json"].write_text(json.dumps(traces, indent=2, sort_keys=True) + "\n")
  paths["summary_md"].write_text(summary_markdown(classified, validations, ledger, sessions, smoke, traces))
  return paths

def summary_markdown(observations:list[dict[str, Any]], validations:list[dict[str, Any]], ledger:dict[str, Any],
                     sessions:list[dict[str, Any]], smoke:dict[str, Any], traces:dict[str, Any]) -> str:
  lines = [
    "# Primitive-local observability summary",
    "",
    f"- generated: `{ledger['generated_at']}`",
    f"- commit: `{ledger['commit']}`",
    f"- observations: `{len(observations)}`",
    f"- validation: `{'PASS' if all(not v['errors'] for v in validations) else 'FAIL'}`",
    f"- search sessions: `{len(sessions)}`",
    f"- runner smoke: `{'PASS' if smoke.get('pass') else 'FAIL'}`",
    "",
    "## Verdict Ledger",
    "",
    "| primitive | phase | role | verdict | bottleneck | next action | evidence |",
    "|---|---|---|---|---|---|---:|",
  ]
  for obs in sorted(observations, key=lambda o: (o["primitive"], o["phase"], o["role"], o["id"])):
    cls = obs["classification"]
    lines.append(f"| `{obs['primitive']}` | `{obs['phase']}` | `{obs['role']}` | `{obs['gate']['verdict']}` | "
                 f"`{obs['bottleneck_inference']}` | `{cls['next_action']}` | `{max(obs['evidence_levels'])}` |")
  lines += [
    "",
    "## Reconstructed Required States",
    "",
    "- q8/MMVQ lifecycle: deferred behind codegen capability.",
    "- pure-tinygrad WMMA bounded sweep: killed/refuted.",
    "- Tensile extraction TPE-5: pass/generalizes.",
    "- TPE-6 block transfer: redirect to graph integration.",
    "- TPE-7a rebindable node: pass; in-model graph capture remains the next gate.",
    "- spec decode shortcut: closed.",
    "",
    "## Runner Registry",
    "",
  ]
  for sess in sessions:
    lines.append(f"- `{sess['id']}`: `{sess['primitive_target']}` via `{sess['candidate_generator']}`")
  lines += ["", "## Runner Smoke", ""]
  for row in smoke.get("sessions", []):
    lines.append(f"- `{row['session_id']}`: `{'PASS' if row['pass'] else 'FAIL'}` "
                 f"({len(row.get('artifacts', []))} replay artifacts)")
  lines += [
    "",
    "## Trace / Counter Plugin Inventory",
    "",
    f"- mode: `{traces['mode']}`",
    f"- rocprofv3: `{traces['tools'].get('rocprofv3') or 'missing'}`",
    f"- rocprof-compute: `{traces['tools'].get('rocprof-compute') or 'missing'}`",
    f"- tinygrad SQTT example files: `{len(traces.get('tinygrad_sqtt_examples', []))}`",
    f"- rocprof trace artifacts: `{len(traces.get('rocprof_trace_artifacts', []))}`",
    f"- PMU probe: `{traces.get('pmu_probe', {}).get('verdict') or 'missing'}`",
    f"- HCQ attribution: `{','.join(traces.get('hcq_attribution', {}).get('classification') or ['missing'])}`",
    "",
    "## Principle Check",
    "",
    "- read-only over existing artifacts by default;",
    "- no model route/default changes;",
    "- correctness and device time remain decision authority;",
    "- root-cause claims are evidence-level labeled;",
    "- optional counters/traces are plugins, not blockers.",
    "",
  ]
  return "\n".join(lines)

def main() -> int:
  ap = argparse.ArgumentParser(description="Build primitive-local observability ledger from existing artifacts")
  ap.add_argument("--out", type=pathlib.Path, default=OUTDIR)
  ap.add_argument("--print-summary", action="store_true")
  args = ap.parse_args()
  observations = collect_observations()
  paths = write_outputs(observations, args.out)
  validation = json.loads(paths["validation_json"].read_text())
  if args.print_summary: print(paths["summary_md"].read_text())
  if not validation["pass"]:
    print(json.dumps(validation, indent=2))
    return 2
  print(f"wrote {len(observations)} observations to {_rel(paths['ledger_jsonl'])}")
  print(f"summary: {_rel(paths['summary_md'])}")
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
