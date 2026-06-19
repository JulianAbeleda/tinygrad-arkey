#!/usr/bin/env python3
"""Read-only primitive lifecycle search seed ledger.

This sits above kernel search. It does not generate kernels, run hardware, or
route model defaults. It records lifecycle candidates: producer format,
consumer primitive, routing boundary, quality policy, and refutations.

Run:
  PYTHONPATH=. python3 extra/qk_lifecycle_search.py
"""
from __future__ import annotations

import datetime, json, pathlib, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
OUT = BENCH / "qk-lifecycle-search"

VALID_STATES = {
  "proposed", "diagnostic", "open", "pass_research", "pass_strong_policy_gated",
  "shipped", "refuted", "closed", "deferred", "policy_bound", "project_level",
}


def _read_json(rel:str) -> dict[str, Any]:
  try:
    obj = json.loads((ROOT / rel).read_text())
    return obj if isinstance(obj, dict) else {}
  except Exception:
    return {}


def _git_commit() -> str:
  try:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                          check=False, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=10).stdout.strip() or "unknown"
  except Exception:
    return "unknown"


def _speedup_by_ctx() -> dict[str, float]:
  base = _read_json("bench/q8-ffn-handwritten-oracle/decode_wd_baseline.json")
  route = _read_json("bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json")
  out: dict[str, float] = {}
  brows = {int(r["ctx"]): float(r["tok_s_W"]) for r in base.get("rows", []) if "ctx" in r and "tok_s_W" in r}
  for row in route.get("rows", []):
    ctx = int(row.get("ctx", -1))
    if ctx in brows and brows[ctx] > 0:
      out[str(ctx)] = round(float(row["tok_s_W"]) / brows[ctx], 4)
  return out


def _dnll_delta() -> float | None:
  base = _read_json("bench/q8-ffn-handwritten-oracle/nll_baseline.json")
  route = _read_json("bench/q8-ffn-handwritten-oracle/nll_q8_route.json")
  if isinstance(base.get("nll"), (int, float)) and isinstance(route.get("nll"), (int, float)):
    return round(float(route["nll"]) - float(base["nll"]), 6)
  return None


def _candidate(**kw: Any) -> dict[str, Any]:
  if kw["state"] not in VALID_STATES:
    raise ValueError(f"bad lifecycle state {kw['state']!r}")
  return {
    "id": kw["id"],
    "phase": kw["phase"],
    "state": kw["state"],
    "role": kw["role"],
    "producer": kw["producer"],
    "format": kw["format"],
    "consumer": kw["consumer"],
    "routing": kw["routing"],
    "quality": kw["quality"],
    "score": kw["score"],
    "evidence": kw["evidence"],
    "blocked_by_refutations": kw.get("blocked_by_refutations", []),
    "next_action": kw["next_action"],
  }


def build_candidates() -> list[dict[str, Any]]:
  tensile = _read_json("bench/qk-tensile-extraction/inmodel_measurement.json")
  shape = _read_json("bench/qk-tensile-extraction/shape_matrix.json")
  q8_artifact = _read_json("bench/q8-ffn-amd-scheduler-project/result.json")
  codegen = _read_json("bench/amd-schedule-codegen-exhaustion/oracle_matrix.json")
  q8_speedup = _speedup_by_ctx()
  q8_dnll = _dnll_delta()

  candidates = [
    _candidate(
      id="decode_q8_separate_pack",
      phase="decode",
      state="refuted",
      role="Q4_K ffn_gate/up",
      producer={"placement": "separate post-RMSNorm q8 pack", "reuse_count": 2},
      format={"activation": "q8_1 side buffer", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "native dot4 MMVQ", "owner": "tinygrad/hand artifact"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "dNLL <= 0.01 if speed passes", "observed": "not worth routing"},
      score={"speed_gate": "FAIL", "expected_e2e": "<=0 or sub-gate", "rank": 90},
      evidence=["docs/qk-q8-activation-lifecycle-verdict-20260618.md",
                "docs/q8-mmvq-lifecycle-deep-result-20260619.md"],
      blocked_by_refutations=["separate q8 pack cost erases dot4 gain"],
      next_action="Do not reopen without a fused producer.",
    ),
    _candidate(
      id="decode_q8_artifact_lifecycle",
      phase="decode",
      state="pass_research",
      role="Q4_K ffn_gate/up",
      producer={"placement": "hipcc/LLD fused RMSNorm/q8 producer artifact", "reuse_count": 2},
      format={"activation": "q8_1", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "hipcc/LLD fused gate+up dot4 artifact", "owner": "external artifact"},
      routing={"mode": "Q8_FFN_HANDWRITTEN=1 research flag", "default_safe": True,
                 "artifact_dependency": True, "no_hip_runtime_in_process": True},
      quality={"gate": "dNLL <= 0.01", "observed_delta": q8_dnll},
      score={"speed_gate": "PASS_RESEARCH", "ctx_speedup": q8_speedup,
               "isolated_lifecycle_us": q8_artifact.get("summary", {}).get("lifecycle_us"),
               "rank": 20},
      evidence=["docs/q8-ffn-handwritten-a4-decode-result-20260619.md",
                "docs/q8-ffn-artifact-import-route-result-20260619.md",
                "bench/q8-ffn-handwritten-oracle/decode_wd_q8_route.json",
                "bench/q8-ffn-amd-scheduler-project/result.json"],
      blocked_by_refutations=["native tinygrad ASM/COMGR variants fail perf"],
      next_action="Policy decision: accept research-only artifact route or keep default off.",
    ),
    _candidate(
      id="decode_q8_native_codegen",
      phase="decode",
      state="project_level",
      role="Q4_K ffn_gate/up",
      producer={"placement": "tinygrad-owned fused RMSNorm/apply side-channel", "reuse_count": 2},
      format={"activation": "q8_1", "weights": "Q4_K", "lossy": True},
      consumer={"primitive": "tinygrad-owned dot4 MMVQ", "owner": "renderer/AMD scheduler"},
      routing={"mode": "future native route", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "same as artifact route", "observed_delta_authority": q8_dnll},
      score={"speed_gate": "UNKNOWN_NATIVE", "oracle_ctx_speedup": q8_speedup,
               "rank": 30},
      evidence=["docs/q8-ffn-route-a-scheduler-codegen-result-20260619.md",
                "docs/q8-ffn-route-a-pmu-sqtt-evidence-result-20260619.md",
                "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"],
      blocked_by_refutations=["no bounded AMD DSL feature closes artifact gap",
                              "producer multi-granularity reduction not UOp-expressible"],
      next_action="Only fund as AMD scheduler/codegen project, not primitive search.",
    ),
    _candidate(
      id="decode_spec_verify_shortcut",
      phase="decode",
      state="closed",
      role="speculative decode verify",
      producer={"placement": "draft tokens then batched verify", "reuse_count": "K+1"},
      format={"activation": "batched T", "weights": "existing GGUF", "lossy": False},
      consumer={"primitive": "batched forward verify", "owner": "tinygrad"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "exact", "observed": "correct but too slow"},
      score={"speed_gate": "FAIL", "rank": 95},
      evidence=["docs/qk-spec-verify-component-breakdown-20260618.md"],
      blocked_by_refutations=["verify cost distributed across attention + Q4_K + Q6_K"],
      next_action="Do not treat spec verify as a single lifecycle candidate.",
    ),
    _candidate(
      id="prefill_tensile_artifact_full",
      phase="prefill",
      state="pass_strong_policy_gated",
      role="ffn_gate/up + ffn_down + attn_q/o",
      producer={"placement": "PREFILL_V2 fp16 realized weights", "reuse_count": "T=512"},
      format={"activation": "fp16/fp32 accum", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "extracted rocBLAS/Tensile GEMM through HCQ", "owner": "external artifact"},
      routing={"mode": "PREFILL_TENSILE_GEMM=1 research flag", "default_safe": True,
                 "artifact_dependency": True, "no_hip_runtime_in_process": True},
      quality={"gate": "dNLL <= 0.01", "observed": tensile.get("quality_dNLL")},
      score={"speed_gate": "PASS_STRONG_POLICY_GATED",
               "pp512_speedup_ffn_only": tensile.get("warm_pp512_speedup"),
               "pp512_speedup_all_roles": tensile.get("A5_attn_qo_routed", {}).get("warm_pp512_speedup"),
               "shape_matrix_speedup": shape.get("full_pp_speedup"),
               "rank": 1},
      evidence=["docs/prefill-tensile-research-measurement-result-20260619.md",
                "docs/prefill-tensile-tpe5-shape-matrix-result-20260619.md",
                "bench/qk-tensile-extraction/inmodel_measurement.json",
                "bench/qk-tensile-extraction/shape_matrix.json"],
      blocked_by_refutations=["in-process HIP runtime bridge killed; HCQ artifact route required"],
      next_action="Decide external artifact policy; if accepted, harden shape/fallback matrix.",
    ),
    _candidate(
      id="prefill_tensile_codegen_transfer",
      phase="prefill",
      state="project_level",
      role="fp16 GEMM matmul bucket",
      producer={"placement": "PREFILL_V2 fp16 realized weights", "reuse_count": "T"},
      format={"activation": "fp16", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "tinygrad renderer schedule matching Tensile", "owner": "tinygrad"},
      routing={"mode": "future native prefill route", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "same as Tensile artifact", "observed": "artifact oracle quality passes"},
      score={"speed_gate": "UNKNOWN_NATIVE", "oracle_pp512_speedup": tensile.get("A5_attn_qo_routed", {}).get("warm_pp512_speedup"),
               "rank": 35},
      evidence=["docs/amd-schedule-codegen-exhaustion-result-20260619.md",
                "bench/amd-schedule-codegen-exhaustion/oracle_matrix.json"],
      blocked_by_refutations=["bounded WMMA/LDS knob sweeps plateau near 42 TFLOPS"],
      next_action="Treat as reusable AMD renderer/scheduler project, using Tensile as oracle.",
    ),
    _candidate(
      id="prefill_route_a_asm_lds",
      phase="prefill",
      state="diagnostic",
      role="dependency-free RDNA3 WMMA GEMM",
      producer={"placement": "hand ASM LDS-staged GEMM", "reuse_count": "T"},
      format={"activation": "fp16", "weights": "fp16 realized", "lossy": False},
      consumer={"primitive": "multi-wave LDS WMMA hand schedule", "owner": "tinygrad research ASM"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "fp16 oracle + pp/dNLL once fast", "observed": "correct P0/P1, perf below A2"},
      score={"speed_gate": "DIAGNOSTIC", "rank": 45},
      evidence=["docs/route-a-a3-p0-p1-result-20260619.md",
                "docs/route-a-a3-lds-multiwave-result-20260619.md"],
      blocked_by_refutations=["naive multi-wave LDS occupancy/barrier tax; not yet double-buffered"],
      next_action="Wait for P2 double-buffer/occupancy result before promoting.",
    ),
    _candidate(
      id="prefill_attention_flash_lifecycle",
      phase="prefill",
      state="deferred",
      role="long-prompt attention",
      producer={"placement": "K/V tile in LDS/registers", "reuse_count": "prompt length"},
      format={"activation": "fp16 attention state", "weights": "n/a", "lossy": False},
      consumer={"primitive": "flash-prefill attention", "owner": "future tinygrad kernel"},
      routing={"mode": "not routed", "default_safe": True, "artifact_dependency": False},
      quality={"gate": "exact/dNLL if dtype changes", "observed": "reuse-free variants refuted"},
      score={"speed_gate": "DEFERRED_LONG_CONTEXT", "rank": 60},
      evidence=["docs/qk-machine-search-primitive-rows-20260618.md",
                "docs/performance-primitive-external-research-audit-20260619.md"],
      blocked_by_refutations=["reuse-free attention is not a performance primitive"],
      next_action="Only reopen with an actual locality design and long-prompt Amdahl.",
    ),
  ]

  for row in codegen.get("rows", []):
    if row.get("primitive") == "prefill_tensile_fp16_gemm":
      candidates[5]["score"]["codegen_oracle_state"] = row.get("state")
    if row.get("primitive") == "q8_decode_mmvq_lifecycle":
      candidates[2]["score"]["codegen_oracle_state"] = row.get("state")
  return candidates


def summarize(candidates:list[dict[str, Any]]) -> dict[str, Any]:
  counts: dict[str, int] = {}
  by_phase: dict[str, dict[str, int]] = {}
  for cand in candidates:
    counts[cand["state"]] = counts.get(cand["state"], 0) + 1
    phase = cand["phase"]
    by_phase.setdefault(phase, {})
    by_phase[phase][cand["state"]] = by_phase[phase].get(cand["state"], 0) + 1
  ranked = sorted(candidates, key=lambda c: c["score"].get("rank", 999))
  return {
    "state_counts": counts,
    "state_counts_by_phase": by_phase,
    "top_ranked": [c["id"] for c in ranked[:5]],
    "live_questions": [
      "Is external artifact policy acceptable for research routes?",
      "Does Claude's Route A/P2 dependency-free LDS work beat the current diagnostic state?",
      "Should q8 decode artifact route remain research-only or become a maintained opt-in?",
      "Is a reusable AMD renderer/scheduler project funded, or are native codegen rows closed for now?",
    ],
  }


def write_summary_md(payload:dict[str, Any]) -> None:
  lines = [
    "# Primitive lifecycle search seed - 2026-06-19",
    "",
    "Read-only seed ledger. It does not run hardware or route a model path.",
    "",
    "## State counts",
    "",
  ]
  for state, count in sorted(payload["summary"]["state_counts"].items()):
    lines.append(f"- `{state}`: {count}")
  lines += ["", "## Ranked candidates", ""]
  by_id = {c["id"]: c for c in payload["candidates"]}
  for cid in payload["summary"]["top_ranked"]:
    cand = by_id[cid]
    lines.append(f"- `{cid}`: `{cand['state']}`; next: {cand['next_action']}")
  lines += ["", "## Live questions", ""]
  lines += [f"- {q}" for q in payload["summary"]["live_questions"]]
  lines.append("")
  (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  candidates = build_candidates()
  payload = {
    "schema": "primitive_lifecycle_search_seed_v1",
    "date": datetime.date.today().isoformat(),
    "commit": _git_commit(),
    "scope_doc": "docs/primitive-lifecycle-search-scope-20260619.md",
    "candidates": candidates,
    "summary": summarize(candidates),
  }
  (OUT / "candidates.json").write_text(json.dumps(payload, indent=2) + "\n")
  write_summary_md(payload)
  print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
  main()
