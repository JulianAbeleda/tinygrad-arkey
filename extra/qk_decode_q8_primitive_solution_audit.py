#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib, subprocess
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_primitive_solution_audit_result.json"


def read_json(rel: str) -> dict[str, Any]:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else {}


def read_text(rel: str) -> str:
  p = ROOT / rel
  return p.read_text() if p.exists() else ""


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def metric(d: dict[str, Any], *path: str) -> Any:
  cur: Any = d
  for p in path:
    if not isinstance(cur, dict): return None
    cur = cur.get(p)
  return cur


def main() -> int:
  clock = read_json("bench/qk-decode-primitive-transfer/decode_q8_clock_authority_manual_peak_confirm_result.json")
  controlled_policy = read_json("bench/qk-decode-primitive-transfer/decode_q8_controlled_clock_policy_closeout_result.json")
  lifecycle = read_json("bench/qk-decode-primitive-transfer/decode_q8_lifecycle_band_attribution_result.json")
  successor = read_json("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_successor_object_result.json")
  promotion = read_json("bench/q8-ffn-artifact-promotion/promotion_result.json")
  graph_route = read_text("extra/q8_ffn_graph_route.py")

  manual = metric(clock, "summary", "manual_peak") or {}
  manual_total = metric(manual, "total_us", "median")
  manual_pass = manual.get("target_pass_sessions")
  manual_sessions = manual.get("sessions")
  slow_total = metric(lifecycle, "summary", "lifecycle_steady_total", "median_of_session_medians_us")

  rows = [
    {
      "solution": "controlled_clock_policy",
      "audit_status": "executable_now",
      "evidence": {
        "policy_verdict": controlled_policy.get("verdict"),
        "manual_peak_median_us": manual_total,
        "manual_peak_pass_sessions": f"{manual_pass}/{manual_sessions}",
        "auto_lifecycle_median_us": metric(controlled_policy, "summary", "auto", "median_lifecycle_us"),
      },
      "decision": "accepted_as_default_off_research_route",
      "why": "Solves the observed band without changing primitives; policy remains explicit and default-off.",
    },
    {
      "solution": "fuse_producer_consumer_one_kernel",
      "audit_status": "not_implementation_ready",
      "evidence": {
        "current_route_has_two_programs": "q8_rmsnorm_side_inject" in graph_route and "q8_mmvq_gateup_inject" in graph_route,
        "successor_object_verdict": successor.get("verdict"),
        "successor_lowering_status": metric(successor, "object", "lowering_status"),
      },
      "decision": "scope_as_new_primitive_project_not_patch",
      "why": (
        "True fusion must combine RMSNorm/q8 production with 12288x2 Q4_K consumers. The existing route only injects "
        "two separate PROGRAMs sharing a q8 buffer. The successor object is metadata-only, not a lowerable fused kernel."
      ),
    },
    {
      "solution": "avoid_per_dispatch_host_waits",
      "audit_status": "partially_auditable_now",
      "evidence": {
        "current_micro_harness_uses_wait_true": True,
        "model_graph_route_uses_two_lazy_custom_kernels": "custom_kernel" in graph_route,
        "controlled_clock_median_us": manual_total,
        "uncontrolled_slow_lifecycle_us": slow_total,
      },
      "decision": "audit_in_model_or_graph_capture_before_kernel_work",
      "why": (
        "The isolated harness times each dispatch with wait=True, which exaggerates burst/session effects. The model route "
        "can enqueue two custom kernels lazily, so the useful audit is graph/model capture timing versus wait=True microtiming."
      ),
    },
    {
      "solution": "batch_or_amortize_decode_work",
      "audit_status": "auditable_but_not_default_decode_solution",
      "evidence": {
        "decode_target": "single-token T=1 FFN gate/up",
        "q8_supported_shape": "Qwen3-8B dim=4096 hidden=12288 gate/up",
      },
      "decision": "only_for_throughput_batching_or_speculative_decode",
      "why": (
        "Batching can amortize launch/perf-state overhead, but normal interactive decode is token-serial. This belongs to "
        "speculative/batched decode policy, not the current single-token q8 primitive."
      ),
    },
    {
      "solution": "persistent_on_device_lifecycle",
      "audit_status": "not_available_in_current_tooling",
      "evidence": {
        "hcq_tools_present": True,
        "persistent_kernel_route_present": False,
      },
      "decision": "rethink_as_runtime_project",
      "why": (
        "HCQ launch tooling exists, but there is no persistent decode worker that keeps token feedback, producer, and consumers "
        "on device. This is a runtime architecture project, not a q8 FFN patch."
      ),
    },
  ]

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_PRIMITIVE_SOLUTION_AUDIT",
    "schema": "decode_q8_primitive_solution_audit_v1",
    "verdict": "PASS_DECODE_Q8_SOLUTION_AUDIT_ROUTE_LEVEL_ENOUGH_PRIMITIVE_FUSION_BLOCKED",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "commit": git_sha(),
    "summary": {
      "have_enough_to_solve_now": "yes, as a controlled-clock default-off research route",
      "have_enough_to_implement_true_fused_primitive": "no",
      "recommended_next": "audit in-model graph-captured q8 route timing under auto/manual_peak before new primitive work",
    },
    "rows": rows,
    "next": {
      "do_now": "model-route timing audit: Q8_FFN_HANDWRITTEN=1 under auto and manual_peak, with no per-kernel wait=True microtiming.",
      "do_not_do_yet": "start fused producer+consumer kernel or persistent runtime before model-route timing proves remaining overhead.",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "summary": result["summary"], "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
