#!/usr/bin/env python3
from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_tools_pending_scope_result.json"


def load(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  return json.loads(path.read_text()) if path.exists() else default


def exists(rel: str) -> bool:
  return (ROOT / rel).exists()


def tool(name: str, *fallbacks: str) -> str | None:
  for item in (name, *fallbacks):
    found = shutil.which(item)
    if found:
      return found
    p = pathlib.Path(item)
    if p.exists():
      return str(p)
  return None


def main() -> int:
  oracle_extract = load("bench/qk-decode-primitive-transfer/decode_oracle_gateup_extract_result.json", {})
  oracle_semantic = load("bench/qk-decode-primitive-transfer/decode_oracle_semantic_map_result.json", {})
  hip_runner = load("bench/qk-decode-primitive-transfer/decode_oracle_hip_runner_result.json", {})
  att = load("bench/qk-decode-primitive-transfer/decode_oracle_att_result.json", {})
  coarse = load("bench/qk-decode-primitive-transfer/decode_oracle_coarse_attribution_result.json", {})
  dnr4_scope = load("bench/qk-decode-primitive-transfer/decode_dnr4_liverange_scope_result.json", {})
  t1 = load("bench/qk-decode-primitive-transfer/decode_dnr4_t1_timing_result.json", {})

  tools = [
    {
      "id": "T0-llvm-code-object",
      "status": "ready",
      "paths": {
        "llvm_objdump": tool("llvm-objdump", "/opt/rocm/llvm/bin/llvm-objdump"),
        "llvm_readobj": tool("llvm-readobj", "/opt/rocm/llvm/bin/llvm-readobj"),
      },
      "answers": ["HSACO identity", "metadata", "symbol presence", "static ISA"],
      "remaining_gap": "does not attribute time",
    },
    {
      "id": "T1-oracle-extraction",
      "status": "ready" if oracle_extract.get("gate_pass") else "blocked",
      "probes": ["extra/qk_decode_oracle_gateup_extract.py"],
      "answers": ["exact q8_mmvq_gateup artifact", "metadata VGPR/SGPR/private/LDS", "grouped ISA contract"],
      "remaining_gap": "metadata counts differ from profiler resource columns; use each only in its own context",
    },
    {
      "id": "T2-semantic-map",
      "status": "ready" if oracle_semantic.get("gate_pass") else "blocked",
      "probes": ["extra/qk_decode_oracle_semantic_map.py"],
      "answers": ["S0-S5 oracle stages", "S3 dot/scale body", "S4/S5 reduction/writeback"],
      "remaining_gap": "native/C7C PCs still need equivalent stage labels for PC timeline comparison",
    },
    {
      "id": "T3-hip-rocprof-runner",
      "status": "ready" if hip_runner.get("gate_pass") else "blocked",
      "probes": ["extra/qk_decode_oracle_hip_runner_probe.py"],
      "answers": ["rocprof-visible oracle dispatch", "kernel-trace resource/timing", "ROCm stack consistency"],
      "remaining_gap": "kernel trace is coarse; no PC stalls",
    },
    {
      "id": "T4-att-thread-trace",
      "status": "blocked",
      "probes": ["extra/qk_decode_oracle_att_probe.py"],
      "answers": ["PC-level oracle timeline", "stage stalls", "wait dependency attribution"],
      "blocking_condition": "missing rocprof trace decoder shared library",
      "evidence": att.get("verdict"),
    },
    {
      "id": "T5-native-resource-ledger",
      "status": "ready",
      "probes": ["extra/qk_decode_native_renderer_dnr3c7a_resource_ledger.py"],
      "answers": ["native allocated VGPR", "live bands", "phase pressure", "private/LDS descriptor"],
      "remaining_gap": "needs DNR4-specific live-interval assertions for each new candidate",
    },
    {
      "id": "T6-same-harness-timing",
      "status": "ready",
      "probes": ["extra/qk_decode_dnr4_t1_timing_probe.py", "extra/qk_decode_native_renderer_dnr3c7d_confirmation_probe.py"],
      "answers": ["material timing movement", "candidate promotion/no-promotion"],
      "remaining_gap": "does not explain why without PMC/ATT",
    },
    {
      "id": "T7-native-PMC",
      "status": "ready_partial",
      "probes": ["extra/qk_decode_native_renderer_dnr3c7b_pmc_ladder.py", "extra/qk_decode_native_renderer_dnr3c7d_confirmation_probe.py"],
      "answers": ["SQ wait/busy/cache/LDS direction on native candidates"],
      "remaining_gap": "not PC-level; cannot directly join oracle HIP PCs to native HCQ PCs",
    },
    {
      "id": "T8-search-BEAM",
      "status": "not_ready",
      "answers": ["candidate exploration once objective exists"],
      "blocking_condition": "needs trusted objective from DNR4-T2 or ATT; static similarity was refuted",
    },
  ]

  pending = [
    {
      "id": "P0-install-att-decoder",
      "status": "blocked_external",
      "why": "ATT is the only path to PC-level oracle stalls.",
      "done_when": "extra/qk_decode_oracle_att_probe.py returns PASS and emits trace artifacts.",
      "depends_on": ["T4-att-thread-trace"],
    },
    {
      "id": "P1-native-stage-label-map",
      "status": "ready",
      "why": "Even without ATT, native/C7C instructions should be labeled with OES-4 S0-S5 names to make DNR4 live intervals precise.",
      "done_when": "native_dnr2, best_static, C7C, and DNR4 candidates report stage-labeled register intervals.",
      "depends_on": ["T2-semantic-map", "T5-native-resource-ledger"],
    },
    {
      "id": "P2-DNR4-T2-dot-body-compression-scope",
      "status": "next",
      "why": "DNR4-T1 reduction reuse was correct but not material. Remaining resource gap is in S2/S3 dot/load/unpack body.",
      "done_when": "a scoped candidate names how to reduce S2/S3 max VGPR/register bands without breaking 16 dot4 correctness.",
      "depends_on": ["P1-native-stage-label-map"],
    },
    {
      "id": "P3-DNR4-T2-structural-candidate",
      "status": "pending",
      "why": "Oracle S2/S3 stays within v0-v25 while native/C7C preload variants use high vector bands.",
      "done_when": "candidate launches, is correct, preserves dot4, and reduces max/static allocated VGPR target toward <=40.",
      "depends_on": ["P2-DNR4-T2-dot-body-compression-scope"],
    },
    {
      "id": "P4-DNR4-T2-same-harness-timing",
      "status": "pending",
      "why": "Structural resource wins are not enough; T1 proved that.",
      "done_when": ">=30us vs native, >=15us vs best static, or >=10us vs C7C, with correctness.",
      "depends_on": ["P3-DNR4-T2-structural-candidate", "T6-same-harness-timing"],
    },
    {
      "id": "P5-DNR4-T2-PMC-confirmation",
      "status": "pending_if_timing_moves",
      "why": "Promotion requires timing plus counter direction.",
      "done_when": "PMC movement agrees with the claimed mechanism, or ATT directly attributes the PC-stage win.",
      "depends_on": ["P4-DNR4-T2-same-harness-timing", "T7-native-PMC"],
    },
    {
      "id": "P6-route-level-decode-decision",
      "status": "pending",
      "why": "If DNR4-T2 does not move, native q8 schedule work should park again and decode should move to route/runtime policy.",
      "done_when": "explicit promote/park decision with timing, resource, and quality policy.",
      "depends_on": ["P4-DNR4-T2-same-harness-timing"],
    },
    {
      "id": "P7-search-objective-definition",
      "status": "not_ready",
      "why": "Search is only useful after DNR4 or ATT names a measurable objective.",
      "done_when": "objective terms are concrete: e.g. max VGPR band, S3 stage latency, waitcnt/stall family, or same-run timing.",
      "depends_on": ["P4-DNR4-T2-same-harness-timing", "P0-install-att-decoder"],
    },
  ]

  gates = {
    "oracle_extract_ready": oracle_extract.get("gate_pass") is True,
    "oracle_semantic_ready": oracle_semantic.get("gate_pass") is True,
    "hip_runner_ready": hip_runner.get("gate_pass") is True,
    "att_blocked_recorded": att.get("verdict") == "BLOCKED_DECODE_ORACLE_ATT_DECODER_LIBRARY_MISSING",
    "coarse_attribution_ready": coarse.get("gate_pass") is True,
    "dnr4_scope_ready": dnr4_scope.get("gate_pass") is True,
    "t1_not_material_recorded": t1.get("verdict") == "BLOCKED_DNR4_T1_STRUCTURAL_ONLY_TIMING_NOT_MATERIAL",
    "next_task_is_t2": pending[2]["id"] == "P2-DNR4-T2-dot-body-compression-scope",
  }

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_TOOLS_AND_PENDING_TASK_SCOPE",
    "schema": "decode_tools_pending_scope_v1",
    "verdict": "PASS_DECODE_TOOLS_PENDING_SCOPE_READY" if all(gates.values()) else "BLOCKED_DECODE_TOOLS_PENDING_SCOPE_INCOMPLETE",
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "tools": tools,
    "pending_tasks": pending,
    "current_decision": {
      "do_next": "P2-DNR4-T2-dot-body-compression-scope",
      "do_not_do_next": [
        "do not run BEAM/search yet",
        "do not continue DNR4-T1 except as cleanup",
        "do not add static count matching patches",
        "do not claim PC-level attribution until ATT decoder is available",
      ],
      "main_blocker": "ATT decoder missing for PC-level oracle stalls",
      "main_executable_path": "DNR4-T2 static/live-range compression followed by same-harness timing",
    },
    "gates": gates,
  }

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "ready_tools": [row["id"] for row in tools if row["status"].startswith("ready")],
    "blocked_tools": [row["id"] for row in tools if row["status"].startswith("blocked") or row["status"] == "not_ready"],
    "next_task": result["current_decision"]["do_next"],
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())

