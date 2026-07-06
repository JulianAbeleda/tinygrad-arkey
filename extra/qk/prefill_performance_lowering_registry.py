"""Performance-focused prefill lowering scope registry (scaffold-only).

This registry tracks the 8B/14B prefill performance recovery phases separately from the strict
default purity/debt registry so strict-default reporting remains unchanged.
"""

from __future__ import annotations

import pathlib
from typing import Any, TypedDict

ROOT = pathlib.Path(__file__).resolve().parents[2]

DOC_PATH = "docs/prefill-performance-lowering-scope-20260706.md"

VALID_OWNER_AREAS = ("scheduler", "codegen", "vocab", "policy")
VALID_STATUSES = ("not_started", "pending", "in_progress", "blocked", "ready", "done")


class PrefillPerformanceLoweringRow(TypedDict):
  id: str
  target: str
  target_label: str
  phase: int
  phase_order: int
  phase_name: str
  owner_area: str
  status: str
  blockers: list[str]
  reuse_files: list[str]
  gates: list[str]
  success_criteria: list[str]
  scope_doc: str


_PERFORMANCE_ROWS: tuple[PrefillPerformanceLoweringRow, ...] = (
  {
    "id": "prefill_performance_target_1_fp16_recovery",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 0,
    "phase_order": 0,
    "phase_name": "target_scope",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "PREFILL_GRAPH_GEMM=1 still routes to raw instruction-list WMMA baseline",
      "No generated performance substrate yet owns prefill-WMMA+LDS staging",
    ],
    "reuse_files": [
      DOC_PATH,
      "extra/qk/prefill_graph_gemm_route.py",
      "extra/qk/prefill_schedule_spec.py",
      "extra/qk/prefill_v2_schedule_search.py",
      "extra/qk/prefill_v2_schedule_table.json",
      "tinygrad/llm/model.py",
      "docs/codegen-wmma-lds-staging-design-20260705.md",
    ],
    "gates": [
      "PURE_MACHINE_SEARCH_ONLY",
      "BUBBLEBEAM_FUTURESIGHT",
    ],
    "success_criteria": [
      "Both targets and all planned phases are tracked in one scope row set.",
      "No changes to strict-default lowering report sources are required for this scaffold.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_baseline",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 1,
    "phase_order": 1,
    "phase_name": "baseline",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "No generated baseline currently beats raw 8B research route in practice",
    ],
    "reuse_files": [
      "docs/8b-prefill-generated-route-closed-20260705.md",
      "docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md",
    ],
    "gates": [
      "baseline_pp512_recovery_gate",
    ],
    "success_criteria": [
      "Current baseline rows remain recorded and reproducible without raw instruction route promotion.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_single_operand_stage",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 2,
    "phase_order": 2,
    "phase_name": "single_operand_lds_staging",
    "owner_area": "codegen",
    "status": "pending",
    "blockers": [
      "Generated fp16 shaped-WMMA LOCAL staging substrate probe passes, but only for a tiny custom kernel",
      "The fp16 prefill TC route has not been integrated with generated LOCAL staging",
      "No medium/route-bound fp16 graph-GEMM performance gate exists yet",
    ],
    "reuse_files": [
      "docs/codegen-wmma-lds-staging-design-20260705.md",
      "tinygrad/schedule/rangeify.py",
      "tinygrad/codegen/opt/postrange.py",
      "tinygrad/llm/model.py",
      "extra/qk/prefill_graph_gemm_single_operand_stage_gate.py",
      "extra/qk/prefill_graph_gemm_fp16_stage_gate.py",
      "extra/qk/prefill_graph_gemm_route_bound_stage_gate.py",
      "bench/prefill-graph-gemm-single-operand-stage/latest.json",
      "bench/prefill-graph-gemm-fp16-single-operand-stage/latest.json",
      "bench/prefill-graph-gemm-route-bound-stage/latest.json",
    ],
    "gates": [
      "prefill_graph_gemm_single_operand_stage_gate",
      "prefill_graph_gemm_fp16_single_operand_stage_gate",
      "prefill_graph_gemm_route_bound_no_raw_ops_ins_gate",
      "prefill_graph_gemm_route_bound_stage_gate",
    ],
    "success_criteria": [
      "Numerically correct small fp16 shaped-WMMA generated single-operand LOCAL staging probe.",
      "Generated AMD kernel shows shared local storage, barrier, and fp16 WMMA binding for the staged operand.",
      "Custom probe execution does not show raw Ops.INS or extra/qk/prefill/wmma.py markers.",
      "A separate fp16 route-bound gate must prove the actual 8B prefill path uses generated staging.",
      "Remaining integration step must bind this mechanism to the fp16 prefill TC route and add medium-shape timing.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_both_operands_stage",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 3,
    "phase_order": 3,
    "phase_name": "both_operands_staging",
    "owner_area": "codegen",
    "status": "pending",
    "blockers": [
      "Generated fp16 both-operand LOCAL staging substrate probe passes, but only for a tiny custom kernel",
      "Both-source staging is not bound to the actual fp16 prefill TC route",
      "No medium/route-bound fp16 graph-GEMM performance gate exists yet",
    ],
    "reuse_files": [
      "docs/codegen-wmma-lds-staging-design-20260705.md",
      "docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md",
      "tinygrad/llm/model.py",
      "extra/qk/prefill_graph_gemm_fp16_stage_gate.py",
      "bench/prefill-graph-gemm-fp16-both-operands-stage/latest.json",
    ],
    "gates": [
      "prefill_graph_gemm_fp16_both_operands_stage_gate",
      "prefill_graph_gemm_route_bound_no_raw_ops_ins_gate",
    ],
    "success_criteria": [
      "Both A and B operands can be staged in a generated fp16 shaped-WMMA custom probe.",
      "Generated AMD kernel shows expected local storage, barriers, and fp16 WMMA binding for both operands.",
      "A separate route-bound gate must prove actual prefill execution does not select raw Ops.INS or extra/qk/prefill/wmma.py.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_coop_partition",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 4,
    "phase_order": 4,
    "phase_name": "cooperative_partition",
    "owner_area": "scheduler",
    "status": "pending",
    "blockers": [
      "Cooperative global-to-LDS partition and per-lane fragment reads are not yet generated",
    ],
    "reuse_files": [
      "tinygrad/schedule/rangeify.py",
      "tinygrad/schedule/wmma.py",
      "docs/handwritten-kernel-exhaustive-lowering-scope-20260706.md",
      "tinygrad/llm/model.py",
    ],
    "gates": [
      "prefill_graph_gemm_coop_partition_gate",
      "prefill_graph_gemm_no_raw_ops_ins_gate",
    ],
    "success_criteria": [
      "8B pp512 materially above current strict-pure runtime when partitioning is enabled.",
      "Route-bound execution stays on generated scheduler/codegen lowering, not raw graph-GEMM research.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_optional_double_buffer",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 5,
    "phase_order": 5,
    "phase_name": "optional_double_buffer",
    "owner_area": "scheduler",
    "status": "not_started",
    "blockers": [
      "Only needed if phase 1-4 leaves >20% residual vs historical graph-GEMM trajectory",
    ],
    "reuse_files": [
      "docs/codegen-wmma-lds-staging-design-20260705.md",
      "tinygrad/codegen/opt/postrange.py",
    ],
    "gates": [
      "prefill_graph_gemm_optional_double_buffer_gate",
    ],
    "success_criteria": [
      "Conditional performance win without introducing race/stall regressions.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_1_promotion",
    "target": "target_1",
    "target_label": "8B resident-fp16 graph-GEMM recovery",
    "phase": 6,
    "phase_order": 6,
    "phase_name": "promotion",
    "owner_area": "policy",
    "status": "blocked",
    "blockers": [
      "8B performance parity must be demonstrated before route authority changes.",
    ],
    "reuse_files": [
      "docs/prefill-performance-lowering-scope-20260706.md",
      "extra/qk/prefill_performance_lowering_report.py",
    ],
    "gates": [
      "route_authority_gate",
    ],
    "success_criteria": [
      "Pure generated path replaces raw graph-GEMM path for strict-default-reachable prefill.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_packed_mmq_recovery",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 0,
    "phase_order": 0,
    "phase_name": "target_scope",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "Generated packed/MMQ substrate still does not dominate direct-packed VALU path",
    ],
    "reuse_files": [
      DOC_PATH,
      "extra/qk/prefill_int8_wmma_spec.py",
      "tinygrad/llm/prefill_routes.py",
      "tinygrad/llm/generated_candidates.py",
      "extra/qk/q4k_prefill_route_spec.py",
      "extra/qk/q6k_prefill_route_spec.py",
      "extra/qk/q4k_wmma_tile_lowering.py",
      "extra/qk/q4k_wmma_full_role_contract_gate.py",
      "docs/route-b-iu8-wmma-mmq-design-20260705.md",
    ],
    "gates": [
      "BUBBLEBEAM_FUTURESIGHT",
      "PURE_MACHINE_SEARCH_ONLY",
    ],
    "success_criteria": [
      "Both targets for this scope are represented without touching strict purity registries.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_baseline",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 1,
    "phase_order": 1,
    "phase_name": "baseline",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "Current direct-packed route remains memory-safe but performance-limited by dequant/VALU",
    ],
    "reuse_files": [
      "docs/prefill-packed-generated-tile-scope-20260704.md",
      "docs/prefill-14b-llama-parity-trace-20260704.md",
    ],
    "gates": [
      "prefill_14b_baseline_gate",
    ],
    "success_criteria": [
      "Current strict-pure and model-authority baseline metrics are fixed as reference points.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_tile_contract",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 2,
    "phase_order": 2,
    "phase_name": "tile_contract",
    "owner_area": "scheduler",
    "status": "done",
    "blockers": [],
    "reuse_files": [
      "docs/q4k-wmma-full-role-lowering-solution-scope-20260705.md",
      "extra/qk/prefill_int8_wmma_spec.py",
      "extra/qk/q4k_prefill_route_spec.py",
      "tinygrad/llm/generated_candidates.py",
      "extra/qk/q4k_wmma_tile_lowering.py",
      "extra/qk/q4k_wmma_full_role_contract_gate.py",
    ],
    "gates": [
      "prefill_14b_tile_contract_gate",
      "extra.qk.q4k_wmma_full_role_contract_gate",
    ],
    "success_criteria": [
      "Tile role shape contract is bounded and compile-time enforceable for both small and large 14B shapes.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_wmma_surface_decision",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 3,
    "phase_order": 3,
    "phase_name": "wmma_surface_decision",
    "owner_area": "vocab",
    "status": "done",
    "blockers": [],
    "reuse_files": [
      "docs/route-b-iu8-wmma-mmq-design-20260705.md",
      "tinygrad/renderer/amd/generate.py",
      "tinygrad/codegen/experimental.py",
      "bench/q4k-wmma-scheduler-surface/latest.json",
    ],
    "gates": [
      "prefill_14b_wmma_surface_gate",
    ],
    "success_criteria": [
      "Planned 14B MMQ lowering surface is selected and backed by existing scheduler/codegen evidence with no raw hand-kernel escape.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_small_lifecycle",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 4,
    "phase_order": 4,
    "phase_name": "small_lifecycle",
    "owner_area": "codegen",
    "status": "done",
    "blockers": [],
    "reuse_files": [
      "extra/qk/prefill_int8_wmma_spec.py",
      "docs/q4k-wmma-full-role-lowering-solution-scope-20260705.md",
      "bench/q4k-wmma-tiled-lifecycle/latest.json",
    ],
    "gates": [
      "prefill_14b_small_lifecycle_gate",
    ],
    "success_criteria": [
      "Small multi-tile lifecycle keeps tile-local RAW, QSUM, and final output lifetime bounded.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_synthetic_shape",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 5,
    "phase_order": 5,
    "phase_name": "synthetic_role_shape",
    "owner_area": "scheduler",
    "status": "pending",
    "blockers": [
      "Synthetic role-shape contract exists, but execution is blocked by scheduler_owned_tile_loop_missing",
    ],
    "reuse_files": [
      "tinygrad/llm/prefill_routes.py",
      "extra/qk/route_manifest.py",
      "tinygrad/llm/generated_candidates.py",
      "extra/qk/q4k_wmma_tile_lowering.py",
      "bench/q4k-wmma-tiled-role-shape-exec/latest.json",
    ],
    "gates": [
      "prefill_14b_synthetic_role_shape_gate",
    ],
    "success_criteria": [
      "14B role-shape gate runs without loading model artifacts and without graph explosion.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_model_authority",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 6,
    "phase_order": 6,
    "phase_name": "model_authority",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "No clear 14B model-authority route policy for MMQ-first packed prefill yet",
    ],
    "reuse_files": [
      "extra/qk/route_manifest.py",
      "tinygrad/llm/prefill_routes.py",
      "tinygrad/llm/generated_candidates.py",
      "docs/prefill-performance-lowering-scope-20260706.md",
    ],
    "gates": [
      "prefill_14b_model_authority_gate",
    ],
    "success_criteria": [
      "Authority test and route selection show packed/MMQ execution for relevant 14B shapes.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_q6_residual_decision",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 7,
    "phase_order": 7,
    "phase_name": "q6_residual_decision",
    "owner_area": "policy",
    "status": "pending",
    "blockers": [
      "No explicit post-MMQ Q6 residual policy (keep direct-packed or promote generated MMQ) is in place",
    ],
    "reuse_files": [
      "docs/route-b-iu8-wmma-mmq-design-20260705.md",
      "docs/prefill-14b-llama-parity-trace-20260704.md",
      "tinygrad/llm/prefill_routes.py",
      "tinygrad/llm/generated_candidates.py",
      "extra/qk/q6k_prefill_route_spec.py",
    ],
    "gates": [
      "prefill_14b_q6_decision_gate",
    ],
    "success_criteria": [
      "Q6_K residual budget is measured and policyized for default route selection.",
    ],
    "scope_doc": DOC_PATH,
  },
  {
    "id": "prefill_performance_target_2_promotion",
    "target": "target_2",
    "target_label": "14B packed/MMQ memory-safe recovery",
    "phase": 8,
    "phase_order": 8,
    "phase_name": "promotion",
    "owner_area": "policy",
    "status": "blocked",
    "blockers": [
      "No >1.25x improvement on strict-pure 14B packed baseline yet",
    ],
    "reuse_files": [
      "docs/prefill-performance-lowering-scope-20260706.md",
      "extra/qk/prefill_performance_lowering_report.py",
    ],
    "gates": [
      "performance_promotion_gate",
    ],
    "success_criteria": [
      "Whole-model 14B authority moves from direct-packed/VALU ceiling to packed/MMQ generated path.",
    ],
    "scope_doc": DOC_PATH,
  },
)


def _validate_rows(rows: tuple[PrefillPerformanceLoweringRow, ...]) -> None:
  seen = set[str]()
  seen_phases: dict[str, list[int]] = {}
  for row in rows:
    if row["id"] in seen:
      raise ValueError(f"duplicate prefill performance row id {row['id']!r}")
    seen.add(row["id"])

    if row["phase"] != row["phase_order"]:
      raise ValueError(f"phase and phase_order must match for row {row['id']!r}")

    if row["owner_area"] not in VALID_OWNER_AREAS:
      raise ValueError(f"invalid owner area {row['owner_area']!r} for row {row['id']!r}")

    if row["status"] not in VALID_STATUSES:
      raise ValueError(f"invalid status {row['status']!r} for row {row['id']!r}")

    if not pathlib.Path(row["scope_doc"]).is_absolute():
      scope_path = ROOT / row["scope_doc"]
    else:
      scope_path = pathlib.Path(row["scope_doc"])
    if not scope_path.exists():
      raise ValueError(f"referenced scope doc {row['scope_doc']!r} for row {row['id']!r} does not exist")
    for reuse_file in row["reuse_files"]:
      reuse_path = pathlib.Path(reuse_file)
      if not reuse_path.is_absolute():
        reuse_path = ROOT / reuse_file
      if not reuse_path.exists():
        raise ValueError(f"referenced reuse file {reuse_file!r} for row {row['id']!r} does not exist")

    seen_phases.setdefault(row["target"], []).append(row["phase"])

  for target, phases in seen_phases.items():
    ordered = sorted(phases)
    if ordered != list(range(ordered[0], ordered[0] + len(ordered))):
      raise ValueError(f"phases for target {target!r} are not contiguous and ordered: {phases}")


def _sanitize_row(row: PrefillPerformanceLoweringRow) -> dict[str, Any]:
  return {
    "id": row["id"],
    "target": row["target"],
    "target_label": row["target_label"],
    "phase": row["phase"],
    "phase_order": row["phase_order"],
    "phase_name": row["phase_name"],
    "owner_area": row["owner_area"],
    "status": row["status"],
    "blockers": list(row["blockers"]),
    "reuse_files": list(row["reuse_files"]),
    "gates": list(row["gates"]),
    "success_criteria": list(row["success_criteria"]),
    "scope_doc": row["scope_doc"],
  }


_validate_rows(_PERFORMANCE_ROWS)


def ids() -> tuple[str, ...]:
  return tuple(row["id"] for row in _PERFORMANCE_ROWS)


def rows() -> list[dict[str, Any]]:
  return [_sanitize_row(r) for r in _PERFORMANCE_ROWS]


def row(row_id: str) -> dict[str, Any]:
  for r in _PERFORMANCE_ROWS:
    if r["id"] == row_id:
      return _sanitize_row(r)
  raise KeyError(f"unknown prefill performance lowering row id {row_id!r}")


def rows_by_target() -> dict[str, list[dict[str, Any]]]:
  grouped: dict[str, list[dict[str, Any]]] = {}
  for r in rows():
    grouped.setdefault(r["target"], []).append(r)
  for items in grouped.values():
    items.sort(key=lambda item: item["phase"])
  return grouped


def build() -> dict[str, Any]:
  all_rows = rows()
  by_target: dict[str, int] = {}
  by_status: dict[str, int] = {}
  by_owner_area: dict[str, int] = {}
  by_phase: dict[int, int] = {}
  for row in all_rows:
    by_target.setdefault(row["target"], 0)
    by_target[row["target"]] += 1
    by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    by_owner_area[row["owner_area"]] = by_owner_area.get(row["owner_area"], 0) + 1
    by_phase[row["phase"]] = by_phase.get(row["phase"], 0) + 1

  pending_rows = [r for r in all_rows if r["status"] in {"pending", "not_started"}]
  blocked_rows = [r["id"] for r in all_rows if r["status"] == "blocked"]
  return {
    "schema": "prefill-performance-lowering-registry.v1",
    "total_rows": len(all_rows),
    "scope_doc": DOC_PATH,
    "rows": all_rows,
    "by_target": by_target,
    "by_status": by_status,
    "by_owner_area": by_owner_area,
    "by_phase": by_phase,
    "pending_rows": [r["id"] for r in pending_rows],
    "blocked_rows": blocked_rows,
    "targets": sorted(by_target),
  }
