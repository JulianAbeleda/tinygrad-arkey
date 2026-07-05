#!/usr/bin/env python3
"""Gate registry -- the single declarative table of qk gates/audits/checks, plus the one runner.

Extends the route_manifest/quant_semantics DATA-module idiom to the experiment surface (anti-re-sprawl +
one-IR-one-engine, structure/Development/tinygrad-coding-overrides.md): a gate is a REGISTRY ROW plus a pure
`build()` in its own module. The runner owns everything the ~90 historical mains cloned: ROOT resolution,
env-before-tinygrad-import ordering, lazy entry import, artifact write (`latest.json`, indent=2 + trailing
newline -- the gate-artifact convention; probe_harness.write_json is the sort_keys probe convention), stdout
echo, traceback capture, and exit-code policy.

A `build()` returns either the verdict dict (runner writes/prints it) or an int exit code (report-only checks
that print their own findings). It must NOT write artifacts or call sys.exit itself.

Usage:
  PYTHONPATH=. python3 -m extra.qk.gate_registry list [--kind KIND] [--gpu|--no-gpu]
  PYTHONPATH=. python3 -m extra.qk.gate_registry run NAME [NAME...]
  PYTHONPATH=. python3 -m extra.qk.gate_registry run --tranche artifact-only
"""
from __future__ import annotations
import argparse, importlib, json, os, pathlib, sys, time, traceback
from dataclasses import dataclass, field

ROOT = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class GateSpec:
  name: str                      # stable id (no _gate/_audit suffix)
  entry: str                     # "extra.qk.module:build" -- imported lazily, AFTER env is applied
  kind: str = "audit"            # gate | audit | microgate | probe | check
  needs_gpu: bool = False        # True: requires DEV=AMD + hardware; False: consumes committed artifacts only
  out_dir: str | None = None     # bench/<out_dir>/latest.json; None = print/exit-code only
  inputs: tuple[str, ...] = ()   # repo-relative artifacts consumed (declared, greppable for retirement checks)
  pass_verdicts: frozenset[str] | None = None  # None = exit 0 whenever build() completes
  env: dict[str, str] = field(default_factory=dict)  # setdefault'd BEFORE entry import (sacred env ordering)
  artifact_name: str = "latest.json"  # override for gates sharing one bench dir with distinct filenames
  snapshot: bool = False         # also write a dated <name>-<ts>.json copy next to the artifact


GATES: tuple[GateSpec, ...] = (
  GateSpec(name="pure_search_gap", entry="extra.qk.pure_search_gap_audit:build",
           out_dir="qk-pure-search-gap",
           inputs=("bench/qk-pure-search-gap", "bench/qk-decode-hotloop-schedule-diff/latest.json",
                   "bench/qk-decode-primitive-space", "bench/qk-decode-isa-vectorization/latest.json",
                   "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json",
                   "bench/qk-decode-pressure-search-ownership/latest.json",
                   "docs/decode-tile-delta-attack-result-20260627.md",
                   "docs/decode-codegen-scheduler-capability-scope.md")),
  GateSpec(name="pure_machine_search_gap", entry="extra.qk.pure_machine_search_gap_audit:build",
           out_dir="qk-pure-machine-search-gap",
           inputs=("bench/canonical-benchmarks.json", "bench/qk-pure-search-gap/latest.json",
                   "bench/qk-prefill-search/prefill_search_readiness.json",
                   "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json",
                   "bench/qk-decode-pressure-search-ownership/latest.json",
                   "docs/prefill-long-context-integration-nonsearch-fix-result-20260624.md",
                   "docs/gemv-pure-search-generated-route-scope.md")),
  GateSpec(name="pressure_search_ownership", entry="extra.qk.decode_pressure_search_ownership_audit:build",
           out_dir="qk-decode-pressure-search-ownership",
           inputs=("bench/qk-pure-search-gap/latest.json", "bench/qk-decode-occupancy-guardrail/latest.json",
                   "bench/qk-decode-outer-b-split-combine/latest.json")),
  GateSpec(name="policy_consistency", entry="extra.qk.policy_consistency_check:build", kind="check",
           inputs=("docs/README.md", "bench/README.md", "docs/current-project-state-handoff-20260624.md")),
  GateSpec(name="outer_b_split_contract", entry="extra.qk.decode_outer_b_split_contract:build",
           out_dir="qk-decode-outer-b-split-combine"),
  GateSpec(name="search_space_manifest", entry="extra.qk.search_space_manifest_check:build", kind="check",
           inputs=("bench/qk-search-spaces/search_profiles.json", "extra/qk/route_manifest.py",
                   "extra/qk/quant_semantics.py")),
  GateSpec(name="surface", entry="extra.qk.surface_audit:build", kind="check"),
  GateSpec(name="hotloop_schedule_diff", entry="extra.qk.decode_hotloop_schedule_diff:build",
           out_dir="qk-decode-hotloop-schedule-diff",
           inputs=("bench/qk-decode-attention-isa-diff/disasm_owned_flash_tile_gqa_whole.txt",
                   "bench/qk-decode-isa-vectorization/disasm_flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.txt")),
  GateSpec(name="gemv_purity", entry="extra.qk.gemv_purity_gate:build", kind="gate",
           out_dir="qk-gemv-purity-gate",
           inputs=("bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json",),
           pass_verdicts=frozenset({"GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV",
                                    "GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3",
                                    "GEMV_NOT_PURE__SEARCH_SELECTED_CUSTOM_BRIDGE",
                                    "GEMV_PURE_SEARCH_GENERATED"})),
  GateSpec(name="primitive_detector", entry="extra.qk.decode_primitive_detector:build",
           out_dir="qk-decode-primitive-space",
           inputs=("bench/qk-decode-attention-fused-score-state-pv-attribution/latest.json",
                   "bench/qk-decode-primitive-space/p1_crosslane_latest.json",
                   "bench/qk-decode-primitive-space/all_primitives_latest.json"),
           pass_verdicts=frozenset({"PRIMITIVE_DETECTOR_READY"})),
  GateSpec(name="attention_reopen", entry="extra.qk.attention_reopen_gate:build", kind="gate",
           out_dir="qk-attention-reopen-gate",
           inputs=("bench/amd-isa-backend-decode-attention-ceiling/latest.json",),
           pass_verdicts=frozenset({"PMS_R7_PASS_ATTENTION_REOPEN_GATE"})),
  GateSpec(name="cache_identity_index", entry="extra.qk.decode_cache_identity_index_gate:build", kind="gate",
           needs_gpu=True, out_dir="qk-decode-cache-identity-index", snapshot=True,
           env={"DEV": "AMD", "JIT": "1"},
           pass_verdicts=frozenset({"CACHE_5D_INDEX_AND_UPCAST_PASS", "CACHE_5D_REG_STORE_DEVEC_PASS"})),
  GateSpec(name="occupancy_guardrail", entry="extra.qk.decode_occupancy_guardrail:build",
           out_dir="qk-decode-occupancy-guardrail",
           inputs=("bench/qk-decode-isa-vectorization/latest.json",
                   "bench/qk-decode-hotloop-schedule-diff/latest.json"),
           pass_verdicts=frozenset({"OCCUPANCY_GUARDRAIL_PASS"})),
  GateSpec(name="attention_block_tile_microgate", entry="extra.qk.decode_attention_block_tile_microgate:build",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-block-tile-microgate", snapshot=True,
           env={"DEV": "AMD", "JIT": "1"},
           pass_verdicts=frozenset({"BLOCK_TILE_MICROGATE_PASS"})),
  GateSpec(name="attention_cross_lane_reduce_store", entry="extra.qk.decode_attention_cross_lane_reduce_store_gate:build",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-cross-lane-reduce-store", snapshot=True,
           env={"DEV": "AMD", "JIT": "1"},
           pass_verdicts=frozenset({"CROSS_LANE_REDUCE_STORE_PASS"})),
  GateSpec(name="attention_fused_xlane_score_pv_microgate",
           entry="extra.qk.decode_attention_fused_xlane_score_pv_microgate:build",
           kind="microgate", needs_gpu=True,
           out_dir="qk-decode-attention-fused-xlane-score-pv-microgate", snapshot=True,
           env={"DEV": "AMD", "JIT": "1"},
           pass_verdicts=frozenset({"FUSED_XLANE_SCORE_PV_MICROGATE_PASS"})),
  GateSpec(name="fused_tile_lifecycle_lowering", entry="extra.qk.fused_tile_lifecycle_lowering_gate:build",
           kind="gate", needs_gpu=True, out_dir="qk-fused-tile-lifecycle-lowering", snapshot=True,
           env={"DEV": "AMD", "JIT": "1"},
           inputs=("bench/qk-decode-attention-fused-score-state-pv-tile/latest.json",)),
  GateSpec(name="q6k_generated_coop", entry="extra.qk.q6k_generated_coop_gate:build", kind="gate",
           needs_gpu=True, out_dir="tg-p3-q6k-generated-coop", env={"DEV": "AMD"},
           pass_verdicts=frozenset({"TG_P3_PASS_Q6K_GENERATED_COOP"})),
  GateSpec(name="int8_wmma_codegen", entry="extra.qk.int8_wmma_codegen_gate:build", kind="gate",
           needs_gpu=True, out_dir="qk-int8-wmma-codegen", env={"DEV": "AMD", "TC": "1", "TC_OPT": "1"},
           pass_verdicts=frozenset({"INT8_WMMA_CODEGEN_PASS"})),
  GateSpec(name="q4k_wmma_tiled_lowering_feasibility",
           entry="extra.qk.q4k_wmma_tiled_lowering_feasibility:build", kind="gate",
           needs_gpu=True, out_dir="q4k-wmma-tiled-lowering-feasibility",
           env={"DEV": "AMD", "TC": "1", "TC_OPT": "1"},
           pass_verdicts=frozenset({"Q4K_WMMA_TILED_LOWERING_FEASIBLE"})),
  GateSpec(name="q4k_wmma_tiled_microgate", entry="extra.qk.q4k_wmma_tiled_microgate:build", kind="microgate",
           needs_gpu=True, out_dir="q4k-wmma-tiled-microgate",
           env={"DEV": "AMD", "TC": "1", "TC_OPT": "1"},
           pass_verdicts=frozenset({"Q4K_WMMA_TILED_MICROGATE_PASS"})),
  GateSpec(name="prefill_generated_schedule", entry="extra.qk.prefill_generated_schedule_gate:build",
           kind="gate", out_dir="tg-p4-prefill-generated-schedule",
           pass_verdicts=frozenset({"TG_P4_PASS_PREFILL_GENERATED_SCHEDULE"})),
  GateSpec(name="generated_quant_binding_audit", entry="extra.qk.generated_quant_binding_audit:build",
           kind="audit", out_dir="generated-quant-runtime-binding-audit",
           inputs=("docs/generated-quant-runtime-architecture-scope-20260705.md",
                   "docs/generated-quant-runtime-execution-map-20260705.md",
                   "extra/qk/route_manifest.py",
                   "tinygrad/llm/runtime_specs.py",
                   "tinygrad/llm/quant_specs.py",
                   "tinygrad/llm/generated_candidates.py"),
           pass_verdicts=frozenset({"GENERATED_QUANT_BINDING_AUDIT_READY"})),
  GateSpec(name="generated_q4k_prefill_e2e", entry="extra.qk.generated_q4k_prefill_e2e_gate:build",
           kind="gate", needs_gpu=True, out_dir="generated-q4k-prefill-e2e",
           inputs=("tinygrad/llm/generated_candidates.py",
                   "tinygrad/llm/prefill_routes.py",
                   "extra/qk/prefill_int8_wmma_spec.py",
                   "extra/qk/prefill_mmq_parity_gate.py",
                   "extra/qk/int8_wmma_codegen_gate.py"),
           pass_verdicts=frozenset({"GENERATED_Q4K_PREFILL_E2E_BLOCKED_GRAPH_EXPLOSION"})),
  GateSpec(name="tg_p8_delta_audit", entry="extra.qk.tg_p8_delta_audit:build",
           out_dir="tg-p8-generated-8b-attention-parity", artifact_name="delta_audit.json",
           inputs=("bench/tg-p8-generated-8b-attention-parity/baseline.json",),
           pass_verdicts=frozenset({"TG_P8_1_PASS_DELTA_CLASSIFIED"})),
  GateSpec(name="decode_attention_a3_1_vdot2", entry="extra.qk.decode_attention_a3_1_vdot2_probe:build",
           kind="probe", needs_gpu=True, out_dir="qk-decode-attention-a3-1-vdot2", snapshot=True,
           env={"DEV": "AMD"},
           pass_verdicts=frozenset({"A3_1_RENDERER_VDOT2_PROBE_PASS", "A3_1_RENDERER_VDOT2_PROBE_INCONCLUSIVE"})),
  GateSpec(name="gemv_g2_representation", entry="extra.qk.gemv_g2_representation_probe:build",
           kind="probe", out_dir="qk-gemv-g2-representation-probe", snapshot=True,
           pass_verdicts=frozenset({"G2_LANEMAP_ADDRESS_BUILDER_PASS"})),
  GateSpec(name="gp3_konly_microgate", entry="extra.qk.gp3_konly_microgate:build", kind="microgate",
           needs_gpu=True, out_dir="gp-track", artifact_name="gp3_microgate.json",
           env={"DEV": "AMD", "JIT": "1"},
           pass_verdicts=frozenset({"GP3_PASS_MICROGATE"})),
  GateSpec(name="tg_p11_reduce_upcast_microgate", entry="extra.qk.tg_p11_reduce_upcast_microgate:build",
           kind="microgate", needs_gpu=True, out_dir="tg-p11-reduce-upcast-accumulator",
           artifact_name="invariant_microgate.json", env={"DEV": "AMD"},
           pass_verdicts=frozenset({"TG_P11_1_PASS_BASELINE_LOWERING"})),
  GateSpec(name="primitive_gap", entry="extra.qk.decode_primitive_gap_gate:build", kind="gate",
           out_dir="qk-decode-primitive-space", artifact_name="gap_latest.json", snapshot=True,
           inputs=("bench/qk-decode-primitive-space/latest.json",),
           pass_verdicts=frozenset({"PRIMITIVE_GAP_READY__SEARCHABLE_PRIMITIVES_PRESENT",
                                    "PRIMITIVE_GAP_CONFIRMED__PHYSICAL_TILE_PRIMITIVES_ABSENT",
                                    "PRIMITIVE_GAP_PARTIAL__ALL_PRIMITIVES_VISIBLE_NOT_IN_FUSED_ROUTE",
                                    "PRIMITIVE_GAP_PARTIAL__P1_LANEMAP_CROSSLANE_VISIBLE_NOT_IN_FUSED_ROUTE",
                                    "PRIMITIVE_GAP_PARTIAL__SOME_PHYSICAL_PRIMITIVES_ABSENT"})),
  GateSpec(name="attention_online_pv_lanemap", entry="extra.qk.decode_attention_online_pv_lanemap:build",
           out_dir="qk-decode-attention-online-pv-lanemap", snapshot=True,
           inputs=("bench/qk-decode-attention-online-pv-tile/latest.json",),
           pass_verdicts=frozenset({"ONLINE_PV_TILE_P3_LANEMAP_READY"})),
  GateSpec(name="asm_scheduler_inc0", entry="extra.qk.asm_scheduler_proofs:build_inc0",
           kind="gate", needs_gpu=True, env={"DEV": "AMD"}),
  GateSpec(name="asm_scheduler_inc1", entry="extra.qk.asm_scheduler_proofs:build_inc1",
           kind="gate", needs_gpu=True, env={"DEV": "AMD"}),
  GateSpec(name="asm_scheduler_inc2", entry="extra.qk.asm_scheduler_proofs:build_inc2",
           kind="gate", needs_gpu=True, env={"DEV": "AMD"}),
  GateSpec(name="asm_scheduler_inc3", entry="extra.qk.asm_scheduler_proofs:build_inc3",
           kind="gate", needs_gpu=True, env={"DEV": "AMD"}),
  GateSpec(name="tg_p9_live_split", entry="extra.qk.tg_p9_live_split:build_live_split",
           kind="microgate", needs_gpu=True, out_dir="tg-p9-pure-attention-primitive-route",
           artifact_name="live_split_microgate.json", env={"DEV": "AMD"},
           pass_verdicts=frozenset({"TG_P9_1_PASS_LIVE_TC_SPLIT_IR"})),
  GateSpec(name="tg_p9_live_split_tile", entry="extra.qk.tg_p9_live_split:build_live_split_tile",
           kind="microgate", needs_gpu=True, out_dir="tg-p9-pure-attention-primitive-route",
           artifact_name="live_split_tile_microgate.json", env={"DEV": "AMD"},
           pass_verdicts=frozenset({"TG_P9_2_PASS_LIVE_SPLIT_TILE"})),
  GateSpec(name="tg_p9_combine", entry="extra.qk.tg_p9_live_split:build_combine",
           kind="microgate", needs_gpu=True, out_dir="tg-p9-pure-attention-primitive-route",
           artifact_name="combine_microgate.json", env={"DEV": "AMD"},
           pass_verdicts=frozenset({"TG_P9_4_PASS_COMBINE_MICROGATE"})),
  GateSpec(name="physical_tile_p1_crosslane", entry="extra.qk.decode_physical_tile:build_p1_crosslane",
           kind="probe", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="p1_crosslane_latest.json", snapshot=True, env={"DEV": "AMD", "V_DOT2_LOWERING": "1"},
           pass_verdicts=frozenset({"P1_CROSSLANE_PASS__LANEMAP_CROSSLANE_VISIBLE",
                                    "P1_CROSSLANE_PASS__EXTRA_PRIMITIVES_PRESENT"})),
  GateSpec(name="physical_tile_pall_route", entry="extra.qk.decode_physical_tile:build_pall_route",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="route_pall_latest.json", snapshot=True, env={"DEV": "AMD", "V_DOT2_LOWERING": "1"},
           pass_verdicts=frozenset({"PALL_ROUTE_BUILDER_READY__ROUTE_NEXT",
                                    "PALL_ROUTE_BLOCKED__MISSING_COMPOSED_PRIMITIVES"})),
  GateSpec(name="physical_tile_pall_lifecycle", entry="extra.qk.decode_physical_tile:build_pall_lifecycle",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="pall_lifecycle_latest.json", snapshot=True, env={"DEV": "AMD", "V_DOT2_LOWERING": "1"},
           pass_verdicts=frozenset({"PALL_LIFECYCLE_BUILDER_READY__ROUTE_NEXT",
                                    "PALL_LIFECYCLE_BLOCKED__NUMERIC",
                                    "PALL_LIFECYCLE_BLOCKED__MISSING_PRIMITIVE_ISA"})),
  GateSpec(name="physical_tile_pall_scaling", entry="extra.qk.decode_physical_tile:build_pall_scaling",
           kind="probe", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="pall_lifecycle_scaling_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1"}),
  GateSpec(name="physical_tile_all_primitives", entry="extra.qk.decode_physical_tile:build_all_primitives",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="all_primitives_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1"},
           pass_verdicts=frozenset({"PALL_PRIMITIVES_VISIBLE__ROUTE_INTEGRATION_NEXT"})),
  GateSpec(name="score_broadcast_direct", entry="extra.qk.decode_score_broadcast:build_direct",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_broadcast_direct_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"},
           pass_verdicts=frozenset({"SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT"})),
  GateSpec(name="score_broadcast_chain", entry="extra.qk.decode_score_broadcast:build_chain",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_broadcast_chain_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"},
           pass_verdicts=frozenset({"SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT"})),
  GateSpec(name="score_broadcast_varjit_chain", entry="extra.qk.decode_score_broadcast:build_varjit_chain",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_broadcast_varjit_chain_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"},
           pass_verdicts=frozenset({"SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT"})),
  GateSpec(name="score_broadcast_control_matrix", entry="extra.qk.decode_score_broadcast:build_control_matrix",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_broadcast_control_matrix_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"},
           pass_verdicts=frozenset({"SCORE_BROADCAST_CONTROL_MATRIX_RECORDED"})),
  GateSpec(name="score_broadcast_model_cache_view", entry="extra.qk.decode_score_broadcast:build_model_cache_view",
           kind="gate", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_broadcast_model_cache_view_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"},
           pass_verdicts=frozenset({"SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT"})),
  GateSpec(name="score_reuse_paths", entry="extra.qk.decode_score_broadcast:build_reuse_paths",
           kind="probe", needs_gpu=True, out_dir="qk-decode-primitive-space",
           artifact_name="score_reuse_paths_latest.json", snapshot=True,
           env={"DEV": "AMD", "V_DOT2_LOWERING": "1", "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1"}),
  GateSpec(name="online_state_pv_p8", entry="extra.qk.decode_attention_online_state_pv:build_p8",
           kind="gate", needs_gpu=True, out_dir="qk-decode-attention-online-state-pv-p8-numeric", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"ONLINE_STATE_PV_P8_NUMERIC_PASS"})),
  GateSpec(name="online_state_pv_p9", entry="extra.qk.decode_attention_online_state_pv:build_p9",
           kind="gate", needs_gpu=True, out_dir="qk-decode-attention-online-state-pv-p9-scalar-numeric", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"ONLINE_STATE_PV_P9_NUMERIC_PASS"})),
  GateSpec(name="online_state_pv_p10", entry="extra.qk.decode_attention_online_state_pv:build_p10",
           kind="gate", needs_gpu=True, out_dir="qk-decode-attention-online-state-pv-p10-xlane-output", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS"})),
  GateSpec(name="online_state_pv_p11", entry="extra.qk.decode_attention_online_state_pv:build_p11",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-online-state-pv-p11-xlane-merge", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"ONLINE_STATE_PV_P11_MERGE_PASS"})),
  GateSpec(name="online_state_pv_p12", entry="extra.qk.decode_attention_online_state_pv:build_p12",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-online-state-pv-p12-xlane-components", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"ONLINE_STATE_PV_P12_COMPONENTS_PASS"})),
  GateSpec(name="xlane_reducer_matrix", entry="extra.qk.decode_attention_online_state_pv:build_p13",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-xlane-reducer-matrix", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"XLANE_REDUCER_MATRIX_PASS"})),
  GateSpec(name="xlane_recurrence_matrix", entry="extra.qk.decode_attention_online_state_pv:build_p14",
           kind="microgate", needs_gpu=True, out_dir="qk-decode-attention-xlane-recurrence-matrix", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"XLANE_RECURRENCE_MATRIX_PASS"})),
  GateSpec(name="split_xlane_output", entry="extra.qk.decode_attention_online_state_pv:build_p15",
           kind="gate", needs_gpu=True, out_dir="qk-decode-attention-split-xlane-output", snapshot=True,
           env={"DEV": "AMD"}, pass_verdicts=frozenset({"SPLIT_XLANE_OUTPUT_PASS"})),
  GateSpec(name="tg_p10_reg_scalar_repro", entry="extra.qk.decode_attention_online_state_pv:build_tg_p10",
           kind="gate", needs_gpu=True, env={"DEV": "AMD", "REG_STORE_DEVEC": "1"}),
)

BY_NAME = {g.name: g for g in GATES}


def run(name: str) -> int:
  spec = BY_NAME[name]
  for k, v in spec.env.items(): os.environ.setdefault(k, v)
  if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
  mod_name, fn_name = spec.entry.split(":")
  try:
    out = getattr(importlib.import_module(mod_name), fn_name)()
  except Exception:
    tb = traceback.format_exc()
    print(tb, file=sys.stderr)
    if spec.out_dir is not None:
      outdir = ROOT / "bench" / spec.out_dir
      outdir.mkdir(parents=True, exist_ok=True)
      (outdir / "harness_error.json").write_text(json.dumps(
        {"gate": name, "verdict": "HARNESS_ERROR", "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "traceback": tb},
        indent=2) + "\n")
    return 2
  if isinstance(out, int): return out
  if spec.out_dir is not None:
    outdir = ROOT / "bench" / spec.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(out, indent=2) + "\n"
    (outdir / spec.artifact_name).write_text(blob)
    if spec.snapshot: (outdir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.json").write_text(blob)
  print(json.dumps(out, indent=2))
  if spec.pass_verdicts is None: return 0
  return 0 if out.get("verdict") in spec.pass_verdicts else 1


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="gate_registry")
  sub = ap.add_subparsers(dest="cmd", required=True)
  lp = sub.add_parser("list")
  lp.add_argument("--kind")
  g = lp.add_mutually_exclusive_group()
  g.add_argument("--gpu", action="store_true")
  g.add_argument("--no-gpu", action="store_true")
  rp = sub.add_parser("run")
  rp.add_argument("names", nargs="*")
  rp.add_argument("--tranche", choices=["artifact-only"])
  args = ap.parse_args(argv)

  if args.cmd == "list":
    for s in GATES:
      if args.kind and s.kind != args.kind: continue
      if args.gpu and not s.needs_gpu: continue
      if args.no_gpu and s.needs_gpu: continue
      print(f"{s.name:40s} {s.kind:10s} {'gpu' if s.needs_gpu else 'artifact-only':13s} bench/{s.out_dir or '-'}")
    return 0

  names = args.names or []
  if args.tranche == "artifact-only": names += [s.name for s in GATES if not s.needs_gpu and s.name not in names]
  if not names: ap.error("run: give NAME(s) or --tranche")
  unknown = [n for n in names if n not in BY_NAME]
  if unknown: ap.error(f"unknown gate(s): {unknown}; see `list`")
  worst = 0
  for n in names:
    rc = run(n)
    print(f"[gate_registry] {n}: exit {rc}", file=sys.stderr)
    worst = max(worst, rc)
  return worst


if __name__ == "__main__":
  raise SystemExit(main())
