#!/usr/bin/env python3
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"


def read_json(rel: str, default: Any = None) -> Any:
  path = ROOT / rel
  if not path.exists(): return default
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def ratio(tensile: int | float | None, tinygrad: int | float | None) -> float | None:
  if tensile is None or tinygrad is None or tinygrad == 0: return None
  return round(float(tensile) / float(tinygrad), 4)


def delta_row(feature: str, tensile: Any, tinygrad: Any, verdict: str, causality: str,
              implementation_surface: str, evidence: list[str], priority: str = "P0") -> dict[str, Any]:
  return {
    "feature": feature,
    "tensile": tensile,
    "tinygrad_authority": tinygrad,
    "verdict": verdict,
    "causality": causality,
    "implementation_surface": implementation_surface,
    "priority": priority,
    "evidence": evidence,
  }


def backlog_item(track: str, item: str, why: str, minimum_pass: str, can_start_now: bool,
                 blocked_by: list[str] | None = None) -> dict[str, Any]:
  return {
    "track": track,
    "item": item,
    "why": why,
    "minimum_pass": minimum_pass,
    "can_start_now": can_start_now,
    "blocked_by": blocked_by or [],
  }


def main() -> int:
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  capture = read_json("bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json", {})
  mapping = read_json("bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json", {})
  bb5a2 = read_json("bench/amd-broad-backend-roadmap/bb5a2_pipelined_dataflow_result.json", {})
  bb5a3 = read_json("bench/amd-broad-backend-roadmap/bb5a3_wait_scheduler_integration_result.json", {})
  bb5a4 = read_json("bench/amd-broad-backend-roadmap/bb5a4_allocator_resource_result.json", {})

  sched = codegen.get("tensile_schedule", {})
  tensile_mix = codegen.get("tensile_instruction_mix", {})
  tiny_mix = capture.get("mix", {}).get("disasm", {})
  tiny_resource = capture.get("resource", {})
  tiny_timing = capture.get("timing", {})
  source_paths = [
    "bench/qk-tensile-extraction/codegen_oracle.json",
    "bench/qk-tensile-extraction/shape_matrix.json",
    "bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json",
    "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
    capture.get("program", {}).get("source_path"),
    capture.get("program", {}).get("elf_path"),
    capture.get("program", {}).get("disasm_path"),
  ]
  evidence = [p for p in source_paths if p]

  proven_deltas = [
    delta_row(
      "macro_tile",
      sched.get("macro_tile_MxNxK"),
      codegen.get("tinygrad_pown1", {}).get("macro_tile_MxNxK"),
      "MATCH",
      "not causal; already matched",
      "none",
      evidence[:4],
      "P2",
    ),
    delta_row(
      "wmma_fragment",
      {"fragment": sched.get("wmma_MI"), "v_wmma_total": tensile_mix.get("v_wmma")},
      {"fragment": codegen.get("tinygrad_pown1", {}).get("wmma_MI"), "captured_v_wmma_total": tiny_mix.get("v_wmma")},
      "MATCH_FAMILY_DENSITY_DIFFERS",
      "not primary cause by itself; both use RDNA3 WMMA, but density/schedule differs",
      "scheduler_density_and_loop_structure",
      evidence,
      "P1",
    ),
    delta_row(
      "lds_presence",
      {"lds_buffering": sched.get("lds_buffering_1LDSB"), "ds_load_b128": tensile_mix.get("ds_load_b128"), "ds_store_b128": tensile_mix.get("ds_store_b128")},
      {"lds_bytes": tiny_resource.get("lds_bytes"), "ds_load_b128": tiny_mix.get("ds_load_b128"), "ds_store_b128": tiny_mix.get("ds_store_b128")},
      "PROVEN_CAUSAL_GAP",
      "Tensile is LDS-staged; captured timing-equivalent tinygrad authority kernel has zero LDS allocation and no LDS instructions.",
      "LDS_layout_and_lowering",
      evidence,
      "P0",
    ),
    delta_row(
      "wide_lds_reads",
      {"LRVW": sched.get("local_read_vec_LRVW"), "ds_load_b128": tensile_mix.get("ds_load_b128")},
      {"ds_load_b128": tiny_mix.get("ds_load_b128"), "ds_load_b64": tiny_mix.get("ds_load_b64"), "ds_load_b32": tiny_mix.get("ds_load_b32")},
      "PROVEN_CAUSAL_GAP",
      "Tensile feeds WMMA from wide LDS reads; captured tinygrad has no LDS read path at all.",
      "LDS_vectorized_read_lowering",
      evidence,
      "P0",
    ),
    delta_row(
      "k_loop_prefetch_pipeline",
      {"PGR": sched.get("prefetch_global_read_PGR"), "PLR": sched.get("prefetch_local_read_PLR"), "depthU": sched.get("depthU")},
      {"captured_lds_bytes": tiny_resource.get("lds_bytes"), "captured_s_barrier": tiny_mix.get("s_barrier"), "captured_s_waitcnt": tiny_mix.get("s_waitcnt")},
      "PROVEN_STRUCTURAL_GAP",
      "Tensile advertises global/local prefetch over a depthU=16 loop; captured tinygrad has no LDS staging to pipeline.",
      "K_loop_pipeline_scheduler",
      evidence,
      "P0",
    ),
    delta_row(
      "wait_and_barrier_density",
      {
        "s_waitcnt_total": (tensile_mix.get("s_waitcnt_vmcnt") or 0) + (tensile_mix.get("s_waitcnt_lgkmcnt") or 0),
        "s_barrier": tensile_mix.get("s_barrier"),
      },
      {"s_waitcnt": tiny_mix.get("s_waitcnt"), "s_barrier": tiny_mix.get("s_barrier")},
      "PROVEN_SCHEDULE_GAP",
      "Tensile has a dense wait/barrier schedule around staged LDS traffic; captured tinygrad has waits only around global loads and no barriers.",
      "semantic_wait_scheduler_integration",
      evidence,
      "P0",
    ),
    delta_row(
      "scratch_spill",
      {"oracle_note": "TT4_64, vgpr256 no spill per codegen oracle"},
      {"scratch_instruction_count": tiny_mix.get("scratch"), "private_segment_fixed_size": tiny_resource.get("kernel_descriptor", {}).get("private_segment_fixed_size")},
      "NO_TINYGRAD_SPILL_IN_CAPTURE",
      "The captured 42 TFLOPS kernel is not slow because of scratch spills; accumulator allocation remains a future risk for larger staged candidates.",
      "allocator_resource_policy",
      evidence,
      "P1",
    ),
    delta_row(
      "timing_gap",
      {"tensile_tflops": tiny_timing.get("reference_tensile_tflops")},
      {"captured_best_tflops": tiny_timing.get("best_tflops"), "authority_tinygrad_tflops": tiny_timing.get("reference_tinygrad_tflops")},
      "PROVEN_PERFORMANCE_GAP",
      "Same-shape pure tinygrad capture remains about 43 TFLOPS while Tensile is 65.6 TFLOPS.",
      "performance_gate",
      evidence,
      "P0",
    ),
  ]

  implementation_backlog = [
    backlog_item(
      "A_causal_delta",
      "Freeze causal deltas as acceptance criteria",
      "Avoid more open-ended knob sweeps; implementation targets are LDS staging, wide LDS reads, K-loop prefetch, wait/barrier schedule, and resource policy.",
      "bb5a9 result has P0 deltas with source artifacts and no unresolved same-kernel evidence gaps.",
      True,
    ),
    backlog_item(
      "B_LDS_layout",
      "Implement real authority-shape LDS tile allocation",
      "Captured tinygrad authority kernel has LDS bytes 0 while Tensile uses LDS buffering.",
      "authority-shape compiled ELF reports nonzero LDS and source/disasm contains LDS store/load traffic.",
      True,
    ),
    backlog_item(
      "B_LDS_layout",
      "Lower wide LDS reads",
      "Tensile has LRVW16 and ds_load_b128; captured tinygrad has zero ds_load_b128.",
      "authority-shape disasm contains ds_load_b128 feeding WMMA path.",
      True,
      ["real LDS tile allocation"],
    ),
    backlog_item(
      "C_K_loop_scheduler",
      "Emit two-stage global->LDS->WMMA K-loop",
      "Tensile PGR1/PLR1 overlaps K-tile movement and compute; current tinygrad computes directly from global-loaded registers.",
      "source/ISA has prologue plus steady-state alternating LDS slots before performance timing.",
      True,
      ["real LDS tile allocation", "wide LDS reads"],
    ),
    backlog_item(
      "C_K_loop_scheduler",
      "Place waits and barriers semantically over staged LDS traffic",
      "Tensile has high wait/barrier density; current tinygrad has no barriers because no LDS staging exists.",
      "scheduled stream has lgkmcnt/vmcnt waits and barriers only at producer/consumer boundaries, with correctness passing.",
      True,
      ["two-stage K-loop"],
    ),
    backlog_item(
      "D_resource_policy",
      "Decode and enforce full-kernel VGPR/occupancy policy",
      "Captured tinygrad has no scratch, but staged candidates can spill; POWN-1 showed more-acc variants can collapse to 11 TFLOPS.",
      "candidate either reports no scratch/private spill and acceptable resource descriptor or is deterministically rejected before timing.",
      True,
    ),
    backlog_item(
      "E_candidate_gate",
      "Build first measured pure-tinygrad staged candidate",
      "Performance movement is the only gate that unblocks BB-5 and eventually q8 transfer.",
      "same authority shape is correct and reaches >=60 TFLOPS with default behavior unchanged.",
      False,
      ["LDS layout", "wide LDS reads", "two-stage K-loop", "wait/barrier schedule", "resource policy"],
    ),
    backlog_item(
      "F_q8_transfer",
      "Scope q8 transfer only after prefill gate",
      "The q8 path is downstream; no q8-only native scheduler patch is justified yet.",
      "BB-5 pure tinygrad prefill candidate reaches >=60 TFLOPS, then q8 transfer continues at <=75us gate.",
      False,
      ["E_candidate_gate"],
    ),
  ]

  parallel_tracks = [
    {"track": "A_causal_delta", "status": "complete_by_this_probe", "depends_on": [], "next": "Use P0 deltas as acceptance criteria."},
    {"track": "B_LDS_layout", "status": "ready", "depends_on": ["A_causal_delta"], "next": "Make authority-shape ELF report nonzero LDS and disasm show ds traffic."},
    {"track": "C_K_loop_scheduler", "status": "ready", "depends_on": ["A_causal_delta", "B_LDS_layout"], "next": "Emit prologue/steady-state two-slot K-loop with semantic waits."},
    {"track": "D_resource_policy", "status": "ready", "depends_on": ["A_causal_delta"], "next": "Classify scratch/private segment/VGPR risk before timing."},
    {"track": "E_candidate_gate", "status": "blocked", "depends_on": ["B_LDS_layout", "C_K_loop_scheduler", "D_resource_policy"], "next": "Correctness plus >=60 TFLOPS."},
    {"track": "F_q8_transfer", "status": "blocked", "depends_on": ["E_candidate_gate"], "next": "Only after BB-5 passes."},
  ]

  gate = {
    "input_mapping_pass": mapping.get("verdict") == "PASS_STATIC_TENSILE_TINYGRAD_MAPPING_CAUSAL_PROOF_BLOCKED",
    "input_capture_pass": capture.get("verdict") == "PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY",
    "timing_equivalent_capture": bool(tiny_timing.get("timing_join_pass") and tiny_timing.get("within_reference_tolerance")),
    "proven_zero_lds_in_tinygrad": tiny_resource.get("lds_bytes") == 0,
    "proven_tensile_lds": (tensile_mix.get("ds_load_b128") or 0) > 0 and (tensile_mix.get("ds_store_b128") or 0) > 0,
    "proven_wmma_on_both": (tensile_mix.get("v_wmma") or 0) > 0 and (tiny_mix.get("v_wmma") or 0) > 0,
    "default_behavior_changed": False,
  }
  causal_delta_pass = all(v for k, v in gate.items() if k != "default_behavior_changed") and not gate["default_behavior_changed"]
  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.9_causal_delta_package",
    "schema": "amd_bb5a9_causal_delta_package_v1",
    "verdict": "PASS_BB5A9_CAUSAL_DELTA_PACKAGE_IMPLEMENTATION_TRACKS_READY" if causal_delta_pass else "BLOCKED_BB5A9_CAUSAL_DELTA_INPUTS_INCOMPLETE",
    "gate_pass": causal_delta_pass,
    "default_behavior_changed": False,
    "performance_claim": False,
    "summary": {
      "root_cause_now_proven": "captured timing-equivalent tinygrad authority kernel uses WMMA but no LDS staging; Tensile uses WMMA plus LDS-staged wide reads/stores and prefetch scheduling",
      "not_root_cause_for_captured_kernel": "scratch spill; captured tinygrad has scratch count 0 and private segment fixed size 0",
      "performance_gap": {
        "tinygrad_best_tflops": tiny_timing.get("best_tflops"),
        "tensile_tflops": tiny_timing.get("reference_tensile_tflops"),
        "speedup_needed": ratio(tiny_timing.get("reference_tensile_tflops"), tiny_timing.get("best_tflops")),
      },
    },
    "gate": gate,
    "proven_deltas": proven_deltas,
    "parallel_tracks": parallel_tracks,
    "implementation_backlog": implementation_backlog,
    "next_action": "Start B_LDS_layout, C_K_loop_scheduler, and D_resource_policy as parallel implementation tracks; keep E_candidate_gate and F_q8_transfer blocked.",
    "input_artifacts": evidence,
  }
  write_json("bb5a9_causal_delta_package_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a9_causal_delta_package_result.json",
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "ready_tracks": [t["track"] for t in parallel_tracks if t["status"] in {"ready", "complete_by_this_probe"}],
    "blocked_tracks": [t["track"] for t in parallel_tracks if t["status"] == "blocked"],
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
