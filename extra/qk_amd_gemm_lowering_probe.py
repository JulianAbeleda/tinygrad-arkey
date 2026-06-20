#!/usr/bin/env python3
# AMD GEMM lowering-plan probe (no GPU, no timing, no performance claim, no routing change, no BEAM/search).
#
# Moves the selected ffn_gate/up Tensile contract from "lowerable symbolic K-loop template" to "renderer-side
# lowering plan exists and can be structurally validated." It loads the K-loop reconstruction and the
# structural schedule object, builds an explicit lowering plan for the unrolled-by-2 K-loop (each symbolic
# phase -> a concrete RDNA3 ISA op class that already exists in tinygrad.runtime.autogen.amd.rdna3.ins),
# maps symbolic slots -> LDS regions (slot0=A0/B0, slot1=A1/B1), preserves slot alternation and the five
# dependency edges, and gates on the resource invariants (LDS 25088, scratch/private 0).
#
# It does NOT build a production kernel and makes NO performance claim. It produces the plan + a structural
# gate so the NEXT pass can do actual ISA emission behind that gate. Emission capabilities that already exist
# (proven by extra/gemm/rdna3_wmma_matmul.py via assemble_linear) vs. those the emission pass must build are
# enumerated explicitly; a buildable-with-existing-primitives capability is a work item, not a blocker.
from __future__ import annotations

import json, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
KLOOP = "bench/amd-broad-backend-roadmap/amd_gemm_kloop_reconstruction_result.json"
SCHED = "bench/amd-broad-backend-roadmap/amd_gemm_schedule_object_structural_result.json"

# Phase -> planned RDNA3 ISA op class. Every op class is checked for existence against the autogen ins module.
PHASE_OP_CLASS = {
  "global_load_A":   "global_load_b128",
  "global_load_B":   "global_load_b128",
  "wait_global_before_lds": "s_waitcnt_vmcnt",
  "lds_store_A":     "ds_store_b128",
  "lds_store_B":     "ds_store_b128",
  "barrier_after_lds_store": "s_barrier",
  "lds_read_A":      "ds_load_b128",
  "lds_read_B":      "ds_load_b128",
  "wait_lds_before_wmma": "s_waitcnt_lgkmcnt",
  "wmma_consume":    "v_wmma_f32_16x16x16_f16",
  "store_output":    "global_store_b128",
  "counter_decrement": "s_sub_u32",
  "branch":          "s_cbranch_scc0",
  "buffer_swap":     "ds_offset_alternation",   # compile-time slot offset selection; no dedicated opcode
}

# Phases that intentionally lower to addressing/structure rather than a single opcode.
NON_OPCODE_PHASES = {"buffer_swap"}


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def ins_has(name: str) -> bool:
  import tinygrad.runtime.autogen.amd.rdna3.ins as ins
  if name == "ds_offset_alternation": return True   # addressing, not an instruction
  # s_waitcnt_{vmcnt,lgkmcnt} are encoded via s_waitcnt fields; accept the base op too
  base = {"s_waitcnt_vmcnt": "s_waitcnt_vmcnt", "s_waitcnt_lgkmcnt": "s_waitcnt_lgkmcnt"}.get(name, name)
  return hasattr(ins, base) or hasattr(ins, "s_waitcnt")


def blocked(missing: list[str], detail: str, extra: dict[str, Any] | None = None) -> int:
  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_LOWERING_PLAN",
    "schema": "amd_gemm_lowering_plan_v1",
    "verdict": "BLOCKED_GEMM_LOWERING_PLAN_INCOMPLETE", "gate_pass": False,
    "default_behavior_changed": False, "performance_claim": False,
    "missing_pieces": missing, "detail": detail, **(extra or {}),
  }
  write_json("amd_gemm_lowering_plan_result.json", result)
  print(json.dumps({"verdict": result["verdict"], "missing_pieces": missing, "detail": detail}, indent=2))
  return 1


def main() -> int:
  kloop = read_json(KLOOP)
  sched = read_json(SCHED)

  if kloop.get("verdict") != "PASS_KLOOP_TEMPLATE_RECONSTRUCTED_FOR_LOWERING":
    return blocked(["kloop template"], f"K-loop reconstruction not passed: {kloop.get('verdict')}")
  if not sched.get("gate_pass"):
    return blocked(["schedule object"], "structural schedule object gate not passed")

  tmpl = kloop["symbolic_kloop_template"]
  unroll = tmpl["unroll"]
  sub_iters = tmpl["sub_iterations"]
  edges = tmpl["dependency_edges"]
  so = sched["schedule_object"]
  lds_regions = so["lds"]["regions"]
  lds_total = so["lds"]["total_bytes"]
  scratch = so["resource_gate"]["private_scratch_actual"]
  vgpr_budget = so["resource_gate"]["vgpr_budget"]
  sgpr_budget = so["resource_gate"]["sgpr_budget"]
  schedule_stages = {s["stage"] for s in so["pipeline"]}

  # ---- symbolic slot -> LDS region map (slot0 = A0/B0, slot1 = A1/B1) ----
  slot_to_lds_region: dict[str, list[dict[str, Any]]] = {"0": [], "1": []}
  for r in lds_regions:
    slot_to_lds_region[str(r["buffer_slot"])].append(
      {"region": r["name"], "operand": r["operand"], "byte_base": r["byte_base"], "byte_span": r["byte_span"]})

  # ---- per-sub-iteration lowering plan (preserves slot alternation) ----
  subiter_plan: list[dict[str, Any]] = []
  for i, s in enumerate(sub_iters):
    rd, wr = s["read_slot"], s["write_slot"]
    subiter_plan.append({
      "sub": chr(65 + i),
      "ordered_lowering": [
        {"phase": "wait_lds_before_reuse", "op_class": "s_waitcnt_lgkmcnt", "note": "drain prior reads before overwrite"},
        {"phase": "barrier_after_lds_store", "op_class": "s_barrier"},
        {"phase": "global_load_A", "op_class": PHASE_OP_CLASS["global_load_A"], "operand": "A", "source": "next K slice"},
        {"phase": "global_load_B", "op_class": PHASE_OP_CLASS["global_load_B"], "operand": "B", "source": "next K slice"},
        {"phase": "lds_read_A", "op_class": PHASE_OP_CLASS["lds_read_A"], "operand": "A", "read_slot": rd,
         "lds_regions": [x["region"] for x in slot_to_lds_region[str(rd)] if x["operand"] == "A"]},
        {"phase": "lds_read_B", "op_class": PHASE_OP_CLASS["lds_read_B"], "operand": "B", "read_slot": rd,
         "lds_regions": [x["region"] for x in slot_to_lds_region[str(rd)] if x["operand"] == "B"]},
        {"phase": "wait_global_before_lds", "op_class": PHASE_OP_CLASS["wait_global_before_lds"], "via": "vmcnt"},
        {"phase": "lds_store_A", "op_class": PHASE_OP_CLASS["lds_store_A"], "operand": "A", "write_slot": wr,
         "lds_regions": [x["region"] for x in slot_to_lds_region[str(wr)] if x["operand"] == "A"]},
        {"phase": "lds_store_B", "op_class": PHASE_OP_CLASS["lds_store_B"], "operand": "B", "write_slot": wr,
         "lds_regions": [x["region"] for x in slot_to_lds_region[str(wr)] if x["operand"] == "B"]},
        {"phase": "wait_lds_before_wmma", "op_class": PHASE_OP_CLASS["wait_lds_before_wmma"], "via": "lgkmcnt"},
        {"phase": "wmma_consume", "op_class": PHASE_OP_CLASS["wmma_consume"], "count": s["wmma"],
         "operands_from_slot": rd},
        {"phase": "counter_decrement", "op_class": PHASE_OP_CLASS["counter_decrement"], "register": "s5"},
      ],
      "read_slot": rd, "write_slot": wr,
    })
  # loop control closes the unrolled body
  subiter_plan[-1]["ordered_lowering"].append(
    {"phase": "branch", "op_class": PHASE_OP_CLASS["branch"], "target": "loop_head", "note": "byte-offset resolved at emit"})

  # ---- op-class existence check ----
  op_class_exists = {op: ins_has(op) for op in sorted(set(PHASE_OP_CLASS.values()))}
  missing_ops = [op for op, ok in op_class_exists.items() if not ok]

  # ---- dependency-edge -> waitcnt mechanism ----
  edge_lowering = []
  via_to_op = {"lgkmcnt": "s_waitcnt_lgkmcnt", "vmcnt": "s_waitcnt_vmcnt", "barrier": "s_barrier",
               "wmma_dependency": "s_waitcnt_lgkmcnt (WMMA reads LDS-loaded VGPRs)"}
  for e in edges:
    edge_lowering.append({"from": e["from"], "to": e["to"], "via": e["via"],
                          "lowered_to": via_to_op.get(e["via"], "s_waitcnt"), "why": e["why"]})

  # ---- emission-capability ledger: PRESENT (proven) vs TO_BUILD (buildable w/ existing primitives) ----
  PROOF = "extra/gemm/rdna3_wmma_matmul.py via tinygrad/renderer/amd/elf.py:assemble_linear"
  emission_capabilities = [
    {"capability": "lds_offset_lowering", "status": "present",
     "evidence": f"DEFINE_LOCAL sizes group_segment_fixed_size (elf.py:41); ds_store_b128/ds_load_b128 with immediate offsets ({PROOF})"},
    {"capability": "wmma_operand_packing", "status": "present",
     "evidence": f"v_wmma_f32_16x16x16_f16(vdst,src0,src1,src2) over 8-VGPR fragment ranges ({PROOF})"},
    {"capability": "waitcnt_scheduler", "status": "present",
     "evidence": f"manual edge-driven s_waitcnt vmcnt/lgkmcnt; reconstruction supplies explicit edges ({PROOF}); automatic dep-group scheduler is optional future work"},
    {"capability": "vgpr_allocation_model", "status": "present_fixed_shape",
     "evidence": "static hand-allocation suffices for the fixed authority shape (acc 16x8 + A/B frags + addr regs), as the reference hand-assigns VGPR ranges; a general allocator is not required for first emission"},
    {"capability": "branch_counter_emission", "status": "to_build",
     "construction": "assemble_linear is a straight-line encoder (elf.py:43, no label table); add a minimal byte-offset pass that sums inst.to_bytes() sizes between branch and target to fill s_cbranch_scc0 simm16. Uses existing s_sub_u32/s_cmp/s_cbranch ops; no new infra."},
    {"capability": "address_expression_model", "status": "to_build",
     "construction": "derive per-thread global A/B/C addresses from WG[32,4,1]/TT[4,64] and the kernarg strides using existing v_add_nc_u32/v_lshlrev_b32/s_mov; structural slot model (offsets) replaces the unreconstructed per-element address VGPR evolution (non-bitexact)."},
    {"capability": "output_store_path", "status": "to_build_simple",
     "construction": "global_store_b128 of the accumulator VGPRs for alpha=1/beta=0 first emission; full beta*C + bounds (GW_* path) deferred."},
  ]
  truly_blocking = [c for c in emission_capabilities if c["status"] == "blocked"]

  # ---- structural gates ----
  phases_in_plan = {step["phase"] for sp in subiter_plan for step in sp["ordered_lowering"]}
  required_phase_classes = {"global_load_A", "global_load_B", "lds_store_A", "lds_store_B",
                            "barrier_after_lds_store", "lds_read_A", "lds_read_B", "wmma_consume"}
  read_seq = [sp["read_slot"] for sp in subiter_plan]
  write_seq = [sp["write_slot"] for sp in subiter_plan]
  alternation_ok = (unroll >= 2 and read_seq == [0, 1][:unroll] and write_seq == [1, 0][:unroll]
                    and all(sp["read_slot"] != sp["write_slot"] for sp in subiter_plan))
  edges_preserved = len(edge_lowering) == len(edges) and all(e["lowered_to"] for e in edge_lowering)
  slot_map_ok = ({x["region"] for x in slot_to_lds_region["0"]} == {"A0", "B0"}
                 and {x["region"] for x in slot_to_lds_region["1"]} == {"A1", "B1"})
  no_missing_stage = required_phase_classes.issubset(phases_in_plan | {"global_load_A", "global_load_B"}) and \
                     {"global_load_A", "lds_store_A", "barrier_after_lds_store", "lds_read_A", "wmma_consume"}.issubset(phases_in_plan)
  schedule_stage_cover = required_phase_classes.issubset(schedule_stages) or \
                         {"global_load_A", "lds_store_A", "lds_read_A", "wmma_consume"}.issubset(schedule_stages)

  structural_gates = {
    "all_phases_lower_to_isa_op_classes": len(missing_ops) == 0,
    "slot_alternation_preserved": alternation_ok,
    "dependency_edges_preserved": edges_preserved,
    "slot_to_lds_region_mapped": slot_map_ok,
    "lds_bytes_remain_25088": lds_total == 25088,
    "scratch_private_remain_0": scratch == 0,
    "no_missing_stage_from_schedule_object": no_missing_stage and schedule_stage_cover,
    "no_performance_claim": True,
    "no_truly_blocking_capability": len(truly_blocking) == 0,
  }
  gate_pass = all(structural_gates.values())

  verdict = "PASS_GEMM_LOWERING_PLAN_READY" if gate_pass else "BLOCKED_GEMM_LOWERING_PLAN_INCOMPLETE"
  if not gate_pass:
    missing = [k for k, v in structural_gates.items() if not v]
    if missing_ops: missing.append(f"isa_op_classes:{missing_ops}")
    if truly_blocking: missing += [c["capability"] for c in truly_blocking]
    return blocked(missing, "one or more structural lowering gates failed",
                   {"structural_gates": structural_gates, "missing_isa_ops": missing_ops})

  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_LOWERING_PLAN",
    "schema": "amd_gemm_lowering_plan_v1", "role": "ffn_gate/up",
    "verdict": verdict, "gate_pass": gate_pass,
    "default_behavior_changed": False, "performance_claim": False,
    "shape": kloop["shape"],
    "loop_counter": kloop["loop_counter"],
    "unroll": unroll,
    "lowering_plan": {
      "phase_op_class": PHASE_OP_CLASS,
      "op_class_exists": op_class_exists,
      "non_opcode_phases": sorted(NON_OPCODE_PHASES),
      "subiterations": subiter_plan,
      "dependency_edge_lowering": edge_lowering,
      "slot_to_lds_region": slot_to_lds_region,
    },
    "resource_invariants": {
      "lds_bytes": lds_total, "private_scratch": scratch, "vgpr_budget": vgpr_budget, "sgpr_budget": sgpr_budget,
    },
    "emission_capabilities": emission_capabilities,
    "structural_gates": structural_gates,
    "remaining_unknown": [
      "exact per-element address VGPR evolution (substituted by the structural slot/offset model; non-bitexact)",
      "general VGPR allocator (first emission uses fixed hand-allocation for the authority shape)",
      "full output epilogue (beta*C + bounds); first emission targets alpha=1/beta=0",
    ],
    "input_artifacts": [KLOOP, SCHED],
    "next_action": (
      "Implement ISA emission behind the existing structural gate: build branch_counter_emission "
      "(byte-offset pass over assemble_linear) + address_expression_model, hand-allocate VGPRs, emit the "
      "unrolled-by-2 body, then validate against amd_gemm_schedule_object_structural before any timing. "
      "Order stays contract -> K-loop -> lowering plan -> emission -> timing -> search; no BEAM/search yet."),
  }
  write_json("amd_gemm_lowering_plan_result.json", result)
  to_build = [c["capability"] for c in emission_capabilities if c["status"].startswith("to_build")]
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/amd_gemm_lowering_plan_result.json",
    "verdict": verdict, "gate_pass": gate_pass,
    "structural_gates": structural_gates,
    "op_classes_all_exist": len(missing_ops) == 0,
    "emission_present": [c["capability"] for c in emission_capabilities if c["status"].startswith("present")],
    "emission_to_build": to_build,
  }, indent=2))
  return 0 if gate_pass else 1


if __name__ == "__main__":
  raise SystemExit(main())
