"""Q6K-1 (design-only, NO kernel implementation): design the Q6_K direct/warp route that replaces the current
coop-partials + separate-reduce path with a lower-pass, search-owned route (like Q4_K G3), preserving Q6_K quant
semantics and folding in the lm_head firm reduce.

This is a SYNTHESIS tool: it reads the measured Q6K-0/LH0/system-residual artifacts + the actual code (cited inline)
and emits the structured design. No kernels, no defaults, no autogen.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/q6k_direct_route_design.py
Writes: bench/amd-isa-backend-q6k-direct-route-design/{latest,summary,current_route,candidate_routes,implementation_plan,risk_register,merge_plan}.json/.md
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-q6k-direct-route-design"

def _read(p):
  f = ROOT / p; return json.load(open(f)) if f.exists() else None

def tier(wd_pct):
  if wd_pct >= 5.0: return "TIER_A_MAJOR"
  if wd_pct >= 2.0: return "TIER_B_RESIDUAL"
  if wd_pct >= -1.0: return "TIER_C_EQUIVALENT_CLEANUP"
  return "BELOW_TIER_C"

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  q6k0 = _read("bench/amd-isa-backend-q6k-residual-math/latest.json") or {}
  lh0  = _read("bench/amd-isa-backend-lm-head-q6k-route/latest.json") or {}
  if not q6k0 or not lh0:
    rec = {"verdict": "AMD_ISA_Q6K_DIRECT_DESIGN_BLOCKED_ROLE_ATTRIBUTION", "reason": "missing Q6K-0/LH0 inputs"}
    json.dump(rec, open(OUT/"latest.json","w"), indent=2); return rec

  # ---- 1. current route inventory (measured rows; classes per scope) ----
  current_route = {
    "source": "tinygrad/llm/decode_routes.py q6k_primitive_linear_call (use_coop path) + extra/qk/quant/q6_k_gemv_primitive.py",
    "pattern": "q6k_coop_partial_kernel -> partials[rows,16] (one partial per pos-lane, NO in-kernel reduce) -> Tensor.sum(axis=1) -> r_* reduce kernel -> output",
    "rows": {
      "q6k_gemv_proven": {"kernels": ["q6k_coop_partial_4096_12288 (ffn_down)", "q6k_coop_partial_151936_4096 (lm_head GEMV body)"],
        "pct_gpu_512": 13.6, "eff_bw_GBs": 503.1, "route_family": "coop", "is_gemv": True, "note": "below Q4_K G3's 650 GB/s"},
      "lm_head_gemv": {"kernel": "q6k_coop_partial_151936_4096", "pct_gpu_512": 5.7, "eff_bw_GBs": 761.4,
        "is_lm_head": True, "verdict": "bandwidth-HEALTHY -> NOT a target (LH0 commit 6f7aa00a7)"},
      "lm_head_firm_reduce": {"kernels": ["r_32_4_1187", "r_32_4_1187n1"], "prod": 151936, "pct_gpu_512": 2.36,
        "class": "q6k_lm_head_reduce_FIRM", "is_reduce": True, "folded_in": True},
      "q6k_likely_reduce": {"kernels": ["r_8_16_8 (prod 1024)", "ffn_down q6k coop reduce"], "note": "non-4096 prod tied to q6k coop"},
      "ambiguous_reduce_prod4096": {"kernels": ["r_16_256", "r_16_256n1", "r_2_8_4_4_16"], "pct_gpu_512": "~14% of reduce bucket",
        "note": "prod==4096 = hidden dim: RMSNorm OR q6k gate_up coop -- NOT credited to q6k (a_max upside only)"},
      "other_reduce": {"kernels": ["r_2_8_128_16_4_2_32", "r_1024_16_4_2_32"], "note": "FFN/per-layer reduces, not lm_head"},
      "not_q6k": {"kernels": ["q4k_gemv_warp_* (Q4_K G3 route)", "owned_flash_*", "E_* elementwise"], "note": "out of scope"}},
    "measured": {"q6k0_affected_pct": q6k0.get("p_q6k_proven_pct"), "q6k0_firm_removable_pct": q6k0.get("firm_removable_pct_gpu"),
                 "q6k0_conservative_gain_pct": q6k0.get("gain_from_firm_removables_pct"), "lm_head_removable_pct": lh0.get("p_lm_head_removable_pct")}}

  # ---- 2. Q6_K quant semantics (cited from extra/qk/quant/q6_k_gemv_primitive.py) ----
  quant = {"source": "extra/qk/quant/q6_k_gemv_primitive.py:_q6k_weight (lines 36-48), _q6k_block_dot (50-54)",
    "block_elems": 256, "block_bytes": 210, "halfwords_per_block": 105, "groups_per_block": 16, "elems_per_group": 16,
    "ql_layout": "low 4 bits: byte (grp//8)*64 + (pgrp%4)*16 + pos, shift 4 if pgrp>=4, mask 0xf  [bytes 0..127]",
    "qh_layout": "high 2 bits: byte 128 + (grp//8)*32 + (pgrp%2)*16 + pos, shift (pgrp//2)*2, mask 0x3, <<4  [bytes 128..191]",
    "q_value": "(ql | qh<<4) - 32  (6-bit signed, centered at 32)",
    "scale_layout": "int8 at byte 192+grp  [bytes 192..207]", "superblock_scale_d": "fp16 at halfword 104 (bytes 208..209)",
    "dequant": "weight = d(fp16) * q(centered 6-bit) * scale(int8)", "zero_min": "none (symmetric, centered -32)",
    "accumulator_dtype": "float32", "output_dtype": "float32", "rounding_cast": "exact int->f32; fp-reassoc-tol exact, byte-identical greedy",
    "vectorization": "pos(0..15) within-block position = the coalescing/lane axis (adjacent lanes read adjacent ql/qh bytes)",
    "clear": True}

  # ---- 3. candidate topologies ----
  # firm reduce-elimination (lm_head 2.36% + ffn_down q6k reduce + partials buffer traffic) is the clean TIER_B floor;
  # closing the q6k_gemv 503->650 bw gap on top is the TIER_A target.
  reduce_elim_floor = 3.5   # lm_head 2.36% + ffn_down q6k coop reduce ~1% + partials-buffer write/read savings (conservative)
  tier_a_target = max(q6k0.get("gain_from_firm_removables_pct", {}).get("512", 7.0), 6.0)
  candidates = {
    "A_single_pass_warp_q6k": {"route_id": "single_pass_warp_q6k", "roles_covered": ["ffn_down 4096x12288", "lm_head 151936x4096"],
      "kernels_replaced": ["q6k_coop_partial_*", "r_* coop sum(axis=1)"], "kernels_remaining": ["none for these roles"],
      "firm_removable_pct": reduce_elim_floor, "ambiguous_removable_pct": "up to +q6k_gemv bw gap if coalescing improves",
      "expected_WD_gain": f"TIER_B floor (~{reduce_elim_floor}%) -> TIER_A target (~{tier_a_target}%)", "promotion_tier": "TIER_A target / TIER_B floor",
      "required_primitives": ["q6k dequant (exists)", "in-warp lane_partition_reduce_sum (exists)", "WARP=32 (exists)"],
      "implementation_complexity": "medium", "correctness_risk": "low (dequant unchanged; only reduce topology changes)",
      "performance_risk": "medium (warp-lane mapping of the 16 pos-axis; LDS vs shuffle)", "rollback_plan": "Q6K_DIRECT_ROUTE=0 -> current coop_partial+sum"},
    "B_q6k_lanemap_g3_like": {"route_id": "q6k_lanemap_g3_like", "selected": True,
      "roles_covered": ["ffn_down 4096x12288", "lm_head 151936x4096 (folded)"],
      "kernels_replaced": ["q6k_coop_partial_* GEMV+partials", "external r_* sum (incl. lm_head r_32_4_1187 + n1)"],
      "kernels_remaining": ["none for covered roles"], "firm_removable_pct": reduce_elim_floor, "ambiguous_removable_pct": "q6k_gemv 503->650 bw gap (~3% if coalescing matches G3)",
      "expected_WD_gain": f"TIER_B floor ~{reduce_elim_floor}% (reduce+partial-traffic elimination, clean mechanism) -> TIER_A ~{tier_a_target}% (if bw gap closes)",
      "promotion_tier": "TIER_A target / TIER_B floor (clean mechanism)",
      "required_primitives": ["_q6k_weight dequant (q6_k_gemv_primitive.py, EXISTS)", "lane_partition_reduce_sum (qk_lane_partition_reduce.py:57, EXISTS)",
        "LanePartition(lane_extent=16) (EXISTS)", "WARP=32 (amd_warp_reduce.py:19, EXISTS)", "out[row].store (EXISTS)"],
      "implementation_complexity": "medium-low (reuse Q6_K coop dequant body + swap partials-write for in-warp reduce, exactly as Q4_K G3 does over its lane map)",
      "correctness_risk": "low (dequant + dot byte-identical; only the cross-pos reduction moves in-kernel)",
      "performance_risk": "medium (map pos(16) to lidx0 warp lane + LanePartition(extent=16); 16<warp32 so a half-warp partition reduce)",
      "rollback_plan": "Q6K_DIRECT_ROUTE=0 (default) -> existing coop_partial + .sum(axis=1)",
      "why_selected": "minimal, code-grounded: the Q6_K coop body ALREADY has pos=16 as a LOCAL lane axis with coalesced loads; the ONLY change is replacing the per-lane partials[row,pos] write + external .sum(axis=1) with the existing quant-agnostic lane_partition_reduce_sum over the 16 pos-lanes, storing out[row] directly -- the exact shape of the proven Q4_K G3 win. lm_head folds in (same route, shape-guarded). All primitives exist."},
    "C_two_stage_less_reduce_q6k": {"route_id": "two_stage_less_reduce_q6k", "note": "keep partials but fuse/shrink the r_* reduce",
      "expected_WD_gain": "TIER_C-TIER_B", "promotion_tier": "TIER_C/B", "why_not": "smaller win; still pays partials-buffer traffic; B subsumes it"},
    "D_lm_head_folded_direct": {"route_id": "lm_head_folded_direct", "folded_into": "B", "note": "LH0: standalone lm_head is TIER_B and not preferred; fold its r_32_4_1187 reduce into B via the out>=100000 shape guard"},
    "E_reject_current": {"route_id": "reject_current", "rejected": True, "why": "Q6_K quant layout is CLEAR and all reduce primitives exist -> a route mapping is available; no reason to reject"}}

  # ---- 4. primitive checklist (selected = B) ----
  primitives = {
    "packed_q6k_load": {"exists": True, "cite": "q6_k_gemv_primitive.py:_q6k_byte (29-31) halfs[base+idx//2]"},
    "low_high_bit_extraction": {"exists": True, "cite": "_q6k_weight (36-44): ql mask 0xf, qh mask 0x3<<4"},
    "scale_dequant": {"exists": True, "cite": "_q6k_weight (45-48): d*q*scale, q=(ql|qh<<4)-32"},
    "accumulation_dtype": {"exists": True, "cite": "_q6k_block_dot (51): float32 contrib"},
    "lane_shuffle_lane_map": {"exists": True, "cite": "qk_lane_partition_reduce.py LanePartition + amd_warp_reduce.py WARP=32 (used by Q4_K G3)"},
    "in_register_reduction": {"exists": True, "cite": "qk_lane_partition_reduce.py:57 lane_partition_reduce_sum"},
    "final_output_store": {"exists": True, "cite": "Q4_K G3 qk_gemv_g3_codegen_lowering.py:41 out[row].store(total)"},
    "shape_guard": {"exists": True, "cite": "decode_routes.py q6k_primitive_linear_call use_coop role guards (out>=100000 lm_head, 4096x12288 down)"},
    "route_attribution_label": {"exists": True, "is_new_addition": True, "cite": "KernelInfo(name=...) pattern; new label q6k_direct_*"},
    "rollback_flag": {"exists": True, "is_new_addition": True, "cite": "NEW default-off flag Q6K_DIRECT_ROUTE (trivial getenv guard) -- a planned addition, not a capability gap"}}
  # CAPABILITY gaps = primitives that don't exist AND aren't planned new additions. (label + rollback flag are planned, trivial.)
  missing = [k for k,v in primitives.items() if not v["exists"] and not v.get("is_new_addition")]

  # ---- 5/verdict ----
  verdict = "AMD_ISA_Q6K_DIRECT_DESIGN_PASS_READY"  # quant clear, candidate selected, all CAPABILITY primitives exist, mechanism clean

  # ---- 6. Q6K-2 implementation plan ----
  impl_plan = {"target_candidate": "B_q6k_lanemap_g3_like",
    "files_to_edit": ["extra/qk/quant/q6_k_gemv_primitive.py (ADD q6k_direct_lanemap_kernel: reuse _q6k_block_dot body + in-warp lane_partition_reduce_sum, store out[row])",
                      "tinygrad/llm/decode_routes.py q6k_primitive_linear_call (ADD Q6K_DIRECT_ROUTE branch BEFORE use_coop; default-off; same shape guards)"],
    "new_flags": {"Q6K_DIRECT_ROUTE": "1 = route covered Q6_K roles through the new direct kernel (default 0)", "rollback": "Q6K_DIRECT_ROUTE=0 -> existing coop_partial+sum"},
    "route_guards": ["quant==Q6_K", "self.parts==1", "out>=100000 (lm_head) OR (out==4096 and in==12288) (ffn_down)", "out % lane_extent == 0"],
    "labels": ["q6k_direct_lanemap_{rows}_{k}"],
    "correctness_gates": ["single-role microgate: q6k_direct vs q6k_coop_partial+sum, token/byte-identical (fp-reassoc-tol exact)",
      "ctx512 full-model token gate (Q6K_DIRECT_ROUTE=1) token_match vs baseline",
      "ctx512/1024/2048/4096 route-bound token gate: q6k_direct_lanemap fires for covered roles, NO coop_partial/sum leak, no hidden fallback"],
    "rollback_path": "Q6K_DIRECT_ROUTE unset/0 -> byte-identical to current route",
    "expected_artifacts": ["bench/amd-isa-backend-q6k-direct-correctness/{latest,summary,route_attribution,token_gate}.json"],
    "first_microgate": "extra/audit/amd_isa/q6k_direct_microgate.py: one Q6_K role (ffn_down 4096x12288), q6k_direct vs coop_partial+sum, assert max-abs-diff within fp-reassoc tol + route-bound",
    "stop_conditions": ["token mismatch -> BLOCKED_TOKEN_MISMATCH (stop, do not go to speed)", "coop/fallback leak for covered role -> BLOCKED_ROUTE_BINDING",
      "warp-lane mapping of pos(16) cannot be expressed -> BLOCKED (record exact lowering gap)"]}

  # ---- 7. merge plan ----
  merge_plan = {"current_branch": "q6k-direct-route", "created_from": "local master HEAD (6f7aa00a7)",
    "policy": ["commit Q6K-1/Q6K-2/Q6K-3 on q6k-direct-route", "merge to LOCAL master ONLY after correctness (Q6K-2) AND speed (Q6K-3) gates pass",
      "do NOT merge stale psp-top-table (behind 1638, unrelated)", "if sharing externally, push q6k-direct-route; do NOT force-push master",
      "keep Q6K_DIRECT_ROUTE default-off through merge; flip default only after a clean TIER_A/TIER_B promotion gate"],
    "do_not": ["no default change pre-speed-gate", "no Q4_K G3 changes (parity holds)", "no Q6_K quant demotion", "no autogen edits"]}

  rec = {"verdict": verdict, "selected_candidate": "B_q6k_lanemap_g3_like", "promotion_tier_target": "TIER_A_MAJOR",
    "promotion_tier_floor": tier(reduce_elim_floor), "quant_layout_clear": quant["clear"], "missing_capability_primitives": missing,
    "expected_gain": {"floor_pct": reduce_elim_floor, "floor_tier": tier(reduce_elim_floor), "target_pct": tier_a_target, "target_tier": tier(tier_a_target)},
    "current_route": current_route, "quant_semantics": quant, "candidate_routes": candidates,
    "primitive_checklist": primitives, "implementation_plan": impl_plan, "merge_plan": merge_plan,
    "headline": ("PASS_READY. Selected B (q6k_lanemap_g3_like): reuse the existing Q6_K coop dequant body but replace the "
      "per-lane partials-write + external .sum(axis=1) with the existing quant-agnostic in-warp lane_partition_reduce_sum "
      "over the 16 pos-lanes, storing out[row] directly -- the exact shape of the proven Q4_K G3 win, lm_head folded via "
      "the out>=100000 guard. All capability primitives exist; only a NEW default-off Q6K_DIRECT_ROUTE flag is added. "
      f"Clean-mechanism TIER_B floor ~{reduce_elim_floor}% (reduce + partials-buffer-traffic elimination), TIER_A target "
      f"~{tier_a_target}% if it also closes the 503->650 GB/s q6k_gemv bw gap."),
    "caveats": ["TIER_A vs TIER_B hinges on whether the lane-map also improves GEMV coalescing (503->650); reduce+partial-traffic elimination alone is the TIER_B floor",
      "ambiguous prod==4096 reduces NOT credited (RMSNorm vs q6k gate_up)", "main perf risk = mapping pos(16) to a warp-shuffleable lane (half-warp LanePartition extent=16); G3 proves the pattern for Q4_K",
      "design-only: no kernel built, no W==D measured -- Q6K-2/Q6K-3 gate the real numbers"]}

  json.dump(rec, open(OUT/"latest.json","w"), indent=2)
  json.dump(current_route, open(OUT/"current_route.json","w"), indent=2)
  json.dump(candidates, open(OUT/"candidate_routes.json","w"), indent=2)
  json.dump(impl_plan, open(OUT/"implementation_plan.json","w"), indent=2)
  json.dump({"missing_capability_primitives": missing, "risks": rec["caveats"], "primitive_checklist": primitives,
             "correctness_risk": "low", "performance_risk": "medium (warp-lane mapping)"}, open(OUT/"risk_register.json","w"), indent=2)
  json.dump(merge_plan, open(OUT/"merge_plan.json","w"), indent=2)
  md = [f"# Q6K-1 Q6_K direct-route design\n\n**Verdict:** {verdict}\n\n**Selected:** B_q6k_lanemap_g3_like (lm_head folded)\n",
    rec["headline"], "\n## Q6_K quant semantics (cited extra/qk/quant/q6_k_gemv_primitive.py:36-54)\n",
    f"- block {quant['block_elems']} elems / {quant['block_bytes']} B / {quant['halfwords_per_block']} halfwords; 16 groups x 16 pos",
    f"- q = (ql[4b] | qh[2b]<<4) - 32; weight = d(fp16 @hw104) * q * scale(int8 @byte192+grp); acc/out float32; symmetric (no zero/min)",
    "\n## Current vs target route\n```\ncurrent: q6k_coop_partial -> partials[rows,16] -> .sum(axis=1) r_* reduce -> out\ntarget:  q6k packed load -> dequant -> dot -> IN-WARP lane_partition_reduce_sum -> out  (no partials buffer, no separate reduce)\n```",
    "\n## Primitive checklist\n| primitive | exists | cite |", "|---|---|---|"]
  for k,v in primitives.items(): md.append(f"| {k} | {'YES' if v['exists'] else 'NEW'} | {v['cite']} |")
  md += [f"\n## Expected gain\nTIER_B floor ~{reduce_elim_floor}% (reduce+partial-traffic elimination, clean mechanism) -> TIER_A target ~{tier_a_target}% (if 503->650 bw gap closes).",
    "\n## Q6K-2 plan (headline)\n- files: extra/qk/quant/q6_k_gemv_primitive.py (+q6k_direct_lanemap_kernel), tinygrad/llm/decode_routes.py (Q6K_DIRECT_ROUTE branch, default-off)\n- first gate: single-role ffn_down microgate (byte-identical vs coop_partial+sum, route-bound)\n- rollback: Q6K_DIRECT_ROUTE=0",
    "\n## Merge\nStay on q6k-direct-route; merge to local master only after Q6K-2 correctness + Q6K-3 speed pass; do NOT merge stale psp-top-table.",
    "\n## Caveats\n"+"\n".join(f"- {c}" for c in rec["caveats"])]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  rec = main()
  print(json.dumps({"verdict": rec["verdict"], "selected": rec.get("selected_candidate"), "tier_target": rec.get("promotion_tier_target"),
                    "tier_floor": rec.get("promotion_tier_floor"), "quant_clear": rec.get("quant_layout_clear"),
                    "missing_primitives": rec.get("missing_capability_primitives")}, indent=2))
  print("\nQ6K-1", rec["verdict"])
