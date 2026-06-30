"""TG0 — G3 provenance audit: split the promoted Q4_K G3 GEMV route into GENERATED vs TEMPLATED vs STILL-HUMAN, so the
'machine authors the route' track (TG1-TG7) targets exactly the human-designed surface and nothing else.

Builds on PMS-R5 (bench/qk-lanemap-template-audit/latest.json), which proved the EMISSION is a lossless template. TG0
adds the third bucket the select->author bridge needs: which design choices in the lane-map are still a HUMAN's, i.e.
what a topology grammar (TG1/TG2) must rediscover from {quant facts + shape + GPU} rather than have hardcoded.

  generated     : produced automatically by tinygrad codegen (UOp -> AMDGCN lowering, regalloc, schedule, waitcnt).
  templated     : emitted programmatically from a parameterized spec (the LaneMap); R5-proven lossless. Mechanical.
  quant_data    : fixed by the quant format (Q4_K block layout). Not a free choice -> TG3 makes it data-driven.
  gpu_data      : fixed by the target (wave width). Not a free choice -> TG5 makes it a target feature.
  still_human   : the actual DESIGN a person made = the lane-map TOPOLOGY (how K is decomposed across lanes/groups and
                  how the cross-lane reduction is structured). THIS is the bridge target for TG1/TG2.

Audit-only; no kernels, no GPU. Run: PYTHONPATH=. python3 extra/qk_g3_provenance_audit.py
Writes: bench/qk-g3-provenance-audit/{latest.json,summary.md}
"""
import json, pathlib
from extra.qk_artifact_cache import emit_artifact
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-g3-provenance-audit"

# component -> (bucket, evidence). Sourced from extra/qk_gemv_g2_lanemap.py + qk_gemv_g3_codegen_lowering.py + R5 audit.
COMPONENTS = {
  "uop_to_amdgcn_lowering":      ("generated",   "tinygrad renderer lowers the UOp program to AMDGCN; regalloc/schedule/waitcnt automatic."),
  "kernel_emission_from_lanemap":("templated",   "q4k_g3_lanemap_gemv_kernel emits the UOp program from Q4KGateUpLaneMap; R5 proved byte-identical (UOp .key) reconstruction from the template."),
  "rows_n":                      ("templated",   "= out_features; a per-role template parameter (gate_up 12288 / down 4096 / qo 4096)."),
  "k":                           ("templated",   "= in_features; a per-role template parameter (gate_up 4096 / down 12288 / qo 4096)."),
  "dequant_dot_body":            ("templated",   "_q4k_block_dot_packed_load reused verbatim; a callable the template invokes."),
  "output_store":                ("templated",   "out[row].store(total).sink(...); mechanical."),
  "qk_k_256":                    ("quant_data",  "Q4_K super-block element count (256). Fixed by the quant format."),
  "q4k_words_per_block_36":      ("quant_data",  "36 uint32 words/Q4_K block. Fixed by the quant layout."),
  "q4k_quant_word_base_4":       ("quant_data",  "scale/min words precede the 32 quant words. Quant layout."),
  "lane_extent_wave32":          ("gpu_data",    "wave width = 32 on gfx1100. Fixed by the target GPU."),
  # --- the human design surface ---
  "block_groups_4":              ("still_human", "K decomposed into 4 block-groups across the wave. A TILING CHOICE (constrained by lane_extent == block_groups * words_per_group)."),
  "words_per_group_8":           ("still_human", "8 packed words per group own the lane. A TILING CHOICE (G2.validate hardcodes ==8 for Q4_K gate/up)."),
  "axis_role_assignment":        ("still_human", "which axes are GLOBAL/LOCAL/REDUCE: row=GLOBAL, block_group+word_col=LOCAL, local_block+group_pair=REDUCE. A SCHEDULE-SHAPE CHOICE."),
  "cross_lane_reduction":        ("still_human", "the wave reduction that sums per-lane partials into out[row]. A REDUCTION-TOPOLOGY CHOICE."),
  "packed_word_lane_index":      ("still_human", "the formula mapping (block_group, word_col, local_block, group_pair) -> which uint32 word each lane loads (coalesced). The core LANE-OWNERSHIP design."),
}

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  by_bucket = {}
  for comp, (bucket, ev) in COMPONENTS.items():
    by_bucket.setdefault(bucket, []).append({"component": comp, "evidence": ev})
  still_human = [c["component"] for c in by_bucket.get("still_human", [])]
  # the free design DOF a topology grammar must span to rediscover G3 (quant_data + gpu_data are inputs, not DOF)
  topology_dof = {
    "k_to_lane_decomposition": "factor k_blocks into (block_groups x words_per_group x local_block x group_pair) s.t. block_groups*words_per_group == lane_extent",
    "axis_roles": "assign each factor to GLOBAL | LOCAL | REDUCE",
    "lane_ownership_index": "the coalesced packed-word index per lane (derivable from the decomposition + quant packing)",
    "reduction_pattern": "cross-lane (ds_bpermute wave reduce) vs partials+separate-reduce",
  }
  rec = {
    "verdict": "TG0_PASS_G3_PROVENANCE_PINNED",
    "headline": "G3 is GENERATED emission of a TEMPLATED spec, but the lane-map TOPOLOGY is STILL HUMAN. The select->author bridge (TG1/TG2) only needs to make that topology machine-discoverable; everything else is already mechanical or data.",
    "bucket_counts": {b: len(v) for b, v in by_bucket.items()},
    "by_bucket": by_bucket,
    "still_human_surface": still_human,
    "topology_dof_for_grammar": topology_dof,
    "bridge_milestone": "TG2 succeeds if a topology grammar over topology_dof, given only {quant facts (TG3) + shape + GPU features (TG5)}, REGENERATES the G2 LaneMap that G3 uses (lossless per R5) WITHOUT the block_groups=4/words_per_group=8/axis-roles being hardcoded. Not 'beat G3' -- 'rediscover G3'.",
    "implications": {
      "TG1": "LaneMapTemplate IR must expose the 4 topology_dof as free fields (R5's schema already parameterizes rows/k/lane_extent; extend it to block_groups/words_per_group/axis_roles/reduction).",
      "TG3": "qk_k/words_per_block/quant_word_base move from G2-hardcoded constants to a QuantSpec the template reads.",
      "TG5": "lane_extent (wave32) moves to a TargetSpec feature.",
      "TG2": "the candidate author enumerates legal topology_dof combos (constrained by quant+target), emits each via TG1, and the evaluator (PMS-R2) gates them; success = the G3 topology is in the enumerated+winning set, not pre-baked.",
    },
    "citations": ["bench/qk-lanemap-template-audit/latest.json (R5 lossless template)", "extra/qk_gemv_g2_lanemap.py (the human-designed lane map)", "extra/qk_gemv_g3_codegen_lowering.py (the emitter)"],
    "caveat": "audit-only / static. 'still_human' = design choices, not correctness claims. block_groups/words_per_group are constrained (product == lane_extent) but the specific 4x8 split + the axis-role assignment were a person's decision, which is exactly the surface TG2 must search.",
  }
  emit_artifact(OUT, rec, kind="derived_artifact", inputs={"audit": "g3_provenance"},
                code_paths=["extra/qk_g3_provenance_audit.py"])
  md = [f"# TG0 — G3 provenance audit\n\n**Verdict:** {rec['verdict']}\n\n{rec['headline']}\n",
        "## Provenance buckets\n| bucket | count | meaning |", "|---|---|---|",
        "| generated | %d | tinygrad codegen (automatic) |" % rec["bucket_counts"].get("generated",0),
        "| templated | %d | emitted from the LaneMap spec (R5 lossless) |" % rec["bucket_counts"].get("templated",0),
        "| quant_data | %d | fixed by the Q4_K format (TG3 makes data-driven) |" % rec["bucket_counts"].get("quant_data",0),
        "| gpu_data | %d | fixed by the target wave (TG5 makes a target feature) |" % rec["bucket_counts"].get("gpu_data",0),
        "| **still_human** | %d | **the lane-map topology — the bridge target** |" % rec["bucket_counts"].get("still_human",0),
        "\n## Still-human surface (what TG1/TG2 must make machine-authorable)\n" + "\n".join(f"- `{c}`" for c in still_human),
        "\n## Topology DOF a grammar must span\n" + "\n".join(f"- **{k}**: {v}" for k,v in topology_dof.items()),
        f"\n## Bridge milestone\n{rec['bridge_milestone']}\n",
        "## Caveat\n"+rec["caveat"]]
  (OUT/"summary.md").write_text("\n".join(md))
  return rec

if __name__ == "__main__":
  r = main()
  print(json.dumps({"verdict": r["verdict"], "bucket_counts": r["bucket_counts"], "still_human_surface": r["still_human_surface"]}, indent=2))
  print("\nTG0", r["verdict"])
