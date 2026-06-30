"""W4: weight-path search-space recommendation. Given the probe matrix (W3) + gap decomposition (W2), emit the ONE
recommended next implementation target as a concrete search space (axes + ranges + the gate it must pass), plus the
refuted/deferred levers with reasons. Audit-only -- this builds the SCOPE for the next phase, it does not implement it.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_weight_path_search_scope_builder.py
Writes: bench/amd-isa-backend-weight-path-ceiling/search_space_recommendation.json
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-isa-backend-weight-path-ceiling"

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  rec = {"scope": "W4 weight-path search-space recommendation",
    "selected_next_target": "offline_weight_layout_reshuffle_for_q4k_gemv",
    "one_line": "Reshuffle packed Q4_K weights offline into a layout whose lane-map is a naturally-coalesced descriptor read, so a SEARCH-generated GEMV can approach achievable bandwidth without hand-tuning the thread-map.",
    "why_this_one": ["decode is weight-memory-bound; Q4_K weight GEMVs are ~58% of GPU-compute (ffn_down ~24.8% + gate_up ~16.6% dominate)",
                     "shipped owned-warp reaches only ~63% of the 820 GB/s achievable bw (54% of peak) -> matching achievable is +58% @ctx512 / +73% @ctx4096 full-decode (>>10%)",
                     "history: the owned edge is packed-word coalescing + in-register dequant lifecycle + block-group-K/thread-map -> all LAYOUT/representation, exactly what an offline reshuffle controls",
                     "fits the north star: it lets the pure-machine search OWN the GEMV (the layout makes the good thread-map discoverable) rather than hand-writing it"],
    "search_axes": [
      {"axis": "packed_word_group", "options": ["32-weight super-block native", "64", "128 (Marlin tile)"], "note": "K-group that one wavefront iteration consumes"},
      {"axis": "lane_to_weight_map", "options": ["interleaved (Marlin)", "block-linear", "transpose-on-store"], "note": "how the 64 lanes map to packed nibbles -> coalescing"},
      {"axis": "scale_zero_placement", "options": ["co-located per super-block", "separate stream", "pre-scaled fp16 side-table"], "note": "dequant operand locality in the K-loop"},
      {"axis": "k_tiling", "options": ["single-wg full-K", "coop K-split (for 12288 down)"], "note": "occupancy is NOT binding now; only revisit if reshuffle exposes it"}],
    "gate_it_must_pass": ["token-match vs the owned oracle (correctness first, like Phase G/H)",
                          "W==D decode tok/s vs owned at ctx512 AND ctx4096 (must not regress either; target -> achievable ceiling)",
                          "default HIP/owned path unchanged; DEV=AMD:ISA opt-in; no autogen edits",
                          "offline reshuffle is a pre-pass artifact, not a per-decode cost"],
    "refuted_or_deferred": [
      {"lever": "generated-G3 codegen parity (WP2)", "status": "CHEAP FOLLOW-UP FIRST", "reason": "route exists, speed-vs-owned unmeasured; one BUBBLEBEAM_FUTURESIGHT capture resolves whether codegen alone (no layout) already matches owned -> do this before the layout project to avoid double work"},
      {"lever": "in-register dequant lifecycle (WP6)", "status": "REFUTED for owned", "reason": "owned already does it"},
      {"lever": "fuse gate+up (WP8)", "status": "DEFERRED", "reason": "weight-bw-bound; saves only launch/activation at bs=1"},
      {"lever": "K-split/coop (WP7)", "status": "DEFERRED", "reason": "occupancy not binding (LDS reclaim moved W==D ~0)"},
      {"lever": "Q6_K (WP3)", "status": "DEFERRED", "reason": "~6-11% of wall vs Q4_K dominance"},
      {"lever": "reduce/cross-lane, generic scheduler, attention tile", "status": "REFUTED", "reason": "prior audits: not the bottleneck / <1% of weight floor / ~2x off owned"}],
    "verdict": "AMD_ISA_WEIGHT_W4_PASS_SEARCH_SPACE_READY"}
  json.dump(rec, open(OUT/"search_space_recommendation.json", "w"), indent=2)
  return rec

if __name__ == "__main__":
  rec = main()
  print("selected:", rec["selected_next_target"]); print("cheap follow-up first:", rec["refuted_or_deferred"][0]["lever"])
  print("\nW4", rec["verdict"])
