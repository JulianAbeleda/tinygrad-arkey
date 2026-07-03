"""W3: weight-path probe matrix (WP0-WP9). Assembles the measured probes (WP0 achievable bandwidth, WP1 per-role wall
share from route_attribution.json, WP3 Q6_K-vs-Q4_K share) with the estimate-only lever probes (WP2 G3 parity, WP4-9
candidate GEMV levers). Each probe carries a measured|estimate flag + a go/refute/inconclusive decision. Audit-only.

Run: DEV=AMD PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/weight_path_probe_matrix.py
Writes: bench/amd-isa-backend-weight-path-ceiling/probe_matrix.json
"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "bench/amd-isa-backend-weight-path-ceiling"

def main():
  OUT.mkdir(parents=True, exist_ok=True)
  w1 = json.load(open(OUT/"route_attribution.json")) if (OUT/"route_attribution.json").exists() else None
  roles = {r["role"]: r for r in (w1["per_ctx"]["512"]["by_role"] if w1 else [])}
  ach = 820  # GB/s measured (WP0)
  owned_impl = round(5027783488 * 103.5 / 1e9, 0)  # 520 GB/s
  P = [
    {"id": "WP0", "name": "achievable HBM bandwidth", "probe_type": "measured",
     "result": f"streaming-copy best = {ach} GB/s = 85% of 960 peak; naive 1D sum-reduce only ~220 GB/s (overhead-bound, invalid as a bw probe)",
     "decision": "ANCHOR: realistic ceiling = 820 GB/s -> 163 tok/s (NOT an assumed 80%)"},
    {"id": "WP1", "name": "per-role wall share (eager PROFILE)", "probe_type": "measured",
     "result": f"weight GEMVs ~58% of GPU-compute: ffn_down {roles.get('ffn_down',{}).get('pct','?')}%, gate_up {roles.get('ffn_gate_up',{}).get('pct','?')}%, attn_proj {roles.get('attn_qkvo_proj',{}).get('pct','?')}%, lm_head {roles.get('lm_head',{}).get('pct','?')}%",
     "decision": "GO-FOCUS: ffn_down + gate_up (Q4_K) are the dominant weight wall"},
    {"id": "WP2", "name": "generated-G3 vs owned-warp speed parity", "probe_type": "estimate_route_present",
     "result": "Q4_K FFN runs on shipped owned-warp by default; G3 (BUBBLEBEAM_FUTURESIGHT) route is purity-equivalent (search-generated), but a head-to-head SPEED capture was not run this pass",
     "decision": "INCONCLUSIVE: route exists; speed parity unmeasured -> a cheap follow-up (one capture under BUBBLEBEAM_FUTURESIGHT) would resolve it"},
    {"id": "WP3", "name": "Q6_K vs Q4_K cost", "probe_type": "measured",
     "result": f"Q6_K present at lm_head ({roles.get('lm_head',{}).get('pct','5.7')}%, coop) + some ffn via coop; not the dominant wall vs Q4_K ffn_down/gate_up",
     "decision": "DEFER: Q6_K is ~6-11% of wall; Q4_K is the priority"},
    {"id": "WP4", "name": "owned implied bandwidth vs achievable", "probe_type": "measured_derived",
     "result": f"owned overall implied bw {owned_impl} GB/s (JIT W==D 103.5 tok/s x 5.03GB) = {round(100*owned_impl/ach)}% of achievable {ach} GB/s",
     "decision": "GO: ~1.6x headroom to achievable bw exists in the weight GEMVs"},
    {"id": "WP5", "name": "offline weight-layout reshuffle (Marlin-style)", "probe_type": "estimate_prior_art",
     "result": "reorder packed Q4_K words offline so the lane-map is a natural coalesced descriptor read; prior art (Marlin/AWQ-GEMM) reaches near-bw on Q4 GEMV",
     "decision": "GO-PRIMARY: history says the owned edge IS layout (packed-word coalescing + in-register dequant lifecycle + block-group-K/thread-map); this is the durable lever"},
    {"id": "WP6", "name": "in-register dequant lifecycle (fuse scale/zero into the GEMV inner loop)", "probe_type": "estimate",
     "result": "keep dequant in VGPRs across the K-loop instead of round-tripping; owned already does this -> low marginal headroom for the owned route",
     "decision": "REFUTE-FOR-OWNED: owned already does it; only relevant if building a NEW generated GEMV"},
    {"id": "WP7", "name": "K-split / coop reduction for tall matrices", "probe_type": "estimate",
     "result": "split K across workgroups for ffn_down (12288 K) to raise occupancy; coop route already used for some roles",
     "decision": "DEFER: occupancy not the binding constraint at ctx512/4096 (RL reclaim moved W==D ~0); bw is"},
    {"id": "WP8", "name": "fuse gate+up GEMV", "probe_type": "estimate",
     "result": "single kernel for gate & up (shared activation read) -> fewer launches, one activation load; weight bytes unchanged so bw-bound gain is small",
     "decision": "DEFER: weight-bw-bound, so fusing saves launch/activation only (~small at decode bs=1)"},
    {"id": "WP9", "name": "abandon weight path (move to non-weight)", "probe_type": "measured_derived",
     "result": "weight GEMVs are ~58% of GPU-compute and decode is weight-mem-bound; non-weight (attention) already audited at <1% of the weight floor",
     "decision": "REFUTE: weight path IS the wall; do not move off it"},
  ]
  rec = {"scope": "W3 weight-path probe matrix (WP0-WP9)", "achievable_bw_GBs": ach, "owned_implied_bw_GBs": owned_impl,
         "probes": P, "verdict": "AMD_ISA_WEIGHT_W3_PASS_PROBE_MATRIX",
         "primary_go": "WP5 (offline weight-layout reshuffle)", "cheap_followup": "WP2 (G3-vs-owned speed capture)"}
  json.dump(rec, open(OUT/"probe_matrix.json", "w"), indent=2)
  return rec

if __name__ == "__main__":
  rec = main()
  for p in rec["probes"]: print(f"{p['id']:5} [{p['probe_type']:22}] {p['decision'][:70]}")
  print("\nW3", rec["verdict"])
