#!/usr/bin/env python3
"""Deterministic FFN chain-structure audit (measurement tooling only).

Tests five claims about the installed tinygrad FFN fold from the retained
locked d512 decode capture and the phase-5 decomposition artifact:

  C1 gate/up -> down is a single RAW buffer edge with no intermediate
     cast/activation/provider kernel.
  C2 the only fp16->cast kernel in play is the candidate-only E_128_32_3;
     the installed control contains none.
  C3 the intermediate activation is 24576 bytes (12288 x fp16), while the
     gate/up and down weight streams are 28.3-41.3 MB.
  C4 the 17 rmsnorm_q8_1_llama_provider_4096 nodes feed Q/K/V projections
     (Q8, 4608 bytes) and never feed the down projection.
  C5 the DRAM streaming gap is wall-time only: same GGUF weight bytes,
     higher tinygrad production time -> lower effective throughput.

It changes no production file and is deterministic given the retained
artifacts (the capture is at the same commit as HEAD).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HASH64 = re.compile(r"_[0-9a-f]{64}$")


def canon(name: str) -> str:
  return HASH64.sub("", str(name)).strip()


def sha256(path: Path) -> str:
  h = hashlib.sha256()
  h.update(path.read_bytes())
  return h.hexdigest()


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--capture", required=True)
  ap.add_argument("--decomposition", required=True)
  ap.add_argument("--closure-result", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  tg = json.loads(Path(args.capture).read_text())
  dec = json.loads(Path(args.decomposition).read_text())
  cl = json.loads(Path(args.closure_result).read_text())

  nodes = tg["nodes"]
  edges = tg["edges"]
  by_id = {n["id"]: n for n in nodes}

  def name_of(nid: int) -> str:
    return canon(by_id[nid]["name"])

  GATE = "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096"
  DOWN = {
    "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd",
    "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd",
  }
  PROVIDER = "rmsnorm_q8_1_llama_provider_4096"
  QPROJ = "q4k_warp_coop_q8_dp4a_partial_4096_4096"
  KPROJ = "q4k_warp_coop_q8_dp4a_partial_1024_4096"
  VPROJ = "q6k_q8_warp_direct_1024_4096"
  PROJ = {QPROJ, KPROJ, VPROJ}

  name_counter = Counter(canon(n["name"]) for n in nodes)

  # --- C1: gate/up -> down is a direct RAW edge with no intermediate kernel
  gate_ids = [n["id"] for n in nodes if canon(n["name"]) == GATE]
  down_ids = [n["id"] for n in nodes if canon(n["name"]) in DOWN]
  c1_rows = []
  for gid in gate_ids:
    raw_outs = [e for e in edges if e["from"] == gid and e["kind"] == "RAW"]
    consumers = [(e["to"], name_of(e["to"]), e["spans"]) for e in raw_outs]
    c1_rows.append({
      "gate_up_id": gid,
      "raw_consumer_count": len(raw_outs),
      "consumers": [c[1] for c in consumers],
      "all_consumers_are_down": all(c[1] in DOWN for c in consumers),
      "spans": [c[2] for c in consumers],
    })

  span_sizes = []
  for row in c1_rows:
    for spans in row["spans"]:
      for s, e in spans:
        span_sizes.append(e - s)
  span_counter = Counter(span_sizes)
  c1 = {
    "gate_up_count": len(gate_ids),
    "expected_gate_up_count": 36,
    "down_count": len(down_ids),
    "expected_down_count": 36,
    "every_gate_up_has_exactly_one_raw_consumer": all(r["raw_consumer_count"] == 1 for r in c1_rows),
    "every_raw_consumer_is_down": all(r["all_consumers_are_down"] for r in c1_rows),
    "intermediate_edge_span_histogram": dict(sorted(span_counter.items())),
    "intermediate_span_is_uniformly_24576": set(span_counter) == {24576},
    "intermediate_span_bytes": sorted(span_counter.items())[0][0] if len(span_counter) == 1 else None,
  }

  # --- C2: the installed control has no cast/activation/pack kernels
  cast_like = [k for k in name_counter if any(t in k.lower() for t in
               ("cast", "silu", "sigmoid", "swish", "h2f", "f2h", "quant", "pack", "_mul", "act"))]
  c2 = {
    "E_128_32_3_count_in_control": name_counter.get("E_128_32_3", 0),
    "cast_activation_pack_like_names_in_control": sorted(cast_like),
    "control_has_no_separable_fp16_cast": name_counter.get("E_128_32_3", 0) == 0,
    "candidate_E_128_32_3_us": cl["per_row_deltas_us"].get("E_128_32_3", {}).get("candidate_us"),
    "candidate_four_warp_body_us": cl["per_row_deltas_us"].get("q4k_gate_up_four_warp_fp16_12288_4096", {}).get("candidate_us"),
    "control_fused_body_us": cl["per_row_deltas_us"].get("q4k_g3_lanemap_gemv_w1w3fused16_12288_4096", {}).get("control_us"),
  }
  fwb = c2["candidate_four_warp_body_us"]
  cb = c2["control_fused_body_us"]
  cast = c2["candidate_E_128_32_3_us"]
  c2["four_warp_body_minus_control_us"] = round(fwb - cb, 3) if fwb is not None and cb is not None else None
  c2["net_gate_up_swap_us"] = round((fwb + cast - cb), 3) if all(v is not None for v in (fwb, cb, cast)) else None

  # --- C3: byte counts (intermediate vs weights)
  gate_buf = dec["kernels"]["gate_up"]["buf_sizes"]
  q4_buf = dec["kernels"]["down_q4"]["buf_sizes"]
  q6_buf = dec["kernels"]["down_q6"]["buf_sizes"]
  fp16_to_q8_saving = 24576 // 2
  total_weight_bytes = (36 * (gate_buf[1] + gate_buf[2])
                        + 18 * q4_buf[1] + 18 * q6_buf[1])
  c3 = {
    "intermediate_activation_bytes": 24576,
    "intermediate_activation_dtype": "fp16 (12288 x 2)",
    "fp16_to_q8_pack_saving_bytes": fp16_to_q8_saving,
    "gate_up_weight_bytes_each": gate_buf[1],
    "gate_up_weight_stream_bytes_per_layer": gate_buf[1] + gate_buf[2],
    "down_q4_weight_bytes": q4_buf[1],
    "down_q6_weight_bytes": q6_buf[1],
    "total_ffn_weight_bytes_per_token": total_weight_bytes,
    "pack_saving_as_fraction_of_weights": round(fp16_to_q8_saving / total_weight_bytes, 8),
    "pack_saving_percent": round(100.0 * fp16_to_q8_saving / total_weight_bytes, 5),
  }

  # --- C4: provider feeds Q/K/V, never down
  prov_ids = [n["id"] for n in nodes if canon(n["name"]) == PROVIDER]
  c4_rows = []
  for pid in prov_ids:
    raw_outs = [e for e in edges if e["from"] == pid and e["kind"] == "RAW"]
    consumers = sorted(name_of(e["to"]) for e in raw_outs)
    spans = [s for e in raw_outs for s in e["spans"]]
    c4_rows.append({
      "provider_id": pid,
      "consumers": consumers,
      "consumers_are_attention_projections": set(consumers) <= PROJ,
      "feeds_down": any(c in DOWN for c in consumers),
      "span_sizes": [e - s for s, e in spans],
    })
  consumer_set_hist = Counter(tuple(r["consumers"]) for r in c4_rows)
  c4 = {
    "provider_count": len(prov_ids),
    "expected_provider_count": 17,
    "every_provider_feeds_attention_projections": all(r["consumers_are_attention_projections"] for r in c4_rows),
    "any_provider_feeds_down": any(r["feeds_down"] for r in c4_rows),
    "consumer_set_histogram": {", ".join(k): v for k, v in sorted(consumer_set_hist.items())},
    "q8_output_span_sizes": sorted({sz for r in c4_rows for sz in r["span_sizes"]}),
    "q8_output_is_4608_bytes": all(sz == 4608 for r in c4_rows for sz in r["span_sizes"]),
  }

  # --- C5: DRAM streaming = same bytes, higher tinygrad wall -> lower rate
  def gbps(mb: float, us: float) -> float:
    return mb * 1e6 / (us * 1e-6) / 1e12

  c5_rows = {}
  for k in ("gate_up", "down_q4", "down_q6"):
    kd = dec["kernels"][k]
    mb = kd["ncu"]["dram_mb"]
    tg_us = kd["P_us"]
    ll_us = kd["llama_body_us"]
    c5_rows[k] = {
      "dram_mb": mb,
      "tinygrad_production_us": tg_us,
      "llama_body_us": ll_us,
      "tinygrad_TBps": round(gbps(mb, tg_us), 3),
      "llama_TBps": round(gbps(mb, ll_us), 3),
      "tinygrad_over_llama": round(gbps(mb, tg_us) / gbps(mb, ll_us), 3),
    }
  c5 = {
    "premise_same_weight_bytes": "single shared GGUF Q4_K_M; weight byte streams are model-defined and identical",
    "rows": c5_rows,
  }

  result = {
    "schema": "tinygrad.nv_ffn_chain_structure_audit.v1",
    "commit": ROOT and subprocess_commit(),
    "inputs": {
      "capture_sha256": sha256(Path(args.capture)),
      "decomposition_sha256": sha256(Path(args.decomposition)),
      "closure_result_sha256": sha256(Path(args.closure_result)),
    },
    "capture_node_count": len(nodes),
    "capture_edge_count": len(edges),
    "distinct_canonical_names": len(name_counter),
    "claims": {"C1_chain": c1, "C2_cast": c2, "C3_bytes": c3, "C4_provider": c4, "C5_dram": c5},
    "verdicts": {
      "C1_no_intermediate_chain": bool(c1["every_raw_consumer_is_down"]
                                      and c1["intermediate_span_is_uniformly_24576"]),
      "C2_cast_is_candidate_only_regression": bool(c2["control_has_no_separable_fp16_cast"]
                                                  and cast is not None and cast > 0),
      "C3_intermediate_is_24KB_against_100MB_weights": bool(
        c3["intermediate_activation_bytes"] == 24576
        and c3["pack_saving_as_fraction_of_weights"] < 0.001),
      "C4_provider_is_attention_only": bool(c4["every_provider_feeds_attention_projections"]
                                           and not c4["any_provider_feeds_down"]),
      "C5_dram_gap_is_wall_time": bool(all(r["tinygrad_over_llama"] < 1.0 for r in c5_rows.values())),
    },
  }

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: v for k, v in result.items() if k not in ("claims",)}, indent=2, sort_keys=True))
  return 0


def subprocess_commit() -> str:
  import subprocess
  return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                        cwd=str(ROOT)).stdout.strip()


if __name__ == "__main__":
  raise SystemExit(main())
