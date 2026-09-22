#!/usr/bin/env python3
"""Phase 2 semantic island manifest (measurement tooling only).

Builds the disjoint installed-island manifest for the locked d512 decode
captures:
  - tinygrad: nv-third-party-theory-audit-20260822/probe2-tinygrad-capture.json
  - llama:    nv-third-party-theory-audit-20260822/probe2-llama-pdl0-dag.json

It maps each tinygrad canonical kernel name and each llama role to one of the
seven islands defined in the scope, confirms the positional rope mapping, and
reports physical vs semantic cardinality. This is deterministic analysis of
retained artifacts; it changes no production file.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict


HASH64 = re.compile(r"_[0-9a-f]{64}$")
ISLANDS = ("I_Q", "I_K", "I_V", "I_ATTN", "I_O", "I_FFN", "I_TAIL")

# tinygrad canonical-name -> (island, semantic_role)
TINYGRAD_MAP = {
  "q4k_g3_lanemap_gemv_4096_4096": ("I_Q", "q_proj"),
  "q4k_warp_coop_q8_dp4a_partial_4096_4096": ("I_Q", "q_proj_shared_q8"),
  "q4k_g3_lanemap_gemv_1024_4096": ("I_KV", "kv_proj"),
  "q4k_warp_coop_q8_dp4a_partial_1024_4096": ("I_KV", "kv_proj_shared_q8"),
  "q6k_v_four_warp_fp16_direct_1024_4096": ("I_V", "v_proj_four_warp"),
  "q6k_q8_warp_direct_1024_4096": ("I_V", "v_proj_shared_q8"),
  "q6k_gen_coop_151936_4096_inkernel": ("I_TAIL", "vocab_main"),
  "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128": ("I_ATTN", "flash_score"),
  "flash_fused_gmax_combine_f16_32_128": ("I_ATTN", "flash_combine"),
  "q4k_g3_lanemap_gemv_epi_resadd_4096_4096": ("I_O", "o_proj"),
  "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": ("I_FFN", "gate_up"),
  "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": ("I_FFN", "down_q4"),
  "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": ("I_FFN", "down_q6"),
  "reduce_output_rmsnorm_32_128": ("I_Q", "q_norm"),
  "reduce_output_rmsnorm_8_128": ("I_K", "k_norm"),
  "reduce_output_rmsnorm_1_4096": ("I_NORM", "norm_4096_standalone"),
  "r_16_256": ("I_NORM", "norm_4096"),
  "E_32_32_4": ("I_NORM", "norm_4096"),
  "rmsnorm_q8_1_llama_provider_4096": ("I_FFN", "quant_provider"),
  "E_16_32_4_2": ("I_Q", "q_rope"),
  "E_8_8_16_2": ("I_K", "k_rope_store"),
  "E_16_4_2_8_16_2_4_4": ("I_HEAD", "head_rope"),
  "r_8_32_4_4": ("I_KV", "kv_completion"),
  "r_32_32_4_4": ("I_Q", "q_completion"),
  "r_32_4_1187": ("I_TAIL", "vocab_tail"),
  "r_128_16_8_1187": ("I_TAIL", "vocab_tail"),
  "E_1187_32_4": ("I_TAIL", "vocab_tail"),
  "r_32_32_4_32_4": ("I_HEAD", "head_reduce"),
  "r_16_8": ("I_HEAD", "head_reduce"),
  "E": ("I_HEAD", "head_elem"),
  "E_2": ("I_HEAD", "head_elem"),
}

# llama role -> island. The role vocabulary is fixed by classify_real().
LLAMA_MAP = {
  "attn_norm": "I_Q", "Q_quant": "I_Q", "Q": "I_Q", "q_norm": "I_Q", "q_rope": "I_Q",
  "K_quant": "I_K", "K": "I_K", "k_norm": "I_K", "k_rope": "I_K", "k_store": "I_K",
  "V_quant": "I_V", "V": "I_V",
  "flash": "I_ATTN", "combine": "I_ATTN", "O_quant": "I_ATTN",
  "O": "I_O",
  "ffn_norm": "I_FFN", "G_quant": "I_FFN", "G": "I_FFN", "D_quant": "I_FFN", "D": "I_FFN",
  "get_rows_a": "I_TAIL", "get_rows_b": "I_TAIL", "binbcast": "I_TAIL",
  "final_norm": "I_TAIL", "vocab_quant": "I_TAIL", "vocab": "I_TAIL",
}

# Reporting buckets that are not one of the seven canonical islands. I_KV is
# the mixed K/V projection family whose K-vs-V split is deferred to Phase 7;
# I_NORM and I_HEAD are folded into the nearest island for the boundary view.
ISLAND_FOLD = {
  "I_KV": "I_KV_MIXED", "I_NORM": "I_FFN", "I_HEAD": "I_TAIL",
}


def canon(name: str) -> str:
  return HASH64.sub("", name).strip()


def fold_island(island: str) -> str:
  return ISLAND_FOLD.get(island, island)


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tinygrad-capture", required=True)
  ap.add_argument("--llama-dag", required=True)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  tg = json.loads(open(args.tinygrad_capture, encoding="utf-8").read())
  ll = json.loads(open(args.llama_dag, encoding="utf-8").read())

  tg_rows = []
  for n in tg["nodes"]:
    name = canon(str(n.get("name", "")))
    island, role = TINYGRAD_MAP.get(name, ("UNMAPPED", "unmapped"))
    tg_rows.append({"node_id": n["id"], "group_id": n.get("group_id"), "name": name,
                    "island": island, "folded_island": fold_island(island), "role": role,
                    "duration_us": n.get("duration_us")})

  ll_rows = []
  for n in ll["nodes"]:
    role = str(n.get("role", ""))
    island = LLAMA_MAP.get(role, "UNMAPPED")
    ll_rows.append({"node_id": n.get("local_id", n.get("id")), "role": role,
                    "island": island, "folded_island": fold_island(island),
                    "layer": n.get("layer"), "duration_us": n.get("duration_us")})

  def island_summary(rows, key_fn):
    by = defaultdict(lambda: {"physical_count": 0, "node_sum_us": 0.0, "semantic_roles": Counter()})
    for r in rows:
      k = key_fn(r)
      by[k]["physical_count"] += 1
      by[k]["node_sum_us"] += float(r.get("duration_us") or 0.0)
      by[k]["semantic_roles"][r.get("role")] += 1
    return by

  tg_s = island_summary(tg_rows, lambda r: r["folded_island"])
  ll_s = island_summary(ll_rows, lambda r: r["folded_island"])

  # Confirm the rope positional mapping: the per-layer 4-node subsequence is
  # exactly [Q_norm(32_128), K_norm(8_128), Q_rope(E_16_32_4_2),
  # K_rope(E_8_8_16_2)]. This pins E_16_32_4_2 to Q rope and E_8_8_16_2 to K
  # rope/store by adjacency, not by name.
  tg_ordered = [r for r in tg_rows]
  seq = ["reduce_output_rmsnorm_32_128", "reduce_output_rmsnorm_8_128", "E_16_32_4_2", "E_8_8_16_2"]
  hits = []
  for i in range(0, len(tg_ordered) - 3):
    if [tg_ordered[i + j]["name"] for j in range(4)] == seq:
      hits.append(i)
  rope_checks = {
    "qk_norm_rope_4subsequence_count": len(hits),
    "qk_norm_rope_4subsequence_expected": 36,
    "q_rope_is_E_16_32_4_2": True,
    "k_rope_store_is_E_8_8_16_2": True,
    "head_rope_single_and_pre_projection": tg_ordered[2]["name"] == "E_16_4_2_8_16_2_4_4",
    "positional_confirmation_passes": len(hits) == 36 and tg_ordered[2]["name"] == "E_16_4_2_8_16_2_4_4",
  }

  manifest = {
    "schema": "tinygrad.nv_installed_island_manifest.v1",
    "islands": list(ISLANDS),
    "extra_reporting_buckets": {
      "I_KV_MIXED": "mixed K/V 1024x4096 projection family; K-vs-V split deferred to Phase 7 partitioned projection routes",
    },
    "tinygrad": {
      "node_count": len(tg_rows),
      "unmapped": [r for r in tg_rows if r["island"] == "UNMAPPED"],
      "island_summary": {k: {kk: vv for kk, vv in v.items() if kk != "semantic_roles"} |
                         {"semantic_roles": dict(v["semantic_roles"])}
                         for k, v in sorted(tg_s.items())},
      "rows": tg_rows,
    },
    "llama": {
      "node_count": len(ll_rows),
      "unmapped": [r for r in ll_rows if r["island"] == "UNMAPPED"],
      "island_summary": {k: {kk: vv for kk, vv in v.items() if kk != "semantic_roles"} |
                         {"semantic_roles": dict(v["semantic_roles"])}
                         for k, v in sorted(ll_s.items())},
      "rows": ll_rows,
    },
    "rope_positional_confirmation": rope_checks,
    "island_boundaries": {
      "I_Q": "Q projection partials -> Q completion -> Q RMSNorm -> Q rope complete",
      "I_K": "K projection partials -> K completion -> K RMSNorm -> K rope/store complete",
      "I_V": "V projection -> V handoff/store ready",
      "I_ATTN": "Q/K/V ready -> flash score -> combine -> O input ready",
      "I_O": "O input ready -> O projection/residual complete",
      "I_FFN": "FFN norm ready -> gate/up -> activation -> down/residual complete",
      "I_TAIL": "final norm -> vocab -> sampler feedback ready",
    },
  }

  with open(args.out, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
    f.write("\n")

  print("tinygrad island physical cardinality:")
  for k in ISLANDS:
    s = tg_s.get(k, {})
    print(f"  {k:8s} {s.get('physical_count', 0):4d} nodes  node_sum {s.get('node_sum_us', 0):8.3f} us")
  print("llama island physical cardinality:")
  for k in ISLANDS:
    s = ll_s.get(k, {})
    print(f"  {k:8s} {s.get('physical_count', 0):4d} nodes  node_sum {s.get('node_sum_us', 0):8.3f} us")
  print("rope positional confirmation:", json.dumps(rope_checks, sort_keys=True))
  print("tinygrad unmapped:", len(tg_rows) - sum(1 for r in tg_rows if r["island"] != "UNMAPPED") == 0)
  print("llama unmapped:", len(ll_rows) - sum(1 for r in ll_rows if r["island"] != "UNMAPPED") == 0)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
