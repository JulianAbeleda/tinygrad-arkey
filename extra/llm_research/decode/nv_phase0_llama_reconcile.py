#!/usr/bin/env python3
"""Reconcile the current Phase 0 tinygrad census against the llama PDL-off oracle.

The retained llama capture (``probe2-llama-pdl0-dag.json``) carries a disjoint
per-node ``role``.  This tool maps every current tinygrad kernel name to the
same common role vocabulary, sums a single representative steady replay, and
reports the tinygrad-vs-llama delta per role.  Pure offline computation over
retained evidence; it touches no GPU.
"""
from __future__ import annotations

import argparse, json, pathlib, re, statistics, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from extra.llm_research.decode.nv_gateup_fourwarp_profile_closure import (
  canon, _replay_metrics, _complete_replays)
from extra.llm_research.decode.nv_full_token_role_census import (
  LLAMA_PDL0_DAG, _llama_roles, _llama_common, COMMON_ROLE)

SCHEMA = "tinygrad.nv_phase0_llama_reconcile.v1"

# Current production spelling -> disjoint role.  The fused Q/K norm+rope bodies
# fold norm AND rope into one kernel; llama counts norm and rope separately, so
# these two rows are reported with an explicit fused-body caveat.
TINYGRAD_ROLE_CURRENT: dict[str, str] = {
  "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": "gate_up",
  "q4k_g3_lanemap_gemv_w1w3vec16_12288_4096": "gate_up",
  "q4k_gate_up_four_warp_vec_fp16_12288_4096": "gate_up",
  "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": "down",
  "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": "down",
  "q4k_fp16_mmvq_direct_vec_4096_12288_epi_ffnresadd": "down",
  "q6k_fp16_packed_lanemap_u4_4096_12288_epi_ffnresadd": "down",
  "q4k_g3_lanemap_gemv_epi_resadd_4096_4096": "o_proj",
  "q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096": "o_proj",
  "q6k_gen_coop_151936_4096_inkernel": "vocab",
  "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128": "flash_score",
  "q4k_g3_lanemap_gemv_4096_4096": "q_proj",
  "q4k_g3_lanemap_gemv_vec_4096_4096": "q_proj",
  "q4k_warp_coop_q8_dp4a_partial_4096_4096": "q_proj",
  "q4k_warp_coop_q8_dp4a_direct_4096_4096": "q_proj",
  "q4k_g3_lanemap_gemv_1024_4096": "kv_proj",
  "q4k_g3_lanemap_gemv_vec_1024_4096": "kv_proj",
  "q4k_warp_coop_q8_dp4a_partial_1024_4096": "kv_proj",
  "q4k_warp_coop_q8_dp4a_direct_1024_4096": "kv_proj",
  "q6k_v_four_warp_fp16_direct_1024_4096": "kv_proj",
  "q4k_g3_lanemap_gemv_pair_vec_1024_4096": "kv_proj",
  "q4k_warp_coop_q8_dp4a_pair_direct_1024_4096": "kv_proj",
  "q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096": "kv_proj",
  "q6k_q8_warp_direct_1024_4096": "kv_proj",
  "flash_fused_gmax_combine_f16_32_128": "flash_combine",
  "reduce_output_rmsnorm_rope_32_128": "q_norm",   # fused norm+rope body
  "reduce_output_rmsnorm_rope_8_128": "k_norm",    # fused norm+rope body
  "reduce_output_rmsnorm_rope_kv_cache_8_128": "k_norm",  # fused norm+rope+cache sink
  "reduce_output_rmsnorm_1_4096": "norm_4096",
  "r_16_256": "norm_4096",
  "E_32_32_4": "norm_4096",
  "rmsnorm_q8_1_llama_provider_4096": "quant",
  "rmsnorm_native_1_4096": "norm_4096",
  "E_16_32_4_2": "rope_store",
  "E_8_8_16_2": "rope_store",
  "E_16_4_2_8_16_2_4_4": "rope_store",
  "r_8_32_4_4": "kv_completion",
  "r_32_32_4_4": "q_completion",
  "r_32_4_1187": "vocab",
  "r_128_16_8_1187": "vocab",
  "E_1187_32_4": "vocab",
  "r_32_32_4_32_4": "other",
  "r_16_8": "other",
  "native_finite_fp32_argmax_151936_t1024": "vocab",
  "E": "other",
  "E_2": "other",
}


def _median_replay(replays: list[list[dict]]) -> list[dict]:
  if not replays:
    raise SystemExit("no replays")
  sums = [sum(float(e["duration"]) for e in r) for r in replays]
  target = statistics.median(sums)
  best = min(replays, key=lambda r: abs(sum(float(e["duration"]) for e in r) - target))
  return best


def _tinygrad_roles_current(replay: list[dict]) -> tuple[dict, list[dict]]:
  out: dict[str, dict] = defaultdict(lambda: {"count": 0, "us": 0.0})
  unmapped: list[dict] = []
  for node in replay:
    name = canon(str(node.get("name", "")))
    role = TINYGRAD_ROLE_CURRENT.get(name)
    if role is None:
      unmapped.append({"name": name, "duration_us": float(node.get("duration", 0.0))})
      continue
    out[role]["count"] += 1
    out[role]["us"] += float(node.get("duration", 0.0))
  return out, unmapped


def _fold_tinygrad(tg: dict[str, dict]) -> dict[str, dict]:
  fold = {
    "q_proj": ("q_proj", "q_completion"),
    "kv_proj": ("kv_proj", "kv_completion"),
    "vocab": ("vocab",),
    "quant": ("quant",),
    "other": ("other",),
  }
  common: dict[str, dict] = defaultdict(lambda: {"tg_count": 0, "tg_us": 0.0})
  for role, spec in COMMON_ROLE.items():
    srcs = fold.get(role, (role,))
    for s in srcs:
      if s in tg:
        common[role]["tg_count"] += tg[s]["count"]
        common[role]["tg_us"] += tg[s]["us"]
  return common


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile-jsonl", required=True, type=pathlib.Path)
  ap.add_argument("--llama-dag", type=pathlib.Path, default=LLAMA_PDL0_DAG,
                  help="llama weighted DAG to compare (default: retained PDL-off oracle)")
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  args = ap.parse_args()

  lines = [json.loads(l) for l in args.profile_jsonl.read_text().splitlines() if l.strip()]
  replays = _complete_replays(lines)
  steady = replays[args.warmup:] if len(replays) > args.warmup else replays
  replay = _median_replay(steady)
  metrics = _replay_metrics(replay)

  tg, unmapped = _tinygrad_roles_current(replay)
  if unmapped:
    print("UNMAPPED tinygrad kernels (fail-closed):")
    for u in sorted(unmapped, key=lambda x: -x["duration_us"]):
      print(f"  {u['duration_us']:8.2f} us  {u['name']}")
    return 2

  ll = _llama_roles(json.loads(args.llama_dag.read_text()))
  tg_common = _fold_tinygrad(tg)
  ll_common = _llama_common(ll)

  tg_node_sum = sum(v["us"] for v in tg.values())
  ll_node_sum = sum(v["us"] for v in ll.values())

  rows = []
  for role, (label, _) in COMMON_ROLE.items():
    tc = tg_common.get(role, {})
    lc = ll_common.get(role, {})
    t_us, l_us = tc.get("tg_us", 0.0), lc.get("ll_us", 0.0)
    rows.append({
      "role": role, "label": label,
      "tinygrad_count": tc.get("tg_count", 0), "tinygrad_us": round(t_us, 3),
      "llama_count": lc.get("ll_count", 0), "llama_us": round(l_us, 3),
      "delta_us": round(t_us - l_us, 3),
    })
  rows.sort(key=lambda r: -r["delta_us"])

  result = {
    "schema": SCHEMA,
    "source_profile_jsonl": str(args.profile_jsonl),
    "llama_source": str(args.llama_dag),
    "representative_replay": {
      "node_count": metrics["node_count"], "node_sum_us": metrics["node_sum_us"],
      "union_us": metrics["union_us"], "overlap_us": metrics["overlap_us"],
      "span_us": metrics["span_us"],
    },
    "closure": {
      "tinygrad_node_sum_us": round(tg_node_sum, 3),
      "llama_node_sum_us": round(ll_node_sum, 3),
      "node_sum_delta_us": round(tg_node_sum - ll_node_sum, 3),
    },
    "rows": rows,
    "fused_body_caveat": ("reduce_output_rmsnorm_rope_{32,8}_128 fold RMSNorm AND RoPE into one "
                          "kernel; llama reports q_norm/k_norm and q_rope/k_rope separately, so the "
                          "q_norm/k_norm deltas are not like-for-like."),
  }
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

  print(f"tinygrad node_sum={tg_node_sum:.2f} llama={ll_node_sum:.2f} delta={tg_node_sum-ll_node_sum:.2f}")
  for r in rows:
    print(f"  {r['label']:<28} tg={r['tinygrad_us']:>8.2f} ({r['tinygrad_count']:>3})  "
          f"ll={r['llama_us']:>8.2f} ({r['llama_count']:>3})  delta={r['delta_us']:>+8.2f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
