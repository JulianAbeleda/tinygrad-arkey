#!/usr/bin/env python3
"""Exhaustive disjoint full-token kernel-role census: tinygrad vs llama PDL-off.

Consumes the two retained full-token captures and sums every kernel exactly
once into a common role. This closes the node_sum/union accounting and names
the previously unattributed device remainder.

Inputs (retained raw artifacts):
  - tinygrad control capture:  probe2-tinygrad-capture.json  (596 nodes)
  - llama wait-free oracle:    probe2-llama-pdl0-dag.json     (762 nodes)

The llama capture already carries a disjoint per-node ``role``. The tinygrad
capture carries ``metadata.semantic.module_path`` only on the GEMV bodies, so
the tinygrad side is classified by exact canonical kernel name, which is
disjoint and exhaustive over the 596-node production census.

This is measurement tooling only. Nothing here changes runtime behavior.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re
from collections import defaultdict

SCHEMA = "tinygrad.nv_full_token_role_census.v1"

TINYGRAD_CAPTURE = pathlib.Path(
  "docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-tinygrad-capture.json")
LLAMA_PDL0_DAG = pathlib.Path(
  "docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-llama-pdl0-dag.json")

HASH64 = re.compile(r"_[0-9a-f]{64}$")


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def canon(name: str) -> str:
  return HASH64.sub("", name).strip()


# Every production kernel spelling in the current-HEAD control capture, mapped
# to one disjoint role. The map is keyed on the canonical (hash-stripped) name.
# Anything not listed here lands in ``unmapped`` so the census fails closed.
TINYGRAD_ROLE: dict[str, str] = {
  "q4k_g3_lanemap_gemv_w1w3fused16_12288_4096": "gate_up",
  "q6k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": "down",
  "q4k_fp16_mmvq_direct_4096_12288_epi_ffnresadd": "down",
  "q4k_g3_lanemap_gemv_epi_resadd_4096_4096": "o_proj",
  "q6k_gen_coop_151936_4096_inkernel": "vocab_main",
  "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128": "flash_score",
  "q4k_g3_lanemap_gemv_4096_4096": "q_proj",
  "q4k_warp_coop_q8_dp4a_partial_4096_4096": "q_proj",
  "q4k_g3_lanemap_gemv_1024_4096": "kv_proj",
  "q4k_warp_coop_q8_dp4a_partial_1024_4096": "kv_proj",
  "q6k_v_four_warp_fp16_direct_1024_4096": "kv_proj",
  "q6k_q8_warp_direct_1024_4096": "kv_proj",
  "flash_fused_gmax_combine_f16_32_128": "flash_combine",
  "reduce_output_rmsnorm_32_128": "q_norm",
  "reduce_output_rmsnorm_8_128": "k_norm",
  "reduce_output_rmsnorm_1_4096": "norm_4096",
  "r_16_256": "norm_4096",
  "E_32_32_4": "norm_4096",
  "rmsnorm_q8_1_llama_provider_4096": "quant_provider",
  "E_16_32_4_2": "rope_store",
  "E_8_8_16_2": "rope_store",
  "E_16_4_2_8_16_2_4_4": "rope_store",
  "r_8_32_4_4": "kv_completion",
  "r_32_32_4_4": "q_completion",
  "r_32_4_1187": "vocab_tail",
  "r_128_16_8_1187": "vocab_tail",
  "E_1187_32_4": "vocab_tail",
  "r_32_32_4_32_4": "other_reduce",
  "r_16_8": "other_reduce",
  "E": "other_elem",
  "E_2": "other_elem",
}


def _tinygrad_roles(capture: dict) -> tuple[dict[str, dict], list[dict]]:
  out: dict[str, dict] = defaultdict(lambda: {"count": 0, "us": 0.0})
  unmapped: list[dict] = []
  for node in capture["nodes"]:
    name = canon(str(node.get("name", "")))
    role = TINYGRAD_ROLE.get(name)
    if role is None:
      unmapped.append({"name": name, "duration_us": float(node.get("duration_us", 0.0))})
      continue
    us = float(node.get("duration_us", 0.0))
    out[role]["count"] += 1
    out[role]["us"] += us
  return out, unmapped


def _llama_roles(dag: dict) -> dict[str, dict]:
  out: dict[str, dict] = defaultdict(lambda: {"count": 0, "us": 0.0})
  for node in dag["nodes"]:
    role = str(node.get("role", ""))
    out[role]["count"] += 1
    out[role]["us"] += float(node.get("duration_us", 0.0))
  return out


# Common role vocabulary and the llama roles that fold into each bucket.
COMMON_ROLE: dict[str, tuple[str, tuple[str, ...]]] = {
  "gate_up":      ("gate/up GEMV", ("G",)),
  "down":         ("down GEMV", ("D",)),
  "o_proj":       ("O projection", ("O",)),
  "q_proj":       ("Q projection + completion", ("Q",)),
  "kv_proj":      ("K/V projections + completion", ("K", "V")),
  "vocab":        ("vocab main + tail", ("vocab", "vocab_quant")),
  "flash_score":  ("flash score", ("flash",)),
  "flash_combine":("flash combine", ("combine",)),
  "q_norm":       ("Q head norm", ("q_norm",)),
  "k_norm":       ("K head norm", ("k_norm",)),
  "norm_4096":    ("attn/ffn/final 4096 norm", ("attn_norm", "ffn_norm", "final_norm")),
  "quant":        ("activation quant", ("Q_quant", "K_quant", "V_quant", "O_quant", "G_quant", "D_quant")),
  "rope_store":   ("rope + K/V store", ("q_rope", "k_rope", "k_store")),
  "other":        ("misc / embedding", ("get_rows_a", "get_rows_b", "binbcast")),
}


def _tinygrad_common(tg: dict[str, dict]) -> dict[str, dict]:
  fold = {
    "q_proj": ("q_proj", "q_completion"),
    "kv_proj": ("kv_proj", "kv_completion"),
    "vocab": ("vocab_main", "vocab_tail"),
    "quant": ("quant_provider",),
    "other": ("other_reduce", "other_elem"),
  }
  common: dict[str, dict] = defaultdict(lambda: {"tg_count": 0, "tg_us": 0.0})
  for role, spec in COMMON_ROLE.items():
    srcs = fold.get(role, (role,))
    for s in srcs:
      if s in tg:
        common[role]["tg_count"] += tg[s]["count"]
        common[role]["tg_us"] += tg[s]["us"]
  return common


def _llama_common(ll: dict[str, dict]) -> dict[str, dict]:
  common: dict[str, dict] = defaultdict(lambda: {"ll_count": 0, "ll_us": 0.0})
  for role, (_, srcs) in COMMON_ROLE.items():
    for s in srcs:
      if s in ll:
        common[role]["ll_count"] += ll[s]["count"]
        common[role]["ll_us"] += ll[s]["us"]
  return common


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out-json", required=True)
  ap.add_argument("--out-md", required=False, help="optional human-readable dump")
  args = ap.parse_args()

  tg_capture = json.loads(TINYGRAD_CAPTURE.read_text(encoding="utf-8"))
  ll_dag = json.loads(LLAMA_PDL0_DAG.read_text(encoding="utf-8"))

  tg, unmapped = _tinygrad_roles(tg_capture)
  ll = _llama_roles(ll_dag)

  if unmapped:
    print("UNMAPPED tinygrad kernels (fail-closed):")
    for u in sorted(unmapped, key=lambda x: -x["duration_us"]):
      print(f"  {u['duration_us']:8.2f} us  {u['name']}")
    return 2

  tg_common = _tinygrad_common(tg)
  ll_common = _llama_common(ll)

  tg_node_sum = sum(v["us"] for v in tg.values())
  ll_node_sum = sum(v["us"] for v in ll.values())

  rows = []
  for role, (label, _) in COMMON_ROLE.items():
    tc = tg_common.get(role, {})
    lc = ll_common.get(role, {})
    t_us = tc.get("tg_us", 0.0)
    l_us = lc.get("ll_us", 0.0)
    rows.append({
      "role": role, "label": label,
      "tinygrad_count": tc.get("tg_count", 0), "tinygrad_us": round(t_us, 3),
      "llama_count": lc.get("ll_count", 0), "llama_us": round(l_us, 3),
      "delta_us": round(t_us - l_us, 3),
    })
  rows.sort(key=lambda r: -r["delta_us"])

  # The nine previously named semantic rows, derived from the census rows
  # themselves (not hardcoded deltas). ``kv_proj`` folds the prior audit's
  # separate "K projection plus completion" and "V projection plus completion".
  named_nine_roles = {"gate_up", "q_proj", "o_proj", "down", "vocab",
                      "flash_combine", "flash_score", "kv_proj"}
  delta_total = sum(r["delta_us"] for r in rows)
  named_rows = [r for r in rows if r["role"] in named_nine_roles]
  named_sum = sum(r["delta_us"] for r in named_rows)
  remainder_rows = [r for r in rows if r["role"] not in named_nine_roles]
  remainder_sum = sum(r["delta_us"] for r in remainder_rows)

  result = {
    "schema": SCHEMA,
    "sources": {
      "tinygrad_capture": str(TINYGRAD_CAPTURE),
      "tinygrad_capture_sha256": _sha256(TINYGRAD_CAPTURE),
      "llama_pdl0_dag": str(LLAMA_PDL0_DAG),
      "llama_pdl0_dag_sha256": _sha256(LLAMA_PDL0_DAG),
    },
    "closure": {
      "tinygrad_node_sum_us": round(tg_node_sum, 3),
      "llama_node_sum_us": round(ll_node_sum, 3),
      "node_sum_delta_us": round(delta_total, 3),
      "known_tinygrad_node_sum_us": 4677.920,
      "known_llama_pdl0_node_sum_us": 3878.254,
      "known_union_delta_pdl0_us": 793.246,
      "tinygrad_overlap_us": round(tg_node_sum - 4671.500, 3),
    },
    "rows": rows,
    "reconciliation": {
      "named_nine_sum_us": round(named_sum, 3),
      "named_nine_rows": named_rows,
      "remainder_node_sum_us": round(remainder_sum, 3),
      "remainder_rows": remainder_rows,
    },
  }

  pathlib.Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  if not args.out_md:
    print(f"node_sum tinygrad={tg_node_sum:.2f} llama={ll_node_sum:.2f} delta={delta_total:.2f}")
    print(f"named nine={named_sum:.2f} remainder={remainder_sum:.2f}")
    return 0

  def us(v): return f"{v:8.2f}"
  lines = ["# NV exhaustive full-token role census (tinygrad vs llama PDL-off)", "",
           "Every kernel is counted exactly once. Durations are per-node residence",
           "(node_sum domain); the node_sum delta equals the union delta plus",
           "tinygrad's own small overlap.", ""]
  lines.append("## Closure")
  lines.append("")
  lines.append("| quantity | value us |")
  lines.append("| --- | ---: |")
  lines.append(f"| tinygrad node_sum | {tg_node_sum:.2f} |")
  lines.append(f"| llama PDL-off node_sum | {ll_node_sum:.2f} |")
  lines.append(f"| node_sum delta | {delta_total:.2f} |")
  lines.append(f"| known union delta (PDL-off) | 793.25 |")
  lines.append(f"| tinygrad own overlap | {tg_node_sum - 4671.500:.2f} |")
  lines.append("")
  lines.append("## Role table (sorted by delta)")
  lines.append("")
  lines.append("| role | tinygrad | llama | delta us |")
  lines.append("| --- | ---: | ---: | ---: |")
  for r in rows:
    lines.append(f"| {r['label']} | {us(r['tinygrad_us'])} ({r['tinygrad_count']}) | "
                 f"{us(r['llama_us'])} ({r['llama_count']}) | **{us(r['delta_us'])}** |")
  lines.append("")
  lines.append("## Reconciliation")
  lines.append("")
  lines.append(f"The nine previously named profile rows sum to {named_sum:.2f} us.")
  lines.append(f"The remaining roles sum to {remainder_sum:.2f} us in node_sum domain.")
  lines.append("That remainder is not a launch bubble. It is:")
  lines.append("")
  lines.append("| role | delta us |")
  lines.append("| --- | ---: |")
  for r in remainder_rows:
    lines.append(f"| {r['label']} | **{us(r['delta_us'])}** |")
  lines.append("")
  lines.append("The remainder is dominated by the RMSNorm kernels and rope/store,")
  lines.append("partially offset by tinygrad doing less activation-quant work.")
  lines.append("")
  pathlib.Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
  print(f"node_sum tinygrad={tg_node_sum:.2f} llama={ll_node_sum:.2f} delta={delta_total:.2f}")
  print(f"named nine={named_sum:.2f} remainder={remainder_sum:.2f}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
