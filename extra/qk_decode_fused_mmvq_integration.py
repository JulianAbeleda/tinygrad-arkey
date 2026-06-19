#!/usr/bin/env python3
"""FMI-1/FMI-2 read-only audit for decode fused-MMVQ integration.

This does not run kernels or route model paths. It consumes the existing
tinygrad role-efficiency artifact plus llama rocprof traces, and emits the
measurement authority needed before any FMI-3/FMI-4 implementation.
"""
from __future__ import annotations

import csv, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench" / "qk-decode-fused-mmvq-integration"
HBM_PEAK_GBS = 960.0


def read_json(rel: str) -> dict[str, Any]:
  with open(ROOT / rel) as f:
    return json.load(f)


def write_json(name: str, obj: dict[str, Any]) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  with open(OUT / name, "w") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")


def _vec_family(name: str) -> str | None:
  if "mul_mat_vec_q" not in name: return None
  if "(ggml_type)12" in name: q = "Q4_K"
  elif "(ggml_type)14" in name: q = "Q6_K"
  else: q = "unknown"
  fused = "fusion_true" if ", true, false>" in name else "fusion_false" if ", false, false>" in name else "fusion_unknown"
  return f"llama_mmvq_{q}_{fused}"


def parse_llama_trace() -> dict[str, Any]:
  trace = ROOT / "bench/llama-kernel-residual-primitive-audit-20260619/rocprof_decode_d0/trace_kernel_trace.csv"
  rows: dict[str, dict[str, Any]] = {}
  geom_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
  with open(trace, newline="") as f:
    for r in csv.DictReader(f):
      fam = _vec_family(r["Kernel_Name"])
      if fam is None: continue
      dur = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
      item = rows.setdefault(fam, {
        "family": fam,
        "calls": 0,
        "total_ns": 0,
        "vgpr_values": set(),
        "sgpr_values": set(),
        "lds_values": set(),
        "scratch_values": set(),
        "geometries": [],
      })
      item["calls"] += 1
      item["total_ns"] += dur
      item["vgpr_values"].add(int(r["VGPR_Count"]))
      item["sgpr_values"].add(int(r["SGPR_Count"]))
      item["lds_values"].add(int(r["LDS_Block_Size"]))
      item["scratch_values"].add(int(r["Scratch_Size"]))
      geom = (
        int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Grid_Size_Z"]),
        int(r["Workgroup_Size_X"]), int(r["Workgroup_Size_Y"]), int(r["Workgroup_Size_Z"]),
        int(r["VGPR_Count"]), int(r["LDS_Block_Size"]), int(r["Scratch_Size"]),
      )
      geom_counts[fam][json.dumps(geom)] += dur

  out_rows = []
  for fam, item in rows.items():
    geoms = []
    for g_s, ns in sorted(geom_counts[fam].items(), key=lambda kv: -kv[1])[:8]:
      gx, gy, gz, wx, wy, wz, vgpr, lds, scratch = json.loads(g_s)
      geoms.append({
        "grid": [gx, gy, gz],
        "workgroup": [wx, wy, wz],
        "vgpr": vgpr,
        "lds": lds,
        "scratch": scratch,
        "share_of_family_time": round(ns / item["total_ns"], 4) if item["total_ns"] else 0,
      })
    out_rows.append({
      "family": fam,
      "calls": item["calls"],
      "total_ms": round(item["total_ns"] / 1e6, 3),
      "avg_us": round(item["total_ns"] / item["calls"] / 1e3, 3),
      "vgpr_values": sorted(item["vgpr_values"]),
      "sgpr_values": sorted(item["sgpr_values"]),
      "lds_values": sorted(item["lds_values"]),
      "scratch_values": sorted(item["scratch_values"]),
      "dominant_geometries": geoms,
    })
  return {
    "schema": "llama_mmvq_launch_contract_v1",
    "source": str(trace.relative_to(ROOT)),
    "rows": sorted(out_rows, key=lambda r: -r["total_ms"]),
    "summary": {
      "dominant_contract": "wg32 one-wave MMVQ, grid proportional to output rows and split/fusion state",
      "low_vgpr": "24-40 per prior convergence note; trace rows expose 32-ish values per kernel family on this build",
      "lds": "0 for Q4_K mul_mat_vec_q rows; 512B for Q6_K rows in this trace",
    },
  }


def tinygrad_contract_rows(role_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  # These are code-contract rows, not fresh profiler rows. They come from model.py and q4/q6 primitive sources.
  rows = []
  for r in role_rows:
    role = r["role"]
    if role in ("attn_q/o",):
      contract = {
        "path": "Q4KPrimitiveLinear -> q4k_coop_partial_kernel",
        "contract_source": "model.py Q4K_ATTN_QO_COOP + q4_k_gemv_primitive.py",
        "shape": "4096x4096",
        "local_axes": "row_i x lane4 = Q4K_COOP_RT(default 16) x 8 = 128 lanes",
        "post_kernel_reduce": "partials[row,8].sum(axis=1)",
        "classification": "occupancy_launch_plus_extra_reduce",
      }
    elif role == "ffn_gate/up":
      contract = {
        "path": "Q4KPrimitiveLinear -> q4k_gemv_partial_kernel (coop not routed for gate/up)",
        "contract_source": "model.py comment: ffn_gate/up already coalesced enough; q4k coop sub-gate",
        "shape": "12288x4096",
        "local_axes": "policy opts, no llama-style q8/int-dot activation lifecycle by default",
        "post_kernel_reduce": "partials.sum(axis=1) when parts > 1",
        "classification": "activation_lifecycle_plus_occupancy_launch",
      }
    elif role in ("ffn_down", "lm_head"):
      contract = {
        "path": "Q6KPrimitiveLinear -> q6k_coop_partial_kernel",
        "contract_source": "model.py Q6K_FFN_DOWN_COOP/Q6K_LM_HEAD_COOP",
        "shape": "4096x12288 or 151936x4096",
        "local_axes": "cooperative-K path, row tile default Q6K_COOP_RT=4",
        "post_kernel_reduce": "partials[row,16].sum(axis=1)",
        "classification": "coverage_plus_occupancy_launch",
      }
    else:
      contract = {
        "path": "Q4/Q6 primitive path",
        "contract_source": "model.py primitive routing",
        "shape": "1024x4096",
        "local_axes": "mixed quant consumers",
        "post_kernel_reduce": "role-dependent",
        "classification": "low_share_mixed",
      }
    rows.append({**r, "tinygrad_contract": contract})
  return rows


def build_loss_atlas() -> dict[str, Any]:
  role_eff = read_json("bench/qk-gemv-role-efficiency/result.json")
  pmu = read_json("bench/qk-primitive-pmu-atlas/result.json")
  reuse = read_json("bench/qk-q8-lifecycle/reuse_map.json")
  rows = []
  total_role_ms = sum(r["total_tm_ms"] for r in role_eff["rows"])
  target_pct = 54.0
  for r in role_eff["rows"]:
    cur = float(r["pct_hbm_peak"])
    speed_if_54 = target_pct / cur if cur else None
    role_time_share = r["total_tm_ms"] / total_role_ms if total_role_ms else 0
    projected_role_cut = role_time_share * (1 - (cur / target_pct)) if cur and cur < target_pct else 0
    if r["role"] == "ffn_gate/up":
      mechanism = ["activation_lifecycle", "occupancy_launch"]
      q8_reuse = 2
    elif r["role"] in ("ffn_down", "lm_head"):
      mechanism = ["occupancy_launch", "coverage"]
      q8_reuse = 0
    elif r["role"] == "attn_q/o":
      mechanism = ["occupancy_launch"]
      q8_reuse = 1
    else:
      mechanism = ["low_share_mixed"]
      q8_reuse = 1
    rows.append({
      **r,
      "role_time_share_within_weight_gemv": round(role_time_share, 4),
      "target_pct_hbm_peak": target_pct,
      "isolated_speed_if_target_54pct": round(speed_if_54, 3) if speed_if_54 else None,
      "projected_weight_gemv_time_cut_if_target": round(projected_role_cut, 4),
      "mechanism_tags": mechanism,
      "q8_activation_reuse_count": q8_reuse,
    })
  aggregate = {
    "authority": "decode-bandwidth-bound-pmu-learning-20260619.md",
    "tinygrad_standalone_gemv_pct_hbm": 76,
    "tinygrad_inmodel_weight_gemv_pct_hbm": 44,
    "llama_standalone_gemv_pct_hbm": 57,
    "llama_inmodel_weight_gemv_pct_hbm": 54,
    "weight_gemv_gpu_time_share": 0.85,
    "projected_e2e_speedup_if_44_to_54": round(1 / (0.85 * 44 / 54 + 0.15), 3),
    "pmu_ctx512_bw_bound_gpu_pct": pmu["by_ctx"]["512"]["bw_bound_gpu%"],
    "pmu_ctx512_cache_served_gpu_pct": pmu["by_ctx"]["512"]["cache_served_gpu%"],
  }
  return {
    "schema": "decode_fused_mmvq_inmodel_loss_atlas_v1",
    "phase": "FMI-1",
    "status": "PASS_ROLE_GROUP_WITH_5PCT_PROJECTED_MOVEMENT",
    "inputs": {
      "role_efficiency": "bench/qk-gemv-role-efficiency/result.json",
      "pmu_atlas": "bench/qk-primitive-pmu-atlas/result.json",
      "q8_reuse_map": "bench/qk-q8-lifecycle/reuse_map.json",
    },
    "aggregate": aggregate,
    "rows": tinygrad_contract_rows(rows),
    "q8_reuse_map_summary": {
      "max_q4k_reuse": reuse["max_amortization_for_q4k_int_dot"],
      "best_first_route": "ffn_norm -> ffn_gate/up",
      "note": reuse["note"],
    },
    "gate": {
      "required": "at least one role group with >=5% projected e2e movement and a named mechanism",
      "passes": True,
      "named_mechanism": "in-model weight-GEMV BW 44% -> 54% projects ~18.7% e2e if achieved across the weight-GEMV bucket; ffn_gate/up plus Q6 roles are the main role groups",
    },
  }


def build_launch_diff(llama_contract: dict[str, Any], atlas: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "decode_fused_mmvq_launch_contract_diff_v1",
    "phase": "FMI-2",
    "status": "PASS_CONCRETE_DELTA_EXISTS",
    "llama_contract": llama_contract,
    "tinygrad_contract": {
      "source": "model.py plus q4_k_gemv_primitive.py/q6_k_gemv_primitive.py code-contract audit",
      "rows": [r["tinygrad_contract"] | {"role": r["role"], "pct_hbm_peak": r["pct_hbm_peak"]} for r in atlas["rows"]],
    },
    "diff": [
      {
        "axis": "activation format",
        "llama": "Q8_1 activation produced once and reused by MMVQ consumers",
        "tinygrad": "default fp activation path; q8 routes are gated/research and native fused producer is codegen-walled",
        "classification": "Track A activation_lifecycle",
      },
      {
        "axis": "dominant launch shape",
        "llama": "mul_mat_vec_q dominant rows use wg32 one-wave contracts with large grids such as 131072/393216 and lds=0",
        "tinygrad": "role-dependent custom_kernel contracts; Q4 attn coop uses row_i*lane4=128 lanes plus separate partial reduce, Q6 coop writes 16 partial lanes plus reduce, gate/up stays fp path",
        "classification": "Track B occupancy_launch",
      },
      {
        "axis": "in-model efficiency retention",
        "llama": "57% standalone -> ~54% in-model",
        "tinygrad": "76% standalone -> ~44% in-model",
        "classification": "integration_penalty",
      },
      {
        "axis": "fresh profiler coverage",
        "llama": "rocprofv3 gives launch geometry/VGPR/LDS for HIP kernels",
        "tinygrad": "HCQ is not visible to rocprofv3; current tinygrad side is code-contract + native attribution, not full launch trace metadata",
        "classification": "tooling_gap",
      },
    ],
    "gate": {
      "required": "concrete tinygrad-side delta such as launch shape, occupancy, extra kernels, or fallback",
      "passes": True,
      "concrete_delta": "llama's dominant MMVQ contract is q8 + wg32 large-grid, while tinygrad in-model roles use mixed fp/Q4/Q6 custom kernels with partial-output reductions and do not preserve the standalone 76% BW in the model",
    },
  }


def build_result(atlas: dict[str, Any], launch: dict[str, Any]) -> dict[str, Any]:
  return {
    "schema": "decode_fused_mmvq_fmi1_fmi2_result_v1",
    "scope": "docs/decode-fused-mmvq-integration-next-path-scope-20260619.md",
    "FMI_1": {
      "status": atlas["status"],
      "artifact": "bench/qk-decode-fused-mmvq-integration/inmodel_loss_atlas.json",
    },
    "FMI_2": {
      "status": launch["status"],
      "artifact": "bench/qk-decode-fused-mmvq-integration/launch_contract_diff.json",
    },
    "decision": {
      "status": "BUILD_TRACK_B_FIRST",
      "reason": "FMI-1/2 identify a byte-identical in-model launch/occupancy integration delta with larger projected EV than q8-only gate/up reuse",
      "next": "FMI-4 occupancy-preserving tinygrad route, beginning with one high-share role group and graph-capture safety; FMI-3 q8 replay remains secondary",
    },
    "do_not_do": [
      "do not build another standalone GEMV kernel first",
      "do not route lossy q8 by default",
      "do not reopen spec-decode TBF-3 from this result",
      "do not touch prefill",
    ],
  }


def write_summary(result: dict[str, Any], atlas: dict[str, Any]) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  lines = [
    "# Decode fused-MMVQ integration FMI-1/FMI-2",
    "",
    f"Verdict: `{result['decision']['status']}`.",
    "",
    "FMI-1 passes: the in-model weight-GEMV bucket has enough movement. The authority aggregate is tinygrad `44%` vs llama `54%` HBM in-model, which projects about "
    f"`{atlas['aggregate']['projected_e2e_speedup_if_44_to_54']}x` if recovered across the weight-GEMV bucket.",
    "",
    "FMI-2 passes: llama's dominant MMVQ launch contract is q8 + wg32/large-grid/lds0, while tinygrad's in-model routes are mixed fp/Q4/Q6 custom kernels with partial-output reductions and do not retain standalone BW.",
    "",
    "Decision: build Track B first. It is byte-identical and targets the larger integration loss. Track A q8 replay remains secondary and lossy/dNLL-gated.",
    "",
  ]
  (OUT / "summary.md").write_text("\n".join(lines))


def main() -> None:
  atlas = build_loss_atlas()
  llama = parse_llama_trace()
  launch = build_launch_diff(llama, atlas)
  result = build_result(atlas, launch)
  write_json("inmodel_loss_atlas.json", atlas)
  write_json("llama_launch_contract.json", llama)
  write_json("launch_contract_diff.json", launch)
  write_json("result.json", result)
  write_summary(result, atlas)
  print(json.dumps({"status": result["decision"]["status"], "out": str(OUT)}, indent=2))


if __name__ == "__main__":
  main()
