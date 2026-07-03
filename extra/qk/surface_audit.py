from __future__ import annotations
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent

PROMOTED = {
  "flash_decode.py", "flash_decode_fused_combine.py", "live_split_geometry.py", "route_manifest.py",
  "q4k_lane_partition_gemv.py", "q4k_packed_gemv_wd.py", "q4k_scheduler_gemv.py", "q6k_route_spec.py",
  "quant_semantics.py", "quantize.py",
}
SUPPORT = {
  "amd_warp_reduce.py", "artifact_cache.py", "asm_scheduler.py", "bubblebeam_futuresight.py", "clock_pin.py",
  "experiment_matrix.py", "flash_common.py", "flash_kernels.py", "gate_registry.py", "harness_contract.py",
  "isa_helpers.py", "kv_load.py", "lane_partition_reduce.py", "lanemap_template.py",
  "asm_scheduler_proofs.py", "decode_physical_tile.py", "decode_score_broadcast.py",
  "decode_attention_online_state_pv.py", "tg_p9_live_split.py",
  "layout.py", "layout_fn.py", "modes.py", "native_isa_block_tile_graph_node.py", "nll_eval.py", "paths.py",
  "probe_harness.py", "pure_search_guard.py", "reg_store_devec.py", "warp_reduce_lowering.py",
}

ACTIVE_PATTERNS = (
  r".*_audit\.py$", r".*_gate\.py$", r".*_microgate\.py$", r".*_check\.py$", r".*_profile\.py$",
  r".*_detector\.py$", r".*_contract\.py$", r".*_diagnostic.*\.py$", r".*_consistency_.*\.py$",
  r".*_reachability_.*\.py$", r".*_discovery\.py$", r".*_attribution_.*\.py$",
  r".*_correctness\.py$", r".*_guardrail\.py$", r".*_overhead\.py$", r".*_bench\.py$",
  r".*_refresh\.py$", r".*_hardening\.py$", r".*_selective\.py$", r".*_route\.py$", r".*_spec\.py$",
  r".*_synced\.py$", r".*_parity\.py$", r".*_role_attribution\.py$",
)
REFUTED_PATTERNS = (
  r"tg_p\d+_.*\.py$", r".*_ab\.py$", r".*_wd\.py$", r".*_probe\.py$", r".*_repro\.py$",
  r".*_matrix\.py$", r".*_diff\.py$", r".*_trace\.py$", r".*_lifecycle_.*\.py$",
  r"decode_attention_online_.*\.py$", r"decode_physical_tile_.*\.py$", r"decode_score_broadcast_.*\.py$",
  r"decode_attention_split_.*\.py$", r".*_one_case\.py$",
  r"asm_scheduler_inc\d+_test\.py$",
)

def classify(path:pathlib.Path) -> str|None:
  name = path.name
  if name in PROMOTED: return "promoted"
  if name in SUPPORT or name.startswith(("layout_", "codegen_", "coalesced_", "cooperative_", "fdot2_", "gemv_g")): return "support"
  if any(re.fullmatch(p, name) for p in ACTIVE_PATTERNS): return "active"
  if any(re.fullmatch(p, name) for p in REFUTED_PATTERNS): return "refuted"
  return None

def build() -> int:
  files = [p for p in ROOT.glob("*.py") if p.name not in {"__init__.py", "surface_audit.py"}]
  groups: dict[str, list[str]] = {"promoted": [], "active": [], "support": [], "refuted": [], "unclassified": []}
  for path in sorted(files):
    groups[classify(path) or "unclassified"].append(path.name)
  print(json.dumps({k: len(v) for k, v in groups.items()}, indent=2, sort_keys=True))
  if groups["unclassified"]:
    print("unclassified qk files:", ", ".join(groups["unclassified"]), file=sys.stderr)
    return 1
  return 0

if __name__ == "__main__":
  sys.path.insert(0, str(ROOT.parents[1]))
  from extra.qk.gate_registry import run
  raise SystemExit(run("surface"))
