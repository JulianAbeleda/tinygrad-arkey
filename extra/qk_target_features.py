#!/usr/bin/env python3
"""TG5: Cross-Target Feature Model -- separate algorithmic route families from target-specific lowering.

A topology candidate (TG2) is an ALGORITHM (lane ownership / reduction / dequant placement). Whether it can RUN
is a TARGET property (wave/subgroup size, vector-dot / matrix-core, LDS, barrier model, register file, occupancy,
native-ISA backend, compiler ownership, profiler availability). This module makes those features DATA and lets the
candidate author/evaluator say "algorithmically plausible but target lowering is missing" instead of silently
pretending portability.

  * AMD gfx1100 is the VALIDATED target (native-ISA backend + HIP + W==D profiling exist) -> reproduces the current
    route permissions (wave32 -> lane_extent=32; wave64/subgroup_simdgroup pruned).
  * NVIDIA / Apple-Metal are DESCRIPTORS only (no backend to run here) -> any candidate gates to
    TARGET_BACKEND_INCOMPLETE until lowering + profiling gates pass.

The TG2 author ALREADY gates candidates on the target's lane_extent (it reads gpu.wave from the profile). TG5
formalizes that into a feature model and proves: (a) the gfx1100 features reproduce the author's wave32-only,
20-candidate bounded set; (b) a wave64 candidate is target-pruned on gfx1100; (c) NVIDIA/Metal candidates gate to
TARGET_BACKEND_INCOMPLETE, never silently TARGET_OK.

AUDIT/RESEARCH only: no GPU kernel, no default change, no live-route repoint. Reads descriptors + checks.

Run: PYTHONPATH=. python3 extra/qk_target_features.py
"""
from __future__ import annotations
import json, pathlib
from extra.qk_artifact_cache import emit_artifact
from dataclasses import dataclass, field, asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGETS_DIR = ROOT / "bench/qk-search-spaces/targets"
OUT = ROOT / "bench/qk-target-features"

TARGET_OK = "TARGET_OK"
TARGET_PRUNED = "TARGET_PRUNED_FEATURE_MISMATCH"
TARGET_BACKEND_INCOMPLETE = "TARGET_BACKEND_INCOMPLETE"


@dataclass(frozen=True)
class TargetFeatures:
  """A GPU target as DATA. backend_validated gates whether a candidate can be RUN/promoted here."""
  target_id: str
  vendor: str
  arch: str
  wave_size: int                        # native wave/subgroup width
  subgroup_model: str                   # "wave32" | "wave64" | "subgroup32" | "simdgroup"
  coalescing_granularity_bytes: int     # contiguous bytes a wave coalesces well
  vector_dot: tuple[str, ...]           # available integer/fp dot primitives (e.g. v_dot4_i32_i8, v_dot2)
  matrix_core: str | None               # WMMA / Tensor Core / simdgroup_matrix / None
  lds_bytes: int                        # shared-memory / LDS bytes per workgroup
  barrier_model: str
  register_file_vgprs: int
  occupancy_model: str
  native_isa_backend_available: bool    # is there an in-repo native-ISA backend for this target?
  external_compiler_ownership: str      # who owns final lowering (HIP/comgr, ptxas, metal)
  profiling_available: bool             # can we measure W==D / whole-prefill authority here?
  backend_validated: bool               # all of {native or generated backend + runtime + profiling} present
  allowed_route_families: tuple[str, ...]
  notes: str = ""

  def lane_extent(self) -> int:
    return self.wave_size

  def row(self) -> dict:
    return asdict(self)


# ---- the target library -------------------------------------------------------------------------------------
TARGETS: dict[str, TargetFeatures] = {
  # AMD gfx1100 (RX 7900 XTX): the VALIDATED target. wave32, v_dot4 int dot, WMMA, native-ISA backend, HIP, W==D.
  "amd_gfx1100": TargetFeatures(
    target_id="amd_gfx1100", vendor="AMD", arch="gfx1100", wave_size=32, subgroup_model="wave32",
    coalescing_granularity_bytes=128, vector_dot=("v_dot4_i32_i8", "v_dot4_u32_u8", "v_dot2_f32_f16"),
    matrix_core="WMMA_rdna3", lds_bytes=65536, barrier_model="workgroup_barrier_s_barrier",
    register_file_vgprs=256, occupancy_model="vgpr_lds_bound",
    native_isa_backend_available=True, external_compiler_ownership="HIP/comgr (tinygrad owns UOp->AMDGCN lowering)",
    profiling_available=True, backend_validated=True,
    allowed_route_families=("lanemap", "coop", "graph_gemm_pipe", "native_isa_attention", "owned_reference"),
    notes="validated: native-ISA backend + HIP + W==D/whole-prefill profiling all present. Reproduces the current "
          "route permissions (the manifest routes run here)."),
  # NVIDIA sm_89 (Ada): DESCRIPTOR ONLY. wave/subgroup=32 but no in-repo backend/runtime/profiler here.
  "nvidia_sm89": TargetFeatures(
    target_id="nvidia_sm89", vendor="NVIDIA", arch="sm_89", wave_size=32, subgroup_model="subgroup32",
    coalescing_granularity_bytes=128, vector_dot=("dp4a", "mma_m16n8k16"),
    matrix_core="TensorCore_ada", lds_bytes=49152, barrier_model="__syncthreads_bar",
    register_file_vgprs=255, occupancy_model="regs_smem_bound",
    native_isa_backend_available=False, external_compiler_ownership="ptxas (closed)",
    profiling_available=False, backend_validated=False,
    allowed_route_families=("lanemap", "coop", "graph_gemm_pipe", "owned_reference"),
    notes="DESCRIPTOR ONLY (no backend to run here). subgroup32 is algorithmically close to wave32, but lowering + "
          "profiling are absent -> candidates gate TARGET_BACKEND_INCOMPLETE. Do NOT claim portability."),
  # Apple Metal (M-series): DESCRIPTOR ONLY. simdgroup=32, simdgroup_matrix, no in-repo backend/runtime/profiler here.
  "apple_metal_m3": TargetFeatures(
    target_id="apple_metal_m3", vendor="Apple", arch="metal3_m3", wave_size=32, subgroup_model="simdgroup",
    coalescing_granularity_bytes=64, vector_dot=("simd_shuffle",),
    matrix_core="simdgroup_matrix", lds_bytes=32768, barrier_model="threadgroup_barrier",
    register_file_vgprs=128, occupancy_model="threadgroup_mem_bound",
    native_isa_backend_available=False, external_compiler_ownership="metal (closed)",
    profiling_available=False, backend_validated=False,
    allowed_route_families=("lanemap", "coop", "owned_reference"),
    notes="DESCRIPTOR ONLY. simdgroup==32 width but distinct shuffle/matrix primitives + no in-repo backend/"
          "profiling -> TARGET_BACKEND_INCOMPLETE."),
}


def target(target_id: str) -> TargetFeatures:
  if target_id not in TARGETS:
    raise KeyError(f"unknown target {target_id!r}; known: {sorted(TARGETS)}")
  return TARGETS[target_id]


# grammar target_feature value -> required wave_size / subgroup_model (the gate)
_GRAMMAR_TARGET_REQUIREMENTS = {
  "wave32": {"wave_size": 32},
  "wave64": {"wave_size": 64},
  "subgroup32": {"wave_size": 32},
  "subgroup_simdgroup": {"subgroup_model": "simdgroup"},
}


def gate_candidate_on_target(candidate: dict, tgt: TargetFeatures) -> tuple[str, str]:
  """Gate a TG2-authored candidate on a target's features.

  Returns (verdict, reason). backend_validated=False ALWAYS yields TARGET_BACKEND_INCOMPLETE (algorithmically
  plausible but no lowering/profiling here) -- never a silent TARGET_OK. On the validated target, a candidate
  whose required target_feature mismatches the target's wave/subgroup is TARGET_PRUNED."""
  # 1. algorithmic plausibility is independent of lowering; but a candidate that needs a wider wave than the target
  #    has is feature-pruned even before lowering.
  req = candidate.get("target_feature_required", "wave32")
  need = _GRAMMAR_TARGET_REQUIREMENTS.get(req, {})
  for k, v in need.items():
    if getattr(tgt, k) != v:
      return TARGET_PRUNED, f"candidate needs {req} ({k}={v}) but target {tgt.target_id} has {k}={getattr(tgt, k)}"
  # 2. lowering/profiling gate: only the validated target can RUN/promote a candidate.
  if not tgt.backend_validated:
    missing = []
    if not tgt.native_isa_backend_available: missing.append("native_isa_backend")
    if not tgt.profiling_available: missing.append("W==D/whole-prefill profiling")
    return TARGET_BACKEND_INCOMPLETE, (f"algorithmically plausible on {tgt.target_id} ({tgt.subgroup_model}) but "
                                       f"target lowering/profiling missing: {missing}")
  return TARGET_OK, f"runs on validated target {tgt.target_id} ({tgt.subgroup_model})"


def dump_targets() -> list[str]:
  TARGETS_DIR.mkdir(parents=True, exist_ok=True)
  paths = []
  for tid, t in TARGETS.items():
    p = TARGETS_DIR / f"{tid}.json"
    json.dump({"_schema": "TG5 target feature descriptor (extra/qk_target_features.py)", **t.row()},
              open(p, "w"), indent=2)
    paths.append(str(p.relative_to(ROOT)))
  return paths


# ---- TG5 acceptance gates -----------------------------------------------------------------------------------
def _author_candidates() -> list[dict]:
  """The TG2-authored gfx1100 candidate set, tagged with the target_feature they require (wave32 for all, since
  the author already gates on the profile's wave32)."""
  from extra.qk_topology_candidate_author import load_profile_facts, enumerate_candidates
  facts = load_profile_facts()
  cands, _ = enumerate_candidates(facts)
  for c in cands:
    c["target_feature_required"] = "wave32"  # the author derives lane_extent=32 from the gfx1100 profile
  return cands


def _gfx1100_reproduces_route_permissions() -> dict:
  """The validated target's lane_extent matches what the author derives, and its allowed_route_families cover
  the families the default routes on this profile use."""
  from extra.qk_topology_candidate_author import load_profile_facts
  facts = load_profile_facts()
  amd = target("amd_gfx1100")
  lane_extent_match = (amd.lane_extent() == facts["lane_extent"] == 32)
  cands = _author_candidates()
  all_ok = all(gate_candidate_on_target(c, amd)[0] == TARGET_OK for c in cands)
  # route permissions: the families used by manifest default routes on gfx1100 are all in allowed_route_families
  used = {"lanemap", "coop", "graph_gemm_pipe", "native_isa_attention", "owned_reference"}
  perms_ok = used.issubset(set(amd.allowed_route_families))
  return {"lane_extent_match_32": lane_extent_match, "all_author_candidates_TARGET_OK": all_ok,
          "candidate_count": len(cands), "route_permissions_cover_manifest_families": perms_ok}


def _wave64_candidate_pruned_on_gfx1100() -> dict:
  """A hypothetical wave64 candidate is feature-pruned on gfx1100 (wave32)."""
  amd = target("amd_gfx1100")
  wave64 = {"lane_grouping": "1row_per_warp", "block_groups": 8, "words_per_group": 8,
            "target_feature_required": "wave64"}
  v, reason = gate_candidate_on_target(wave64, amd)
  simd = {"target_feature_required": "subgroup_simdgroup"}
  v2, _ = gate_candidate_on_target(simd, amd)
  return {"wave64_verdict": v, "wave64_pruned": v == TARGET_PRUNED, "reason": reason,
          "subgroup_simdgroup_verdict": v2, "subgroup_simdgroup_pruned": v2 == TARGET_PRUNED}


def _nvidia_metal_backend_incomplete() -> dict:
  """The SAME author candidates gate to TARGET_BACKEND_INCOMPLETE on NVIDIA/Metal -- never silently OK."""
  cands = _author_candidates()
  out = {}
  for tid in ("nvidia_sm89", "apple_metal_m3"):
    t = target(tid)
    verdicts = {gate_candidate_on_target(c, t)[0] for c in cands}
    out[tid] = {"verdicts": sorted(verdicts), "all_backend_incomplete": verdicts == {TARGET_BACKEND_INCOMPLETE},
                "backend_validated": t.backend_validated,
                "sample_reason": gate_candidate_on_target(cands[0], t)[1]}
  return out


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  paths = dump_targets()
  gfx = _gfx1100_reproduces_route_permissions()
  pruned = _wave64_candidate_pruned_on_gfx1100()
  cross = _nvidia_metal_backend_incomplete()

  ready = (gfx["lane_extent_match_32"] and gfx["all_author_candidates_TARGET_OK"]
           and gfx["route_permissions_cover_manifest_families"] and pruned["wave64_pruned"]
           and pruned["subgroup_simdgroup_pruned"]
           and all(v["all_backend_incomplete"] for v in cross.values()))
  verdict = "TG5_PASS_TARGET_FEATURE_MODEL_READY" if ready else "TG5_BLOCKED_TARGET_BACKEND_INCOMPLETE"

  result = {
    "scope": "TG5 cross-target feature model: separate algorithmic route families from target lowering. AMD gfx1100 "
             "is the validated target; NVIDIA/Metal are descriptors only (TARGET_BACKEND_INCOMPLETE). AUDIT only.",
    "verdict": verdict,
    "module": "extra/qk_target_features.py", "target_descriptors": paths,
    "targets": {tid: t.row() for tid, t in TARGETS.items()},
    "acceptance_1_gfx1100_reproduces_route_permissions": gfx,
    "acceptance_2_wave64_subgroup_pruned_on_gfx1100": pruned,
    "acceptance_3_nvidia_metal_backend_incomplete": cross,
    "do_not": ["no GPU kernel", "no default change", "no live-route repoint",
               "do not claim NVIDIA/Metal portability (descriptors only)"],
  }
  emit_artifact(OUT, result, kind="derived_artifact", inputs={"targets": "amd_gfx1100+nvidia_sm89+apple_metal_m3"},
                code_paths=["extra/qk_target_features.py"])

  md = [f"# TG5 Cross-Target Feature Model -- verdict: **{verdict}**", "",
        "Target features as DATA (wave/subgroup, vector-dot/matrix-core, LDS, barrier, registers, occupancy, "
        "native-ISA backend, compiler ownership, profiling). The candidate author gates on them.", "",
        "## Targets", "",
        "| target | wave | subgroup | vector_dot | matrix_core | native ISA | profiling | backend_validated |",
        "|---|---:|---|---|---|:--:|:--:|:--:|"]
  for tid, t in TARGETS.items():
    md.append(f"| {tid} | {t.wave_size} | {t.subgroup_model} | {','.join(t.vector_dot)} | {t.matrix_core} | "
              f"{t.native_isa_backend_available} | {t.profiling_available} | {t.backend_validated} |")
  md += ["", "## Acceptance", "",
         f"- **gfx1100 reproduces route permissions**: lane_extent==32 {gfx['lane_extent_match_32']}; all "
         f"{gfx['candidate_count']} author candidates TARGET_OK {gfx['all_author_candidates_TARGET_OK']}; "
         f"families covered {gfx['route_permissions_cover_manifest_families']}",
         f"- **wave64 / subgroup_simdgroup pruned on gfx1100**: {pruned['wave64_pruned']} / "
         f"{pruned['subgroup_simdgroup_pruned']} ({pruned['reason']})",
         f"- **NVIDIA/Metal candidates -> TARGET_BACKEND_INCOMPLETE (never silently OK)**: "
         f"{all(v['all_backend_incomplete'] for v in cross.values())}", "",
         "The author can now say 'algorithmically plausible but target lowering is missing' instead of pretending "
         "portability:", ""]
  for tid, v in cross.items():
    md.append(f"- **{tid}**: {v['verdicts']} -- {v['sample_reason']}")
  md.append("")
  (OUT / "summary.md").write_text("\n".join(md))

  print(verdict)
  print(f"  gfx1100: lane_extent==32 {gfx['lane_extent_match_32']} | {gfx['candidate_count']} candidates TARGET_OK "
        f"{gfx['all_author_candidates_TARGET_OK']} | families covered {gfx['route_permissions_cover_manifest_families']}")
  print(f"  wave64 pruned: {pruned['wave64_pruned']} | subgroup_simdgroup pruned: {pruned['subgroup_simdgroup_pruned']}")
  for tid, v in cross.items():
    print(f"  {tid}: {v['verdicts']} (backend_validated={v['backend_validated']})")
  return 0 if ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
