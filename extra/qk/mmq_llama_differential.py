#!/usr/bin/env python3
"""Fail-closed structural comparison for tinygrad and translated llama Q4_K MMQ.

This module compares claims, not results.  In particular, correctness evidence
is deliberately outside the vocabulary: equal output numbers cannot fill in a
missing launch, staging, synchronization, ownership, or lifecycle fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from extra.qk.mmq_llama_oracle import LLAMA_MMQ_CUH, LlamaMMQOracleGeometry
from extra.qk.q4k_q8_mmq_uop import Q4KQ8MMQWMMASpec


SCHEMA = "tinygrad.mmq_llama_structural_differential.v1"
LLAMA_SOURCE_COMMIT = "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
REQUIRED_DIMENSIONS = (
  "tile_m", "tile_n", "tile_k", "waves_workgroup", "lds_q4_panel", "lds_q8_panel", "barriers",
  "q8_ds_semantics", "accumulator_ownership", "k_lifecycle", "dot_primitive", "writeback", "stream_k",
  "resource_fields",
)


@dataclass(frozen=True)
class StructuralDifferential:
  equivalent: bool
  gaps: tuple[dict[str, Any], ...]
  oracle: Mapping[str, Any]
  candidate: Mapping[str, Any]

  @property
  def status(self) -> str: return "EQUIVALENT" if self.equivalent else "BLOCKED"

  def to_json(self) -> dict[str, Any]:
    return {"schema": SCHEMA, "status": self.status, "equivalent": self.equivalent,
            "numeric_correctness_considered": False, "gaps": list(self.gaps),
            "oracle": dict(self.oracle), "candidate": dict(self.candidate)}


def llama_q4k_q8_structural_descriptor(
    geometry: LlamaMMQOracleGeometry = LlamaMMQOracleGeometry()) -> dict[str, Any]:
  """The bounded AMD MMA contract translated from local ``mmq.cuh`` anchors."""
  geometry.validate()
  return {
    "schema": SCHEMA, "descriptor_id": "llama.q4_k_q8_1.mmq.amd_mma",
    "source": LLAMA_MMQ_CUH, "source_commit": LLAMA_SOURCE_COMMIT,
    "dimensions": {
      "tile_m": geometry.mmq_y,
      "tile_n": geometry.mmq_x,
      "tile_k": geometry.iter_k,
      "waves_workgroup": {"block": (32, 8, 1), "waves": geometry.nwarps, "wave_size": geometry.warp_size,
                          "workgroup_threads": geometry.nwarps * geometry.warp_size},
      "lds_q4_panel": {"present": True, "carrier": "tile_x", "loader": "load_tiles_q4_K",
                       "representation": "decoded_q4", "row_stride_ints": 76, "rows": 128,
                       "bytes": 38912, "offset_bytes": 18944, "row_layout": "aos_interleaved",
                       "row_components": (("qs", 0, 256), ("dm", 256, 32), ("padding", 288, 16))},
      "lds_q8_panel": {"present": True, "carrier": "tile_y", "layout": "block_q8_1_mmq",
                       "rows": 128, "row_bytes": 144, "bytes": 18432, "offset_bytes": 512,
                       "padded_bytes": 18432, "row_layout": "aos_interleaved",
                       "row_components": (("ds", 0, 16), ("qs", 16, 128)),
                       "halves_per_k_iteration": 2, "half_k_elements": 128},
      "barriers": {"per_k_iteration": 4, "sequence": (
        "stage_q4_and_q8_half0", "barrier", "dot_half0", "barrier",
        "stage_q8_half1", "barrier", "dot_half1", "barrier")},
      "q8_ds_semantics": {"format": "Q8_1", "layout": "DS4", "values": "signed_int8",
                          "scale_and_sum": True, "sum_semantic": "sum_original_fp",
                          "sum_timing": "before_quantization", "sum_group_elements": 32},
      "accumulator_ownership": {"storage": "thread_private_sum", "dtype": "float32",
                                "elements_per_thread": geometry.mmq_x * geometry.mmq_y //
                                                       (geometry.nwarps * geometry.warp_size),
                                "tile_c": (geometry.tile_c_i, geometry.tile_c_j), "layout": "J_MAJOR"},
      "k_lifecycle": {"step": geometry.iter_k, "q4_loads_per_step": 1, "q4_residency": "both_q8_halves",
                      "q8_half_loads_per_step": 2, "q8_half_elements": 128,
                      "scale_group_elements": 32, "scale_groups_per_step": 8,
                      "integer_accumulator_scope": "one_scale_group", "integer_reset_per_scale_group": True,
                      "float_correction_timing": "immediately_after_each_scale_group",
                      "accumulate_float_across_steps": True, "writeback_after_loop": True},
      "dot_primitive": {"dispatch": "vec_dot_q8_1_q8_1_mma", "isa": "v_wmma_i32_16x16x16_iu8",
                        "signed_A": True, "signed_B": True, "accumulator": "int32",
                        "intrinsic_k": 16, "semantic_dot_group_k": 32, "wmma_per_scale_group": 2,
                        "postscale_accumulator": "float32", "subtile": (geometry.tile_c_i, geometry.tile_c_j)},
      "writeback": {"function": "mmq_write_back_mma", "owner": "wave_tile_c_fragment",
                    "sum_index": "(j0/tile_C::J+n)*tile_C::ne+l",
                    "dst_index": "ids_dst[j]*stride+i", "role_tails": False},
      "stream_k": {"enabled": False, "target": "gfx1100_rdna3", "runtime_path": "conventional_tiling",
                   "grid": {"x": "ceil(nrows_x/128)", "y": "ceil(ncols_max/128)",
                            "z": "channels*samples"},
                   "generic_source_support": True, "enable_condition": "nvidia_volta_plus_or_cdna"},
      "resource_fields": {"required": ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills",
                                               "sgpr_spills", "wavefront_size", "workgroup_threads"),
                          "launch_bounds_threads": geometry.nwarps * geometry.warp_size,
                          "lds_bytes": 57856, "lds_ids_bytes": 512, "lds_q8_padded_bytes": 18432,
                          "lds_q4_bytes": 38912},
    },
    "source_anchors": {
      "lds_and_k": "mul_mat_q_process_tile", "q4_loader": "load_tiles_q4_K",
      "dot_dispatch": "mmq_type_traits<..., GGML_TYPE_Q4_K>", "writeback": "mmq_write_back_mma",
      "stream_k": "mul_mat_q / mul_mat_q_stream_k_fixup",
    },
  }


def current_direct_uop_descriptor(spec: Q4KQ8MMQWMMASpec, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
  """Describe only facts exposed by the current authored UOps/final evidence."""
  spec.validate()
  dims: dict[str, Any] = {
    "tile_m": spec.m, "tile_n": spec.n, "tile_k": spec.k,
    "q8_ds_semantics": {"format": "Q8_1", "layout": "row_major", "values": "signed_int8",
                        "scale_and_sum": False},
  }
  if evidence is not None:
    launch, resources = evidence.get("launch"), evidence.get("resources")
    if isinstance(launch, Mapping) and isinstance(resources, Mapping):
      local = launch.get("local_size")
      if isinstance(local, (list, tuple)) and local and isinstance(resources.get("wavefront_size"), int):
        threads = 1
        for value in local: threads *= value
        dims["waves_workgroup"] = {"waves": threads // resources["wavefront_size"],
                                   "wave_size": resources["wavefront_size"], "workgroup_threads": threads}
    required = llama_q4k_q8_structural_descriptor()["dimensions"]["resource_fields"]["required"]
    if isinstance(resources, Mapping) and all(field in resources for field in required[:-1]) and "waves_workgroup" in dims:
      dims["resource_fields"] = {"required": required,
                                 "launch_bounds_threads": dims["waves_workgroup"]["workgroup_threads"]}
    final_isa = evidence.get("final_isa")
    if isinstance(final_isa, Mapping) and final_isa.get("wmma_mnemonic") == "v_wmma_i32_16x16x16_iu8":
      # The mnemonic does not encode the builtin's signed-A/signed-B control
      # operands, grouping, reset scope, or correction recurrence.
      dims["dot_primitive"] = {"isa": "v_wmma_i32_16x16x16_iu8"}
  return {"schema": SCHEMA, "descriptor_id": spec.name, "dimensions": dims,
          "evidence_attached": evidence is not None}


def compare_structures(candidate: Mapping[str, Any], oracle: Mapping[str, Any] | None = None) -> StructuralDifferential:
  """Require every dimension and exact structural equality; unknown always blocks."""
  target = llama_q4k_q8_structural_descriptor() if oracle is None else oracle
  candidate_dims = candidate.get("dimensions") if isinstance(candidate, Mapping) else None
  oracle_dims = target.get("dimensions") if isinstance(target, Mapping) else None
  if not isinstance(candidate_dims, Mapping): candidate_dims = {}
  if not isinstance(oracle_dims, Mapping): raise ValueError("oracle has no dimensions mapping")
  gaps: list[dict[str, Any]] = []
  for dimension in REQUIRED_DIMENSIONS:
    if dimension not in oracle_dims or oracle_dims[dimension] is None:
      raise ValueError(f"oracle missing required dimension {dimension}")
    if dimension not in candidate_dims or candidate_dims[dimension] is None:
      gaps.append({"dimension": dimension, "kind": "missing", "expected": oracle_dims[dimension]})
    elif candidate_dims[dimension] != oracle_dims[dimension]:
      gaps.append({"dimension": dimension, "kind": "mismatch", "expected": oracle_dims[dimension],
                   "actual": candidate_dims[dimension]})
  return StructuralDifferential(not gaps, tuple(gaps), target, candidate)


def compare_current_direct_uop(spec: Q4KQ8MMQWMMASpec, evidence: Mapping[str, Any] | None = None) -> StructuralDifferential:
  return compare_structures(current_direct_uop_descriptor(spec, evidence))
