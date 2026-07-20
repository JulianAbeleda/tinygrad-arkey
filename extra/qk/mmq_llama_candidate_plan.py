"""Exact source-derived llama Q4_K/Q8_1 MMQ representation plan.

This module is deliberately inert: it owns typed data and a projection, not an
emitter, lowering, route, or claim that tinygrad emits the represented kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, json
from typing import Any

from extra.qk.mmq_llama_differential import (LLAMA_SOURCE_COMMIT, SCHEMA as STRUCTURAL_SCHEMA,
  llama_q4k_q8_structural_descriptor)
from extra.qk.mmq_llama_oracle import LLAMA_MMQ_CUH
from extra.qk.mmq_llama_packed_operands import Q4_K_DECODED_LDS_ROW, Q4_K_GLOBAL_BLOCK, Q8_1_DS4_ROW, SOURCE_ANCHORS
from tinygrad import dtypes
from extra.qk.kernel_pipeline import (DotUpdateRecurrencePlan, HierarchicalKernelPipelinePlan,
  HierarchicalPipelineRole, hierarchical_lifecycle_events)
from extra.qk.kernel_writeback import WMMAWritebackDescriptor, WMMAWritebackLayout, WMMAWritebackProof
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.uop.ops import Ops
from extra.qk.kernel_vocabulary import (KernelCandidateContext, KernelLDSArenaRegion, KernelLDSRecordComponent,
  KernelLDSRecordLayout, KernelLDSWindow, KernelTileGeometry)

PLAN_SCHEMA = "tinygrad.mmq_llama_candidate_plan.v1"
DESCRIPTOR_ID = "llama.q4_k_q8_1.mmq.amd_mma"


def _rdna3_f16_tc():
  return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)


# Retained alias: some call sites still spell the selector by its historical (int8-era) name.
_rdna3_i8_tc = _rdna3_f16_tc


Q8_RECORD_TRANSFORM = PackedOperandRecordTransform(
  "llama.q8_1.ds4.source_to_lds_record.v1", Q8_1_DS4_ROW, Q8_1_DS4_ROW)
Q4_DECODED_WITH_FRAGMENT_VIEW = PackedOperandTransform("llama.q4_k.decoded_lds_row.char_fragment_view.v1", (
  PackedOperandComponent("qs", dtypes.int, 0, 256, "64x_int32_with_signed_char_fragment_view", 4, 16),
  PackedOperandComponent("dm", dtypes.half, 256, 32, "8x_half2_scale_min_corrections", 4, 16),
  PackedOperandComponent("padding", dtypes.int, 288, 16, "4x_int32_padding", 4, 16)))
Q4_RECORD_TRANSFORM = PackedOperandRecordTransform(
  "llama.q4_k.global_packed_to_decoded_lds_record.v1", Q4_K_GLOBAL_BLOCK, Q4_DECODED_WITH_FRAGMENT_VIEW)


def _geometry() -> KernelTileGeometry:
  # Phase-1 fp16 bring-up geometry: mirror the hand kernel (wmma.py
  # build_gemm_lds2_q4k, PART I.1 of the implementation plan) LDS layout exactly
  # -- two single-buffered fp16 regions, SA=SB=BK*2=64 B/row, 128 rows each,
  # 16384 B total, instead of the int8 kernel's ids/q8/q4 57856 B arena.
  #
  # This intentionally shrinks the hierarchical stage's outer/phase/group K
  # decomposition from the int8 kernel's K256 = 2 phases * 4 K32 groups down to
  # exactly one K32 group (outer_k=phase_k=group_k=32, phase_count=1): the hand
  # kernel never keeps a full-K256 operand resident in LDS (both A and B are
  # re-decoded and re-stored every K32 group), so matching its ~16 KB budget is
  # structurally incompatible with the old persistent(K256)/overwriteable(K128)
  # hierarchy at K256 granularity. The generic HierarchicalPackedRecordStage /
  # HierarchicalPackedRecordStageDescriptor machinery still applies unmodified
  # at this finer granularity -- an outer K256 loop is expected to invoke this
  # recurrence eight times (K256 = 8 * K32); wiring that outer loop is later work.
  a = KernelLDSArenaRegion("A", 0, 8192, 16, KernelLDSRecordLayout(128, 64, (
    KernelLDSRecordComponent("value", dtypes.half, 0, 64, 16),)))
  b = KernelLDSArenaRegion("B", 8192, 16384, 16, KernelLDSRecordLayout(128, 64, (
    KernelLDSRecordComponent("value", dtypes.half, 0, 64, 16),)))
  return KernelTileGeometry((128, 128, 32), (8, 1), 256, 32,
    (KernelLDSWindow("A", 0, 8192, 64), KernelLDSWindow("B", 8192, 16384, 64)), (), (a, b))


@dataclass(frozen=True)
class LlamaMMQCandidatePlan:
  geometry: KernelTileGeometry
  q8_transform: PackedOperandRecordTransform
  q4_transform: PackedOperandRecordTransform
  lifecycle: HierarchicalKernelPipelinePlan
  recurrence: DotUpdateRecurrencePlan
  tensor_core: object
  writeback: WMMAWritebackDescriptor
  source_commit: str = LLAMA_SOURCE_COMMIT
  classification: str = "representation_plan_only"
  emitted: bool = False

  def __post_init__(self) -> None:
    if self.classification != "representation_plan_only" or self.emitted:
      raise ValueError("candidate plan cannot claim emitted kernel equivalence")
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source commit is not pinned")
    WMMAWritebackProof.prove(self.writeback)

  def structural_descriptor(self) -> dict[str, Any]:
    """Project the oracle-shaped JSON solely from this plan's typed fields."""
    g, q8r, q4r = self.geometry, self.geometry.lds_region("q8"), self.geometry.lds_region("q4")
    q8c, q4c = q8r.records.components, q4r.records.components  # type: ignore[union-attr]
    phases, groups, dots = self.recurrence.phase_count, self.recurrence.groups_per_phase, self.recurrence.dot_substeps
    return {
      "schema": STRUCTURAL_SCHEMA, "descriptor_id": DESCRIPTOR_ID, "source": LLAMA_MMQ_CUH,
      "source_commit": self.source_commit,
      "dimensions": {
        "tile_m": g.tile[0], "tile_n": g.tile[1], "tile_k": g.tile[2],
        "waves_workgroup": {"block": (g.wave_size, g.waves[0], g.waves[1]), "waves": g.waves[0]*g.waves[1],
                            "wave_size": g.wave_size, "workgroup_threads": g.threads},
        "lds_q4_panel": {"present": True, "carrier": "tile_x", "loader": "load_tiles_q4_K",
          "representation": "decoded_q4", "row_stride_ints": q4r.records.stride_bytes//dtypes.int.itemsize,
          "rows": q4r.records.rows, "bytes": q4r.end-q4r.base, "offset_bytes": q4r.base,
          "row_layout": "aos_interleaved", "row_components": tuple((x.component, x.offset_bytes, x.size_bytes) for x in q4c)},
        "lds_q8_panel": {"present": True, "carrier": "tile_y", "layout": "block_q8_1_mmq",
          "rows": q8r.records.rows, "row_bytes": q8r.records.stride_bytes, "bytes": q8r.end-q8r.base,
          "offset_bytes": q8r.base, "padded_bytes": q8r.end-q8r.base, "row_layout": "aos_interleaved",
          "row_components": tuple((x.component, x.offset_bytes, x.size_bytes) for x in q8c),
          "halves_per_k_iteration": phases, "half_k_elements": g.tile[2]//phases},
        "barriers": {"per_k_iteration": phases*2, "sequence": (
          "stage_q4_and_q8_half0", "barrier", "dot_half0", "barrier",
          "stage_q8_half1", "barrier", "dot_half1", "barrier")},
        "q8_ds_semantics": {"format": "Q8_1", "layout": "DS4", "values": "signed_int8",
          "scale_and_sum": True, "sum_semantic": "sum_original_fp", "sum_timing": "before_quantization",
          "sum_group_elements": g.tile[2]//(phases*groups)},
        "accumulator_ownership": {"storage": "thread_private_sum", "dtype": _structural_dtype(self.recurrence.persistent_dtype),
          "elements_per_thread": g.tile[0]*g.tile[1]//g.threads, "tile_c": (16, 16), "layout": "J_MAJOR"},
        "k_lifecycle": {"step": g.tile[2], "q4_loads_per_step": 1, "q4_residency": "both_q8_halves",
          "q8_half_loads_per_step": phases, "q8_half_elements": g.tile[2]//phases,
          "scale_group_elements": g.tile[2]//(phases*groups), "scale_groups_per_step": phases*groups,
          "integer_accumulator_scope": "one_scale_group", "integer_reset_per_scale_group": True,
          "float_correction_timing": "immediately_after_each_scale_group", "accumulate_float_across_steps": True,
          "writeback_after_loop": True},
        "dot_primitive": {"dispatch": "vec_dot_q8_1_q8_1_mma", "isa": "v_wmma_i32_16x16x16_iu8",
          "signed_A": True, "signed_B": True, "accumulator": _structural_dtype(self.recurrence.dot_dtype),
          "intrinsic_k": g.tile[2]//(phases*groups*dots), "semantic_dot_group_k": g.tile[2]//(phases*groups),
          "wmma_per_scale_group": dots, "postscale_accumulator": _structural_dtype(self.recurrence.persistent_dtype), "subtile": (16, 16)},
        "writeback": {"function": "mmq_write_back_mma", "owner": "wave_tile_c_fragment",
          "sum_index": "(j0/tile_C::J+n)*tile_C::ne+l", "dst_index": "ids_dst[j]*stride+i", "role_tails": False},
        "stream_k": {"enabled": False, "target": "gfx1100_rdna3", "runtime_path": "conventional_tiling",
          "grid": {"x": "ceil(nrows_x/128)", "y": "ceil(ncols_max/128)", "z": "channels*samples"},
          "generic_source_support": True,
          "enable_condition": "nvidia_volta_plus_or_cdna"},
        "resource_fields": {"required": ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills",
          "wavefront_size", "workgroup_threads"), "launch_bounds_threads": g.threads, "lds_bytes": g.lds_bytes,
          "lds_ids_bytes": g.lds_region("ids").end-g.lds_region("ids").base,
          "lds_q8_padded_bytes": q8r.end-q8r.base, "lds_q4_bytes": q4r.end-q4r.base}},
      "source_anchors": {"lds_and_k": "mul_mat_q_process_tile", "q4_loader": "load_tiles_q4_K",
        "dot_dispatch": "mmq_type_traits<..., GGML_TYPE_Q4_K>", "writeback": "mmq_write_back_mma",
        "stream_k": "mul_mat_q / mul_mat_q_stream_k_fixup"}}

  def to_json(self) -> dict[str, Any]:
    return {"schema": PLAN_SCHEMA, "classification": self.classification, "emitted": self.emitted,
            "identity": self.identity(), "source_anchors": SOURCE_ANCHORS, "structural_plan": self.structural_descriptor()}

  def identity(self) -> str:
    payload = {"schema": PLAN_SCHEMA, "commit": self.source_commit, "geometry": repr(self.geometry),
      "q8": self.q8_transform.identity, "q4": self.q4_transform.identity, "lifecycle": repr(self.lifecycle),
      "recurrence": repr(self.recurrence), "tc": repr(self.tensor_core), "writeback": repr(self.writeback)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()

  def context(self) -> KernelCandidateContext:
    return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", self.identity(), self.geometry,
      self.lifecycle, packed_operand_a=self.q4_transform, packed_operand_b=self.q8_transform)


def llama_mmq_candidate_plan() -> LlamaMMQCandidatePlan:
  geometry, tc = _geometry(), _rdna3_f16_tc()
  # Single K32 group per stage/epoch (phase-1 bring-up; see _geometry()).  Both
  # A and B are re-decoded/re-stored every group, so there is no true
  # multi-phase residency left at this granularity -- phase_count=1 collapses
  # the generic hierarchical persistent/overwriteable lifecycle to exactly one
  # produce/publish/consume/release cycle per role, matching the hand kernel's
  # per-K32-group double-barrier cadence (wmma.py I.3/I.4).
  lifecycle = HierarchicalKernelPipelinePlan(HierarchicalPipelineRole("A", "outer_epoch"),
                                              HierarchicalPipelineRole("B", "inner_phase"), 1)
  assert len(hierarchical_lifecycle_events(lifecycle)) == 7
  recurrence = DotUpdateRecurrencePlan(dtypes.float.vec(8), dtypes.float.vec(8), 1, 1, 2, Ops.WMMA)
  # No ids-gather LDS region in the fp16 hand kernel (I.6): output is dense
  # row-major, not a token/expert-id lookup, so the writeback has no ids_region.
  writeback = WMMAWritebackDescriptor(geometry, tc, dtypes.float, 8, WMMAWritebackLayout("col", "row", 128), None)
  return LlamaMMQCandidatePlan(geometry, Q8_RECORD_TRANSFORM, Q4_RECORD_TRANSFORM, lifecycle, recurrence, tc, writeback)


def _structural_dtype(dtype) -> str:
  scalar = dtype.scalar()
  if scalar == dtypes.float: return "float32"
  if scalar == dtypes.int: return "int32"
  raise ValueError(f"no structural dtype spelling for {dtype}")


__all__ = ["DESCRIPTOR_ID", "LlamaMMQCandidatePlan", "PLAN_SCHEMA", "Q4_RECORD_TRANSFORM", "Q8_RECORD_TRANSFORM",
           "llama_mmq_candidate_plan"]
