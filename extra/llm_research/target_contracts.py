"""Census pins for the target-schedule derivation (T1).

Every C-class schedule field (declared emitter contract, never computed) is
listed here with the value the promoted AMD artifact carries and a citation to
the emitter/lowering that implements it. The unit tests assert that the
promoted template, the capability rows, and this table agree, so the census in
``docs/task_workflow/input/target-schedule-derivation-scope-20260801.md``
becomes a code object instead of prose.

Status discipline: ``"declared"`` means the value names an existing emitter
contract for this target and the citation is the load-bearing part.
``"pending"`` means the value is carried so the v1 schema validates and
admission can run, but it is not yet measured for this target; a pending field
must never be sold as a proven fact. Only the AMD gfx1100 rows are fully
declared today; NV/Metal declare only what the renderer facts and the C1-C5
qualification runs prove.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ScheduleContract:
  """One declared C-class field: its value, the citation, and its honesty status."""
  value: Any
  citation: str
  status: str

  def __post_init__(self) -> None:
    if not isinstance(self.citation, str) or not self.citation:
      raise ValueError("contract citation must be a non-empty string")
    if self.status not in ("declared", "pending"):
      raise ValueError("contract status must be 'declared' or 'pending'")


# Field keys are the dotted schedule paths of the v1 payload. Values are the
# exact promoted-template literals; the tests walk them against the artifact.
TARGET_SCHEDULE_CONTRACTS: dict[tuple[str, str], dict[str, ScheduleContract]] = {
  ("AMD", "gfx1100"): {
    "lane_ownership": ScheduleContract("rdna3_wmma_f32_16x16x16_f16_lds2_static",
      "cstyle.py rdna3 packed-WMMA fragment layout used by the prefill WMMA-LDS generator; coincides "
      "with wmma.fragment_layout in the LDS family (the five-buffer row proves the fields are distinct)", "declared"),
    "cooperative_load.lane_mapping": ScheduleContract("cooperative_row_stride_64_b128",
      "kernel_lds.py cooperative b128 row-stride load contract; the string admission's capability_lane_map "
      "gate reads", "declared"),
    "lds.banks": ScheduleContract(32,
      "HIPRenderer.lds_bank_dwords = 32 (cstyle.py): 32 dword-wide LDS banks, ISA-documented", "declared"),
    "lds.padding": ScheduleContract(16,
      "kernel_lds.py b128 cooperative-load transport contract: row stride = tk*itemsize + 16 keeps every "
      "row 16-byte aligned", "declared"),
    "pipeline.epoch_graph": ScheduleContract(
      [{"epoch": "body", "slot": 0, "produce": ["a", "b"], "wait": ["global", "lds"],
        "barrier": "before_fragment_load", "consume": ["a", "b"]}],
      "KernelStage1PipelinePlan slot/epoch contract: produce a,b; wait global+lds; barrier "
      "before_fragment_load; consume a,b", "declared"),
    "wmma.fragment_layout": ScheduleContract("rdna3_wmma_f32_16x16x16_f16_lds2_static",
      "cstyle.py rdna3 packed-WMMA branch (admission capability_tc)", "declared"),
    "wmma.accumulator_ownership": ScheduleContract("wmma_accum_wm_x_wn_8_vgprs",
      "the WMMA accumulator layout the prefill generator emits: wm x wn tile, 8 vgprs per lane", "declared"),
    "dependency_policy.waitcnt": ScheduleContract({"vm": 0, "lgkm": 0},
      "AMD s_waitcnt vocabulary: vmcnt 0 / lgkmcnt 0 before fragment load; both counters are AMD-only", "declared"),
    "dependency_policy.barriers": ScheduleContract(["before_fragment_load", "after_wmma_before_slot_reuse"],
      "KernelStage1PipelinePlan: the barrier names the prefill pipeline emits between epoch stages", "declared"),
    "epilogue.lane_mapping": ScheduleContract("wmma_accumulator_scalar_b16",
      "b16 accumulator scalar epilogue contract (cstyle.py rdna3 epilogue)", "declared"),
    "epilogue.vector_width": ScheduleContract(1,
      "cstyle.py rdna3 scalar epilogue: one fp32 value per lane", "declared"),
    "residency.preload": ScheduleContract(["a", "b"],
      "promoted template policy: both operands are preloaded into LDS before the fragment loop "
      "(KernelStage1PipelinePlan)", "declared"),
    "residency.resident": ScheduleContract(["accumulator"],
      "promoted template policy: the accumulator stays register-resident across slots "
      "(KernelStage1PipelinePlan)", "declared"),
    "residency.reuse": ScheduleContract({"a": 4, "b": 2},
      "promoted template policy at tile 128x128x32 / waves 4x2: a reused 4x, b reused 2x per K window", "declared"),
    "numerical_mode": ScheduleContract("ieee_fp16_acc_fp32",
      "fp16 operands with IEEE fp32 accumulation, matching the (half, float) tc descriptor the rows derive", "declared"),
    "static_constraints.max_vgpr_per_thread": ScheduleContract(256,
      "AMD RDNA3 wave32 ISA register file: 256 dwords per thread (hardware fact)", "declared"),
    "static_constraints.allow_spill": ScheduleContract(False,
      "the prefill schedule generator emits spill-free kernels (REGALLOC contract)", "declared"),
  },
  ("CUDA", "sm120"): {
    "lane_ownership": ScheduleContract("cuda_mma_f32_8x16x16_f16_lds2_static",
      "LDS-family coincidence with wmma.fragment_layout (distinct in the five-buffer row, so the "
      "coincidence is preserved, not assumed)", "declared"),
    "cooperative_load.lane_mapping": ScheduleContract("cooperative_row_stride_64_b128",
      "carried b128 row-stride load pattern so the mint admits; not yet measured on sm120", "pending"),
    "lds.banks": ScheduleContract(32,
      "carried AMD-shaped value; CUDARenderer declares no lds_bank_dwords", "pending"),
    "lds.padding": ScheduleContract(16,
      "assumed b128 alignment; not yet measured on sm120", "pending"),
    "pipeline.epoch_graph": ScheduleContract(
      [{"epoch": "body", "slot": 0, "produce": ["a", "b"], "wait": ["global", "lds"],
        "barrier": "before_fragment_load", "consume": ["a", "b"]}],
      "carried pipeline-stage contract; not yet measured on sm120", "pending"),
    "wmma.fragment_layout": ScheduleContract("cuda_mma_f32_8x16x16_f16_lds2_static",
      "cuda.py mma.sync lowering for the (8,16,16) half->float descriptor (admission capability_tc)", "declared"),
    "wmma.accumulator_ownership": ScheduleContract("wmma_accum_wm_x_wn_8_vgprs",
      "carried AMD-shaped accumulator vocabulary; NV accumulator layout not yet measured", "pending"),
    "dependency_policy.waitcnt": ScheduleContract({"vm": None, "lgkm": None},
      "no AMD counter vocabulary on CUDA; carried null until an NV dependency policy is measured", "pending"),
    "dependency_policy.barriers": ScheduleContract(["before_fragment_load", "after_wmma_before_slot_reuse"],
      "carried barrier names; NV barrier policy not yet measured", "pending"),
    "epilogue.lane_mapping": ScheduleContract("wmma_accumulator_scalar_b16",
      "carried epilogue vocabulary; NV epilogue not yet measured", "pending"),
    "epilogue.vector_width": ScheduleContract(1,
      "carried scalar epilogue; NV epilogue not yet measured", "pending"),
    "residency.preload": ScheduleContract(["a", "b"],
      "carried preload policy; NV residency not yet measured", "pending"),
    "residency.resident": ScheduleContract(["accumulator"],
      "carried residency policy; NV residency not yet measured", "pending"),
    "residency.reuse": ScheduleContract({"a": 4, "b": 2},
      "carried AMD-geometry reuse; NV reuse not yet measured", "pending"),
    "numerical_mode": ScheduleContract("ieee_fp16_acc_fp32",
      "IEEE fp32 accumulation is the CUDA mma.sync default; unmeasured in this repo's NV lowering", "pending"),
    "static_constraints.max_vgpr_per_thread": ScheduleContract(256,
      "carried AMD value; NV per-thread register bound unmeasured", "pending"),
    "static_constraints.allow_spill": ScheduleContract(False,
      "carried spill policy; NV lowering unmeasured", "pending"),
  },
  ("Metal", "m4_10c"): {
    "lane_ownership": ScheduleContract("metal_simdgroup_matrix_f32_8x8x8_f16_lds2_static",
      "LDS-family coincidence with wmma.fragment_layout (distinct in the five-buffer row, so the "
      "coincidence is preserved, not assumed)", "declared"),
    "cooperative_load.lane_mapping": ScheduleContract("cooperative_row_stride_64_b128",
      "the b128 row-stride load contract the M1b/M1c/M1d Metal dispatch compiled and ran with", "declared"),
    "lds.banks": ScheduleContract(32,
      "carried AMD-shaped value; MetalRenderer declares no lds_bank_dwords (cstyle.py)", "pending"),
    "lds.padding": ScheduleContract(16,
      "stride contract carried by the M1b/M1c/M1d Metal dispatch (stride 80); bank behavior unmeasured", "pending"),
    "pipeline.epoch_graph": ScheduleContract(
      [{"epoch": "body", "slot": 0, "produce": ["a", "b"], "wait": ["global", "lds"],
        "barrier": "before_fragment_load", "consume": ["a", "b"]}],
      "carried pipeline-stage contract; Metal stage barriers not yet measured", "pending"),
    "wmma.fragment_layout": ScheduleContract("metal_simdgroup_matrix_f32_8x8x8_f16_lds2_static",
      "cstyle.py simdgroup_multiply_accumulate lowering for the (8,8,8) half->float descriptor "
      "(admission capability_tc)", "declared"),
    "wmma.accumulator_ownership": ScheduleContract("wmma_accum_wm_x_wn_8_vgprs",
      "carried AMD-shaped accumulator vocabulary; Metal accumulator layout not yet measured", "pending"),
    "dependency_policy.waitcnt": ScheduleContract({"vm": None, "lgkm": None},
      "no AMD counter vocabulary on Metal; carried null until a Metal dependency policy is measured", "pending"),
    "dependency_policy.barriers": ScheduleContract(["before_fragment_load", "after_wmma_before_slot_reuse"],
      "carried barrier names; Metal barrier policy not yet measured", "pending"),
    "epilogue.lane_mapping": ScheduleContract("wmma_accumulator_scalar_b16",
      "carried epilogue vocabulary; Metal epilogue not yet measured", "pending"),
    "epilogue.vector_width": ScheduleContract(1,
      "carried scalar epilogue; Metal epilogue not yet measured", "pending"),
    "residency.preload": ScheduleContract(["a", "b"],
      "carried preload policy; Metal residency not yet measured", "pending"),
    "residency.resident": ScheduleContract(["accumulator"],
      "carried residency policy; Metal residency not yet measured", "pending"),
    "residency.reuse": ScheduleContract({"a": 4, "b": 2},
      "carried AMD-geometry reuse; Metal reuse not yet measured", "pending"),
    "numerical_mode": ScheduleContract("ieee_fp16_acc_fp32",
      "carried numerical mode; Metal accumulation mode not yet measured", "pending"),
    "static_constraints.max_vgpr_per_thread": ScheduleContract(256,
      "carried AMD value; Metal per-thread register bound unmeasured", "pending"),
    "static_constraints.allow_spill": ScheduleContract(False,
      "carried spill policy; Metal lowering unmeasured", "pending"),
  },
}

# Every C-class field in the scope's census. A field missing from this set is a
# census regression; a field with no row entry for a target is an undeclared
# target.
CONTRACT_FIELD_KEYS = tuple(next(iter(TARGET_SCHEDULE_CONTRACTS.values())).keys())


def contract_row(backend: str, arch: str) -> dict[str, ScheduleContract]:
  """The declared contract row for a target; undeclared targets fail closed."""
  try:
    return TARGET_SCHEDULE_CONTRACTS[(backend, arch)]
  except KeyError:
    raise ValueError(f"no declared schedule contracts for target {backend}:{arch}") from None
