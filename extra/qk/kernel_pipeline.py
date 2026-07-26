"""Typed plans and ownership proofs for staged kernel pipelines.

Promoted to core codegen under LR-050 (docs/task_workflow/input/lowering-architecture-refactor-scope-20260726.md
Phase 5): every contract that used to live in this file (the dot/update recurrence plan+proof, the hierarchical
produce/publish/consume/release lifecycle, and the scheduler output tile loop) is generic -- built only on core
UOp/AxisType/Ops concepts, with no wave width, ISA intrinsic, or device-specific assumption anywhere in them --
so it now lives in `tinygrad/codegen/opt/kernel_pipeline.py` alongside the resource/pipeline contract LR-021
harvested earlier. This module re-exports rather than forking, so existing callers (including
`test/unit/test_hierarchical_kernel_pipeline.py`, `test/unit/test_grouped_dot_update_pipeline.py`, and
`test/unit/test_scheduler_output_tile_loop.py`) keep working unchanged.
"""
from __future__ import annotations

from tinygrad.codegen.opt.kernel_pipeline import (
  AttachmentT as AttachmentT,
  HierarchicalLifetime as HierarchicalLifetime,
  HierarchicalOp as HierarchicalOp,
  DotUpdateRecurrencePlan as DotUpdateRecurrencePlan,
  DotUpdateAttachment as DotUpdateAttachment,
  DotUpdateGroupContext as DotUpdateGroupContext,
  DotUpdateGroupRecord as DotUpdateGroupRecord,
  DotUpdateRecurrenceGraph as DotUpdateRecurrenceGraph,
  DotUpdateRecurrenceProof as DotUpdateRecurrenceProof,
  prove_dot_update_recurrence as prove_dot_update_recurrence,
  build_dot_update_recurrence as build_dot_update_recurrence,
  HierarchicalPipelineRole as HierarchicalPipelineRole,
  HierarchicalKernelPipelinePlan as HierarchicalKernelPipelinePlan,
  HierarchicalLifecycleEvent as HierarchicalLifecycleEvent,
  HierarchicalLifecycleProof as HierarchicalLifecycleProof,
  hierarchical_lifecycle_events as hierarchical_lifecycle_events,
  prove_hierarchical_lifecycle as prove_hierarchical_lifecycle,
  SchedulerOutputTileLoop as SchedulerOutputTileLoop,
  SchedulerOutputTileIndices as SchedulerOutputTileIndices,
  build_scheduler_output_tile_owner as build_scheduler_output_tile_owner,
  build_scheduler_output_tile_loop as build_scheduler_output_tile_loop,
  PINNED_WMMA_VGPR_BUDGET as PINNED_WMMA_VGPR_BUDGET,
  resource_plan_for_scheduler_tile_loop as resource_plan_for_scheduler_tile_loop,
  validate_scheduler_tile_loop_pressure as validate_scheduler_tile_loop_pressure,
)
