#!/usr/bin/env python3
"""Static coalescing predicate -- the SEED of the layout/mapping IR (docs/layout-mapping-ir-design-20260625.md).

Promoted to core codegen under LR-050 (docs/task_workflow/input/lowering-architecture-refactor-scope-20260726.md
Phase 5): `axis_stride`/`is_coalesced`/`vector_width` are pure symbolic-index-algebra predicates with no
backend-specific assumptions, so they now live in `tinygrad/codegen/late/coalesced_load.py`. This module re-exports
them rather than forking the implementation, so `extra.qk.bubblebeam_futuresight` and
`extra.qk.coalesced_load_lowering` keep working unchanged.
"""
from __future__ import annotations
from tinygrad.codegen.late.coalesced_load import axis_stride as axis_stride, is_coalesced as is_coalesced, \
  vector_width as vector_width
