#!/usr/bin/env python3
"""Coalesced vector-load lowering -- the bandwidth PRIMITIVE for generated kernels.

Promoted to core codegen under LR-050 (docs/task_workflow/input/lowering-architecture-refactor-scope-20260726.md
Phase 5): the implementation now lives in `tinygrad/codegen/late/coalesced_load.py` (pure UOp/AxisType transform,
no backend-specific assumptions). This module re-exports `coalesce_loads` rather than forking it, so existing
`extra/qk` callers (and the historical `tinygrad.codegen.experimental` wrapper) keep working unchanged. The
opt-in `COALESCED_LOAD_LOWERING` gate and the `ren.target.device == "AMD"` validation-scope restriction still
live at the call site in `tinygrad/codegen/__init__.py`, which now imports the core pass directly.
"""
from __future__ import annotations
from tinygrad.codegen.late.coalesced_load import coalesce_loads as coalesce_loads
