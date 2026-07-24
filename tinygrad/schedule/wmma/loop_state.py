"""Centralized loop-state read/write and packed fragment loader primitives.

Step 1 (this module) is deliberately left empty: the wr/rd/fr/state_write/
state_read/fragment closures currently duplicated across the five kernels in
``wmma/kernels.py`` are NOT yet centralized here -- that is Step 2 of the
wmma modularization scope (docs/wmma-modularization-scope-20260724.md). This
module exists now only so the package layout and import DAG (loop_state is a
leaf that fragments/softmax/kernels may depend on) are already in place.
"""
from __future__ import annotations
