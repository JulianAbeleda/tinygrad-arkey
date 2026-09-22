#!/usr/bin/env python3
"""Alias layer over the canonical flash-decode descriptor module.

P1 (docs/task_workflow/input/nv-search-genericization-flash-shape-scope-20260818.md): the production module
`tinygrad.llm.flash_decode_attention` is the single canonical owner of the flash geometry descriptors
(FlashDecodeTileSpec/FlashCombineSpec/FlashDecodeAttentionSpec/LiveSplitGeometrySpec and the describe/emit
helpers). This module previously redefined all of them against `extra/llm_research/flash_kernels.py`; it now
exists only so existing research imports (test_flash_buffer_roles.py, test_flash_decode_attention_spec.py,
decode_hd_sweep_numerics.py, decode_codegen_identity_check.py, extra/audit/lowering_baseline.py) keep
resolving. It adds no behavior and imports nothing from `extra` itself, so the canonical module never has a
research dependency (see test_production_module_has_no_research_import).
"""
from tinygrad.llm.flash_decode_attention import (BufferRole, FlashCombineSpec, FlashDecodeAttentionSpec,
                                                 FlashDecodeTileSpec, LiveSplitGeometrySpec,
                                                 describe_flash_decode_attention, emit_flash_decode_combine,
                                                 emit_flash_decode_tile)

__all__ = ["BufferRole", "FlashCombineSpec", "FlashDecodeAttentionSpec", "FlashDecodeTileSpec",
           "LiveSplitGeometrySpec", "describe_flash_decode_attention", "emit_flash_decode_combine",
           "emit_flash_decode_tile"]
