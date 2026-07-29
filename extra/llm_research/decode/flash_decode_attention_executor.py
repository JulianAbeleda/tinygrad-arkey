#!/usr/bin/env python3
"""Compatibility import for the promoted production flash-decode executor.

Research tools historically imported this path. Qualification must execute the
same implementation inference uses, so the implementation now lives solely in
``tinygrad.llm.flash_decode_attention``. The sibling EXP descriptor remains an
explicit parity oracle and is not reached through this adapter.
"""
from tinygrad.llm.flash_decode_attention import flash_decode_live_split_block_tile

__all__ = ["flash_decode_live_split_block_tile"]
