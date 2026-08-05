"""Research import surface for the closed-default production candidate.

The implementation lives in ``tinygrad.llm.q4k_ffn_down_mmvq`` so the
production-model qualification and the isolated microgate compile identical
UOps. This module intentionally adds no route selector.
"""
from tinygrad.llm.q4k_ffn_down_mmvq import *  # noqa: F401,F403

