from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)


# NOTE: coalesce_loads was promoted to core codegen (LR-050) -- tinygrad/codegen/__init__.py now imports
# tinygrad.codegen.late.coalesced_load directly instead of going through this lazy extra.qk adapter.
