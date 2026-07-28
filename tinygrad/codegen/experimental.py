from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)


# NOTE: coalesce_loads was promoted to core codegen (LR-050) -- tinygrad/codegen/__init__.py now imports
# tinygrad.codegen.late.coalesced_load directly instead of going through this lazy extra.qk adapter.
def warp_reduce_pm(): return _attr("extra.qk.warp_reduce_lowering", "pm_warp_reduce")
def reg_store_devec_pm(): return _attr("extra.qk.reg_store_devec", "pm_reg_store_devec")
def fdot2_pm(): return _attr("extra.qk.fdot2_lowering", "pm_fdot2")
def line_lower_fdot2(*args, **kwargs): return _attr("extra.qk.fdot2_lowering", "line_lower_fdot2")(*args, **kwargs)
def lower_fdot2_add(*args, **kwargs): return _attr("extra.qk.fdot2_lowering", "lower_fdot2_add")(*args, **kwargs)
def list_schedule(*args, **kwargs): return _attr("extra.qk.codegen_list_scheduler", "list_schedule")(*args, **kwargs)
def structural_ops(): return _attr("extra.qk.codegen_list_scheduler", "_STRUCTURAL")
