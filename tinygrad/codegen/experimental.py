from __future__ import annotations

import importlib
from functools import cache


@cache
def _attr(module:str, name:str):
  return getattr(importlib.import_module(module), name)


def unroll_recurrence(*args, **kwargs): return _attr("extra.qk.codegen_recurrence_unroll", "unroll_recurrence")(*args, **kwargs)
def outer_b_split(*args, **kwargs): return _attr("extra.qk.codegen_outer_b_lds_split", "outer_b_split")(*args, **kwargs)
def coalesce_loads(*args, **kwargs): return _attr("extra.qk.coalesced_load_lowering", "coalesce_loads")(*args, **kwargs)
def warp_reduce_pm(): return _attr("extra.qk.warp_reduce_lowering", "pm_warp_reduce")
def reg_store_devec_pm(): return _attr("extra.qk.reg_store_devec", "pm_reg_store_devec")
def fdot2_pm(): return _attr("extra.qk.fdot2_lowering", "pm_fdot2")
def line_lower_fdot2(*args, **kwargs): return _attr("extra.qk.fdot2_lowering", "line_lower_fdot2")(*args, **kwargs)
def lower_fdot2_add(*args, **kwargs): return _attr("extra.qk.fdot2_lowering", "lower_fdot2_add")(*args, **kwargs)
def list_schedule(*args, **kwargs): return _attr("extra.qk.codegen_list_scheduler", "list_schedule")(*args, **kwargs)
def structural_ops(): return _attr("extra.qk.codegen_list_scheduler", "_STRUCTURAL")
def amd_isa_extension_descriptors(default): return _attr("extra.qk.codegen_extensions", "amd_isa_extension_descriptors")(default)
