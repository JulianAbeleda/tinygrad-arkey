"""Bounded host-side search manifest for non-LDS prefill roles.

This describes the existing ``prefill_v2_schedule_search`` surface only; it
does not admit candidates or alter runtime defaults.
"""
from dataclasses import dataclass
from extra.qk.model_profiles import qwen3_8b_q4k_m_gfx1100_profile

@dataclass(frozen=True)
class NonLDSRoleSearch:
  role: str
  shape: tuple[int, int, int]
  knobs: tuple[str, ...] = ("UPCAST_M", "UPCAST_N", "LOCAL", "UNROLL")

def qwen3_8b_nonlds_searches():
  p = qwen3_8b_q4k_m_gfx1100_profile()
  return tuple(NonLDSRoleSearch(r.role, (512, r.N, r.K)) for r in p.roles if r.role in {"attn_qo", "ffn_down", "attn_kv"})

def search_command(scope: NonLDSRoleSearch) -> str:
  m, n, k = scope.shape
  return ("PYTHONPATH=. DEV=AMD TC=1 ALLOW_DEVICE_USAGE=1 "
          "python3 extra/qk/prefill_v2_schedule_search.py "
          f"--shapes {n},{k} --out bench/prefill-nonlds-search/{scope.role}.json")
