"""Canonical logical-role vocabulary shared by LLM runtime and research tools."""
from __future__ import annotations

DENSE_PROJECTION_ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv", "lm_head")
PROGRAM_WORKLOAD_ROLES = (*DENSE_PROJECTION_ROLES, "attention", "generic")

_ROLE_ALIASES = {
  "ffn_gate": "ffn_gate_up", "ffn_up": "ffn_gate_up", "ffn_gate_up": "ffn_gate_up",
  "ffn_gate_shexp": "ffn_gate_up", "ffn_up_shexp": "ffn_gate_up",
  "ffn_down": "ffn_down", "ffn_down_shexp": "ffn_down",
  "attn_q": "attn_qo", "attn_output": "attn_qo", "attn_qo": "attn_qo",
  "attn_k": "attn_kv", "attn_v": "attn_kv", "attn_kv": "attn_kv",
  "output": "lm_head", "lm_head": "lm_head",
  "attention_tile": "attention", "attention_combine": "attention", "attention": "attention",
  # Legacy runtime-spec payloads used ``unknown`` as the wildcard/fallback role. Keep accepting it at this
  # normalization boundary, but only emit the canonical ``generic`` value.
  "unknown": "generic", "generic": "generic",
}

def normalize_program_role(role_or_name:str) -> str:
  value = str(role_or_name or "")
  leaf = value[:-len(".weight")] if value.endswith(".weight") else value
  leaf = leaf.rsplit(".", 1)[-1]
  return _ROLE_ALIASES.get(leaf, value)

__all__ = ["DENSE_PROJECTION_ROLES", "PROGRAM_WORKLOAD_ROLES", "normalize_program_role"]
