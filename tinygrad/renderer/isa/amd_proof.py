from __future__ import annotations

_hook = None

def install_amd_isa_proof_hook(hook) -> None:
  global _hook
  _hook = hook

def _proof_record(kind:str, x, inst, extra:dict|None=None) -> None:
  if _hook is not None: _hook.record(kind, x, inst, extra)

def _proof_record_inst(kind:str, logical_op:str, inst, extra:dict|None=None) -> None:
  if _hook is not None: _hook.record_inst(kind, logical_op, inst, extra)

def _proof_carrier_meta(u) -> dict:
  return {} if _hook is None else _hook.carrier_meta(u)

def _store_owner_tag_from_store_arg(x):
  return x.tag if _hook is None else _hook.store_owner_tag(x)

def _store_owner_meta_from_ins(x) -> dict:
  return {} if _hook is None else _hook.store_owner_meta(x.tag)
