"""Research-only AMD ISA proof metadata and manifest lifecycle.

Importing this module installs a passive renderer hook. The production AMD ISA
renderer neither imports this module nor enables proof collection by default.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from tinygrad.helpers import getenv
from tinygrad.renderer.isa.amd import install_amd_isa_proof_hook

AMD_ISA_PROOF_MANIFEST:list[dict] = []
_CAPTURE:ContextVar[tuple[list[dict], int]|None] = ContextVar("amd_isa_proof_capture", default=None)
OPERAND_PATH_TAG = "amd_operand_path"
OPERAND_PATH_FIELDS = frozenset(("operand_id", "source_operand_id", "fetch_group", "cache_policy", "width_bytes",
                                 "vector_width_bytes", "retained_fragment", "semantic_owner", "semantic_ownership"))

def reset_amd_isa_proof_manifest() -> None: AMD_ISA_PROOF_MANIFEST.clear()
def amd_isa_proof_manifest() -> tuple[dict, ...]: return tuple(AMD_ISA_PROOF_MANIFEST)

@contextmanager
def capture_amd_isa_proof_manifest(*, max_rows:int=4096):
  if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 0:
    raise ValueError("max_rows must be a non-negative integer")
  rows:list[dict] = []
  token = _CAPTURE.set((rows, max_rows))
  try: yield rows
  finally: _CAPTURE.reset(token)

def _freeze(value):
  if isinstance(value, dict): return frozenset((key, _freeze(item)) for key, item in value.items())
  if isinstance(value, list): return tuple(_freeze(item) for item in value)
  return value

def _thaw(value):
  if isinstance(value, frozenset): return {key: _thaw(item) for key, item in value}
  if isinstance(value, tuple): return tuple(_thaw(item) for item in value)
  return value

def amd_isa_operand_path_tag(tag, **metadata):
  unknown = set(metadata) - OPERAND_PATH_FIELDS
  if unknown: raise ValueError(f"unknown AMD ISA operand-path metadata: {sorted(unknown)}")
  payload = tuple(sorted((key, _freeze(value)) for key, value in metadata.items()))
  return (tag, (OPERAND_PATH_TAG, payload)) if not isinstance(tag, tuple) else tag + ((OPERAND_PATH_TAG, payload),)

def _operand_path_meta(tag) -> dict:
  for candidate in tag if isinstance(tag, tuple) else (tag,):
    if isinstance(candidate, tuple) and len(candidate) == 2 and candidate[0] == OPERAND_PATH_TAG:
      raw = candidate[1]
      if isinstance(raw, tuple):
        try: raw = dict(raw)
        except (TypeError, ValueError): return {}
      return {key: _thaw(value) for key, value in raw.items() if key in OPERAND_PATH_FIELDS} if isinstance(raw, dict) else {}
    if isinstance(candidate, dict) and OPERAND_PATH_TAG in candidate:
      raw = candidate[OPERAND_PATH_TAG]
      return {key: _thaw(value) for key, value in raw.items() if key in OPERAND_PATH_FIELDS} if isinstance(raw, dict) else {}
  return {}

def _register_index(value) -> int|None:
  index = getattr(getattr(value, "reg", None), "index", None)
  return index if isinstance(index, int) and not isinstance(index, bool) else None

def _store_owner_meta(tag) -> dict:
  if isinstance(tag, frozenset):
    try: tag = dict(tag)
    except Exception: pass
  if isinstance(tag, dict) and "store_owner" in tag: owner = tag["store_owner"]
  elif isinstance(tag, tuple) and len(tag) >= 2 and tag[0] == "store_owner": owner = tag[1]
  else: return {}
  if isinstance(owner, tuple):
    try: owner = dict(owner)
    except Exception: pass
  return {"store_owner": dict(owner)} if isinstance(owner, dict) else {"store_owner": owner}

class _ProofHook:
  @staticmethod
  def enabled() -> bool: return _CAPTURE.get() is not None or bool(getenv("AMD_ISA_PROOF_MANIFEST", 0))
  @staticmethod
  def append(row:dict) -> None:
    if (capture := _CAPTURE.get()) is None: AMD_ISA_PROOF_MANIFEST.append(row)
    else:
      rows, limit = capture
      if len(rows) >= limit: raise ValueError(f"AMD ISA proof exceeds max_rows={limit}")
      rows.append(row)
  def record(self, kind, x, inst, extra=None) -> None:
    if not self.enabled(): return
    row = {"schema": "amd-isa-renderer-proof-manifest-row.v1", "kind": kind,
           "logical_op": x.arg.name if hasattr(x.arg, "name") else str(x.arg), "emitted": str(inst),
           "dest_reg": _register_index(x), "source_regs": [_register_index(s) for s in x.src], **_operand_path_meta(x.tag)}
    if extra is not None: row.update(extra)
    self.append(row)
  def record_inst(self, kind, logical_op, inst, extra=None) -> None:
    if not self.enabled(): return
    row = {"schema": "amd-isa-renderer-proof-manifest-row.v1", "kind": kind, "logical_op": logical_op, "emitted": str(inst)}
    if extra is not None: row.update(extra)
    self.append(row)
  @staticmethod
  def carrier_meta(u) -> dict:
    if u is not None and isinstance(u.arg, tuple) and u.arg[:1] == ("wmma_acc",):
      return {"carrier_kind": "wmma_acc", "define_reg_id": u.arg[1], "subtile": u.arg[2], "element": u.arg[3], "physical_vgpr": u.arg[4]}
    return {}
  @staticmethod
  def store_owner_tag(x):
    if isinstance(x.arg, tuple) and len(x.arg) >= 2 and x.arg[0] == "store_owner":
      owner = tuple(sorted(x.arg[1].items())) if isinstance(x.arg[1], dict) else x.arg[1]
      return frozenset((("store_owner", owner),))
    return x.tag
  @staticmethod
  def store_owner_meta(tag) -> dict: return _store_owner_meta(tag)

install_amd_isa_proof_hook(_ProofHook())

__all__ = ["amd_isa_operand_path_tag", "amd_isa_proof_manifest", "capture_amd_isa_proof_manifest", "reset_amd_isa_proof_manifest"]
