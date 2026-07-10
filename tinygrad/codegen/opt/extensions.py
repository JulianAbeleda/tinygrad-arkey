from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol

from tinygrad.codegen import experimental
from tinygrad.uop.ops import UOp

class PostRangeExtension(Protocol):
  name: str
  def tc_local_stage_mode(self) -> str: ...
  def tc_local_stage_with_planned_local(self) -> bool: ...
  def tc_local_stage_post_opt(self) -> bool: ...
  def prefill_dbuf_lds_addr_serial(self, enabled:bool) -> bool: ...
  def tc_local_stage_owned_stage_meta(self, operand_idx:int) -> bool: ...
  def tc_local_stage_owned_stage_emit_mode(self, operand_idx:int) -> str: ...
  def tc_local_stage_pipe_primitive_disabled_for_ranges(self, stage_ranges:tuple[UOp, ...]) -> bool: ...
  def warmstart_pipe_primitive_no_local_stage_key(self, key:tuple[frozenset[int], int]) -> bool: ...
  def warmstart_local_stage_allowed_key(self, key:tuple[frozenset[int], int], local_stage_keys:Any,
                                        local_stage_deny_keys:set[tuple[frozenset[int], int]]) -> bool: ...
  def prefill_dbuf_peel_allowed(self, has_tensor_core_opt:bool, has_wmma:bool) -> bool: ...

class DevectorizerExtension(Protocol):
  name: str
  def disables_ptr_group(self, buf:UOp) -> bool: ...
  def preserves_stage_tag(self, uop:UOp) -> bool: ...
  def preserves_wmma_proof_tag(self, uop:UOp) -> bool: ...

@dataclass(frozen=True)
class EmptyDevectorizerExtension:
  name: str = "empty"
  def disables_ptr_group(self, buf:UOp) -> bool: return False
  def preserves_stage_tag(self, uop:UOp) -> bool: return False
  def preserves_wmma_proof_tag(self, uop:UOp) -> bool: return False

@dataclass(frozen=True)
class EmptyPostRangeExtension:
  name: str = "empty"
  def tc_local_stage_mode(self) -> str: return ""
  def tc_local_stage_with_planned_local(self) -> bool: return False
  def tc_local_stage_post_opt(self) -> bool: return False
  def prefill_dbuf_lds_addr_serial(self, enabled:bool) -> bool: return False
  def tc_local_stage_owned_stage_meta(self, operand_idx:int) -> bool: return False
  def tc_local_stage_owned_stage_emit_mode(self, operand_idx:int) -> str: return ""
  def tc_local_stage_pipe_primitive_disabled_for_ranges(self, stage_ranges:tuple[UOp, ...]) -> bool: return False
  def warmstart_pipe_primitive_no_local_stage_key(self, key:tuple[frozenset[int], int]) -> bool: return False
  def warmstart_local_stage_allowed_key(self, key:tuple[frozenset[int], int], local_stage_keys:Any,
                                        local_stage_deny_keys:set[tuple[frozenset[int], int]]) -> bool:
    return local_stage_keys is None or key in local_stage_keys
  def prefill_dbuf_peel_allowed(self, has_tensor_core_opt:bool, has_wmma:bool) -> bool: return has_tensor_core_opt or has_wmma

@dataclass(frozen=True)
class CodegenExtensionRegistry:
  postrange: tuple[PostRangeExtension, ...] = field(default_factory=tuple)
  devectorizer: tuple[DevectorizerExtension, ...] = field(default_factory=tuple)

  def _postrange_ext(self) -> PostRangeExtension:
    return self.postrange[0] if self.postrange else EMPTY_POSTRANGE_EXTENSION

  def tc_local_stage_mode(self) -> str: return self._postrange_ext().tc_local_stage_mode()
  def tc_local_stage_with_planned_local(self) -> bool: return self._postrange_ext().tc_local_stage_with_planned_local()
  def tc_local_stage_post_opt(self) -> bool: return self._postrange_ext().tc_local_stage_post_opt()
  def prefill_dbuf_lds_addr_serial(self, enabled:bool) -> bool: return self._postrange_ext().prefill_dbuf_lds_addr_serial(enabled)
  def tc_local_stage_owned_stage_meta(self, operand_idx:int) -> bool: return self._postrange_ext().tc_local_stage_owned_stage_meta(operand_idx)
  def tc_local_stage_owned_stage_emit_mode(self, operand_idx:int) -> str: return self._postrange_ext().tc_local_stage_owned_stage_emit_mode(operand_idx)
  def tc_local_stage_pipe_primitive_disabled_for_ranges(self, stage_ranges:tuple[UOp, ...]) -> bool:
    return self._postrange_ext().tc_local_stage_pipe_primitive_disabled_for_ranges(stage_ranges)
  def warmstart_pipe_primitive_no_local_stage_key(self, key:tuple[frozenset[int], int]) -> bool:
    return self._postrange_ext().warmstart_pipe_primitive_no_local_stage_key(key)
  def warmstart_local_stage_allowed_key(self, key:tuple[frozenset[int], int], local_stage_keys:Any,
                                        local_stage_deny_keys:set[tuple[frozenset[int], int]]) -> bool:
    return self._postrange_ext().warmstart_local_stage_allowed_key(key, local_stage_keys, local_stage_deny_keys)
  def prefill_dbuf_peel_allowed(self, has_tensor_core_opt:bool, has_wmma:bool) -> bool:
    return self._postrange_ext().prefill_dbuf_peel_allowed(has_tensor_core_opt, has_wmma)

  def disables_ptr_group(self, buf:UOp) -> bool:
    return any(ext.disables_ptr_group(buf) for ext in self.devectorizer)

  def preserves_stage_tag(self, uop:UOp) -> bool:
    return any(ext.preserves_stage_tag(uop) for ext in self.devectorizer)

  def preserves_wmma_proof_tag(self, uop:UOp) -> bool:
    return any(ext.preserves_wmma_proof_tag(uop) for ext in self.devectorizer)

EMPTY_DEVECTORIZER_EXTENSION = EmptyDevectorizerExtension()
EMPTY_POSTRANGE_EXTENSION = EmptyPostRangeExtension()
DEFAULT_CODEGEN_EXTENSION_REGISTRY = CodegenExtensionRegistry()

def get_codegen_extension_registry() -> CodegenExtensionRegistry:
  return experimental.codegen_extension_registry(DEFAULT_CODEGEN_EXTENSION_REGISTRY)
