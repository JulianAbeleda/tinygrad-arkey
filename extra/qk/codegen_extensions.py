from __future__ import annotations
from dataclasses import dataclass

from tinygrad.codegen.opt.extensions import CodegenExtensionRegistry
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import getenv
from tinygrad.renderer.isa.extensions import AMDISARendererExtensionDescriptor
from tinygrad.uop.ops import Ops, UOp
from extra.qk.amd_isa_renderer_policy import PREFILL_AMD_ISA_RENDERER_POLICY


@dataclass(frozen=True)
class PrefillDevectorizerExtension:
  name: str = "prefill"
  def disables_ptr_group(self, buf:UOp) -> bool:
    return buf.addrspace == AddrSpace.LOCAL and (
      (getenv("PREFILL_DBUF_D3A_POST", 0) and buf.op is Ops.DEFINE_LOCAL and buf.arg in (990, 991, 993)) or
      (getenv("PREFILL_TC_LOCAL_STAGE_B_TILEKEY", 0) and buf.op is Ops.DEFINE_LOCAL and buf.arg in (991, 993))
    )
  def preserves_stage_tag(self, uop:UOp) -> bool: return False
  def preserves_wmma_proof_tag(self, uop:UOp) -> bool: return False

@dataclass(frozen=True)
class PrefillPostRangeExtension:
  name: str = "prefill"
  def tc_local_stage_mode(self) -> str:
    return str(getenv("PREFILL_TC_LOCAL_STAGE", "")).strip().lower()
  def tc_local_stage_with_planned_local(self) -> bool:
    return bool(getenv("PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", 0))
  def tc_local_stage_post_opt(self) -> bool:
    return bool(getenv("PREFILL_TC_LOCAL_STAGE_POST", 0))
  def prefill_dbuf_lds_addr_serial(self, enabled:bool) -> bool:
    return bool(enabled and getenv("PREFILL_DBUF_LDS_ADDR_SERIAL", 0))
  def tc_local_stage_owned_stage_meta(self, operand_idx:int) -> bool:
    return bool(getenv("PREFILL_DBUF_OWNED_AB_STAGE_META", 0) or
                getenv("PREFILL_DBUF_OWNED_A_STAGE_META" if operand_idx == 0 else "PREFILL_DBUF_OWNED_B_STAGE_META", 0))
  def tc_local_stage_owned_stage_emit_mode(self, operand_idx:int) -> str:
    return str(getenv("PREFILL_DBUF_OWNED_A_STAGE_EMIT" if operand_idx == 0 else "PREFILL_DBUF_OWNED_B_STAGE_EMIT", "")).strip().lower()
  def tc_local_stage_pipe_primitive_disabled_for_ranges(self, stage_ranges:tuple[UOp, ...]) -> bool:
    if not getenv("PREFILL_WMMA_PIPE_PRIMITIVE", 0): return False
    if str(getenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "1")).strip().lower() in ("", "0", "false", "off", "no"): return False
    dims = {r.vmax + 1 for r in stage_ranges}
    return bool(dims & {1024, 4096}) and 12288 not in dims
  def warmstart_pipe_primitive_no_local_stage_key(self, key:tuple[frozenset[int], int]) -> bool:
    if not getenv("PREFILL_WMMA_PIPE_PRIMITIVE", 0): return False
    if str(getenv("PREFILL_WMMA_PIPE_ATTN_KV_NO_LOCAL_STAGE", "1")).strip().lower() in ("", "0", "false", "off", "no"): return False
    out_dims, red = key
    pipe_ns = {1024, 4096}
    return (512 in out_dims and bool(out_dims & pipe_ns) and red in (4096, 12288)) or (bool(out_dims & pipe_ns) and red == 1)
  def warmstart_local_stage_allowed_key(self, key:tuple[frozenset[int], int], local_stage_keys,
                                        local_stage_deny_keys:set[tuple[frozenset[int], int]]) -> bool:
    return (not self.warmstart_pipe_primitive_no_local_stage_key(key)) and key not in local_stage_deny_keys and (
      local_stage_keys is None or key in local_stage_keys)
  def prefill_dbuf_peel_allowed(self, has_tensor_core_opt:bool, has_wmma:bool) -> bool:
    return has_tensor_core_opt or has_wmma


PREFILL_DEVECTORIZER_EXTENSION = PrefillDevectorizerExtension()
PREFILL_POSTRANGE_EXTENSION = PrefillPostRangeExtension()

def codegen_extension_registry(default:CodegenExtensionRegistry) -> CodegenExtensionRegistry:
  return CodegenExtensionRegistry(postrange=default.postrange+(PREFILL_POSTRANGE_EXTENSION,),
                                  devectorizer=default.devectorizer+(PREFILL_DEVECTORIZER_EXTENSION,))

PREFILL_AMD_ISA_RENDERER_EXTENSION = AMDISARendererExtensionDescriptor(
  "prefill", proof_tags=("wmma_frag_proof", "wmma_frag_buffer_proof"),
  local_buffer_ids=frozenset((990, 991, 993)),
  machine_search_hooks=("dbuf_d3a_stage", "wmma_kmajor_phase", "wmma_kmajor_stage_steal"),
  renderer_policy=PREFILL_AMD_ISA_RENDERER_POLICY)

def amd_isa_extension_descriptors(default:tuple[AMDISARendererExtensionDescriptor, ...]) -> tuple[AMDISARendererExtensionDescriptor, ...]:
  return default + (PREFILL_AMD_ISA_RENDERER_EXTENSION,)
