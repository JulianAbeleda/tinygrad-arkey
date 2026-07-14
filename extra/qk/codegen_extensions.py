from __future__ import annotations
from tinygrad.renderer.isa.extensions import AMDISARendererExtensionDescriptor
from extra.qk.amd_isa_renderer_policy import PREFILL_AMD_ISA_RENDERER_POLICY

PREFILL_AMD_ISA_RENDERER_EXTENSION = AMDISARendererExtensionDescriptor(
  "prefill", proof_tags=("wmma_frag_proof", "wmma_frag_buffer_proof"),
  local_buffer_ids=frozenset((990, 991, 993)),
  machine_search_hooks=("dbuf_d3a_stage", "wmma_kmajor_phase", "wmma_kmajor_stage_steal"),
  renderer_policy=PREFILL_AMD_ISA_RENDERER_POLICY)

def amd_isa_extension_descriptors(default:tuple[AMDISARendererExtensionDescriptor, ...]) -> tuple[AMDISARendererExtensionDescriptor, ...]:
  return default + (PREFILL_AMD_ISA_RENDERER_EXTENSION,)
