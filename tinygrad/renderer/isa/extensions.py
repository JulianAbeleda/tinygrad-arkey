from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from tinygrad.codegen import experimental

@dataclass(frozen=True)
class AMDISARendererExtensionDescriptor:
  name: str
  proof_tags: tuple[str, ...] = field(default_factory=tuple)
  local_buffer_ids: frozenset[int] = field(default_factory=frozenset)
  machine_search_hooks: tuple[str, ...] = field(default_factory=tuple)
  renderer_policy: Any|None = None

DEFAULT_AMD_ISA_EXTENSION_DESCRIPTORS: tuple[AMDISARendererExtensionDescriptor, ...] = ()

def get_amd_isa_extension_descriptors() -> tuple[AMDISARendererExtensionDescriptor, ...]:
  return experimental.amd_isa_extension_descriptors(DEFAULT_AMD_ISA_EXTENSION_DESCRIPTORS)
