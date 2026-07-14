import pathlib, re

from tinygrad.renderer.isa.extensions import (
  AMDISARendererExtensionDescriptor, DEFAULT_AMD_ISA_EXTENSION_DESCRIPTORS, get_amd_isa_extension_descriptors,
)
from extra.qk.codegen_extensions import PREFILL_AMD_ISA_RENDERER_EXTENSION

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+extra\.(?:qk|audit|reference)", re.MULTILINE)
def test_default_amd_isa_extension_descriptors_are_empty():
  assert DEFAULT_AMD_ISA_EXTENSION_DESCRIPTORS == ()
  desc = AMDISARendererExtensionDescriptor("future")
  assert desc.proof_tags == ()
  assert desc.local_buffer_ids == frozenset()
  assert desc.machine_search_hooks == ()

def test_live_amd_isa_extension_descriptors_route_prefill_policy_through_adapter():
  assert get_amd_isa_extension_descriptors() == (PREFILL_AMD_ISA_RENDERER_EXTENSION,)
  desc = get_amd_isa_extension_descriptors()[0]
  assert desc.renderer_policy is not None
  assert "wmma_frag_proof" in desc.proof_tags

def test_extension_interfaces_do_not_import_extra_qk():
  offenders = []
  for rel in ("tinygrad/renderer/isa/extensions.py",):
    path = ROOT / rel
    if ROUTE_IMPORT.search(path.read_text()):
      offenders.append(rel)
  assert offenders == []
