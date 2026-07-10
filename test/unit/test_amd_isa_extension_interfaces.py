import pathlib, re

from tinygrad.codegen.opt.extensions import (
  DEFAULT_CODEGEN_EXTENSION_REGISTRY, EMPTY_DEVECTORIZER_EXTENSION, get_codegen_extension_registry,
)
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.isa.extensions import (
  AMDISARendererExtensionDescriptor, DEFAULT_AMD_ISA_EXTENSION_DESCRIPTORS, get_amd_isa_extension_descriptors,
)
from tinygrad.uop.ops import Ops, UOp
from extra.qk.codegen_extensions import (
  PREFILL_AMD_ISA_RENDERER_EXTENSION, PREFILL_DEVECTORIZER_EXTENSION, PREFILL_POSTRANGE_EXTENSION,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
ROUTE_IMPORT = re.compile(r"^\s*(?:from|import)\s+extra\.(?:qk|audit|reference)", re.MULTILINE)

def test_core_codegen_extension_registry_default_is_empty():
  assert DEFAULT_CODEGEN_EXTENSION_REGISTRY.postrange == ()
  assert DEFAULT_CODEGEN_EXTENSION_REGISTRY.devectorizer == ()

  uop = UOp(Ops.NOOP)
  assert not DEFAULT_CODEGEN_EXTENSION_REGISTRY.disables_ptr_group(uop)
  assert not DEFAULT_CODEGEN_EXTENSION_REGISTRY.preserves_stage_tag(uop)
  assert not DEFAULT_CODEGEN_EXTENSION_REGISTRY.preserves_wmma_proof_tag(uop)

def test_live_codegen_extension_registry_routes_prefill_extensions_through_adapter():
  registry = get_codegen_extension_registry()
  assert registry.postrange == (PREFILL_POSTRANGE_EXTENSION,)
  assert registry.devectorizer == (PREFILL_DEVECTORIZER_EXTENSION,)

def test_prefill_devectorizer_extension_disables_ptr_group_for_existing_buffer_id_predicate(monkeypatch):
  for key in ("PREFILL_DBUF_D3A_POST", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY"):
    monkeypatch.delenv(key, raising=False)
  getenv.cache_clear()

  a_buf = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(2048, AddrSpace.LOCAL), arg=990)
  b_buf = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(2048, AddrSpace.LOCAL), arg=991)
  other_buf = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(2048, AddrSpace.LOCAL), arg=992)
  reg_buf = UOp(Ops.DEFINE_REG, dtypes.half.ptr(2048, AddrSpace.REG), arg=991)

  registry = get_codegen_extension_registry()
  assert not registry.disables_ptr_group(a_buf)
  monkeypatch.setenv("PREFILL_DBUF_D3A_POST", "1")
  getenv.cache_clear()
  assert registry.disables_ptr_group(a_buf)
  assert registry.disables_ptr_group(b_buf)
  assert not registry.disables_ptr_group(other_buf)
  assert not registry.disables_ptr_group(reg_buf)

  monkeypatch.delenv("PREFILL_DBUF_D3A_POST", raising=False)
  monkeypatch.setenv("PREFILL_TC_LOCAL_STAGE_B_TILEKEY", "1")
  getenv.cache_clear()
  assert not registry.disables_ptr_group(a_buf)
  assert registry.disables_ptr_group(b_buf)

def test_empty_devectorizer_extension_predicates_are_false():
  uop = UOp(Ops.NOOP)
  assert EMPTY_DEVECTORIZER_EXTENSION.name == "empty"
  assert not EMPTY_DEVECTORIZER_EXTENSION.disables_ptr_group(uop)
  assert not EMPTY_DEVECTORIZER_EXTENSION.preserves_stage_tag(uop)
  assert not EMPTY_DEVECTORIZER_EXTENSION.preserves_wmma_proof_tag(uop)

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
  for rel in ("tinygrad/codegen/opt/extensions.py", "tinygrad/renderer/isa/extensions.py"):
    path = ROOT / rel
    if ROUTE_IMPORT.search(path.read_text()):
      offenders.append(rel)
  assert offenders == []
