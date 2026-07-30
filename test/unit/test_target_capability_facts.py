"""TG2: declare the missing target capabilities (docs/task_workflow/input/
target-capability-policy-decoupling-scope-20260730.md). These pin the three declarative facts -- cross-lane
shuffle availability, lane (wavefront/simdgroup) width, and `max_indirect_buffer_offset` -- for METAL, AMD, and
CPU. No admission/eligibility logic is exercised here: these are readability pins only.

No AMD hardware is available on this machine (scope section 8), so AMD facts are verified structurally: by
constructing the real `HIPRenderer`/`AMDLLVMRenderer` class and reading its declared attributes, and by feeding
that real renderer through the existing `device_facts` probe with a faked `Device[...]` lookup -- never by
executing an AMD kernel.
"""
from types import SimpleNamespace

from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import ClangRenderer, HIPRenderer, MetalRenderer
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.codegen.late.warp_reduce import WARP_SHFL_XOR_TAG  # noqa: F401  (sanity: TG1 tag still lives here)
from tinygrad.llm.device_facts import _tinygrad_target_probe
from tinygrad.runtime.graph.metal import METAL_ICB_OFFSET_MAX


def _amd(arch="gfx1100"): return HIPRenderer(Target.parse(f"AMD:HIP:{arch}"))
def _metal(): return MetalRenderer(Target.parse("METAL:METAL:Apple9"))
def _cpu(): return ClangRenderer(Target.parse("CPU:CLANG:x86_64,znver2"))


# ---- Fact 1: cross-lane shuffle availability, derived from TG1's provider, not restated -------------------

def test_shuffle_availability_agrees_with_tg1_provider_for_every_renderer():
  """`supports_warp_shfl_xor` must be a read of the TG1 attribute, not a second source of truth: for every
  renderer, the derived property must equal `warp_shfl_xor is not None` exactly."""
  for renderer in (_amd(), _metal(), _cpu()):
    assert renderer.supports_warp_shfl_xor == (getattr(renderer, "warp_shfl_xor", None) is not None)
  # CUDARenderer.__init__ needs a real NVRTC library unavailable here (see test_warp_shfl_xor_renderer_lowering.py);
  # the class attribute itself needs no hardware to read.
  assert CUDARenderer.warp_shfl_xor is not None


def test_shuffle_available_on_amd_metal_cuda_unavailable_on_cpu():
  assert _amd().supports_warp_shfl_xor is True
  assert _metal().supports_warp_shfl_xor is True
  assert CUDARenderer.supports_warp_shfl_xor.fget(CUDARenderer) is True  # unbound: no NVRTC needed
  assert _cpu().supports_warp_shfl_xor is False


def test_shuffle_availability_is_a_property_not_a_restated_bool():
  """Guard against a second source of truth: this must be a computed property reading the TG1 attribute,
  never an independently-set bare bool that could drift from it."""
  assert isinstance(type(_amd()).supports_warp_shfl_xor, property)


# ---- Fact 2: lane width, with "unreported" explicitly distinguishable from a known 32 ----------------------

def test_amd_gfx1100_wave_size_is_32():
  assert _amd("gfx1100").wave_size == 32


def test_amd_cdna_wave_size_is_64_not_defaulted_to_32():
  """gfx942/gfx950 (CDNA) are wave64 -- proves wave_size is a real per-architecture fact, not a blanket 32."""
  assert _amd("gfx942").wave_size == 64
  assert _amd("gfx950").wave_size == 64


def test_metal_wave_size_is_explicitly_unreported_and_distinguishable_from_32():
  """Scope section 3.3: Metal's simdgroup is 32-wide in hardware, but that is not modelled here -- it must
  read as None (unreported), never silently default to 32, and None must be distinguishable from a renderer
  that actually reports 32."""
  metal = _metal()
  assert metal.wave_size is None
  assert metal.wave_size != 32
  assert metal.wave_size is not _amd().wave_size  # None vs 32: not the same value, not the same identity


def test_cpu_wave_size_is_unreported():
  assert _cpu().wave_size is None


def test_amd_wave_size_flows_through_the_existing_device_facts_probe(monkeypatch):
  """Reuse tinygrad/llm/device_facts.py (scope 3.2): this is the one existing hook
  (`getattr(renderer, "wave_size", None)` in _tinygrad_target_probe) that was already reading this attribute
  name -- TG2 only needed to populate it on the real renderer, not add a parallel facts object. Structural-only:
  no AMD hardware, so `Device[...]` is faked and rocminfo is forced unavailable."""
  amd = _amd()
  opened = SimpleNamespace(renderer=amd, is_aql=False, arch=amd.target.arch)

  class FakeDevices:
    def __getitem__(self, _device): return opened

  import tinygrad.device, tinygrad.llm.device_facts
  monkeypatch.setattr(tinygrad.device, "Device", FakeDevices())
  monkeypatch.setattr(tinygrad.llm.device_facts.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
  facts = _tinygrad_target_probe("AMD")
  assert facts["backend"] == "AMD" and facts["architecture"] == "gfx1100"
  assert facts["wave_size"] == 32


def test_metal_wave_size_flows_through_the_probe_as_unreported(monkeypatch):
  opened = SimpleNamespace(renderer=_metal(), is_aql=None)

  class FakeDevices:
    def __getitem__(self, _device): return opened

  import tinygrad.device
  monkeypatch.setattr(tinygrad.device, "Device", FakeDevices())
  facts = _tinygrad_target_probe("METAL")
  assert facts["backend"] == "METAL" and facts["wave_size"] is None


# ---- Fact 3: max_indirect_buffer_offset, reusing METAL_ICB_OFFSET_MAX, not restating the literal -----------

def test_metal_icb_offset_limit_equals_the_existing_constant():
  assert _metal().max_indirect_buffer_offset == METAL_ICB_OFFSET_MAX == 0xFFFFFFFF


def test_backends_without_an_icb_constraint_report_no_limit_not_a_sentinel():
  """AMD and CPU have no indirect-command-buffer offset constraint; this must read as None ("no limit"),
  never as a numeric sentinel (e.g. 0) that could be mistaken for a real bound of zero."""
  assert _amd().max_indirect_buffer_offset is None
  assert _cpu().max_indirect_buffer_offset is None
