"""MetalDevice.invalidate_caches: the cold-cache flush time_call(clear_l2=True) looks for by name.

Metal has no cache control, so the device evicts by streaming 128 MiB through a blit fill and copy. These pin
the contract (the name time_call reads, the mode a measurement reports, a pass that runs and leaves the device
usable). Whether it evicts is a measurement, recorded in the commit that added it, not a unit assertion."""
import sys

import pytest

from extra.llm_research.flash_live import NO_FLUSH_FALLBACK, flush_mode


def test_a_device_without_the_call_reports_the_fallback_by_name():
  class Bare: pass
  assert flush_mode(Bare()) == NO_FLUSH_FALLBACK


def test_the_metal_device_reports_its_scrub_mode_without_opening_a_gpu():
  from tinygrad.runtime.ops_metal import MetalDevice
  assert callable(MetalDevice.invalidate_caches)
  assert MetalDevice.CACHE_SCRUB_BYTES == 128 << 20
  assert flush_mode(MetalDevice) == "blit_scrub_128MiB_fill_and_copy"


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a Metal GPU")
def test_invalidate_caches_runs_and_the_device_still_computes():
  from tinygrad import Tensor
  from tinygrad.device import Device
  dev = Device["METAL"]
  dev.invalidate_caches(); dev.invalidate_caches()
  assert dev._scrub_pass == 2
  assert (Tensor.ones(64, device="METAL") * 3).sum().item() == 192
