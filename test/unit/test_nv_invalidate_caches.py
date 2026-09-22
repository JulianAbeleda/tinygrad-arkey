"""NVDevice.invalidate_caches evicts by streaming, not by the driver call alone.

Measured on an RTX 5090 (96 MiB L2, 2026-09-22): the driver's FB_FLUSH_GPU_CACHE writes back but does not
invalidate; a 64 MiB working set still read at 2.03 TB/s after it, above the card's DRAM rate, so it was still in
L2. The device now also fills a buffer larger than any NVIDIA L2 with a store kernel; a copy-engine copy did not evict. These pin the contract
without a GPU; whether it evicts is a measurement, recorded in the commit that added it."""
import pytest

from extra.llm_research.flash_live import flush_mode

ops_nv = pytest.importorskip("tinygrad.runtime.ops_nv", reason="the NV runtime's bindings do not import here")


def test_the_nv_scrub_is_larger_than_the_biggest_nvidia_l2():
  # GB202 (RTX 5090) reports 100,663,296 bytes of L2; the scrub must exceed it with room to spare.
  assert ops_nv.NVDevice.L2_SCRUB_BYTES >= 2 * 100_663_296
  assert ops_nv.NVDevice.L2_SCRUB_BYTES == 256 << 20


def test_the_nv_device_names_its_flush_as_driver_write_back_plus_scrub():
  assert callable(ops_nv.NVDevice.invalidate_caches)
  assert flush_mode(ops_nv.NVDevice) == "driver_write_back_then_store_scrub_256MiB"
