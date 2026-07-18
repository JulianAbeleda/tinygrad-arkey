import ctypes

from tinygrad.runtime.autogen import hsa
from tinygrad.runtime.ops_amd import _publish_aql_packet
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.system import System


def test_aql_packet_body_is_visible_before_valid_header(monkeypatch):
  storage = (ctypes.c_ubyte * 64)(*([0xA5] * 64))
  slot = MMIOInterface(ctypes.addressof(storage), 64, fmt='B')
  header_setup = (
    (hsa.HSA_PACKET_TYPE_KERNEL_DISPATCH << hsa.HSA_PACKET_HEADER_TYPE) |
    (3 << 16)
  )
  packet = header_setup.to_bytes(4, "little") + bytes(range(4, 64))
  snapshots = []

  monkeypatch.setattr(System, "memory_barrier", lambda: snapshots.append(bytes(storage)))
  _publish_aql_packet(slot, packet)

  assert len(snapshots) == 1
  invalid_header = hsa.HSA_PACKET_TYPE_INVALID << hsa.HSA_PACKET_HEADER_TYPE
  assert int.from_bytes(snapshots[0][:4], "little") == invalid_header
  assert snapshots[0][4:] == packet[4:]
  assert bytes(storage) == packet


def test_aql_packet_publication_rejects_partial_packet_or_slot():
  storage = (ctypes.c_ubyte * 64)()
  slot = MMIOInterface(ctypes.addressof(storage), 64, fmt='B')
  for bad_slot, bad_packet in ((slot, bytes(63)), (slot.view(size=32), bytes(64))):
    try: _publish_aql_packet(bad_slot, bad_packet)
    except ValueError: pass
    else: raise AssertionError("partial AQL publication must fail closed")
