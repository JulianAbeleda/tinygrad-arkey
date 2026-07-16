import struct

from tinygrad import Device
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER, gguf_load
from tinygrad.llm.memory_semantics import MODEL_PARAMETER
from tinygrad.llm.physical_memory_ledger import PhysicalMemoryLedger
from tinygrad.schedule.memory import collect_memory_plan_manifests


def _write_one_tensor_gguf(path) -> None:
  name = b"weight"
  header = b"GGUF" + struct.pack("<iqq", 3, 1, 0)
  header += struct.pack("<Q", len(name)) + name
  header += struct.pack("<IQiQ", 1, 4, 0, 0)  # one dimension, four f32 values, offset zero
  data_start = (len(header) + 31) // 32 * 32
  path.write_bytes(header + bytes(data_start-len(header)) + struct.pack("<4f", 1, 2, 3, 4))


def test_selected_gguf_copy_owns_physical_destination_and_carries_semantic_result(tmp_path):
  path = tmp_path/"selected.gguf"
  _write_one_tensor_gguf(path)
  ledger = PhysicalMemoryLedger()

  with ledger.active(), collect_memory_plan_manifests() as manifests:
    _, state = gguf_load(path)

  rows = [row for manifest in manifests for row in manifest.buffers]
  assert {row.device for row in rows} >= {"DISK:"+str(path), Device.DEFAULT}
  backing_copy = next(manifest for manifest in manifests if any(row.device.startswith("DISK:") for row in manifest.buffers))
  assert backing_copy.buffers and all(row.semantic_owner == MODEL_PARAMETER for row in backing_copy.buffers)

  destination = [event for event in ledger.events if event.event == "alloc" and event.device == Device.DEFAULT and
                 event.requested_nbytes == path.stat().st_size and event.owner == MODEL_PARAMETER_ALLOCATION_OWNER]
  assert len(destination) == 1
  assert state["weight"].tolist() == [1.0, 2.0, 3.0, 4.0]
