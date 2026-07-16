import struct
from types import SimpleNamespace

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.gguf import MODEL_PARAMETER_ALLOCATION_OWNER, gguf_load_with_metadata
from tinygrad.llm.memory_semantics import MODEL_PARAMETER, RUNTIME_ACTIVATION, mark_memory_semantic, memory_semantic_owner, model_parameter
from tinygrad.llm.model_route_plan import build_model_route_plan
from extra.qk.physical_memory_ledger import PhysicalMemoryLedger
from tinygrad.llm.qk_primitives import (
  Q4KPrimitiveLinear, Q6KPrimitiveLinear, _install_q4k_primitives, _install_q6k_primitives,
  _model_parameter_materialization,
)
from extra.qk.schedule_memory_manifest import collect_memory_plan_manifests
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def _owned_allocations(ledger):
  return [event for event in ledger.events if event.event == "alloc"]


def test_materialization_inherits_model_semantic_and_physical_lifetime():
  source = model_parameter(Tensor.empty(8, dtype=dtypes.float32))
  ledger = PhysicalMemoryLedger()
  with ledger.active():
    derived = _model_parameter_materialization(source, source.cast(dtypes.float16).contiguous())

  assert memory_semantic_owner(derived) == MODEL_PARAMETER
  assert _owned_allocations(ledger)[-1].owner == MODEL_PARAMETER_ALLOCATION_OWNER


def test_materialization_rejects_non_parameter_semantic_source():
  source = mark_memory_semantic(Tensor.empty(8), RUNTIME_ACTIVATION)
  try: _model_parameter_materialization(source, source.clone())
  except ValueError as exc: assert "MODEL_PARAMETER" in str(exc)
  else: raise AssertionError("non-parameter source was accepted")


def test_q4_q6_prefill_clones_are_model_owned(monkeypatch):
  monkeypatch.delenv("PREFILL_PACKED_STREAM", raising=False)
  source = model_parameter(Tensor.empty(8, dtype=dtypes.float16))
  q4 = Q4KPrimitiveLinear(source, None, Tensor.empty(8, dtype=dtypes.uint32), 1, 1, 1, (), "ignored", 32, 0, "q4_ondemand")
  q6 = Q6KPrimitiveLinear(source, None, Tensor.empty(8, dtype=dtypes.uint16), 1, 1, 1, (), "ignored", 16, 0, "q4_ondemand")
  ledger = PhysicalMemoryLedger()
  with ledger.active(): q4_words, q6_halfs = q4.prefill_packed_weight(), q6.prefill_packed_weight()

  assert memory_semantic_owner(q4_words) == memory_semantic_owner(q6_halfs) == MODEL_PARAMETER
  assert all(event.owner == MODEL_PARAMETER_ALLOCATION_OWNER for event in _owned_allocations(ledger))
  assert not hasattr(q4.q4k_storage, "__dict__") and not hasattr(q6.q6k_storage, "__dict__")


@pytest.mark.parametrize("ggml_type,block_bytes,name,installer", [
  (12, 144, "blk.0.ffn_gate.weight", _install_q4k_primitives),
  (14, 210, "blk.0.ffn_down.weight", _install_q6k_primitives),
])
def test_shared_packed_custom_kernel_consumes_semantic_buffer_view_without_copy(tmp_path, ggml_type, block_bytes, name, installer):
  # Production topology: gguf_load_with_metadata supplies one shared backing,
  # primitive installation selects a per-tensor packed view, and the logical
  # ownership wrapper must remain transparent to its concrete buffer identity.
  path, encoded_name = tmp_path/f"shared-{ggml_type}.gguf", name.encode()
  packed_nbytes = 256 * 256 // 256 * block_bytes
  header = b"GGUF" + struct.pack("<iqq", 3, 1, 0)
  header += struct.pack("<Q", len(encoded_name)) + encoded_name
  header += struct.pack("<IQQiQ", 2, 256, 256, ggml_type, 0)
  data_start = (len(header) + 31) // 32 * 32
  path.write_bytes(header + bytes(data_start-len(header)) + bytes(packed_nbytes))

  _, _, meta = gguf_load_with_metadata(path)
  make_linear = lambda: SimpleNamespace(weight=Tensor.empty(256, 256, dtype=dtypes.float16), bias=None)
  model = SimpleNamespace(blk=[SimpleNamespace(ffn_gate=make_linear(), ffn_down=make_linear())])
  linear = installer(model, path, meta, storage_mode="shared", route_plan=build_model_route_plan(meta))[0]
  packed = linear.prefill_packed_weight()
  words = packed.to(packed.device)
  assert memory_semantic_owner(words) == MODEL_PARAMETER

  def consume_packed(out:UOp, source:UOp) -> UOp:
    return UOp.sink(out[0].store(source[0]), arg=KernelInfo(name="shared_q4_owner_probe"))

  result = Tensor.empty(1, dtype=dtypes.uint32, device=words.device).custom_kernel(words, fxn=consume_packed)[0]
  with collect_memory_plan_manifests() as manifests: scheduled, _ = Tensor.linear_with_vars(result)
  source_id = f"buffer:{words.uop.buf_uop.key.hex()}"
  consumers = [call for call in scheduled.src if call.op is Ops.CALL and
               any(f"buffer:{arg.buf_uop.key.hex()}" == source_id for arg in call.src[1:])]
  assert len(consumers) == 1 and consumers[0].src[0].op is Ops.SINK
  assert consumers[0].src[0].arg.name == "shared_q4_owner_probe"
  rows = [row for manifest in manifests for row in manifest.buffers if row.identity == source_id]
  assert len(rows) == 1 and rows[0].byte_range == (0, packed_nbytes)
  assert rows[0].semantic_owner == MODEL_PARAMETER
  # No second packed-size destination exists: the opaque kernel receives the
  # shared backing view directly through its argument slot.
  assert sum(row.rounded_bytes == packed_nbytes and row.semantic_owner == MODEL_PARAMETER
             for manifest in manifests for row in manifest.buffers) == 1
