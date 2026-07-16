from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import (
  TransformerBlock, _GGUF_TENSOR_OWNER, _KV_CACHE_OWNER, _RUNTIME_PERSISTENT_OWNER, _bind_state_dict_owners,
)
from extra.qk.physical_memory_ledger import PhysicalMemoryLedger


def _allocation_owner(tensor:Tensor):
  ledger = PhysicalMemoryLedger()
  buffer = tensor.uop.buffer
  with ledger.active():
    buffer.allocate()
    buffer.deallocate()
  return next(event.owner for event in ledger.events if event.event == "alloc")


def test_state_dict_binding_survives_lazy_allocation_and_tied_base():
  shared = Tensor.empty(8, device="CPU")
  tied = shared.reshape(2, 4).cast(dtypes.float16)
  _bind_state_dict_owners({"token_embd.weight": shared, "output.weight": tied})
  assert _allocation_owner(shared) == _GGUF_TENSOR_OWNER


def test_transformer_lazy_state_buffers_have_structural_model_owners():
  config = SimpleNamespace(dim=8, n_heads=2, n_kv_heads=1, head_dim=4, v_head_dim=4, rope_dim=4,
    max_context=8, rope_theta=10000.0, norm_eps=1e-5, hidden_dim=16, n_experts=0, qkv_bias=False,
    qk_norm=0, attn_output_gate=False, kv_quant=True, num_blocks=1, admit=None)
  block = TransformerBlock.__new__(TransformerBlock)
  block.config = config
  block._init_state(Tensor.empty(1, 1, config.dim, dtype=dtypes.float32, device="CPU"))

  assert _allocation_owner(block.cache_kv) == _KV_CACHE_OWNER
  assert _allocation_owner(block.cache_kv_scale) == _KV_CACHE_OWNER
  assert _allocation_owner(block.freqs_cis) == _RUNTIME_PERSISTENT_OWNER
