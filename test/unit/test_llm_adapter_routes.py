from tinygrad import nn
from tinygrad.llm.adapter import install_lora


class Model:
  def __init__(self):
    self.output = nn.Linear(4, 8, bias=False)
    self._decode_direct_greedy_promoted = True
    self._decode_submit_ahead_promoted = True
    self._decode_vocab_top1_lease = True
    self._decode_native_argmax_threads = 256
    self._decode_native_argmax_lease = 256
    self._decode_packed_argmax_promoted = True


def test_output_lora_closes_unqualified_output_shortcuts():
  model = Model()
  install_lora(model, ["output"], rank=2, alpha=2)
  assert model._decode_direct_greedy_promoted is False
  assert model._decode_submit_ahead_promoted is False
  assert model._decode_vocab_top1_lease is False
  assert model._decode_native_argmax_threads == 0
  assert model._decode_native_argmax_lease == 0
  assert model._decode_packed_argmax_promoted is False
  assert model._output_adapter_active is True
