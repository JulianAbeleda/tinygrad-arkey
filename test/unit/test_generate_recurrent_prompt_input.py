"""Recurrent (GatedDeltaNet) models prefill one token at a time through the T==1 decode JIT, the same TinyJit the
generated tokens use. TinyJit's input contract is structural, so a prompt token must be handed over in the feedback
token's own spelling (a realized (1,1) buffer), not as a start_pos-sliced view of the prompt tensor. Seen on the
RTX 5090 with Qwen3.5-27B: "args mismatch in JIT" at the first generated token."""
from tinygrad import Tensor, UOp, dtypes
from tinygrad.engine.jit import _jit_input_descriptors
import inspect

from tinygrad.llm.model import Transformer, _generation_input_slice, _recurrent_prompt_token


def _descriptor(t:Tensor):
  (view, variables, dtype, device), = _jit_input_descriptors([t.uop])
  return view, variables, dtype


def test_recurrent_prompt_token_has_the_feedback_token_input_contract():
  prompt = Tensor([[7, 8, 9, 10] + [0] * 12], dtype=dtypes.int32, device="CPU").realize()
  feedback = Tensor([[3, 1, 2]], dtype=dtypes.int32, device="CPU").argmax(-1, keepdim=True).realize()
  for pos in (0, 1, 3):
    token = _recurrent_prompt_token(prompt, pos)
    assert token.shape == (1, 1) and token.dtype == feedback.dtype and int(token.item()) == [7, 8, 9, 10][pos]
    assert _descriptor(token) == _descriptor(feedback)
  # the symbolic slice the chunked/dense path keeps is a different contract (it carries start_pos): that is the mismatch
  sliced = _generation_input_slice(prompt, UOp.variable("start_pos", 0, 15).bind(1), UOp.variable("toks", 1, 1).bind(1), 1)
  assert _descriptor(sliced)[1] != () or _descriptor(sliced)[0] is not _descriptor(feedback)[0]
  assert _descriptor(sliced) != _descriptor(feedback)


def test_recurrent_models_never_take_a_prefill_v2_chunk():
  # GatedDeltaNetBlock._attention asserts T==1: generate forces chunk_size=1 for recurrent models and both prefill-v2
  # branches (a ubatch-wide t[:, sp:sp+ubatch] chunk) must be closed to them, or a 512-token prompt asserts there.
  src = inspect.getsource(Transformer.generate)
  assert src.count("self.config.prefill_v2 and not self.has_recurrent_block and prefill_v2_target_admitted(") == 2
  assert "if self.has_recurrent_block: chunk_size = 1" in src
