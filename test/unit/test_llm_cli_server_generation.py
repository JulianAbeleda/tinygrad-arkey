from types import SimpleNamespace
import threading

from tinygrad.llm.cli import Handler


class MutatingModel:
  def get_start_pos(self, tokens): return 0
  def generate(self, tokens, **kwargs):
    tokens.append(9)
    yield 9


class Tokenizer:
  def stream_decoder(self): return lambda token_id=None: "x" if token_id is not None else ""
  def is_end(self, token_id): return False


def test_stream_generation_does_not_mutate_prompt_usage():
  metrics = {"last_prompt_tokens": None, "last_completion_tokens": None, "last_cached_prefix_tokens": None,
             "last_prefill_tok_s": None, "last_decode_tok_s": None, "last_finish_reason": None}
  state = SimpleNamespace(model=MutatingModel(), tok=Tokenizer(), prefill_chunk_size=64, metrics=metrics,
                          remote_metrics=False, cancel_event=threading.Event())
  handler = Handler.__new__(Handler)
  handler.server, handler.path = SimpleNamespace(state=state), "/v1/chat/completions"
  prompt = [1, 2, 3]
  assert list(handler._stream_tokens(prompt, 1, 0.0)) == [("delta", "x"), ("finish", "length")]
  assert prompt == [1, 2, 3]
  assert metrics["last_prompt_tokens"] == 3
  assert metrics["last_completion_tokens"] == 1
