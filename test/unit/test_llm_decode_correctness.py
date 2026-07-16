import numpy as np
import pytest

from tinygrad import Tensor, nn
from extra.llm.model_e2e_bench import qualify_decode_correctness
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer, TransformerConfig
import tinygrad.llm.model as model_module


def _config() -> TransformerConfig:
  return TransformerConfig(num_blocks=1, dim=8, hidden_dim=16, n_heads=2, n_kv_heads=1, norm_eps=1e-5,
    vocab_size=32, head_dim=4, rope_theta=10001.0, rope_dim=4, v_head_dim=4, max_context=8)


def _set_phase(model:Transformer, prefill:bool) -> None:
  for block in model.blk:
    block._is_prefill, block._prefill_v2 = prefill, False
    block._use_flash, block._ring_full = False, False


def _models_with_shared_weights() -> tuple[Transformer, Transformer, dict[str, Tensor]]:
  Tensor.manual_seed(123)
  source = Transformer(_config())
  state = nn.state.get_state_dict(source)
  full, incremental = Transformer(_config()), Transformer(_config())
  nn.state.load_state_dict(full, state, verbose=False)
  nn.state.load_state_dict(incremental, state, verbose=False)
  return full, incremental, state


def test_incremental_decode_logits_match_full_causal_prefix_at_every_position():
  with Context(DEV="CPU"):
    full, incremental, _ = _models_with_shared_weights()
    _set_phase(full, True)
    _set_phase(incremental, False)
    tokens = [1, 2, 3, 4, 5]

    full_logits = full.logits(Tensor([tokens]), 0).realize().numpy()
    incremental_logits = np.stack([
      incremental.logits(Tensor([[token]]), position).realize().numpy()[:, 0]
      for position, token in enumerate(tokens)
    ], axis=1)

    np.testing.assert_allclose(incremental_logits, full_logits, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(incremental_logits.argmax(-1), full_logits.argmax(-1))


def test_generate_jit_replay_matches_full_prefix_greedy_oracle():
  with Context(DEV="CPU", JIT=1):
    _, model, state = _models_with_shared_weights()
    prompt, reference_tokens, expected = [1, 2, 3], [1, 2, 3], []
    for _ in range(4):
      oracle = Transformer(_config())
      nn.state.load_state_dict(oracle, state, verbose=False)
      _set_phase(oracle, True)
      token = int(oracle.logits(Tensor([reference_tokens]), 0)[:, -1, :].argmax(-1).item())
      expected.append(token)
      reference_tokens.append(token)

    # First request warms/captures; the second replays the same decode JIT from
    # an independent prompt-side input and exercises prefix/cache recovery.
    for _ in range(2):
      got = [token for _, token in zip(range(4), model.generate(prompt.copy(), chunk_size=3, temperature=0.0))]
      assert got == expected


def test_model_benchmark_decode_correctness_qualification_passes():
  evidence = qualify_decode_correctness()
  assert evidence["passed"] is True
  assert all(tokens == evidence["oracle_token_ids"] for tokens in evidence["jit_replay_token_ids"])


def test_model_benchmark_rejects_legacy_graph_visible_persistent_ownership(monkeypatch):
  monkeypatch.setattr(model_module, "_mark_physical_semantic", lambda tensor, mark: mark(tensor))
  with pytest.raises(RuntimeError, match="decode correctness qualification failed"):
    qualify_decode_correctness()
