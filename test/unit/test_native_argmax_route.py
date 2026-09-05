import json
from types import SimpleNamespace
import pytest

from tinygrad import Tensor
import tinygrad.llm.model as model

from tinygrad.llm.model_route_plan import (decode_native_argmax_promoted, decode_native_argmax_threads,
                                           load_decode_native_argmax_promotion)


def test_native_argmax_shipped_target_policy():
  assert decode_native_argmax_promoted(("NV", "sm_120"))
  assert not decode_native_argmax_promoted(("AMD", "gfx1100"))
  assert not decode_native_argmax_promoted(("CUDA", "sm_120"))


def test_native_argmax_policy_loader_is_closed_default(tmp_path):
  path = tmp_path / "policy.json"
  path.write_text(json.dumps({"schema":"boltbeam.route_policy.v1",
                              "promoted_targets":[{"backend":"NV", "architecture":"sm_120"}]}))
  assert load_decode_native_argmax_promotion(str(path)) == frozenset({("NV", "sm_120")})
  path.write_text(json.dumps({"schema":"boltbeam.route_policy.v1"}))
  assert load_decode_native_argmax_promotion(str(path)) == frozenset()


def test_native_argmax_explicit_rollback():
  enabled = lambda _name, default=0: default
  disabled = lambda name, default=0: 1 if name == "TINYGRAD_NATIVE_ARGMAX_DISABLE" else default
  assert decode_native_argmax_threads(("NV", "sm_120"), enabled) == 1024
  assert decode_native_argmax_threads(("NV", "sm_120"), disabled) == 0
  assert decode_native_argmax_threads(("AMD", "gfx1100"), enabled) == 0


@pytest.mark.parametrize("rows", [1, 2])
def test_greedy_prefill_passes_terminal_row_to_native_argmax(monkeypatch, rows):
  logits = Tensor([[[9., 0., 0.]] * (rows-1) + [[0., 5., 1.]]], device="CPU")
  seen = []
  def native_argmax(value, threads):
    assert value.shape == (1, 3)
    seen.append(threads)
    return value.argmax(-1, keepdim=True)
  monkeypatch.setattr(model, "native_argmax_finite_fp32", native_argmax)
  instance = SimpleNamespace(logits=lambda *_: logits, _decode_native_argmax_lease=1024)
  result = model.Transformer.forward_greedy(instance, Tensor([[1, 2]], device="CPU"), 0, Tensor([0.], device="CPU"))
  assert result.numpy().tolist() == [[1]]
  assert seen == [1024]
