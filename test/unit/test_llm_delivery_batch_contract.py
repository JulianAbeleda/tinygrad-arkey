import inspect
import pytest

from tinygrad.llm.model import Transformer


def test_delivery_batch_is_explicit_and_streaming_default():
  param = inspect.signature(Transformer.generate).parameters["delivery_batch"]
  assert param.default == 1


@pytest.mark.parametrize("kwargs,match", [
  ({"delivery_batch":0}, "positive integer"),
  ({"delivery_batch":2}, "expected_output_tokens"),
  ({"delivery_batch":2, "expected_output_tokens":4, "temperature":0.5}, "requires greedy"),
  ({"delivery_batch":2, "expected_output_tokens":4, "diagnostic_full_logits":True}, "requires greedy"),
])
def test_delivery_batch_rejects_ambiguous_contracts_before_model_access(kwargs, match):
  gen = Transformer.generate(object(), [], **kwargs)
  with pytest.raises(ValueError, match=match): next(gen)
