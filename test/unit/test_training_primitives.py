import unittest

import numpy as np

from tinygrad import Tensor, nn
from tinygrad.llm.memory_semantics import runtime_output


class TestTrainingPrimitives(unittest.TestCase):
  def test_backward_computes_expected_gradient(self):
    x = Tensor([1.0, 2.0, 3.0], device="CPU").is_param_()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0])

  def test_backward_passes_through_memory_semantic(self):
    x = Tensor([1.0, 2.0, 3.0], device="CPU").is_param_()
    runtime_output(x.square()).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0])

  def test_backward_aliases_of_param_accumulate_once(self):
    # a live alias of a param (Tensor(p.uop.base), Tensor(p.uop)) used to get its own grad sharing p.grad's buffer, so the
    # second backward raised "unordered repeated write epochs" (or summed wrong)
    x = Tensor([[1.0, 2.0], [3.0, 4.0]], device="CPU")
    for make_alias in (lambda p: Tensor(p.uop.base), lambda p: Tensor(p.uop), lambda p: p.reshape(4)):
      p = Tensor.ones(2, 2, device="CPU").contiguous().realize().is_param_()
      alias = make_alias(p)
      for it in range(1, 4):
        (x @ p.T).sum().backward()
        np.testing.assert_allclose(p.grad.realize().numpy(), it * np.array([[4.0, 6.0], [4.0, 6.0]]))
      self.assertIsNone(alias.grad)

  def test_adam_reduces_loss(self):
    weight = Tensor([0.0], device="CPU").is_param_()
    optimizer = nn.optim.Adam([weight], lr=0.1)
    previous_training = Tensor.training
    Tensor.training = True
    try:
      initial = float(((weight - 3.0) ** 2).sum().numpy())
      for _ in range(20):
        loss = ((weight - 3.0) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      final = float(((weight - 3.0) ** 2).sum().numpy())
    finally:
      Tensor.training = previous_training
    self.assertLess(final, initial)


if __name__ == "__main__": unittest.main()
