import unittest

import numpy as np

from tinygrad import Tensor, nn


class TestTrainingPrimitives(unittest.TestCase):
  def test_backward_computes_expected_gradient(self):
    x = Tensor([1.0, 2.0, 3.0], device="CPU").is_param_()
    (x * x).sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), [2.0, 4.0, 6.0])

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
