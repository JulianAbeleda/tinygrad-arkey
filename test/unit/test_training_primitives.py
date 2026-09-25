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

  def test_assign_reads_unrealized_fill(self):
    # the RMW kernel's load shares the store's INDEX(PARAM); _sink_param_io saw slot 0 as write-only, so the dead-item
    # pass dropped the ones() fill and x.assign(x*0.9) read uninitialized memory (0.0, not 0.9)
    a, b = Tensor.ones((1,), device="CPU"), Tensor.ones((1,), device="CPU")
    a *= 0.9
    b *= 0.999
    np.testing.assert_allclose(a.numpy(), [0.9], rtol=1e-6)
    np.testing.assert_allclose(b.numpy(), [0.999], rtol=1e-6)

  def test_adam_zero_grad_param_stays_finite(self):
    # fresh (lazy-const) Adam state: b2_t lost its fill -> v_hat = 0/(1-1) = NaN on a zero-grad param
    a, b = Tensor.ones(2, 2, device="CPU").contiguous().realize().is_param_(), Tensor.ones(2, device="CPU").contiguous().realize().is_param_()
    optimizer = nn.optim.Adam([a, b], lr=0.1)
    a.grad, b.grad = Tensor.zeros(2, 2, device="CPU").contiguous().realize(), Tensor.ones(2, device="CPU").contiguous().realize()
    with Tensor.train(): optimizer.step()
    np.testing.assert_allclose(a.numpy(), np.ones((2, 2)))
    np.testing.assert_allclose(b.numpy(), np.full(2, 0.9), rtol=1e-5)

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
