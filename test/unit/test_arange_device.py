import unittest
import numpy as np
from tinygrad import Tensor, dtypes

class TestArangeDevice(unittest.TestCase):
  def test_default_is_deviceless(self):
    self.assertIsNone(Tensor.arange(5).device)
    np.testing.assert_equal(Tensor.arange(5).numpy(), np.arange(5))

  def test_device_kwarg(self):
    for args in ((5,), (2, 9, 3), (5.5, 10, 2), (10, 0, -3)):
      t = Tensor.arange(*args, device="CPU")
      self.assertEqual(t.device, "CPU")
      np.testing.assert_allclose(t.numpy(), np.arange(*args))

  def test_device_and_dtype(self):
    t = Tensor.arange(4, dtype=dtypes.float16, device="CPU")
    self.assertEqual((t.device, t.dtype), ("CPU", dtypes.float16))
    np.testing.assert_equal(t.numpy(), np.arange(4, dtype=np.float16))

  def test_empty_with_device(self):
    t = Tensor.arange(3, 3, device="CPU")
    self.assertEqual((t.device, t.shape), ("CPU", (0,)))

  def test_combines_with_device_tensor(self):
    np.testing.assert_equal((Tensor.arange(3, device="CPU") + Tensor.ones(3, device="CPU")).numpy(), [1, 2, 3])

if __name__ == "__main__":
  unittest.main()
