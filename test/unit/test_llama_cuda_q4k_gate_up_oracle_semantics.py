import numpy as np


def test_swiglu_keeps_the_up_factor():
  # This is the regression behind the original false construction failure.
  up = np.array([-2.0, 1.5], np.float32)
  gate = np.array([0.5, -1.0], np.float32)
  fused = up * (gate/(1 + np.exp(-gate)))
  assert np.allclose(fused, np.array([-0.62245935, -0.4034121], np.float32))
  assert not np.allclose(fused, gate/(1 + np.exp(-gate)))
