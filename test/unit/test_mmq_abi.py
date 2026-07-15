import numpy as np
import pytest
from extra.qk.mmq_abi import Q4K_Q8_MMQ_ABI

def test_q4_q8_abi_accepts_exact_operand_contract():
  Q4K_Q8_MMQ_ABI.validate(np.zeros(36, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), m=1, k=256, n=1)

def test_q4_q8_abi_sizes_activations_by_m_not_n():
  Q4K_Q8_MMQ_ABI.validate(np.zeros(72, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), m=1, k=256, n=2)

@pytest.mark.parametrize("k,n", [(256, 0), (256, -1), (0, 1), (250, 1)])
def test_q4_q8_abi_rejects_invalid_shape_contract(k, n):
  with pytest.raises(ValueError):
    Q4K_Q8_MMQ_ABI.validate(np.zeros(36, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), m=1, k=k, n=n)

def test_q4_q8_abi_rejects_non_flat_or_strided_operands():
  with pytest.raises(ValueError):
    Q4K_Q8_MMQ_ABI.validate(np.zeros((2, 18), np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), m=1, k=256, n=1)
  with pytest.raises(ValueError):
    Q4K_Q8_MMQ_ABI.validate(np.zeros(72, np.uint32)[::2], np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), m=1, k=256, n=1)

@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_q4_q8_abi_rejects_wrong_dtype(index):
  args = [np.zeros(36, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32)]
  args[index] = args[index].astype(np.float16)
  with pytest.raises(ValueError): Q4K_Q8_MMQ_ABI.validate(*args, m=1, k=256, n=1)
