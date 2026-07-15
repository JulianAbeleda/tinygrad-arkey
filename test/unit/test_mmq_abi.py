import numpy as np
import pytest
from extra.qk.mmq_abi import Q4K_Q8_MMQ_ABI

def test_q4_q8_abi_accepts_exact_operand_contract():
  Q4K_Q8_MMQ_ABI.validate(np.zeros(36, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32), k=256, n=1)

@pytest.mark.parametrize("index", [0, 1, 2, 3])
def test_q4_q8_abi_rejects_wrong_dtype(index):
  args = [np.zeros(36, np.uint32), np.zeros(256, np.int8), np.ones(8, np.float32), np.zeros(8, np.float32)]
  args[index] = args[index].astype(np.float16)
  with pytest.raises(ValueError): Q4K_Q8_MMQ_ABI.validate(*args, k=256, n=1)
