# Ownership markers (MEMORY_SEMANTIC) are metadata: slicing, assigning into, or moving a marked value must behave as on
# the unmarked value. Each test below was a wrong or failing decode of a hybrid recurrent model (Gated DeltaNet state).
import unittest
import numpy as np
from tinygrad import Tensor
from tinygrad.llm.memory_semantics import runtime_input_materialization

class TestMarkedInputSlice(unittest.TestCase):
  def test_each_slice_reads_its_own_element(self):
    # a second slice became a zero-copy view at offset 0: every recurrent prompt token read token 0
    t = runtime_input_materialization(Tensor([760, 6511, 314, 9338, 369], dtype="int32").reshape(1, 5))
    self.assertEqual([t[:, i:i+1].reshape(1).contiguous().realize().numpy().tolist() for i in range(5)],
                     [[760], [6511], [314], [9338], [369]])
    self.assertEqual([t[0, i].item() for i in (3, 1)], [9338, 6511])

if __name__ == "__main__":
  unittest.main()
