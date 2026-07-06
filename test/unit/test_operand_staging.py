import unittest
from tinygrad.uop.ops import UOp
from tinygrad.uop import Ops
from tinygrad.dtype import dtypes
from tinygrad.codegen.opt.operand_staging import operand_staging_policy, _production_cost, REGISTER, LDS

def _plain_load():
  # INDEX/LOAD off a buffer -> no arithmetic in the slice
  buf = UOp(Ops.DEFINE_REG, dtypes.uint32.ptr(1024), (), 0)
  return buf.index(UOp.const(dtypes.int, 0)).load()

def _dequant_like():
  # multi-ALU operand: shift/mask unpack + scale (sub then mul add) over a load
  packed = _plain_load()
  scale = UOp.const(dtypes.uint32, 4)
  nib = (packed >> UOp.const(dtypes.uint32, 4)) & UOp.const(dtypes.uint32, 15)  # SHR, AND
  return (nib - scale) * scale + scale  # SUB, MUL, ADD

class TestOperandStaging(unittest.TestCase):
  def test_plain_load_is_register(self):
    # a plain INDEX/LOAD operand costs ~0 -> REGISTER even with high reuse
    op = _plain_load()
    self.assertEqual(_production_cost(op), 0)
    self.assertEqual(operand_staging_policy(op, reuse_factor=64), REGISTER)

  def test_multi_alu_is_lds(self):
    # a computed (dequant-like) operand with reuse -> LDS
    op = _dequant_like()
    self.assertGreater(_production_cost(op), 2)
    self.assertEqual(operand_staging_policy(op, reuse_factor=64), LDS)

  def test_low_reuse_is_register(self):
    # decode / M==1: LDS never amortizes, even for an expensive operand
    op = _dequant_like()
    self.assertEqual(operand_staging_policy(op, reuse_factor=1), REGISTER)
    self.assertEqual(operand_staging_policy(op, reuse_factor=0), REGISTER)

  def test_override_respected(self):
    # env escape hatch wins over the computed decision, both directions
    cheap, pricey = _plain_load(), _dequant_like()
    self.assertEqual(operand_staging_policy(cheap, reuse_factor=64, override=LDS), LDS)
    self.assertEqual(operand_staging_policy(pricey, reuse_factor=64, override=REGISTER), REGISTER)

if __name__ == "__main__":
  unittest.main()
