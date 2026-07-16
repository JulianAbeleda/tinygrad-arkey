import unittest

from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import Register
from tinygrad.uop.ops import Ops, UOp


class TestPressureSchedule(unittest.TestCase):
  def _v(self, name, physical):
    return Register(name, physical.index, _cons=(physical,))

  def _ins(self, name, src=(), reg=None):
    return UOp(Ops.INS, dtypes.int32, src=tuple(src), arg=name, tag=(reg,) if reg else ())

  def test_preserves_dependencies(self):
    physical = Register("r", 0)
    a = self._ins("a", reg=self._v("va", physical))
    b = self._ins("b", (a,), self._v("vb", physical))
    c = self._ins("c", (b,))
    independent = self._ins("independent", reg=self._v("vi", physical))

    out = pressure_schedule([a, b, independent, c])
    positions = {u: i for i, u in enumerate(out)}
    self.assertLess(positions[a], positions[b])
    self.assertLess(positions[b], positions[c])
    self.assertEqual(set(out), {a, b, c, independent})

  def test_is_deterministic(self):
    physical = Register("r", 0)
    a = self._ins("a", reg=self._v("va", physical))
    b = self._ins("b", reg=self._v("vb", physical))
    ca = self._ins("ca", (a,))
    cb = self._ins("cb", (b,))
    block = [a, b, ca, cb]
    self.assertEqual(pressure_schedule(block), pressure_schedule(block))

  def test_reduces_peak_constrained_lifetime(self):
    physical = Register("r", 0)
    a = self._ins("a", reg=self._v("va", physical))
    b = self._ins("b", reg=self._v("vb", physical))
    ca = self._ins("ca", (a,))
    cb = self._ins("cb", (b,))
    original = [a, b, ca, cb]
    scheduled = pressure_schedule(original)

    def peak(block):
      pos = {u: i for i, u in enumerate(block)}
      ends = {u: max((pos[x] for x in block if u in x.src), default=pos[u]) for u in block}
      return max(sum(pos[u] <= i <= ends[u] for u in block if u.tag) for i in range(len(block)))

    self.assertLess(peak(scheduled), peak(original))


if __name__ == "__main__":
  unittest.main()
