import unittest

from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.amd import AMDOps, VBASE, _localize_memory_address_recipes
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

  def test_amd_localized_address_tree_follows_memory_prerequisites(self):
    """Private address recipes must not open before their sole effects are ready."""
    ctx = IselContext(UOp.sink())
    src = self._ins("lane", reg=ctx.vreg(VBASE[1:]))
    shared_mul = UOp(Ops.INS, dtypes.int32, (src, UOp.const(dtypes.int32, 4).rtag()), AMDOps.V_IMUL,
                     tag=(ctx.vreg(VBASE[1:]),))
    shared_addr = UOp(Ops.INS, dtypes.int32, (shared_mul, UOp.const(dtypes.int32, 8).rtag()), AMDOps.V_IADD,
                      tag=(ctx.vreg(VBASE[1:]),))
    ready = src
    memories = []
    for i in range(8):
      ready = self._ins(f"ready{i}", (ready,), ctx.vreg(VBASE[1:]))
      memories.append(UOp(Ops.INS, dtypes.int32,
        (shared_addr, UOp.const(dtypes.uint64, 0).rtag(), UOp.const(dtypes.int32, i).rtag(), ready),
        AMDOps.GLOBAL_LOAD, tag=(ctx.vreg(VBASE[1:]),)))
    triggers = tuple(self._ins(f"trigger{i}", reg=ctx.vreg(VBASE[1])) for i in range(2))

    # This reconstructs the pre-fix clone shape: private, but ordered only by
    # the original lane input, so all eight trees open before the ready chain.
    old_memories = []
    for mem in memories:
      mul = shared_mul.replace(tag=(ctx.vreg(VBASE[1:]),))
      addr = shared_addr.replace(src=(mul,) + shared_addr.src[1:], tag=(ctx.vreg(VBASE[1:]),))
      old_memories.append(mem.replace(src=(addr,) + mem.src[1:]))
    old = pressure_schedule(list(UOp.sink(*old_memories, *triggers).toposort()))

    localized = _localize_memory_address_recipes(ctx, UOp.sink(*memories, *triggers))
    self.assertIsNotNone(localized)
    new = pressure_schedule(list(localized.toposort()))

    def metrics(block):
      pos = {u:i for i,u in enumerate(block)}
      mems = [u for u in block if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_LOAD]
      distances = [pos[m] - pos[m.src[0]] for m in mems]
      ends = {u:max((pos[c] for c in block if u in c.src), default=pos[u]) for u in block}
      addresses = [u for u in block if u.op is Ops.INS and u.arg in (AMDOps.V_IMUL, AMDOps.V_IADD)]
      peak = max(sum(pos[u] <= i <= ends[u] for u in addresses) for i in range(len(block)))
      return peak, max(distances)

    old_peak, _ = metrics(old)
    new_peak, new_distance = metrics(new)
    self.assertLessEqual(new_peak, old_peak)
    self.assertEqual(new_distance, 1)


if __name__ == "__main__":
  unittest.main()
