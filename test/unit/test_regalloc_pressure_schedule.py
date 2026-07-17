import unittest

from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import FixedRegisterUse, Register, RegisterSpan
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.amd import AMDOps, VBASE, _localize_memory_address_recipes
from tinygrad.uop.ops import Ops, UOp


class TestPressureSchedule(unittest.TestCase):
  def _v(self, name, physical):
    return Register(name, physical.index, _cons=(physical,))

  def _wide_v(self, name, physical, count=8):
    return Register(name, physical.index, _cons=(physical,), _span=RegisterSpan(count, count))

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

  def test_reduces_weighted_peak_for_overlapping_wide_pipelines(self):
    physical = Register("r", 0)
    def pipeline(n):
      a = self._ins(f"load_a{n}", reg=self._wide_v(f"va{n}", physical))
      b = self._ins(f"load_b{n}", reg=self._wide_v(f"vb{n}", physical))
      wide = tuple(self._v(f"vw{n}_{i}", physical) for i in range(8))
      dot = UOp(Ops.INS, dtypes.int32, (a, b), f"dot{n}", tag=wide)
      converts = tuple(self._ins(f"convert{n}_{i}", (dot,), self._v(f"vc{n}_{i}", physical)) for i in range(8))
      updates = tuple(self._ins(f"update{n}_{i}", (c,), self._v(f"vu{n}_{i}", physical)) for i,c in enumerate(converts))
      return a, b, dot, converts, updates
    p0, p1 = pipeline(0), pipeline(1)
    original = [p0[0], p0[1], p1[0], p1[1], p0[2], p1[2], *p0[3], *p1[3], *p0[4], *p1[4]]
    scheduled = pressure_schedule(original)

    def _register_tuple(u):
      return isinstance(u.tag, tuple) and u.tag and all(isinstance(r, Register) for r in u.tag)
    def peak(block):
      pos = {u:i for i,u in enumerate(block)}
      ends = {u:max((pos[c] for c in block if u in c.src), default=pos[u]) for u in block}
      return max(sum(sum(r.span.count for r in u.tag) for u in block if _register_tuple(u) and pos[u] <= i <= ends[u])
                 for i in range(len(block)))

    self.assertLess(peak(scheduled), peak(original))
    positions = {u:i for i,u in enumerate(scheduled)}
    self.assertLess(max(positions[u] for u in (*p0[3], *p0[4])), positions[p1[2]])

  def test_finishes_oldest_fanout_before_newer_metadata_generation(self):
    """Release an older group's lanes before following a newer producer chain."""
    physical = Register("r", 0)
    gate = self._ins("group0_ready")
    metadata_seed = self._ins("metadata_seed", reg=self._wide_v("metadata_seed", physical))
    inner = tuple(self._ins(f"inner0_{i}", (gate,), self._v(f"inner0_{i}", physical)) for i in range(8))
    outer = tuple(self._ins(f"outer0_{i}", (x,)) for i,x in enumerate(inner))
    kickoff = self._ins("group1_kickoff", (gate, metadata_seed))
    metadata = []
    previous = kickoff
    for i in range(8):
      previous = self._ins(f"metadata1_{i}", (previous,), self._v(f"metadata1_{i}", physical))
      metadata.append(previous)

    scheduled = pressure_schedule([gate, metadata_seed, kickoff, *metadata, *inner, *outer])
    positions = {u:i for i,u in enumerate(scheduled)}
    self.assertLess(max(positions[u] for u in outer), min(positions[u] for u in metadata))

  def test_fixed_wide_lease_is_opened_at_its_ready_consumer(self):
    physical = Register("r", 0)
    lease = self._ins("lease", reg=FixedRegisterUse("fixed", 32, _span=RegisterSpan(8, 8)))
    first = self._ins("first", reg=self._wide_v("first", physical))
    ready = self._ins("ready", (first,), self._wide_v("ready", physical))
    consume = self._ins("consume", (lease, ready), self._wide_v("consume", physical))
    independent = self._ins("independent", reg=self._wide_v("independent", physical))

    scheduled = pressure_schedule([lease, first, ready, independent, consume])
    positions = {u:i for i,u in enumerate(scheduled)}
    self.assertEqual(positions[consume], positions[lease] + 1)
    self.assertLess(positions[ready], positions[lease])

  def test_single_choice_load_lease_waits_for_consumer_prerequisites(self):
    """An early-ready physical fragment stays local to its sole consumer."""
    fragment = Register("fragment", 200)
    addr = self._ins("addr")
    order = self._ins("order")
    release = self._ins("release")
    load = self._ins("load", (addr, order, release), self._wide_v("loaded", fragment, 4))
    other0 = self._ins("other0")
    other1 = self._ins("other1", (other0,))
    other2 = self._ins("other2", (other1,))
    consume = self._ins("matrix", (load, other2), self._wide_v("result", Register("result", 8)))
    tail = self._ins("reuse_fragment", (consume,), self._wide_v("reused", fragment, 4))

    scheduled = pressure_schedule([addr, order, release, load, other0, other1, other2, consume, tail])
    positions = {u:i for i,u in enumerate(scheduled)}
    self.assertEqual(positions[consume], positions[load] + 1)
    self.assertLess(positions[other2], positions[load])
    self.assertLess(max(positions[x] for x in (addr, order, release)), positions[load])

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

  def test_amd_localizes_hardware_id_root_for_every_memory_effect(self):
    ctx = IselContext(UOp.sink())
    special = UOp(Ops.SPECIAL, dtypes.int32, arg="gidx0", tag=(ctx.vreg(VBASE[1:]),))
    root = UOp(Ops.INS, dtypes.int32, (special,), AMDOps.WG_ID, tag=(ctx.vreg(VBASE[1:]),))
    shifted = UOp(Ops.INS, dtypes.int32, (root, UOp.const(dtypes.int32, 4).rtag()), AMDOps.V_LSHR,
                  tag=(ctx.vreg(VBASE[1:]),))
    address = UOp(Ops.INS, dtypes.int32, (shifted, UOp.const(dtypes.int32, 4).rtag()), AMDOps.V_IMUL,
                  tag=(ctx.vreg(VBASE[1:]),))
    memories = tuple(UOp(Ops.INS, dtypes.void,
      (address, UOp.const(dtypes.uint64, 0).rtag(), UOp.const(dtypes.int32, i).rtag()), AMDOps.GLOBAL_STORE) for i in range(3))
    localized = _localize_memory_address_recipes(ctx, UOp.sink(*memories))
    self.assertIsNotNone(localized)
    stores = [u for u in localized.toposort() if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
    roots = [next(u for u in store.src[0].toposort() if u.op is Ops.INS and u.arg is AMDOps.WG_ID) for store in stores]
    self.assertEqual(len(set(roots)), len(stores))


if __name__ == "__main__":
  unittest.main()
