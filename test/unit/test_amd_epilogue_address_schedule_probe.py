import unittest

from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.isa.amd import AMDOps, _chain_epilogue_stores, _serialize_register_stage_writes
from tinygrad.uop.ops import Ops, UOp


class TestAMDEpilogueAddressScheduleProbe(unittest.TestCase):
  def test_stage_pair_loads_follow_previous_fixed_write(self):
    ptr = UOp.const(dtypes.int32, 0)
    writes = []
    for i in range(3):
      vals = []
      for j in range(2):
        load = UOp(Ops.INS, dtypes.float32, src=(ptr, ptr, UOp.const(dtypes.int32, j)), arg=AMDOps.GLOBAL_LOAD,
                   tag=(Register(f"load{i}_{j}", i * 2 + j),))
        vals.append(UOp(Ops.AFTER, dtypes.float32, src=(load,)))
      writes.append(UOp(Ops.INS, dtypes.void, src=tuple(vals) + (UOp.const(dtypes.int32, i),), arg=AMDOps.STAGE_WRITE))
    out = _serialize_register_stage_writes(UOp(Ops.SINK, dtypes.void, src=tuple(writes)))
    got = [u for u in out.toposort() if u.op is Ops.INS and u.arg is AMDOps.STAGE_WRITE]
    self.assertEqual(len(got), 3)
    for i in range(1, 3):
      self.assertTrue(all(got[i-1] in got[i].src[j].src[0].src for j in range(2)))

  def test_chain_preserves_address_and_targets_serialized_store(self):
    base = UOp.const(dtypes.int32, 100)
    stores = []
    for i in range(3):
      seed = UOp(Ops.INS, dtypes.int32, src=(base, UOp.const(dtypes.int32, 8)), arg=AMDOps.V_IADD,
                 tag=(Register(f"seed{i}", i),))
      poison = UOp(Ops.INS, dtypes.int32, src=(base, UOp.const(dtypes.int32, 999)), arg=AMDOps.V_IADD,
                   tag=(Register(f"poison{i}", i + 6),))
      carrier = UOp(Ops.NOOP, dtypes.int32.vec(2), src=(seed, poison))
      addr = UOp(Ops.INS, dtypes.int32, src=(carrier, UOp.const(dtypes.int32, i * 4)), arg=AMDOps.V_OFFSET,
                 tag=(Register(f"addr{i}", i + 3),))
      stores.append(UOp(Ops.INS, dtypes.void,
        src=(addr, UOp.const(dtypes.int32, 7), UOp.const(dtypes.float32, 1), UOp.const(dtypes.int32, 4)),
        arg=AMDOps.GLOBAL_STORE, tag=("store_owner", i)))
    sink = UOp(Ops.SINK, dtypes.void, src=tuple(stores))
    ctx = IselContext(sink); ctx._ncruns = 2
    out = _chain_epilogue_stores(ctx, sink)
    got = [u for u in out.toposort() if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
    self.assertEqual([u.tag[1] for u in got], [0, 1, 2])
    self.assertEqual([u.src[0].src[1].arg for u in got], [0, 4, 8])
    self.assertTrue(all(st.src[0].src[0].src[1].arg == 8 for st in got))
    self.assertFalse(any(u.op is Ops.INS and u.arg is AMDOps.V_IADD and u.src[1].arg == 999 for u in out.toposort()))
    self.assertEqual([len([s for s in u.src[0].src if s.op is Ops.INS and s.arg is AMDOps.GLOBAL_STORE]) for u in got], [0, 1, 1])
    # Every replayed arithmetic instruction is fresh and follows the same continuation as its store's root.
    recipes = [[st.src[0].src[0], st.src[0]] for st in got]
    self.assertTrue(all(r[0].arg is AMDOps.V_IADD and r[1].arg is AMDOps.V_OFFSET for r in recipes))
    self.assertEqual(len({id(u) for recipe in recipes for u in recipe}), 6)
    self.assertEqual(len({u.tag[0] for recipe in recipes for u in recipe}), 6)
    topo = list(out.toposort())
    for i, recipe in enumerate(recipes):
      continuation = got[i].src[2] if i == 0 else got[i-1]
      self.assertTrue(all(continuation in u.src[2:] for u in recipe))
      self.assertTrue(all(topo.index(continuation) < topo.index(u) for u in recipe))


if __name__ == "__main__": unittest.main()
