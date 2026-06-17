#!/usr/bin/env python3
"""WR ladder Phases 1-2: shape-safe warp shuffle + warp reductions (the primitive the stale flash reference
lacked under current tinygrad). Proves extra/amd_warp_reduce compiles, is correct, Ops.PROGRAM + JIT-replayable.
No attention, no model. AMD-gated (ds_bpermute is wave32 gfx1100)."""
import unittest

import numpy as np

from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.helpers import JIT
from tinygrad.uop.ops import UOp, KernelInfo, Ops
from tinygrad.uop.ops import AxisType
from tinygrad.dtype import AddrSpace
from extra.amd_warp_reduce import warp_shfl_xor, warp_reduce_max, warp_reduce_sum, WARP

_DEV_OK = Device.DEFAULT == "AMD"
NB = 8
_LOG2E = 1.4426950408889634

def _kernel(name, body):
  def k(y:UOp, x:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); lane = UOp.special(WARP, "lidx0")
    xg = x.reshape(NB, WARP)[gid]; yg = y.reshape(NB, WARP)[gid]
    return yg[lane].store(body(xg[lane], lane)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return k

def _run(fxn, xnp):
  x = Tensor(xnp).realize()
  return Tensor.empty(NB * WARP, dtype=dtypes.float32).custom_kernel(x, fxn=fxn)[0].realize().numpy().reshape(NB, WARP)

@unittest.skipUnless(_DEV_OK, "ds_bpermute warp primitives are AMD (wave32) gated")
class TestAMDWarpReduce(unittest.TestCase):
  def setUp(self): self.xnp = np.random.default_rng(0).standard_normal((NB, WARP)).astype(np.float32)

  def test_WR1_shuffle(self):
    for off in (1, 2, 8, 16):
      got = _run(_kernel(f"shfl_{off}", lambda v, l, o=off: warp_shfl_xor(v, o, l)), self.xnp)
      ref = self.xnp[:, np.arange(WARP) ^ off]
      self.assertTrue(np.allclose(got, ref), f"xor-shuffle offset {off} wrong")

  def test_WR2_reduce_max(self):
    got = _run(_kernel("wmax", warp_reduce_max), self.xnp)
    ref = np.broadcast_to(self.xnp.max(axis=1, keepdims=True), (NB, WARP))
    self.assertTrue(np.allclose(got, ref), "warp_reduce_max wrong")

  def test_WR2_reduce_sum(self):
    got = _run(_kernel("wsum", warp_reduce_sum), self.xnp)
    ref = np.broadcast_to(self.xnp.sum(axis=1, keepdims=True), (NB, WARP))
    self.assertTrue(np.allclose(got, ref, atol=1e-4), "warp_reduce_sum wrong")

  def test_emits_bpermute_no_cpu_fallback(self):
    from tinygrad.engine.realize import compile_linear
    out = Tensor.empty(NB * WARP, dtype=dtypes.float32).custom_kernel(
      Tensor(self.xnp).realize(), fxn=_kernel("wmax_src", warp_reduce_max))[0]
    src = ""
    for call in compile_linear(out.schedule_linear()).src:
      p = call.src[0]
      if p.op is Ops.PROGRAM and "wmax_src" in p.arg.name:
        src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), ""); break
    self.assertIn("ds_bpermute", src, "warp reduce did not emit ds_bpermute")

  @unittest.skipUnless(JIT, "replay check needs JIT")
  def test_captured_and_replayed(self):
    jf = TinyJit(lambda x: Tensor.empty(NB * WARP, dtype=dtypes.float32).custom_kernel(x, fxn=_kernel("wmax_j", warp_reduce_max))[0])
    for vals in (self.xnp, self.xnp[::-1].copy(), np.ones((NB, WARP), np.float32)):
      got = jf(Tensor(vals).realize()).numpy().reshape(NB, WARP)
      self.assertTrue(np.allclose(got, np.broadcast_to(vals.max(1, keepdims=True), (NB, WARP))), "JIT replay wrong")
    names = [u.src[0].arg.name for u in jf.captured.linear.toposort()
             if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]
    self.assertTrue(any(n.startswith("wmax_j") for n in names), f"warp-reduce kernel not captured: {names}")

def _softmax_row_kernel(NTILES):
  # WR3: online-softmax row STATE (max m + denominator l) over a row of NTILES*WARP scores, register-resident,
  # using the warp reductions. TWO-PASS (max-reduce, then sum-reduce-using-max): each is a single-accumulator
  # reduce. The single-pass ONLINE recurrence (coupled m/l in one group store) hits a store-ordering hazard --
  # m is overwritten before alpha=exp(m_old-m_new) reads it, so alpha collapses to 1 (verified). Two-pass is
  # the shape-safe equivalent (same lesson as Attempt A: sequential single-accumulator beats coupled).
  def k(OM:UOp, OL:UOp, S:UOp) -> UOp:
    gid = UOp.special(NB, "gidx0"); lane = UOp.special(WARP, "lidx0"); Sg = S.reshape(NB, NTILES, WARP)[gid]
    m = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG); m = m.after(m[0].store(-1e30))
    nt1 = UOp.range(NTILES, 0, AxisType.REDUCE)
    mt = warp_reduce_max(Sg[nt1, lane].cast(dtypes.float32), lane)
    m = m.after(m[0].store(m.after(nt1)[0].maximum(mt)).end(nt1)); mg = m[0]
    l = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG); l = l.after(l[0].store(0.0))
    nt2 = UOp.range(NTILES, 1, AxisType.REDUCE)
    p = ((Sg[nt2, lane].cast(dtypes.float32) - mg) * _LOG2E).exp2(); lt = warp_reduce_sum(p, lane)
    l = l.after(l[0].store(l.after(nt2)[0] + lt).end(nt2))
    return UOp.group(OM.reshape(NB)[gid].store(mg), OL.reshape(NB)[gid].store(l[0])).sink(
      arg=KernelInfo(name=f"osm_{NTILES}", opts_to_apply=()))
  return k

@unittest.skipUnless(_DEV_OK, "WR3 online-softmax row is AMD (wave32) gated")
class TestWR3OnlineSoftmaxRow(unittest.TestCase):
  def test_softmax_row_state(self):
    NTILES = 4; KV = NTILES * WARP
    Snp = (np.random.default_rng(1).standard_normal((NB, KV)) * 1.5).astype(np.float32)
    S = Tensor(Snp).realize()
    res = Tensor.empty(NB, dtype=dtypes.float32).custom_kernel(
      Tensor.empty(NB, dtype=dtypes.float32), S, fxn=_softmax_row_kernel(NTILES))
    om = res[0].realize().numpy(); ol = res[1].realize().numpy()
    ref_m = Snp.max(1); ref_l = np.exp(Snp - ref_m[:, None]).sum(1)
    self.assertTrue(np.allclose(om, ref_m), "row max wrong")
    self.assertTrue(np.allclose(ol, ref_l, rtol=1e-4), "softmax denominator wrong")

if __name__ == "__main__":
  unittest.main()
