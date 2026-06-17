#!/usr/bin/env python3
"""LDS tiling primitive arc — Phase 2: prove a custom kernel can use LDS/shared memory + a barrier.

Proves the missing flash-prefill performance primitive (cooperative shared-memory tile reuse) is EXPRESSIBLE
through `Tensor.custom_kernel` directly (no BEAM): a workgroup cooperatively loads a tile GLOBAL->LDS, a
barrier orders it, then lanes read CROSS-LANE from LDS (which is only correct because of the barrier + shared
residency). Asserts correctness, Ops.PROGRAM capture + TinyJit replay, and that the HIP source actually emits
shared memory + an AMD barrier (i.e. real LDS, not a CPU fallback). No attention, no model. Gated to AMD.

Mechanism (see docs/amd-lds-tiling-existing-primitives-20260617.md): AddrSpace.LOCAL placeholder ->
Ops.DEFINE_LOCAL (renders `__attribute__((shared,...))`), UOp.barrier -> Ops.BARRIER (renders
`__builtin_amdgcn_s_barrier`).
"""
import json, pathlib, unittest

import numpy as np

from tinygrad import Tensor, TinyJit, dtypes, Device
from tinygrad.helpers import JIT
from tinygrad.uop.ops import UOp, KernelInfo, Ops
from tinygrad.dtype import AddrSpace

_DEV_OK = Device.DEFAULT == "AMD"
N, BLOCK = 256, 64
NB = N // BLOCK

def lds_smoke_kernel(y:UOp, x:UOp) -> UOp:
  gid = UOp.special(NB, "gidx0"); tid = UOp.special(BLOCK, "lidx0")
  xt = x.reshape(NB, BLOCK)[gid]; yt = y.reshape(NB, BLOCK)[gid]
  lds = UOp.placeholder((BLOCK,), dtypes.float32, slot=0, addrspace=AddrSpace.LOCAL)
  store = lds[tid].store(xt[tid])              # cooperative GLOBAL -> LDS
  lds = lds.after(UOp.barrier(store))          # barrier, then CROSS-LANE reuse (needs the barrier)
  return yt[tid].store(lds[(tid + 1) % BLOCK] * 2.0 + 1.0).sink(arg=KernelInfo(name=f"lds_smoke_{N}", opts_to_apply=()))

def _run(x:Tensor) -> Tensor:
  return Tensor.empty(N, dtype=dtypes.float32).custom_kernel(x, fxn=lds_smoke_kernel)[0]

def _ref(xnp:np.ndarray) -> np.ndarray:
  return (np.roll(xnp.reshape(NB, BLOCK), -1, axis=1) * 2 + 1).reshape(N)

@unittest.skipUnless(_DEV_OK, "LDS custom-kernel proof is AMD-gated")
class TestLDSCustomKernelBridge(unittest.TestCase):
  def test_correct(self):
    x = Tensor(np.arange(N, dtype=np.float32)).realize()
    self.assertTrue(np.allclose(_run(x).numpy(), _ref(x.numpy())), "LDS cross-lane read wrong (barrier/LDS broken)")

  def test_emits_shared_and_barrier(self):
    # the kernel must compile to real LDS + barrier (not a CPU fallback). Inspect the rendered HIP source.
    from tinygrad.engine.realize import compile_linear
    out = _run(Tensor(np.arange(N, dtype=np.float32)).realize())
    compiled = compile_linear(out.schedule_linear())
    src = ""
    for call in compiled.src:
      p = call.src[0]
      if p.op is Ops.PROGRAM and "lds_smoke" in p.arg.name:
        src = next((u.arg for u in p.toposort() if u.op is Ops.SOURCE), ""); break
    self.assertIn("shared", src, "no shared-memory declaration in the kernel source")
    self.assertTrue("barrier" in src or "s_barrier" in src, "no workgroup barrier in the kernel source")

  @unittest.skipUnless(JIT, "replay check needs JIT")
  def test_captured_as_program_and_replayed(self):
    jf = TinyJit(_run)
    for vals in (np.arange(N, dtype=np.float32), np.arange(N, dtype=np.float32)[::-1].copy(), np.ones(N, np.float32)):
      x = Tensor(vals).realize()
      self.assertTrue(np.allclose(jf(x).numpy(), _ref(vals)), "LDS kernel wrong under JIT")
    names = [u.src[0].arg.name for u in jf.captured.linear.toposort()
             if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM]
    self.assertTrue(any(n.startswith("lds_smoke") for n in names), f"LDS kernel not captured as Ops.PROGRAM: {names}")

_REUSE_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "lds-tiling-primitive-20260617" / "result.json"

class TestLDSReuseArtifact(unittest.TestCase):
  """Lock the Phase-3 result: LDS reuse beats redundant HBM reads in the high-reuse regime (the real primitive
  proof). Skip-if-absent; the DEBUG=2 benchmark (extra/qk_lds_reuse_bench.py) stays out of the suite."""
  def test_lds_reuse_wins(self):
    if not _REUSE_ARTIFACT.exists(): self.skipTest(f"no artifact at {_REUSE_ARTIFACT}")
    d = json.loads(_REUSE_ARTIFACT.read_text())
    self.assertTrue(d["all_correct"], "LDS reuse benchmark correctness regressed")
    self.assertTrue(d["verdict"].startswith("PASS"))
    self.assertTrue(d["win_at_high_reuse_W"], "LDS reuse should beat global reads at high W (the flash regime)")
    self.assertGreater(d["best_speedup"], 1.5, "LDS reuse speedup collapsed")

_QK_ARTIFACT = pathlib.Path(__file__).parents[2] / "bench" / "lds-tiling-primitive-20260617" / "phase4-qk" / "result.json"

class TestLDSQkTileReuseArtifact(unittest.TestCase):
  """Lock Phase 4: LDS q.k tile reuse beats global reread (locality inside the real attention math).
  Skip-if-absent; the DEBUG=2 benchmark (extra/lds_qk_tile_reuse.py) stays out of the suite."""
  def test_qk_reuse_wins(self):
    if not _QK_ARTIFACT.exists(): self.skipTest(f"no artifact at {_QK_ARTIFACT}")
    d = json.loads(_QK_ARTIFACT.read_text())
    self.assertTrue(d["all_correct"] and d["lds_emitted"], "Phase-4 correctness / shared+barrier regressed")
    self.assertTrue(d["verdict"].startswith("PASS"))
    self.assertTrue(d["win_at_high_reuse_T"], "LDS q.k reuse should beat global reread at high T")
    self.assertGreater(d["best_speedup"], 1.5, "LDS q.k reuse speedup collapsed")

if __name__ == "__main__":
  unittest.main()
