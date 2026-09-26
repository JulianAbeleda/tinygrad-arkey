"""SIBLING_FUSE: independent stores over the same output ranges share one kernel (tinygrad/schedule/sibling.py).

Positive cases check the kernel count and that values are bit-identical to the unfused schedule. Each invariant
(I1..I5) has a case where the illegal fusion is refused, with the refusal reason taken from the pass itself.
"""
from __future__ import annotations
from types import SimpleNamespace
import unittest.mock

import numpy as np
import pytest

from tinygrad import Tensor, dtypes, nn, GlobalCounters, Device
from tinygrad.helpers import Context
import tinygrad.schedule.sibling as sibling


def _run(*outs:Tensor, fuse:int, pcontig:int=0) -> tuple[int, list[np.ndarray]]:
  with Context(SIBLING_FUSE=fuse, PCONTIG=pcontig):
    GlobalCounters.reset()
    Tensor.realize(*outs)
    return GlobalCounters.kernel_count, [o.numpy() for o in outs]


def _bits(a:np.ndarray) -> np.ndarray: return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _decisions(*outs:Tensor, pcontig:int=0) -> list[str|None]:
  """Refusal reasons the pass gives, for every candidate pair in the scheduled graph (None = fusible)."""
  seen:list[str|None] = []
  orig = sibling.fuse_siblings
  def spy(tsink):
    members = [m for u in tsink.toposort() if (m:=sibling.candidate(u)) is not None]
    for i, a in enumerate(members):
      for b in members[i+1:]: seen.append(sibling.refusal([a], b))
    return orig(tsink)
  with unittest.mock.patch.object(sibling, "fuse_siblings", spy), Context(SIBLING_FUSE=1, PCONTIG=pcontig):
    Tensor.realize(*outs)
  return seen


def _inputs(seed=0):
  rng = np.random.default_rng(seed)
  return [Tensor(rng.standard_normal((32, 64)).astype(np.float32)).realize() for _ in range(2)]


def test_elementwise_siblings_share_one_kernel():
  a, b = _inputs()
  n0, v0 = _run((a + b).contiguous(), (a * b).contiguous(), (a - b).cast(dtypes.bfloat16).contiguous(), fuse=0)
  n1, v1 = _run((a + b).contiguous(), (a * b).contiguous(), (a - b).cast(dtypes.bfloat16).contiguous(), fuse=1)
  assert (n0, n1) == (3, 1)
  for x, y in zip(v0, v1): assert np.array_equal(_bits(x), _bits(y))


def _residual_rmsnorm_hilo(hidden:Tensor, mixed:Tensor, norm:nn.RMSNorm):
  x = hidden + mixed.cast(hidden.dtype)
  n = norm(x)
  hi = n.cast(dtypes.bfloat16)
  return x.contiguous(), hi.contiguous(), (n - hi.float()).cast(dtypes.bfloat16).contiguous()


def test_residual_rmsnorm_hilo_is_one_kernel():
  Tensor.manual_seed(0)
  norm = nn.RMSNorm(256, 1e-5)
  norm.weight = Tensor.rand(256).contiguous().realize()
  hidden = Tensor.randn(8, 1, 256).contiguous().realize()
  mixed = Tensor.randn(8, 1, 256).cast(dtypes.bfloat16).contiguous().realize()
  # fusion is compared at the same PCONTIG: PCONTIG=2 itself (the reduce staged in its consumer instead of a global
  # round trip) may round the normed value differently, which is not this pass's doing
  n0, v0 = _run(*_residual_rmsnorm_hilo(hidden, mixed, norm), fuse=0, pcontig=2)
  n1, v1 = _run(*_residual_rmsnorm_hilo(hidden, mixed, norm), fuse=1, pcontig=2)
  assert n0 == 3 and n1 == 1
  for x, y in zip(v0, v1): assert np.array_equal(_bits(x), _bits(y))


def test_default_off_is_unchanged():
  a, b = _inputs()
  assert _run((a + b).contiguous(), (a * b).contiguous(), fuse=0)[0] == 2


def test_I1_different_output_ranges_refused():
  a, b = _inputs()
  assert "I1 ranges" in _decisions((a + b).contiguous(), a.sum(1).contiguous())
  assert _run((a + b).contiguous(), a.sum(1).contiguous(), fuse=1)[0] == 2


def test_I2_different_devices_refused():
  if "CPU:1" not in [Device.canonicalize("CPU:1")]: pytest.skip("no second CPU device")
  a, b = _inputs()
  a1, b1 = a.to("CPU:1").realize(), b.to("CPU:1").realize()
  assert "I2 device" in _decisions((a + b).contiguous(), (a1 * b1).contiguous())


def test_I3_dependent_outputs_refused():
  a, b = _inputs()
  def outs():
    y = (a + b).contiguous()
    return y, (y * 2).contiguous()
  assert "I3 order" in _decisions(*outs())
  n, (vy, vz) = _run(*outs(), fuse=1)
  assert n == 2 and np.array_equal(vz, vy * 2)


def test_I3_cross_group_cycle_refused():
  # A and B in one kernel, C and D in another, with B waiting on D and C waiting on A: the two kernels would wait on
  # each other. Built from stand-in members, since the scheduler's own graphs never present this shape by accident.
  A, B, C, D = (SimpleNamespace(after=object(), deps=set()) for _ in range(4))
  B.deps, C.deps = {D.after}, {A.after}
  assert sibling._acyclic([[A, B], [C], [D]])
  assert not sibling._acyclic([[A, B], [C, D]])


def test_I4_two_reduces_refused():
  a, _ = _inputs()
  outs = lambda: (a.sum(1).contiguous(), a.max(1).contiguous())
  assert "I4 reduce" in _decisions(*outs())
  n, (s, m) = _run(*outs(), fuse=1)
  assert n == 2 and np.allclose(s, a.numpy().sum(1), atol=1e-4) and np.array_equal(m, a.numpy().max(1))


def test_I4_contraction_is_never_a_candidate():
  a, b = _inputs()
  outs = lambda: ((a @ b.T).contiguous(), (a[:, :32].reshape(32, 32) + 1).contiguous())
  assert _decisions(*outs()) == []  # only the elementwise output is a candidate: no pair to consider
  assert _run(*outs(), fuse=0)[0] == _run(*outs(), fuse=1)[0] == 2


def test_I5_sibling_writes_what_another_reads_refused():
  a, b = _inputs()
  a2 = Tensor(a.numpy()).realize()
  y = (a2 * 2)
  upd = a2.assign(a2 + b)
  assert "I5 memory" in _decisions(y.contiguous(), upd)
