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
from tinygrad.uop.ops import Ops
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


# ***** SIBLING_FUSE=2: shared output ranges in rangeify + row-local lowering (codegen/opt/row_local.py) *****

from tinygrad.codegen.opt.row_local import lane_count


def test_v2_residual_rmsnorm_hilo_is_one_kernel_and_accurate():
  Tensor.manual_seed(1)
  D = 3136
  norm = nn.RMSNorm(D, 1e-5)
  norm.weight = Tensor.rand(D).contiguous().realize()
  hidden = (Tensor.randn(16, 1, D) * 4).contiguous().realize()
  mixed = Tensor.randn(16, 1, D).cast(dtypes.bfloat16).contiguous().realize()
  n0, (x0, hi0, lo0) = _run(*_residual_rmsnorm_hilo(hidden, mixed, norm), fuse=0)
  n2, (x2, hi2, lo2) = _run(*_residual_rmsnorm_hilo(hidden, mixed, norm), fuse=2)
  assert n0 == 5 and n2 == 1
  assert np.array_equal(_bits(x0), _bits(x2))
  x64 = hidden.numpy().astype(np.float64) + mixed.float().numpy().astype(np.float64)
  n64 = x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + 1e-5) * norm.weight.numpy().astype(np.float64)
  err = lambda hi, lo: np.abs(hi.astype(np.float64) + lo.astype(np.float64) - n64).max() / np.abs(n64).max()
  # a different summation order, not a loss of precision: same error against fp64 as the unfused schedule
  assert err(hi2, lo2) < 2 * err(hi0, lo0) + 1e-7


def test_v2_no_workgroup_spanning_shared_stage():
  # PCONTIG=2 alone stages the whole normed tensor in shared memory (602 KB at [32, 3136]); with SIBLING_FUSE=2 the
  # consumers share ranges and nothing is staged
  Tensor.manual_seed(0)
  norm = nn.RMSNorm(256, 1e-5)
  norm.weight = Tensor.rand(256).contiguous().realize()
  hidden = Tensor.randn(8, 1, 256).contiguous().realize()
  mixed = Tensor.randn(8, 1, 256).cast(dtypes.bfloat16).contiguous().realize()
  seen = []
  import tinygrad.codegen as codegen
  orig = codegen.full_rewrite_to_sink
  def spy(ast, ren, optimize=True):
    seen.extend(u for u in ast.toposort() if u.op is Ops.STAGE)
    return orig(ast, ren, optimize)
  with unittest.mock.patch.object(codegen, "full_rewrite_to_sink", spy):
    n, _ = _run(*_residual_rmsnorm_hilo(hidden, mixed, norm), fuse=2, pcontig=2)
  assert n == 1 and seen == []


def test_v2_lane_count():
  assert lane_count(3136, 3136, 512) == 448  # 14 warps: the largest whole-warp common divisor
  assert lane_count(100, 30, 512) == 10      # no whole-warp divisor: the largest common one
  assert lane_count(7, 11, 512) is None
  assert lane_count(200, 200, 512) == 100    # never the whole extent: an L*L shared stage otherwise (160 KB at 200)


def _nv_sources(*outs):
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.cuda import CUDARenderer
  with Context(SIBLING_FUSE=2):
    lin = Tensor.schedule_linear(*outs)
    ren = CUDARenderer(Target.parse("NV:CUDA:sm_120"))
    with unittest.mock.patch.object(type(ren.compiler), "compile", lambda self, src: b""), Context(CACHELEVEL=0):
      progs = [to_program(c.src[0], ren) for c in lin.src if c.op is Ops.CALL and c.src[0].op is Ops.SINK]
  return [next(u.arg for u in p.src if u.op is Ops.SOURCE and isinstance(u.arg, str)) for p in progs]


def test_v2_gpu_lowering_is_one_workgroup_per_row():
  D = 3136
  norm = nn.RMSNorm(D, 1e-5)
  norm.weight = Tensor.empty(D)
  srcs = _nv_sources(*_residual_rmsnorm_hilo(Tensor.empty(32, 1, D), Tensor.empty(32, 1, D, dtype=dtypes.bfloat16), norm))
  assert len(srcs) == 1
  src = srcs[0]
  # rows are the grid, 448 lanes reduce a row through a 448-entry shared buffer and then write its three outputs
  assert "blockIdx.x; /* 32 */" in src and "threadIdx.x; /* 448 */" in src and "float buf1[448]" in src
  assert src.count("__syncthreads") == 1 and "blockIdx.y" not in src


def test_v2_row_local_declines_reduce_that_depends_on_every_output():
  # out[b, d] = sum_r x[b, d, r]: the reduce depends on d, so no lane can be shared; the heuristic keeps the kernel
  import tinygrad.codegen.opt.row_local as rl
  plans = []
  orig = rl.row_local_plan
  def spy(k):
    plans.append(ret:=orig(k))
    return ret
  with unittest.mock.patch.object(rl, "row_local_plan", spy):
    srcs = _nv_sources(Tensor.empty(32, 64, 128).sum(-1).contiguous())
  assert len(srcs) == 1 and plans == [None]


def test_v2_rollout_sampler_adds_no_shared_stage_and_keeps_tokens():
  # Partial realizations under SIBLING_FUSE=2 go to global buffers. A shared-memory stage over realized axes that
  # are not workgroup-local failed to compile on the 4B (62-160 KB of shared memory); on this tiny model the same
  # path made 31 stages.
  from tinygrad.dtype import AddrSpace
  import tinygrad.codegen as codegen
  from test.unit.test_nemotron_h_model import tiny_model
  from tinygrad.llm.nemotron_h_sampler import NemotronHRolloutSampler
  def run(flag):
    stages = []
    orig = codegen.full_rewrite_to_sink
    def spy(ast, ren, optimize=True):
      stages.extend(u for u in ast.toposort() if u.op is Ops.STAGE and getattr(u.arg, "addrspace", None) is AddrSpace.LOCAL)
      return orig(ast, ren, optimize)
    with unittest.mock.patch.object(codegen, "full_rewrite_to_sink", spy), Context(SIBLING_FUSE=flag):
      s = NemotronHRolloutSampler(tiny_model(), batch=3, capacity=16, prefix_capacity=8, prompts=2, rows=2, ring=2, window=4)
      res = s.generate([([3, 17, 5, 42, 9], 2, [5, 7]), ([7, 7, 1], 2, [4, 6])], max_new=8)
    return len(stages), [t for r in res for t, _ in r]
  (s0, t0), (s2, t2) = run(0), run(2)
  assert s2 <= s0 and t2 == t0
