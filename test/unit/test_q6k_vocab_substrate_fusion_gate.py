"""L4 vocab substrate fusion gate tests
(l4-vocab-substrate-fusion-implementation-scope-20260803.md section 4.1): the vocab head
selects the coop in-kernel merge only when the existing single-warp constraint holds
(row_tile * pos lanes <= 32, Q6KGEMVRouteSpec.validate) AND the target is fusion-admitted.
NV sm_120 (row_tile=2) admits the 151936-row vocab head and the fused kernel renders;
AMD row_tile=4 (4*16=64 > 32) and Metal (no fusion admission) stay on external_sum with the
vocab scalar-reduce + scatter chain intact (the fused path emits exactly one program, no
vocab_reduce)."""
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm import decode_routes
from tinygrad.llm.decode_kernels import Q6KGEMVRouteSpec
from tinygrad.llm.qk_layout import Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

VOCAB_ROWS, VOCAB_K = 151936, 4096


class _FakeQ6Storage:
  def __init__(self):
    self.halfs = Tensor.zeros(8, dtype=dtypes.uint16)


class _FakeQ6Linear:
  def __init__(self, capability, *, fusion, n_rows, k=VOCAB_K, parts=1):
    self.q6k_storage = _FakeQ6Storage()
    self.decode_enabled = True
    self.bias = None
    self.in_features = k
    self.out_features = n_rows
    self.parts = parts
    self.opts = ()
    self.route_admission = QKPrimitiveRouteAdmission(capability, True, epilogue_fusion_promoted=fusion)


def _bind(linear):
  x = Tensor.zeros((1, 1, linear.in_features), dtype=dtypes.float16)
  binding = decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, True)
  assert binding is not None
  return binding


def _execute_capture(linear, binding, monkeypatch):
  x = Tensor.zeros((1, 1, linear.in_features), dtype=dtypes.float16)
  captured = []
  def fake_execute(output, *inputs, program):
    captured.append(program)
    return Tensor.zeros(*program.output_spec.shape, dtype=program.output_spec.dtype)
  monkeypatch.setattr(decode_routes, "execute_promoted_program", fake_execute)
  result = decode_routes.Q6K_DECODE_CANDIDATE.execute(linear, x, binding)
  return result, captured


def _emit(program, *, rows, k, in_kernel):
  shape = (rows,) if in_kernel else (rows, program.output_spec.shape[1])
  return program.emitter(
    UOp.placeholder(shape, dtypes.float32, 0),
    UOp.placeholder((rows * (k // Q6_K_BLOCK_ELEMS) * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1),
    UOp.placeholder((k,), dtypes.float16, 2))


def _render_src(ast):
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)
  return next(u.arg for u in to_program(ast, ren).src if u.op is Ops.SOURCE)


def test_nv_sm120_vocab_in_kernel_admits_and_renders(monkeypatch):
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120", wave_size=32, supports_warp_shfl_xor=True)
  linear = _FakeQ6Linear(cap, fusion=True, n_rows=VOCAB_ROWS)
  binding = _bind(linear)
  assert binding.row_tile == 2 and binding.use_coop
  result, programs = _execute_capture(linear, binding, monkeypatch)
  # The fused path emits exactly the in-kernel GEMV: no vocab_reduce, no scatter chain.
  assert [p.output_spec.shape for p in programs] == [(VOCAB_ROWS,)]
  assert result.shape == (1, 1, VOCAB_ROWS)
  ast = _emit(programs[0], rows=VOCAB_ROWS, k=VOCAB_K, in_kernel=True)
  assert ast.arg.name == "q6k_gen_coop_151936_4096_inkernel"
  src = _render_src(ast)
  assert "__shfl_xor_sync" in src


def test_amd_row_tile4_vocab_stays_external_sum(monkeypatch):
  cap = QKPrimitiveCapability(backend="AMD", architecture="gfx1100", wave_size=32, supports_warp_shfl_xor=True)
  linear = _FakeQ6Linear(cap, fusion=True, n_rows=VOCAB_ROWS)
  binding = _bind(linear)
  assert binding.row_tile == 4 and binding.use_coop
  result, programs = _execute_capture(linear, binding, monkeypatch)
  # external_sum (N,16) partials plus the vocab scalar-reduce: the legacy chain is intact.
  assert [p.output_spec.shape for p in programs] == [(VOCAB_ROWS, 16), (VOCAB_ROWS,)]
  assert result.shape == (1, 1, VOCAB_ROWS)
  ast = _emit(programs[0], rows=VOCAB_ROWS, k=VOCAB_K, in_kernel=False)
  assert ast.arg.name == "q6k_gen_coop_151936_4096"
  # The fail-closed backstop: an illegal in-kernel spec still raises on the single-warp gate.
  with pytest.raises(ValueError, match="single warp"):
    Q6KGEMVRouteSpec(rows=VOCAB_ROWS, k=VOCAB_K, row_tile=4, reduction="in_kernel").validate()


def test_metal_vocab_unchanged_external_sum(monkeypatch):
  cap = QKPrimitiveCapability(backend="METAL", architecture=None, wave_size=None, supports_warp_shfl_xor=None)
  linear = _FakeQ6Linear(cap, fusion=False, n_rows=VOCAB_ROWS)
  binding = _bind(linear)
  assert binding.use_coop  # safe-default row_tile=4 divides the vocab rows
  assert not linear.route_admission.fusion_admitted
  result, programs = _execute_capture(linear, binding, monkeypatch)
  assert [p.output_spec.shape for p in programs] == [(VOCAB_ROWS, 16), (VOCAB_ROWS,)]
  assert result.shape == (1, 1, VOCAB_ROWS)
  ast = _emit(programs[0], rows=VOCAB_ROWS, k=VOCAB_K, in_kernel=False)
  assert ast.arg.name == "q6k_gen_coop_151936_4096"


def test_nv_sm120_fused_vocab_census_one_program_not_five(monkeypatch):
  """Unit-level census assertion: the fused NV vocab path emits exactly one program; the
  scalar-reduce and scatter-chain kernels (1 + 4 per token in the prime census) are structurally
  gone from this path. The GPU census gate reports the same 1021 -> 1016 per-token move."""
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120", wave_size=32, supports_warp_shfl_xor=True)
  linear = _FakeQ6Linear(cap, fusion=True, n_rows=VOCAB_ROWS)
  binding = _bind(linear)
  _, programs = _execute_capture(linear, binding, monkeypatch)
  assert len(programs) == 1
  assert programs[0].program_id == f"{binding.candidate_id}.gemv"
