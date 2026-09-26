"""cp.async multi-stage precontract pipelines: rendered structure, stage honouring, and launch-sized LDS (no GPU)."""
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.helpers import Target
from tinygrad.llm.prefill_candidate_runtime import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, RuntimeLocalBytes
import tinygrad.codegen.opt.postrange as pr


def _context(tile, waves, stages, async_copy=True):
  (tm, tn, tk), (wm, wn) = tile, waves
  stride = tk * 2 + 16
  geometry = KernelTileGeometry(tile, waves, wm * wn * 32, 32,
    (KernelLDSWindow("A", 0, tm * stride, stride), KernelLDSWindow("B", tm * stride, (tm + tn) * stride, stride)))
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                KernelStage1PipelinePlan(stages, geometry.lds_bytes, 1, async_copy=async_copy))


def _render(context, dtype=dtypes.bfloat16, m=256, n=256, k=512):
  # Programs are cached by the pre-optimization AST, which does not carry the candidate: give every case its own shape.
  a, w = Tensor.empty(m, k, dtype=dtype, device="CPU"), Tensor.empty(n, k, dtype=dtype, device="CPU")
  (call,) = a.dot(w.T, dtype=dtypes.float).schedule_linear().src
  key = pr.warmstart_key({m, n}, k)
  with pr.warmstart_candidate_state({key: (Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key: context}):
    program = to_program(call.src[0], CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  return next(u.arg for u in program.src if u.op is Ops.SOURCE), program.arg.aux


def _loop_body(src):
  lines = src.splitlines()
  start = next(i for i, line in enumerate(lines) if line.strip().startswith("for ("))
  return lines[:start], lines[start:]


@pytest.mark.parametrize("dtype", [dtypes.bfloat16, dtypes.half])
@pytest.mark.parametrize("stages", [2, 3, 4])
def test_async_ring_renders_cp_async_commit_and_wait_groups_per_stage(dtype, stages):
  context = _context((64, 64, 32), (2, 2), stages)
  src, _ = _render(context, dtype, n=64 * (stages + 2))
  prologue, body = _loop_body(src)
  copies_per_tile = (64 + 64) * (32 * 2 // 16) // 128          # b128 vectors per tile / threads
  assert sum("cp.async.cg.shared.global" in line for line in prologue) == (stages - 1) * copies_per_tile
  assert sum("cp.async.commit_group" in line for line in prologue) == stages - 1
  assert sum("cp.async.cg.shared.global" in line for line in body) == copies_per_tile
  assert sum("cp.async.commit_group" in line for line in body) == 1
  waits = [line.strip() for line in body if "cp.async.wait_group" in line]
  assert waits == [f'asm volatile("cp.async.wait_group {stages-2};" ::: "memory");', 'asm volatile("cp.async.wait_group 0;" ::: "memory");']
  # the next tile's copies are issued before this tile's MMAs so the two overlap
  first_copy = next(i for i, line in enumerate(body) if "cp.async.cg" in line)
  first_mma = next(i for i, line in enumerate(body) if "__WMMA_" in line and "=" in line)
  wait = next(i for i, line in enumerate(body) if "wait_group" in line)
  sync = next(i for i, line in enumerate(body) if "__syncthreads" in line)
  assert wait < sync < first_copy < first_mma
  assert "ld.global" not in src and "LDG" not in src


def test_arena_above_the_static_limit_is_launch_sized():
  small_src, small_aux = _render(_context((64, 64, 32), (2, 2), 3), n=128)    # 30 KB: static
  assert "extern __shared__" not in small_src and small_aux == ()
  big = _context((128, 128, 32), (4, 2), 3)                                    # 60 KB: launch-sized
  src, aux = _render(big, m=384, n=384)
  assert "extern __shared__ __align__(16) nv_bfloat16 buf0[];" in src
  assert aux == (61440,) and isinstance(aux[0], RuntimeLocalBytes)


def test_arena_beyond_the_target_maximum_fails_closed():
  with pytest.raises(Exception, match="exceeds this target.s workgroup-local limits"):
    _render(_context((128, 256, 64), (4, 2), 4), m=128, n=256, k=256)          # 4 x 384 x 144 B > 99 KB


def test_async_plan_validation():
  KernelStage1PipelinePlan(4, 1024, 1, async_copy=True)
  with pytest.raises(ValueError, match="must be 1 or 2"): KernelStage1PipelinePlan(3, 1024, 1)
  with pytest.raises(ValueError, match=r"\[2, 8\]"): KernelStage1PipelinePlan(9, 1024, 1, async_copy=True)
  with pytest.raises(ValueError, match=r"\[2, 8\]"): KernelStage1PipelinePlan(1, 1024, 1, async_copy=True)


def test_targets_without_async_copy_ops_fail_closed(monkeypatch):
  monkeypatch.setattr(CUDARenderer, "async_copy_ops", None)
  with pytest.raises(Exception, match="async_copy_ops"): _render(_context((64, 64, 32), (2, 2), 3), m=192, n=192, k=192)


def test_runtime_local_bytes_is_an_int_the_nv_runtime_extends_by_its_reserve():
  from tinygrad.runtime.ops_nv import NV_RESERVED_SHARED_BYTES
  assert RuntimeLocalBytes(61440) == 61440 and NV_RESERVED_SHARED_BYTES == 1024


def test_candidate_registry_decodes_async_copy_pipelines():
  import json
  from extra.llm_research.mint_typed_candidate_template import _compact
  from tinygrad.llm.prefill_candidate_runtime import NV_ARTIFACT, candidate_registry, expand_compact_candidate_set
  nv = json.loads(NV_ARTIFACT.read_text())
  template = json.loads(json.dumps(nv["template"]))
  template["schedule"]["pipeline"].update(buffer_count=3, async_copy=True)
  template["static_constraints"]["max_lds_bytes"] = 101376
  compact = _compact("async_test", nv["target"], template, [("r", (512, 512, 512))])
  (admission,) = candidate_registry(expand_compact_candidate_set(compact, **nv["target"])).admissions
  assert admission.pipeline_plan.async_copy is True and admission.pipeline_plan.buffer_count == 3
  (sync,) = candidate_registry(expand_compact_candidate_set(_compact("sync_test", nv["target"], nv["template"],
                                                                     [("r", (512, 512, 512))]), **nv["target"])).admissions
  assert sync.pipeline_plan.async_copy is False and sync.pipeline_plan.buffer_count == 2


# --- matrix (ldmatrix) fragment loads and XOR-swizzled LDS windows ---

def _matrix_context(tile, waves, stages, *, swizzle, async_copy=True):
  (tm, tn, tk), (wm, wn) = tile, waves
  stride = tk * 2 + (0 if swizzle else 16)
  geometry = KernelTileGeometry(tile, waves, wm * wn * 32, 32,
    (KernelLDSWindow("A", 0, tm * stride, stride, xor_swizzle=swizzle),
     KernelLDSWindow("B", tm * stride, (tm + tn) * stride, stride, xor_swizzle=swizzle)))
  return KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "0" * 64, geometry,
                                KernelStage1PipelinePlan(stages, geometry.lds_bytes, 1, async_copy=async_copy, matrix_fragments=True))


def test_matrix_fragment_layout_is_derived_from_the_descriptor():
  from tinygrad.codegen.opt.kernel_lds import derive_matrix_fragment_layout
  tcs = {tc.dtype_in: tc for tc in CUDARenderer(Target.parse("NV:CUDA:sm_120")).tensor_cores if tc.dims == (8, 16, 16) and tc.dtype_out == dtypes.float}
  for dtype in (dtypes.bfloat16, dtypes.half):
    a, b = derive_matrix_fragment_layout(tcs[dtype], 0), derive_matrix_fragment_layout(tcs[dtype], 1)
    assert (a.width, a.select_terms) == (4, ((8, 0), (0, 8)))   # m16n8k16 A: rows +8, then K +8
    assert (b.width, b.select_terms) == (2, ((0, 8),))          # B: K +8
  from tinygrad.codegen.opt import tc as tc_mod
  with pytest.raises(ValueError):   # an fp32 (tf32-style) fragment is not b16 matrices
    derive_matrix_fragment_layout(next(t for t in tc_mod.get_cuda("sm_120") if t.dtype_in == dtypes.float), 0)


@pytest.mark.parametrize("row_bytes", [32, 64, 128, 256])
def test_xor_swizzle_is_conflict_free_for_eight_row_groups(row_bytes):
  from tinygrad.codegen.opt.kernel_lds import xor_swizzle_conflict_free
  window = KernelLDSWindow("A", 0, 64 * row_bytes, row_bytes, xor_swizzle=True)
  assert xor_swizzle_conflict_free(window, bank_dwords=32)


def test_xor_swizzle_is_a_permutation_of_each_row():
  from tinygrad.codegen.opt.kernel_lds import lds_row_element
  window = KernelLDSWindow("A", 0, 64 * 64, 64, xor_swizzle=True)
  for row in range(16):
    slots = {lds_row_element(window, row, k, 2, bank_dwords=32) for k in range(32)}
    assert slots == set(range(row * 32, row * 32 + 32))


@pytest.mark.parametrize("swizzle", [False, True])
def test_matrix_fragments_render_one_named_native_load_per_fragment(swizzle):
  src, _ = _render(_matrix_context((64, 64, 32), (2, 2), 3, swizzle=swizzle), m=128, n=64 * (13 if swizzle else 7), k=512)
  _, body = _loop_body(src)
  # warp tile 32x32: 2 A fragments (x4) and 4 B fragments (x2) per K16 substep, 2 substeps; 16 MMAs
  assert sum("tg_ldmatrix_x4(" in line for line in body) == 4 and sum("tg_ldmatrix_x2(" in line for line in body) == 8
  assert sum("__WMMA_" in line and "=" in line for line in body) == 16
  assert not any("nv_bfloat16 val" in line for line in body)   # no scalar LDS fragment element loads
  assert ("^" in src.split("__launch_bounds__")[1]) == swizzle


def test_swizzled_window_needs_the_target_bank_facts(monkeypatch):
  monkeypatch.setattr(CUDARenderer, "lds_bank_dwords", None)
  with pytest.raises(Exception, match="lds_bank_dwords"):
    _render(_matrix_context((64, 64, 32), (2, 2), 3, swizzle=True), m=128, n=64 * 9, k=512)


def test_matrix_fragments_need_native_fragment_loads(monkeypatch):
  monkeypatch.setattr(CUDARenderer, "native_fragment_x4", None)
  with pytest.raises(Exception, match="native_fragment_x2/x4"):
    _render(_matrix_context((64, 64, 32), (2, 2), 3, swizzle=False), m=128, n=64 * 11, k=512)


def test_candidate_registry_decodes_matrix_fragments_and_swizzle_keys():
  import json
  from extra.llm_research.mint_typed_candidate_template import _compact
  from tinygrad.llm.prefill_candidate_runtime import NV_ARTIFACT, candidate_registry, expand_compact_candidate_set
  nv = json.loads(NV_ARTIFACT.read_text())
  template = json.loads(json.dumps(nv["template"]))
  template["schedule"]["pipeline"].update(buffer_count=4, async_copy=True, fragment_load="matrix")
  tk = template["schedule"]["tile"]["k"]
  rows = template["schedule"]["tile"]["m"], template["schedule"]["tile"]["n"]
  template["schedule"]["lds"].update(swizzle="xor_b128", padding=0, strides={"a": tk * 2, "b": tk * 2},
                                     windows={"a": [0, rows[0] * tk * 2], "b": [rows[0] * tk * 2, sum(rows) * tk * 2]})
  template["static_constraints"]["max_lds_bytes"] = 101376
  (admission,) = candidate_registry(expand_compact_candidate_set(_compact("mx", nv["target"], template, [("r", (512, 512, 512))]),
                                                                  **nv["target"])).admissions
  assert admission.pipeline_plan.matrix_fragments and all(w.xor_swizzle for w in admission.geometry.lds_windows)
  template["schedule"]["pipeline"]["fragment_load"] = "bogus"
  with pytest.raises(ValueError, match="fragment load"):
    candidate_registry(expand_compact_candidate_set(_compact("bad", nv["target"], template, [("r", (512, 512, 512))]), **nv["target"]))


def test_xor_bound_is_integer_only_and_tight():
  from tinygrad.uop.ops import UOp
  from tinygrad.dtype import dtypes as dt
  r = UOp.range(6, 991)
  assert (r ^ 3).vmin == 0 and (r ^ 3).vmax == 7
  b = UOp.variable("b", 0, 1, dt.bool) if hasattr(UOp, "variable") else None
  if b is not None: assert (b ^ True).vmax in (True, 1)
