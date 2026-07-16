import os

import numpy as np
import pytest

from extra.qk.prefill.q4k_q8_five_buffer_artifact import build_q4k_q8_five_buffer_artifact
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import compile_q4k_q8_five_buffer_program
from extra.qk.prefill.q4k_q8_five_buffer_pipeline import build_q4k_q8_five_buffer_execution
from extra.qk.runtime_specs import derive_q4k_q8_1_five_buffer_candidate
from test.unit.test_q4k_q8_five_buffer_compile_adapter import _payload
from tinygrad import Tensor
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
from tinygrad.renderer.isa import CompilerCaptureProof, CompilerRegisterLease, IselContext
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, isel_index, isel_store
from tinygrad.uop.ops import Ops, RegisterResidentAccumulator, UOp


@pytest.fixture(scope="module")
def role_program():
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((512, 1024, 5120), role="attn_kv"))
  return compile_q4k_q8_five_buffer_program(entry.payload, entry.canonical_identity)


def test_role_sized_direct_global_descriptor_has_owned_accumulators_and_no_ds(role_program):
  program, admission = role_program
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE)
  proof = next(u.arg for u in program.src if u.op is Ops.LINEAR)
  assert isinstance(proof, CompilerCaptureProof)
  assert proof.authority == "final_regalloc" and len(proof.owned_storage) == 2
  assert any(x.purpose == "fixed_fp32_accumulator" for x in proof.leases)
  spans = [set(range(x.start, x.end)) for x in proof.leases]
  assert all(spans[i].isdisjoint(spans[j]) for i in range(len(spans)) for j in range(i+1, len(spans)))
  assert admission.active_lds_bytes == kernel_descriptor_from_elf(binary).group_segment_fixed_size == proof.lds_bytes == 0
  assert "ds_load" not in source and "ds_store" not in source
  assert source.count("v_wmma_i32_16x16x16_iu8") == 1


def test_assembly_projection_strips_only_exact_proven_define_reg_identity():
  owned = UOp(Ops.DEFINE_REG, dtypes.float.ptr(1, AddrSpace.REG), arg=0)
  unowned = UOp(Ops.DEFINE_REG, dtypes.float.ptr(1, AddrSpace.REG), arg=1)
  sink = UOp.sink(owned, unowned)
  program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD:ISA:gfx1100")))
  leases = tuple(CompilerRegisterLease(role, "vgpr", i, i+1, "test", True, 1, ("begin", "end"))
                 for role, i in (("A", 1), ("B", 2), ("C", 3)))
  proof = CompilerCaptureProof(leases, owned_storage=(owned,))
  projected = AMDISARenderer._assembly_program(program, proof)
  assert [u for u in projected.src[0].toposort() if u.op is Ops.DEFINE_REG] == [unowned]


def test_dynamic_or_unproven_register_index_remains_lds_backed():
  dreg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(8, AddrSpace.REG), arg=0)
  dynamic = UOp(Ops.DEFINE_VAR, dtypes.int32, arg="lane")
  index = dreg.index(dynamic)
  ctx = IselContext(UOp.sink(index))
  selected = isel_index(ctx, index)
  assert selected.arg == "lds" and dreg in ctx._lds


def test_declared_register_resident_accumulator_fails_closed_on_dynamic_index():
  dreg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(8, AddrSpace.REG), arg=0, tag=RegisterResidentAccumulator(Ops.ADD))
  index = dreg.index(UOp(Ops.DEFINE_VAR, dtypes.int32, arg="lane"))
  with pytest.raises(NotImplementedError, match="static index"): isel_index(IselContext(UOp.sink(index)), index)


def test_wmma_owned_accumulator_fails_closed_on_dynamic_index():
  dreg = UOp(Ops.DEFINE_REG, dtypes.float.ptr(8, AddrSpace.REG), arg=0)
  index = dreg.index(UOp(Ops.DEFINE_VAR, dtypes.int32, arg="lane"))
  ctx = IselContext(UOp.sink(index)); ctx._wmmaacc = {id(dreg)}
  with pytest.raises(NotImplementedError, match="static index"): isel_index(ctx, index)


def test_expanded_fixed_accumulator_assignment_keeps_write_ownership():
  order, store = UOp(Ops.NOOP), UOp(Ops.STORE)
  reads, vals = [], []
  for pin in (8, 9, 10):
    meta = UOp(Ops.NOOP, arg=("fixed_acc", "add"))
    reads.append(UOp(Ops.INS, dtypes.float32, src=(order, UOp.const(dtypes.int32, pin).rtag(), meta), arg=AMDOps.ACCUM_READ))
    vals.append(UOp.const(dtypes.float32, pin))
  out = isel_store(IselContext(UOp.sink()), UOp(Ops.NOOP, dtypes.float32.vec(3), src=tuple(reads)), UOp(Ops.NOOP, dtypes.float32.vec(3), src=tuple(vals)), store)
  assert len([u for u in out.toposort() if u.op is Ops.INS and u.arg is AMDOps.ACCUM_WRITE]) == 3


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_bounded_five_buffer_pipeline_correctness_on_amd():
  m = n = 16; k = 256
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((m, n, k)))
  artifact = build_q4k_q8_five_buffer_artifact(m, n, k, seed=31)
  source = np.zeros((m, k), dtype=np.float32)
  source[np.arange(m), np.asarray(artifact.metadata["selected_positions"])] = \
    np.asarray(artifact.metadata["coefficients_fp32"], dtype=np.float32)
  graph = build_q4k_q8_five_buffer_execution(entry.payload, entry.canonical_identity,
    Tensor(artifact.q4_packed_words, device="AMD"), Tensor(source.reshape(-1), device="AMD"))
  np.testing.assert_allclose(graph.output.numpy(), artifact.reference, rtol=3e-4, atol=3e-3)
