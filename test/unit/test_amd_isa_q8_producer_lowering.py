import os

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Context, Target
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, isel_cast, isel_matcher, lower_inst
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_q4k_q8_reference import q8_1_mmq_ds4_quantize_reference
from extra.qk.q4k_q8_activation_producer import (AMD_NATIVE_VGPR_WAVE_REDUCE, PhysicalDS4Q8ActivationSpec,
  produce_physical_ds4_q8_1)


def _renderer(): return AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))


def _mnemonics(prg):
  linear = next(u for u in prg.src if u.op is Ops.LINEAR)
  insts = [u.arg for u in linear.src if not isinstance(u.arg, tuple)]
  assert insts and all(inst.to_bytes() for inst in insts)
  return [str(inst).split("(", 1)[0] for inst in insts]


def test_float32_reciprocal_selects_and_encodes_gfx1100_v_rcp_f32():
  ctx = IselContext(UOp.sink())
  source = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  selected = isel_matcher.rewrite(UOp(Ops.RECIPROCAL, dtypes.float32, (source,)), ctx)
  assert selected is not None and selected.arg is AMDOps.V_RCP

  physical = selected.replace(src=(source.replace(tag=(Register("v1", 1),)),), tag=(Register("v2", 2),))
  encoded = lower_inst(physical)
  assert str(encoded.arg).startswith("v_rcp_f32_e32")
  assert encoded.arg.to_bytes()


def test_float32_to_signed_int8_selects_i32_conversion_and_encodes():
  ctx = IselContext(UOp.sink())
  source = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  selected = isel_cast(ctx, UOp(Ops.CAST, dtypes.int8, (source,)))
  assert selected.arg is AMDOps.V_CVT_F2I and selected.dtype is dtypes.int8

  physical = selected.replace(src=(source.replace(tag=(Register("v1", 1),)),), tag=(Register("v2", 2),))
  encoded = lower_inst(physical)
  assert str(encoded.arg).startswith("v_cvt_i32_f32_e32")
  assert encoded.arg.to_bytes()


def test_reciprocal_and_signed_int8_cast_survive_regalloc_without_spills():
  to_program_cache.clear()
  source = Tensor.empty(32, dtype=dtypes.float32)
  sink = next(u for u in source.reciprocal().cast(dtypes.int8).schedule_linear().toposort() if u.op is Ops.SINK)
  mnemonics = _mnemonics(to_program(sink, _renderer()))
  # The scheduler may vectorize this elementwise kernel; every lane must keep both native operations.
  assert mnemonics.count("v_rcp_f32_e32") >= 1
  assert mnemonics.count("v_cvt_i32_f32_e32") == mnemonics.count("v_rcp_f32_e32")


@pytest.mark.parametrize("shape", ((1, 128), (256, 256)))
def test_physical_ds4_static_to_program(shape):
  to_program_cache.clear()
  source = Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")
  spec = PhysicalDS4Q8ActivationSpec(*shape, wave_reduce_lowering=AMD_NATIVE_VGPR_WAVE_REDUCE)
  output = produce_physical_ds4_q8_1(source, spec)
  sink = next(u for u in output.values.schedule_linear().toposort() if u.op is Ops.SINK)
  program = to_program(sink, _renderer())
  mnemonics = _mnemonics(program)
  assert "v_rcp_f32_e32" in mnemonics
  assert "v_cvt_i32_f32_e32" in mnemonics
  assert "v_trunc_f32_e32" in mnemonics
  assert mnemonics.count("ds_bpermute_b32") == 10
  assert not any(mnemonic.startswith(("ds_load", "ds_store")) for mnemonic in mnemonics)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  assert kernel_descriptor_from_elf(binary).group_segment_fixed_size == 0
  assert not any(u.op is Ops.DEFINE_REG for u in program.src[0].toposort())


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
@pytest.mark.parametrize("case", ("zeros", "ramp", "seeded_boundaries"))
def test_physical_ds4_native_amd_correctness_matrix(case):
  if case == "zeros": source = np.zeros((1, 128), dtype=np.float32)
  elif case == "ramp": source = np.linspace(-127.5, 127.5, 128, dtype=np.float32).reshape(1, 128)
  else:
    source = np.random.default_rng(17).standard_normal((2, 256), dtype=np.float32) * np.float32(19)
    source.reshape(-1)[:8] = np.array([-128, -127.5, -0.5, 0, 0.5, 126.5, 127, 128], dtype=np.float32)
  ref_values, ref_scales, ref_sums = q8_1_mmq_ds4_quantize_reference(source)
  with Context(DEV="AMD:ISA"):
    spec = PhysicalDS4Q8ActivationSpec(*source.shape, wave_reduce_lowering=AMD_NATIVE_VGPR_WAVE_REDUCE)
    output = produce_physical_ds4_q8_1(Tensor(source, device="AMD"), spec)
    got_values, got_scales, got_sums = output.values.numpy(), output.scales.numpy(), output.sums.numpy()
  np.testing.assert_array_equal(got_values, ref_values)
  np.testing.assert_allclose(got_scales, ref_scales, rtol=1e-6, atol=1e-7)
  np.testing.assert_allclose(got_sums, ref_sums, rtol=1e-6, atol=2e-5)
