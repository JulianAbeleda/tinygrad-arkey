import os

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.helpers import Context, Target
from tinygrad.renderer.amd.dsl import v as amd_v
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.amd.elf import kernel_descriptor_from_elf
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, VBASE, isel_bitcast, isel_cast, isel_matcher, lower_inst
from tinygrad.runtime.autogen.amd.rdna3.ins import v_cvt_f16_f32_e32, v_pack_b32_f16
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


def test_scalar_half_allocation_cannot_alias_a_live_dword_vgpr():
  ctx = IselContext(UOp.sink())
  address = UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.int32, 0).rtag(),), AMDOps.V_MOVK,
                tag=(ctx.vreg(VBASE[1:]),))
  source = UOp(Ops.INS, dtypes.float32, (UOp.const(dtypes.float32, 2.0).rtag(),), AMDOps.V_CONST,
               tag=(ctx.vreg(VBASE[1:]),))
  half = isel_cast(ctx, UOp(Ops.CAST, dtypes.half, (source,)))
  assert half.reg.cons and max(r.index for r in half.reg.cons) < 128
  restored = isel_cast(ctx, UOp(Ops.CAST, dtypes.float32, (half,)))
  # Keep the dword address live across the fp16 definition and use.
  final = UOp(Ops.INS, dtypes.int32, (address, restored), AMDOps.V_IADD, tag=(ctx.vreg(VBASE[1:]),))
  uops = list(UOp.sink(final).toposort())
  regalloc = LinearScanRegallocContext(uops, _renderer())

  address_reg = regalloc.reals[uops.index(address)][address.reg]
  half_reg = regalloc.reals[uops.index(half)][half.reg]
  assert half_reg.index < 128
  assert address_reg.index != half_reg.index
  assert not regalloc.spills


def test_scalar_half_restriction_does_not_remove_real_high_dword_vgprs():
  ctx = IselContext(UOp.sink())
  # These are independent physical resources: scalar-half v114.l and whole
  # dword v242.  Only the scalar-half candidate pool is restricted.
  high_dword = UOp(Ops.INS, dtypes.int32, (UOp.const(dtypes.int32, 7).rtag(),), AMDOps.V_MOVK,
                   tag=(ctx.vreg((VBASE[242],)),))
  half = UOp(Ops.INS, dtypes.half, (UOp.const(dtypes.half, 1.0).rtag(),), AMDOps.V_CONST,
             tag=(ctx.vreg((VBASE[114],)),))
  final = UOp(Ops.INS, dtypes.int32, (high_dword, half), AMDOps.V_IADD, tag=(ctx.vreg((VBASE[243],)),))
  uops = list(UOp.sink(final).toposort())
  regalloc = LinearScanRegallocContext(uops, _renderer())

  assert regalloc.reals[uops.index(high_dword)][high_dword.reg].index == 242
  assert regalloc.reals[uops.index(half)][half.reg].index == 114
  assert not regalloc.spills


def test_gfx11_f16_vop_destination_bit7_is_a_half_selector_not_vgpr128():
  # The DSL's explicit high-half spelling and raw encoded destination 128+i
  # produce the same VOP1 bits.  A dtype-only allocator therefore cannot
  # interpret candidate 242 as both real v242 and v114.h.
  explicit_high = v_cvt_f16_f32_e32(amd_v[114].h, amd_v[1])
  encoded_bit7 = v_cvt_f16_f32_e32(amd_v[242], amd_v[1])
  assert explicit_high.to_bytes() == encoded_bit7.to_bytes()


def test_uint16_bitcast_to_half_canonicalizes_a_high_physical_source():
  ctx = IselContext(UOp.sink())
  raw = UOp(Ops.INS, dtypes.uint16, arg=AMDOps.V_AND, tag=(ctx.vreg((VBASE[132],)),))
  selected = isel_bitcast(ctx, UOp(Ops.BITCAST, dtypes.half, (raw,)))
  assert selected.arg is AMDOps.V_HALF_CANON and selected.dtype is dtypes.half
  assert selected.src == (raw,)
  assert selected.reg.cons and max(r.index for r in selected.reg.cons) < 128

  physical = selected.replace(src=(raw.replace(tag=(Register("v132", 132),)),), tag=(Register("v5", 5),))
  encoded = lower_inst(physical)
  assert str(encoded.arg) == "v_mov_b32_e32(v[5], v[132])"
  assert encoded.arg.to_bytes()


def test_vpack_has_explicit_half_selectors_and_keeps_full_physical_sources():
  # V_PACK is VOP3: its 9-bit source fields still name all 256 physical
  # VGPRs, while op_sel independently chooses a high half.  It is therefore
  # unlike scalar-f16 VOP1/VOP2 and needs no raw-register canonicalization.
  physical_high_register = v_pack_b32_f16(amd_v[1], amd_v[132], amd_v[2])
  explicit_high_half = v_pack_b32_f16(amd_v[1], amd_v[4].h, amd_v[2])
  assert physical_high_register.to_bytes() != explicit_high_half.to_bytes()


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


def test_half_multiply_selects_and_encodes_gfx1100_native_v_mul_f16():
  # The pinned llama Q4 metadata recurrence multiplies scale/sum as half2 (record producers: base * code.cast(half)),
  # so the parent MUL is a genuine fp16 multiply.  Without a typed rule it survived isel unselected, which desynced
  # the positional regalloc rewrite rather than reporting itself.  Selection must stay fp16: no fp32 widening.
  ctx = IselContext(UOp.sink())
  a = UOp(Ops.INS, dtypes.half, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  b = UOp(Ops.INS, dtypes.half, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v2", 2),)),))
  source = UOp(Ops.MUL, dtypes.half, (a, b))
  assert all(s.dtype is dtypes.half for s in source.src) and source.dtype is dtypes.half

  selected = isel_matcher.rewrite(source, ctx)
  assert selected is not None and selected.arg is AMDOps.V_MUL_F16
  assert selected.dtype is dtypes.half                      # never widened to fp32
  assert selected.arg is not AMDOps.V_MUL                    # V_MUL lowers to v_mul_f32_e32
  assert not any(u.op is Ops.MUL for u in selected.toposort())
  allocated = isel_matcher.rewrite(selected, ctx)
  assert allocated.reg.cons and max(r.index for r in allocated.reg.cons) < 128

  selected_where = isel_matcher.rewrite(UOp(Ops.WHERE, dtypes.half, (UOp.const(dtypes.bool, True), a, b)), ctx)
  assert selected_where.reg.cons and max(r.index for r in selected_where.reg.cons) < 128

  physical = selected.replace(src=(a.replace(tag=(Register("v1", 1),)), b.replace(tag=(Register("v2", 2),))),
                              tag=(Register("v3", 3),))
  encoded = lower_inst(physical)
  assert str(encoded.arg).startswith("v_mul_f16_e32")        # native fp16 multiply, one rounding per lane
  assert encoded.arg.to_bytes()


def test_half_multiply_by_a_constant_encodes_a_16_bit_literal_not_an_fp32_pattern():
  # gfx11 VOP f16 literals carry the value in the LOW 16 bits.  Encoding a Python float (the fp32 path used by
  # v_mul_f32_e32) puts an fp32 bit pattern there, so the hardware reads a silently wrong constant: 0.3f decodes as
  # -0.0027h.  Only inline constants (0.0/1.0) survive that bug, which is why it hides behind "the program encodes".
  import struct
  ctx = IselContext(UOp.sink())
  a = UOp(Ops.INS, dtypes.half, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  selected = isel_matcher.rewrite(UOp(Ops.MUL, dtypes.half, (a, UOp.const(dtypes.half, 0.3))), ctx)
  assert selected is not None and selected.arg is AMDOps.V_MUL_F16

  physical = selected.replace(src=(a.replace(tag=(Register("v1", 1),)), selected.src[1]), tag=(Register("v2", 2),))
  encoded = lower_inst(physical)
  raw = encoded.arg.to_bytes()
  assert str(encoded.arg).startswith("v_mul_f16_e32") and len(raw) == 8   # 4-byte inst + 4-byte literal dword
  assert struct.unpack("<e", raw[4:6])[0] == struct.unpack("<e", struct.pack("<e", 0.3))[0]
