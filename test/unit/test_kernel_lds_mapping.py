import sys

import pytest

from tinygrad.codegen.opt.kernel_lds import (binary_axis_count, cooperative_store_octet_rows, cooperative_store_row,
                                             cooperative_store_row_rotation, derive_precontract_factors, derive_precontract_shape_factors,
                                             fold_binary_axes, validate_precontract_carriers, validate_wmma_descriptor)
from tinygrad.codegen.opt.tc import amd_cdna_161616, amd_rdna3
from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import HIPRenderer, MetalRenderer
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry


def _geometry():
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))

def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)


def test_logical_rdna3_formulas_match_core_tensor_descriptor():
  tc = _tc()
  validate_wmma_descriptor(tc)
  assert tc.dims == (16, 16, 16) and tc.threads == 32 and tc.elements_per_thread == (16, 16, 8)


def test_metal_descriptor_is_admitted_by_the_same_generic_validator():
  """No RDNA3 numbers here at all: Metal's own dims/threads/elements_per_thread/swizzle are self-consistent
  and its dtype_in/dtype_out pairing is one this precontract path expresses, so it is admitted by the exact
  same function AMD uses -- one mechanism driven by declared facts, not a second frozen constant block."""
  from tinygrad.codegen.opt.tc import metal
  tc = next(tc for tc in metal if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)
  assert (tc.dims, tc.threads, tc.elements_per_thread) == ((8, 8, 8), 32, (2, 2, 2))
  validate_wmma_descriptor(tc)


def test_wave64_cdna_descriptor_is_self_consistent_but_unsupported():
  """A real, valid CDNA descriptor -- self-consistent, and used for actual CDNA lowering elsewhere in
  tc.py -- is still rejected here, because this precontract path's fragment/cooperative-store math is
  proven only for a 32-wide warp and CDNA's wave64 genuinely is not one. This is the 'genuinely
  unsupported descriptor' the generalized validator must still reject: no RDNA3-vs-not special case,
  just the descriptor's own declared thread count."""
  cdna = amd_cdna_161616[0]
  assert cdna.threads == 64
  with pytest.raises(ValueError, match="threads must be 32"): validate_wmma_descriptor(cdna)


def test_binary_axis_count_derives_from_elements_per_thread_not_a_literal_four():
  """PrecontractCandidateContract.assemble used to demand exactly four binary axes per operand --
  a frozen snapshot of RDNA3's own numbers (elements_per_thread=(16,16,8) -> log2 = (4,4,3)), one
  layer below the validator PG0 fixed. Metal's descriptor (elements_per_thread=(2,2,2) -> log2 =
  (1,1,1)) is just as valid a mapping but could never satisfy the literal ``4``."""
  from tinygrad.codegen.opt.tc import metal
  amd, mtl = _tc(), next(tc for tc in metal if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)
  assert tuple(binary_axis_count(amd, i) for i in range(3)) == (4, 4, 3)
  assert tuple(binary_axis_count(mtl, i) for i in range(3)) == (1, 1, 1)


def test_fold_binary_axes_reproduces_the_original_horner_expression():
  """The four-axis fold used to be hand-unrolled as ``((a0*2+a1)*2+a2)*2+a3``. The general
  reduction must reproduce that exact value for RDNA3's four axes, and the same Horner shape for
  any other axis count (Metal folds one axis; the fold degenerates to that axis itself)."""
  a0, a1, a2, a3 = 1, 0, 1, 1
  assert fold_binary_axes((a0, a1, a2, a3)) == ((a0*2+a1)*2+a2)*2+a3
  assert fold_binary_axes((1,)) == 1
  with pytest.raises(ValueError, match="zero binary axes"): fold_binary_axes(())


def test_precontract_factor_derivation_exact_anchor_and_legal_smaller_family():
  exact = derive_precontract_factors(_geometry(), _tc())
  assert (exact.subtiles_m, exact.subtiles_n, exact.waves_m, exact.waves_n, exact.k_substeps,
          exact.vectors_per_row, exact.loads_a, exact.loads_b) == (2, 4, 4, 2, 2, 4, 2, 2)
  smaller = KernelTileGeometry((64, 64, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 10240, 80)))
  factors = derive_precontract_factors(smaller, _tc())
  assert (factors.subtiles_m, factors.subtiles_n, factors.k_substeps, factors.vectors_per_row,
          factors.loads_a, factors.loads_b) == (2, 2, 2, 4, 2, 2)


def test_shape_factor_derivation_is_independent_of_lds_windows():
  from types import SimpleNamespace
  shape_only = SimpleNamespace(tile=(128, 128, 32), waves=(4, 2), threads=256, wave_size=32)
  factors = derive_precontract_shape_factors(shape_only, _tc())
  assert (factors.subtiles_m, factors.subtiles_n, factors.k_substeps,
          factors.vectors_per_row, factors.loads_a, factors.loads_b) == (2, 4, 2, 4, 2, 2)


def test_wmma_carrier_abi_validation_is_storage_independent():
  tc = _tc()
  validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(8), tc=tc)
  with pytest.raises(ValueError, match="fragment carrier"):
    validate_precontract_carriers(dtypes.half.vec(8), dtypes.float.vec(8), tc=tc)
  with pytest.raises(ValueError, match="accumulator carrier"):
    validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(16), tc=tc)


class _DescriptorDrift:
  def __init__(self, base, field, value): self.base, self.field, self.value = base, field, value
  def __getattr__(self, name): return self.value if name == self.field else getattr(self.base, name)


@pytest.mark.parametrize("field,value,match", (
  ("dims", (16, 16, 8), "self-consistent"), ("threads", 64, "self-consistent"),
  ("elements_per_thread", (16, 8, 8), "self-consistent"),
  ("dtype_in", dtypes.bfloat16, "dtype_in"), ("dtype_out", dtypes.half, "dtype_out"),
  ("opts", ("l0",), "self-consistent"),
  ("swizzle", (((), (), ()), ((), (), ())), "self-consistent"),
))
def test_descriptor_fingerprint_drift_fails_closed(field, value, match):
  drift = _DescriptorDrift(_tc(), field, value)
  with pytest.raises(ValueError, match=match): validate_wmma_descriptor(drift)


def test_non_permutation_remap_fails_closed():
  """A swizzle that satisfies every LaneMap.validate() length check (so the shape-only self-consistency
  assertions all pass) but drops one lane/value coordinate and duplicates another is still rejected: the
  precontract math needs an honest bijection, not merely matching part counts."""
  tc = _tc()
  broken_local = tc.swizzle[0][0][:-1] + (tc.swizzle[0][0][0],)  # duplicate the first symbol, drop the last
  broken_swizzle = ((broken_local, tc.swizzle[0][1], tc.swizzle[0][2]), tc.swizzle[1])
  drift = _DescriptorDrift(tc, "swizzle", broken_swizzle)
  with pytest.raises(ValueError, match="permutation"): validate_wmma_descriptor(drift)


def test_renderers_declare_lds_bank_facts_amd_known_metal_unknown():
  """PG1's vocabulary: bank count/cycle-lanes are TG2 capability facts, sourced from each renderer's
  own reporting, never restated in kernel_lds.py. AMD's ISA manuals document both numbers for every
  generation this renderer targets; Apple documents neither for its threadgroup memory, so Metal must
  read back the same explicit-unknown ``None`` that ``wave_size`` uses for an unreported width -- not
  a silent 32/8 borrowed from AMD."""
  amd = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  assert (amd.lds_bank_dwords, amd.lds_bank_cycle_lanes) == (32, 8)
  if sys.platform == "darwin":
    metal = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
    assert (metal.lds_bank_dwords, metal.lds_bank_cycle_lanes) == (None, None)


def test_cooperative_store_rotation_fires_for_amds_shipped_geometry():
  """The real AMD production geometry (stride 80 B, vector 16 B -> vectors_per_row=4, rows a multiple
  of 8) must still take the rotation once AMD's declared facts (32 bank dwords, 8 cycle lanes) are
  threaded through: this is the 'AMD must still take the rotation' regression this change must not
  cause. Same geometry with unknown facts (Metal, or any unreported target) always skips."""
  amd = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  assert cooperative_store_row_rotation(vectors_per_row=4, rows=128, stride_bytes=80, bank_dwords=amd.lds_bank_dwords,
                                        bank_cycle_lanes=amd.lds_bank_cycle_lanes) is True
  if sys.platform == "darwin":
    metal = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
    assert cooperative_store_row_rotation(vectors_per_row=4, rows=128, stride_bytes=80, bank_dwords=metal.lds_bank_dwords,
                                          bank_cycle_lanes=metal.lds_bank_cycle_lanes) is False


def test_cooperative_store_rotation_unknown_facts_never_fall_through_to_amd_arithmetic():
  """Before this change, an unreported target still ran AMD's literal 32/8 bank arithmetic, so a
  probe that happened to reuse AMD's tuned geometry (vectors_per_row==4, rows%8==0) took the
  rotation by numeric accident -- nothing target-specific was ever consulted. Either fact missing
  must return False outright, never fall through to compute ``quads`` from a default."""
  assert cooperative_store_row_rotation(vectors_per_row=4, rows=128, stride_bytes=80, bank_dwords=None, bank_cycle_lanes=8) is False
  assert cooperative_store_row_rotation(vectors_per_row=4, rows=128, stride_bytes=80, bank_dwords=32, bank_cycle_lanes=None) is False
  assert cooperative_store_row_rotation(vectors_per_row=4, rows=128, stride_bytes=80, bank_dwords=None, bank_cycle_lanes=None) is False


def test_cooperative_store_row_defaults_to_unknown_and_is_identity_when_skipped():
  """``cooperative_store_row``'s bank facts default to ``None`` so an unmigrated caller keeps today's
  safe behaviour (identity), and applying known AMD facts still produces the documented permutation
  of an aligned eight-row block (0,4,1,5,2,6,3,7), not merely 'some' rotation."""
  assert cooperative_store_row(5, vectors_per_row=4, rows=128, stride_bytes=80) == 5
  amd = HIPRenderer(Target.parse("AMD:HIP:gfx1100"))
  rotated = [cooperative_store_row(r, vectors_per_row=4, rows=128, stride_bytes=80,
                                   bank_dwords=amd.lds_bank_dwords, bank_cycle_lanes=amd.lds_bank_cycle_lanes) for r in range(8)]
  assert rotated == [0, 4, 1, 5, 2, 6, 3, 7]


def test_cooperative_store_octet_rows_is_parameterized_and_unreferenced():
  """cooperative_store_octet_rows is dead code (PG1a's audit: nothing in this tree calls it), but it
  still hardcoded AMD's 8-lanes-per-cycle fact in its own arithmetic and docstring. Parameterized on
  the same declared fact as the row rotation: known AMD facts reproduce the original 8//vectors_per_row
  answer, and an unknown fact still raises rather than silently defaulting to AMD's 8."""
  assert cooperative_store_octet_rows(4, bank_cycle_lanes=8) == 2
  with pytest.raises(ValueError, match="whole rows"): cooperative_store_octet_rows(4, bank_cycle_lanes=None)
  with pytest.raises(ValueError, match="whole rows"): cooperative_store_octet_rows(3, bank_cycle_lanes=8)


def test_precontract_factor_derivation_rejects_nondivisible_and_bad_windows():
  nondivisible = KernelTileGeometry((80, 64, 16), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 3840, 48), KernelLDSWindow("B", 3840, 6912, 48)))
  with pytest.raises(ValueError, match="whole per-wave"): derive_precontract_factors(nondivisible, _tc())
  uneven = KernelTileGeometry((64, 64, 32), (4, 4), 512, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 10240, 80)))
  with pytest.raises(ValueError, match="divide evenly"): derive_precontract_factors(uneven, _tc())
