import pytest

from tinygrad.codegen.opt.kernel_lds import (derive_precontract_factors, derive_precontract_shape_factors,
                                             validate_precontract_carriers, validate_rdna3_wmma_descriptor)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad import dtypes
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry


def _geometry():
  return KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))

def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)


def test_logical_rdna3_formulas_match_core_tensor_descriptor():
  tc = _tc()
  validate_rdna3_wmma_descriptor(tc)
  assert tc.dims == (16, 16, 16) and tc.threads == 32 and tc.elements_per_thread == (16, 16, 8)


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
  validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(8))
  with pytest.raises(ValueError, match="fragment carrier"):
    validate_precontract_carriers(dtypes.half.vec(8), dtypes.float.vec(8))
  with pytest.raises(ValueError, match="accumulator carrier"):
    validate_precontract_carriers(dtypes.half.vec(16), dtypes.float.vec(16))


class _DescriptorDrift:
  def __init__(self, base, field, value): self.base, self.field, self.value = base, field, value
  def __getattr__(self, name): return self.value if name == self.field else getattr(self.base, name)


@pytest.mark.parametrize("field,value", (
  ("dims", (16, 16, 8)), ("threads", 64), ("elements_per_thread", (16, 8, 8)),
  ("dtype_in", dtypes.bfloat16), ("dtype_out", dtypes.half),
  ("opts", ("l0",)),
  ("swizzle", (((), (), ()), ((), (), ()))),
))
def test_descriptor_fingerprint_drift_fails_closed(field, value):
  drift = _DescriptorDrift(_tc(), field, value)
  with pytest.raises(ValueError, match=field): validate_rdna3_wmma_descriptor(drift)


def test_precontract_factor_derivation_rejects_nondivisible_and_bad_windows():
  nondivisible = KernelTileGeometry((80, 64, 16), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 3840, 48), KernelLDSWindow("B", 3840, 6912, 48)))
  with pytest.raises(ValueError, match="whole per-wave"): derive_precontract_factors(nondivisible, _tc())
  uneven = KernelTileGeometry((64, 64, 32), (4, 4), 512, 32,
    (KernelLDSWindow("A", 0, 5120, 80), KernelLDSWindow("B", 5120, 10240, 80)))
  with pytest.raises(ValueError, match="divide evenly"): derive_precontract_factors(uneven, _tc())
