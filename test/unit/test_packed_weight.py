from __future__ import annotations

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.packed_weight import PackedWeightTile, PackedWeightTransform
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, KernelLDSWindow, KernelTileGeometry, Ops, UOp
from extra.qk.layout import Q4_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES, q4_k_reference, q6_k_reference


def _packed(fmt:str, rows:int, k:int, seed:int) -> Tensor:
  block_bytes = Q4_K_BLOCK_BYTES if fmt == "Q4_K" else Q6_K_BLOCK_BYTES
  rng = np.random.default_rng(seed)
  raw = rng.integers(0, 256, (rows*k//256, block_bytes), dtype=np.uint8)
  # Keep scale multiplication finite and bounded.
  if fmt == "Q4_K":
    raw[:, :4] = np.stack(((rng.standard_normal(len(raw))*5e-4).astype(np.float16).view(np.uint8),
                            (rng.standard_normal(len(raw))*5e-4).astype(np.float16).view(np.uint8)), axis=1).reshape(-1, 4)
  else: raw[:, 208:210] = (rng.standard_normal(len(raw))*5e-4).astype(np.float16).view(np.uint8).reshape(-1, 2)
  return Tensor(raw.flatten().copy()).realize()


@pytest.mark.parametrize("fmt,rows,k", [("Q4_K", 2, 512), ("Q6_K", 2, 512)])
@pytest.mark.parametrize("device", ["CPU", "PYTHON"])
def test_scalar_fp16_producer_matches_reference(fmt:str, rows:int, k:int, device:str):
  packed = _packed(fmt, rows, k, 20260714).to(device)
  desc = PackedWeightTransform(fmt, rows, k)
  storage = packed.bitcast(desc.storage_dtype).contiguous().realize()
  def kernel(out:UOp, source:UOp) -> UOp:
    stores = tuple(out[r, kk].store(desc.dequant(source, r, kk)) for r in range(rows) for kk in range(k))
    return UOp.sink(*stores, arg=KernelInfo(name=f"packed_weight_{fmt.lower()}"))
  got = Tensor.empty(rows, k, dtype=dtypes.float16, device=device).custom_kernel(storage, fxn=kernel)[0].numpy()
  reference = (q4_k_reference if fmt == "Q4_K" else q6_k_reference)(packed.bitcast(dtypes.uint8), rows*k)
  np.testing.assert_array_equal(got, reference.reshape(rows, k).cast(dtypes.float16).numpy())


@pytest.mark.parametrize("fmt,start,width", [("Q4_K", 28, 8), ("Q4_K", 248, 16),
                                               ("Q6_K", 12, 8), ("Q6_K", 248, 16)])
def test_tile_fp16_producer_matches_scalar_across_group_and_block_boundaries(fmt:str, start:int, width:int):
  packed = _packed(fmt, 1, 512, 20260715).to("PYTHON")
  desc = PackedWeightTransform(fmt, 1, 512)
  storage = packed.bitcast(desc.storage_dtype).contiguous().realize()
  def kernel(out:UOp, source:UOp) -> UOp:
    tile = desc.dequant_tile(source, 0, start, width)
    assert isinstance(tile, PackedWeightTile) and tile.value.dtype == dtypes.half.vec(width)
    return UOp.sink(out.index(UOp.const(dtypes.weakint, 0), dtype=dtypes.half.vec(width)).store(tile.value),
                    arg=KernelInfo(name=f"packed_tile_{fmt.lower()}"))
  got = Tensor.empty(width, dtype=dtypes.float16, device="PYTHON").custom_kernel(storage, fxn=kernel)[0].numpy()
  reference = (q4_k_reference if fmt == "Q4_K" else q6_k_reference)(packed.bitcast(dtypes.uint8), 512)
  np.testing.assert_array_equal(got, reference.reshape(512)[start:start+width].cast(dtypes.float16).numpy())


@pytest.mark.parametrize("fmt,start", [("Q4_K", 24), ("Q6_K", 8)])
def test_tile_consumes_shared_native_units(fmt:str, start:int):
  desc, offsets = PackedWeightTransform(fmt, 1, 256), []
  tile = desc.dequant_tile(lambda offset: offsets.append(offset) or UOp.const(desc.storage_dtype, 0), 0, start, 16)
  assert tile.value.op is Ops.STACK and len(tile.value.src) == 16
  assert all(isinstance(x, int) for x in offsets)
  scalar_offsets = []
  for k in range(start, start+16):
    desc.dequant(lambda offset: scalar_offsets.append(offset) or UOp.const(desc.storage_dtype, 0), 0, k)
  assert len(offsets) == len(set(offsets)) < len(scalar_offsets)


@pytest.mark.parametrize("fmt", ["Q4_K", "Q6_K"])
def test_symbolic_tile_bounds_and_ownership(fmt:str):
  desc = PackedWeightTransform(fmt, 2, 512)
  row, base = UOp.range(2, 73), UOp.range(497, 74)
  source = UOp.placeholder((desc.packed_bytes//desc.storage_width,), desc.storage_dtype, 75)
  tile = desc.dequant_tile(source, row, base, 16)
  assert tile.value.dtype == dtypes.half.vec(16)
  assert row in tile.value.backward_slice_with_self and base in tile.value.backward_slice_with_self
  offsets = []
  desc.dequant_tile(lambda offset: offsets.append(offset) or UOp.const(desc.storage_dtype, 0), row, base, 16)
  assert offsets and all(x.vmin >= 0 and x.vmax < desc.packed_bytes//desc.storage_width for x in offsets)


def test_tile_width_and_interval_validation():
  desc = PackedWeightTransform("Q4_K", 1, 256)
  with pytest.raises(ValueError, match="width"): desc.dequant_tile(lambda _: UOp.const(dtypes.uint32, 0), 0, 0, 4)  # type: ignore[arg-type]
  with pytest.raises(IndexError, match="tile"): desc.dequant_tile(lambda _: UOp.const(dtypes.uint32, 0), 0, 249, 8)


def test_addresses_follow_canonical_blocks_and_units():
  q4 = PackedWeightTransform("Q4_K", 2, 512)
  a = q4.address(1, 256+37)
  assert a.block == 3 and a.block_byte == 3*144 and a.payload_byte == 3*144+16+5
  assert a.unit_offsets(4)[0] == a.payload_byte//4
  high = q4.address(1, 256+4*32+3)
  assert high.scale_byte == 3*144+4 and high.min_scale_byte == 3*144+8
  assert high.auxiliary_bytes == (3*144+12,)
  q6 = PackedWeightTransform("Q6_K", 2, 512)
  b = q6.address(1, 256+255)
  assert b.block == 3 and b.block_byte == 3*210 and b.d_byte == 3*210+208
  assert b.unit_offsets(2)[0] == b.payload_byte//2


@pytest.mark.parametrize("kwargs,match", [
  ({"quant_format": "Q5_K", "rows": 1, "k": 256}, "quant_format"),
  ({"quant_format": "Q4_K", "rows": 0, "k": 256}, "rows"),
  ({"quant_format": "Q6_K", "rows": 1, "k": 257}, "block aligned"),
  ({"quant_format": "Q4_K", "rows": 1, "k": 256, "block_bytes": 210}, "block_bytes"),
])
def test_validation(kwargs, match):
  with pytest.raises(ValueError, match=match): PackedWeightTransform(**kwargs)


def test_json_roundtrip_and_geometry_validation():
  desc = PackedWeightTransform("Q6_K", 3, 512)
  row = desc.to_json()
  assert row["storage_dtype"] == dtypes.uint16.name and row["packed_bytes"] == 3*2*210
  assert PackedWeightTransform.from_json(row) == desc
  with pytest.raises(ValueError, match="storage_width"): PackedWeightTransform.from_json({**row, "storage_width": 4})
  with pytest.raises(ValueError, match="missing"): PackedWeightTransform.from_json({"quant_format": "Q4_K"})
  with pytest.raises(ValueError, match="unknown"): PackedWeightTransform.from_json({**row, "model": "x"})


def test_coordinate_validation():
  desc = PackedWeightTransform("Q4_K", 1, 256)
  with pytest.raises(IndexError, match="row"): desc.address(1, 0)
  with pytest.raises(IndexError, match="k"): desc.address(0, 256)
  with pytest.raises(TypeError, match="integer or UOp"): desc.dequant(lambda _: UOp.const(dtypes.uint32, 0), 0, "0")  # type: ignore[arg-type]


@pytest.mark.parametrize("fmt,dtype", [("Q4_K", dtypes.uint32), ("Q6_K", dtypes.uint16)])
def test_symbolic_row_and_k_survive_scalar_decoder(fmt, dtype):
  desc = PackedWeightTransform(fmt, 2, 512)
  row, kk = UOp.range(2, 70), UOp.range(512, 71)
  source = UOp.placeholder((desc.packed_bytes // desc.storage_width,), dtype, 72)
  value = desc.dequant(source, row, kk)
  assert value.dtype == dtypes.float16
  assert row in value.backward_slice_with_self and kk in value.backward_slice_with_self
  indexes = [u for u in value.toposort() if u.op.name == "INDEX" and source in u.backward_slice_with_self]
  assert indexes


@pytest.mark.parametrize("fmt,rows,k", [("Q4_K", 2, 512), ("Q6_K", 2, 512)])
def test_every_symbolic_decoder_load_stays_within_packed_storage(fmt, rows, k):
  desc = PackedWeightTransform(fmt, rows, k)
  row, kk, offsets = UOp.range(rows, 80), UOp.range(k, 81), []
  desc.dequant(lambda offset: offsets.append(offset) or UOp.const(desc.storage_dtype, 0), row, kk)
  assert offsets
  assert all(offset.vmin >= 0 for offset in offsets)
  assert all(offset.vmax < desc.packed_bytes // desc.storage_width for offset in offsets)


def test_candidate_context_carries_validated_packed_transform_only_with_geometry():
  desc = PackedWeightTransform("Q4_K", 16, 256)
  identity = "1" * 64
  with pytest.raises(ValueError, match="geometry"):
    KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity, packed_weight=desc)
  geometry = KernelTileGeometry((128, 128, 32), (2, 2), 128, 32,
    (KernelLDSWindow("A", 0, 8192, 64), KernelLDSWindow("B", 8192, 16384, 64)))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity, geometry=geometry, packed_weight=desc)
  assert context.packed_weight is desc
  with pytest.raises(TypeError, match="PackedWeightTransform"):
    KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity, geometry=geometry, packed_weight={"quant_format":"Q4_K"})
