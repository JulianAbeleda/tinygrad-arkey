import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.register_contracts import LogicalRegisterTile


def test_logical_register_tile_is_backend_neutral_and_stable():
  tile = LogicalRegisterTile("A", dtypes.half, (2, 16), 2, 16, 16, 2, "proven", "row_major_fragment16")
  assert tile.scalar_bytes == 2
  assert tile.tile_elements == 32
  assert tile.fragment_elements == 32
  assert tile.logical_bytes == 128
  assert tile.alignment_bytes == 32
  assert tile.snapshot() == {
    "role": "A", "dtype": "half", "tile_shape": (2, 16), "fragments": 2,
    "lane_width": 16, "carrier_width": 16, "slot_count": 2,
    "slot_addressing": "proven", "layout": "row_major_fragment16",
    "alignment_bytes": 32, "ownership": ("producer", "consumer"),
    "lifetime": ("produce", "consume", "release"), "tile_elements": 32,
    "logical_bytes": 128,
  }


@pytest.mark.parametrize("kwargs", [
  {"slot_addressing": "dynamic"},
  {"tile_shape": (0, 16)},
  {"fragments": 0},
  {"dtype": dtypes.half.vec(16)},
  {"alignment_bytes": 3},
  {"layout": ""},
])
def test_logical_register_tile_rejects_unproven_or_malformed_contracts(kwargs):
  base = dict(role="A", dtype=dtypes.half, tile_shape=(2, 16), fragments=2,
              lane_width=16, carrier_width=16, slot_count=2,
              slot_addressing="static", layout="row_major_fragment16")
  base.update(kwargs)
  with pytest.raises(ValueError): LogicalRegisterTile(**base)


def test_logical_register_tile_allows_sequential_static_slot_contract():
  tile = LogicalRegisterTile("B", dtypes.half, (2, 16), 2, 16, 16, 1, "sequential", "row_major_fragment16")
  assert tile.slot_count == 1 and tile.slot_addressing == "sequential"
