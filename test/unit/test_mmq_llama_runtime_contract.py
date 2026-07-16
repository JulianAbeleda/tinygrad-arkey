"""Reference formulas transcribed from pinned mmq.cuh (anchors asserted below)."""
import itertools

import pytest

from extra.qk.mmq_llama_runtime_contract import (ConventionalRuntimeContract, LLAMA_SOURCE_COMMIT, MMQExtents,
  MMQStrides, MMQTile, SOURCE_ANCHORS)


def contract(rows=5, cols=7, max_cols=7, cx=1, cy=2, sx=1, sy=2):
  return ConventionalRuntimeContract(
    MMQTile(x=3, y=2, qk=4, iter_k=8, q8_pair_elements=8, q8_record_ints=5),
    MMQExtents(16, rows, cols, 11, max_cols, cx, cy, sx, sy),
    MMQStrides(17, 101, 103, 107, 109, 113, 127, 131))


def test_source_pin_and_line_anchors_are_explicit():
  assert LLAMA_SOURCE_COMMIT == "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
  assert set(SOURCE_ANCHORS.values()) == {"mmq.cuh:3478-3517", "mmq.cuh:3564-3577", "mmq.cuh:3583-3587",
    "mmq.cuh:3596-3618", "mmq.cuh:3622-3633", "mmq.cuh:3958-3961", "mmq.cuh:3963-3973",
    "mmq.cuh:3975-3991"}


def test_conventional_grid_and_decode_exhaustive_small_domain():
  # mmq.cuh:3958-3961 and 3583-3587.
  for rows, cols, cy, sy in itertools.product(range(1, 7), range(1, 8), range(1, 4), range(1, 4)):
    c = contract(rows, cols, cols, cy=cy, sy=sy)
    assert c.grid == type(c.grid)((rows+1)//2, (cols+2)//3, cy*sy)
    for bx, by, bz in itertools.product(range(c.grid.x), range(c.grid.y), range(c.grid.z)):
      idx = c.index(bx, by, bz)
      assert (idx.it, idx.jt, idx.wt, idx.zt) == (bx, by, bz//cy, bz%cy)


def test_outer_k_epochs_and_q4_q8_global_addresses():
  # mmq.cuh:3478-3488 and 3503-3515; values are runtime extents/strides.
  c = contract()
  assert c.k_epoch_starts == (0, 2)
  for bx, by, bz, kb in itertools.product(range(c.grid.x), range(c.grid.y), range(c.grid.z), c.k_epoch_starts):
    t = c.conventional_tile(bx, by, bz)
    got = t.addresses(kb, c.tile, c.extents)
    pair = kb*c.tile.qk//c.tile.q8_pair_elements
    first = t.offset_y + c.extents.ncols_y*pair*c.tile.q8_record_ints
    assert (got.q4_block, got.q8_first_int, got.q8_second_int) == (
      t.offset_x+kb, first, first+c.extents.ncols_y*c.tile.q8_record_ints)


def test_identity_offsets_tails_and_destinations_exhaustive():
  # mmq.cuh:3564-3577 and 3589-3594, 3622-3628.
  c = contract(cx=1, cy=2, sx=1, sy=2)
  for bx, by, bz in itertools.product(range(c.grid.x), range(c.grid.y), range(c.grid.z)):
    t = c.conventional_tile(bx, by, bz)
    wt, zt = bz//2, bz%2
    assert t.ids == (0, 1, 2)
    assert t.offset_x == (wt//2)*109 + (zt//2)*101 + bx*2*17
    assert t.offset_y == wt*113 + zt*103 + by*3*5
    assert t.offset_dst == wt*127 + zt*107 + by*3*131 + bx*2
    assert (t.tails.i_max, t.tails.j_max, t.tails.need_check) == (5-bx*2-1, 7-by*3-1, True)
    for i in range(t.tails.i_max+1):
      for j in range(min(c.tile.x, t.tails.j_max+1)):
        assert t.destination(i, j, c.strides) == t.offset_dst+j*131+i


def test_moe_ids_initialization_and_destination_reference():
  # mmq.cuh:3596-3618 and 3622-3627.
  c, ids = contract(cols=9, max_cols=6), tuple(range(100, 120))
  for jt in range(2):
    t = c.conventional_tile(0, jt, 0, expert_bounds=(2, 8), moe_ids=ids)
    assert t.ids == ids[2+jt*3:2+(jt+1)*3]
    assert t.offset_y == (2+jt*3)*5 and t.offset_dst == 0
    assert t.tails.j_max == 6-jt*3-1
    for j in range(min(c.tile.x, t.tails.j_max+1)):
      assert t.destination(0, j, c.strides) == t.ids[j]*131


def test_need_check_matches_oracle_predicate_for_small_rows():
  # mmq.cuh:3975-3991; i_max/j_max are mmq.cuh:3625-3626.
  for rows in range(1, 10):
    c = contract(rows=rows)
    assert c.need_check == (rows % c.tile.y != 0)
    for bx in range(c.grid.x): assert c.conventional_tile(bx, 0, 0).tails.i_max == rows-bx*c.tile.y-1


def test_contract_fails_closed_on_invalid_runtime_assumptions():
  with pytest.raises(ValueError): contract(cx=2, cy=3)
  c = contract()
  with pytest.raises(ValueError): c.index(c.grid.x, 0, 0)
  with pytest.raises(ValueError): c.conventional_tile(0, 0, 0, expert_bounds=(0, 2))
  with pytest.raises(ValueError): c.conventional_tile(0, 0, 0, expert_bounds=(0, 2), moe_ids=(9, 8))
  with pytest.raises(ValueError): c.conventional_tile(0, 0, 0).addresses(1, c.tile, c.extents)
