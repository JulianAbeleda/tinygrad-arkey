from extra.llm_research.prefill.nv_q6_destination_partial import (
  BLOCK, GRID, COLS, ROWS, TILES, destination_index, destination_major_fixup_source, partial_index)


def test_destination_major_partial_layout_is_a_bijection():
  offsets={partial_index(row,col) for row in range(ROWS) for col in range(COLS)}
  assert offsets == set(range(ROWS*COLS))
  outputs={destination_index(tile,row,col) for tile in range(TILES) for row in range(ROWS) for col in range(COLS)}
  assert outputs == set(range(512*4096))


def test_destination_major_lane_mapping_coalesces_both_sides():
  for tile in (0,63,127):
    for slice_index in range(GRID[1]):
      for iteration in (0,15,31):
        col=slice_index*32+iteration
        assert [partial_index(row,col) for row in range(BLOCK[0])] == list(range(col*128,col*128+128))
        base=destination_index(tile,0,col)
        assert [destination_index(tile,row,col) for row in range(ROWS)] == list(range(base,base+128))


def test_destination_major_fixup_is_progress_safe_and_exact_ordered():
  source=destination_major_fixup_source(); lower=source.lower()
  assert source.count("__fadd_rn") == 3
  assert "z=col*128+row" in source
  assert all(x not in lower for x in ("atomic", "membar", "__syncthreads", "while ("))
