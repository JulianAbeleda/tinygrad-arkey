import numpy as np
from extra.llm_research.prefill.nv_q6_structural_segments import *
from extra.llm_research.prefill.nv_generated_q6k_streamk import OWNERS, TILES, K_BLOCKS, streamk_segments

def test_structural_table_is_two_segments_and_covers_streamk():
  rows = structural_owner_table()
  assert len(rows) == OWNERS
  assert all(r.first_tile >= 0 and r.first_end > r.first_begin for r in rows)
  assert all(r.second_tile == -1 or r.second_end > r.second_begin for r in rows)
  assert sum((r.first_end-r.first_begin)+(r.second_end-r.second_begin) for r in rows) == TILES*K_BLOCKS

def test_structural_table_matches_existing_owner_abi():
  rows = structural_owner_table()
  for r in rows:
    segs = streamk_segments(r.owner)
    assert (r.first_tile, r.first_begin, r.first_end) == (segs[0].tile_id, segs[0].begin, segs[0].end)
    if len(segs) == 2:
      assert (r.second_tile, r.second_begin, r.second_end) == (segs[1].tile_id, segs[1].begin, segs[1].end)

def test_structural_python_recurrence_matches_slot_sums():
  values = np.arange(TILES*K_BLOCKS, dtype=np.float32) + 1
  for r in structural_owner_table():
    got = values[r.first_tile*K_BLOCKS+r.first_begin:r.first_tile*K_BLOCKS+r.first_end].sum()
    expected = values[r.first_tile*K_BLOCKS+r.first_begin:r.first_tile*K_BLOCKS+r.first_end].sum()
    assert got == expected
