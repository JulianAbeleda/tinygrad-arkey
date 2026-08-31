from collections import Counter

from tinygrad.codegen.opt.stream_k import StreamKSchedule
from tinygrad.codegen.opt.persistent_accumulator import owner_segments

def _tile_slot_map(schedule):
  table = [[-1, -1, -1] for _ in range(schedule.output_tiles)]
  metadata = {}
  for owner in range(schedule.owners):
    for segment_index, segment in enumerate(owner_segments(schedule, owner)):
      slot = owner * 2 + segment_index
      table[segment.output_tile][sum(x != -1 for x in table[segment.output_tile])] = slot
      metadata[slot] = segment.output_tile
  return tuple(tuple(row) for row in table), metadata

def test_q6_balanced_schedule_contract():
  schedule = StreamKSchedule(128, 1, 48, 1, 1, 1, 170, 1)
  intervals = tuple(schedule.interval(owner) for owner in range(170))
  assert intervals[0][0] == 0 and intervals[-1][1] == 6144
  assert all(a[1] == b[0] for a,b in zip(intervals, intervals[1:]))
  assert {stop-start for start,stop in intervals} == {36, 37}
  assert all(stop > start for start,stop in intervals)

  segments = tuple(segment for owner in range(170) for segment in owner_segments(schedule, owner))
  assert len(segments) == 294
  assert all(len(owner_segments(schedule, owner)) <= 2 for owner in range(170))
  assert all(segment.linear_start < segment.linear_stop for segment in segments)

  contributors = Counter(segment.output_tile for segment in segments)
  assert Counter(contributors.values()) == Counter({2: 90, 3: 38})
  assert set(contributors) == set(range(128))

  for owner in range(170):
    owned = owner_segments(schedule, owner)
    assert tuple(segment.owner for segment in owned) == (owner,) * len(owned)
    start, stop = schedule.interval(owner)
    assert all(segment.partial_slot == owner*2 + int(segment.linear_stop == stop and segment.linear_stop != 48*(segment.output_tile+1) and start % 48 != 0)
               for segment in owned)
    assert tuple((segment.linear_start, segment.linear_stop) for segment in owned) == tuple(
      (max(schedule.interval(owner)[0], 48*segment.output_tile),
       min(schedule.interval(owner)[1], 48*(segment.output_tile+1))) for segment in owned)

def test_q6_tile_slot_map_is_ordered_and_reversible():
  schedule = StreamKSchedule(128, 1, 48, 1, 1, 1, 170, 1)
  table, metadata = _tile_slot_map(schedule)
  assert all(tuple(x for x in row if x != -1) == tuple(sorted(x for x in row if x != -1)) for row in table)
  assert all(len([x for x in row if x != -1]) in (2, 3) for row in table)
  assert sum(x != -1 for row in table for x in row) == 294
  for tile, row in enumerate(table):
    for slot in row:
      if slot != -1: assert metadata[slot] == tile
