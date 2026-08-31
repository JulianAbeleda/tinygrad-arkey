from tinygrad.codegen.opt.persistent_accumulator import PersistentAccumulatorABI, fixup_contributors, persistent_work_item
from tinygrad.codegen.opt.stream_k import StreamKSchedule

SCHEDULE=StreamKSchedule(512,12288,4096,128,128,64,170,8)

def test_representative_persistent_work_is_an_exact_partition():
  seen=[]
  partial_slots=set()
  for owner in range(SCHEDULE.owners):
    start,stop=SCHEDULE.interval(owner)
    for serial in range(stop-start):
      item=persistent_work_item(SCHEDULE,owner,serial)
      seen.append(item.linear)
      if item.partial:
        assert item.partial_slot not in partial_slots
        partial_slots.add(item.partial_slot)
      assert not (item.direct and item.partial)
  assert sorted(seen) == list(range(SCHEDULE.work_units))
  assert all(0 <= slot < 2*SCHEDULE.owners for slot in partial_slots)

def test_each_tile_has_direct_output_or_reverse_owner_fixup():
  for tile in range(SCHEDULE.output_tiles):
    items=[]
    for owner in range(SCHEDULE.owners):
      start,stop=SCHEDULE.interval(owner)
      items += [persistent_work_item(SCHEDULE,owner,s) for s in range(stop-start)
                if (start+s)//SCHEDULE.k_blocks == tile]
    assert sorted(item.k_block for item in items) == list(range(SCHEDULE.k_blocks))
    direct=[item for item in items if item.direct]
    partial=[item for item in items if item.partial]
    assert (len(direct)==1) != bool(partial)
    assert fixup_contributors(SCHEDULE,tile) == tuple(sorted(partial,key=lambda item:item.owner,reverse=True))

def test_workspace_abi_is_fp32_head_tail():
  assert PersistentAccumulatorABI() == PersistentAccumulatorABI("float32",2)
