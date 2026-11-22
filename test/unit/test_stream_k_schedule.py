from tinygrad.codegen.opt.stream_k import StreamKSchedule


def test_llama_shaped_stream_k_is_an_exact_deterministic_cover():
  schedule=StreamKSchedule(512,12288,4096,128,128,64,170,8).validate_exact_cover()
  assert (schedule.tiles_m,schedule.tiles_n,schedule.k_blocks,schedule.output_tiles,schedule.work_units)==(4,96,64,384,24576)
  assert schedule.interval(0)==(0,144) and schedule.interval(169)==(24424,24576)
  for owner in range(schedule.owners):
    begin,end=schedule.interval(owner)
    assert begin%8==end%8==0 and begin<=end
    assert schedule.fixup_predecessors(owner)==tuple(sorted(schedule.fixup_predecessors(owner),reverse=True))


def test_stream_k_rejects_invalid_geometry_and_owner():
  for args in ((512,12288,4095,128,128,64,170,2),(512,12288,4096,128,128,64,30000,2)):
    try: StreamKSchedule(*args)
    except ValueError: pass
    else: raise AssertionError("invalid Stream-K schedule admitted")
