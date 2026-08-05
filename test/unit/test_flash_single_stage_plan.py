from extra.llm_research.decode.flash_single_stage_plan import SingleStageFlashPlan, merge_warp_owned_states, ordered_split_merge


def test_d512_owner_map_is_bijective_and_within_ordinary_launch_limits():
  p = SingleStageFlashPlan()
  p.validate()
  assert p.threads == 512
  assert [p.owner(w) for w in range(16)] == [(s, h) for s in range(4) for h in range(4)]
  assert p.old_workgroups == 32 and p.new_workgroups == 8


def test_d512_shared_memory_bound_and_partial_contract():
  p = SingleStageFlashPlan()
  assert p.tile_bytes == 32768
  assert p.partial_bytes == 8320
  assert p.shared_bytes_upper_bound == 41088
  topo = p.topology()
  assert topo["old_programs_per_layer"] == 2 and topo["candidate_programs_per_layer"] == 1
  assert topo["partial_exchange"]["address_space"] == "LOCAL"


def test_invalid_resource_shape_fails_closed():
  try:
    SingleStageFlashPlan(splits=9).validate()
  except ValueError as e:
    assert "thread limit" in str(e)
  else:
    raise AssertionError("oversized workgroup must fail closed")


def test_warp_owned_merge_preserves_legacy_split_association():
  p = SingleStageFlashPlan()
  states = [([float(10*s+h+i) for i in range(128)], 1.0+s, -2.0+0.25*s+0.01*h)
            for s in range(4) for h in range(4)]
  for h in range(4):
    assert merge_warp_owned_states(p, states, h) == ordered_split_merge([states[s*4+h] for s in range(4)])
