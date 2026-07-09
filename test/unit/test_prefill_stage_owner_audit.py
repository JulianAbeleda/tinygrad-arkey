from extra.qk.prefill import prefill_stage_owner_audit as audit


def test_stage_ownership_summary_marks_postrange_ready():
  stages = [
    {"role": "A", "nbuf": 2, "has_reduce_range": True, "has_global_range": True, "has_unroll_range": True},
    {"role": "B", "nbuf": 2, "has_reduce_range": True, "has_global_range": True, "has_unroll_range": True},
  ]
  wmma = [{"carrier_role": "A", "carrier_nbuf": 2}, {"carrier_role": "B", "carrier_nbuf": 2}]

  out = audit.stage_ownership_summary(stages, wmma)

  assert out["pre_lowering_ownership_ready"] is True
  assert out["full_lowering_ownership_lost"] is False
  assert out["stage_count_by_role"] == {"A": 1, "B": 1}
  assert out["next_required_object"].startswith("RotatedStageOwner")


def test_stage_ownership_summary_marks_full_lowering_loss():
  out = audit.stage_ownership_summary([], [{"carrier_role": None, "carrier_nbuf": None}])

  assert out["pre_lowering_ownership_ready"] is False
  assert out["full_lowering_ownership_lost"] is True


def test_owner_records_extract_rotated_owner_fields():
  records = audit.owner_records([
    {
      "role": "B", "lds_buffer_id": 993, "nbuf": 2, "tile_count": 1, "tile_elems": 256, "stage_id": 7,
      "stage_ranges": ["(0, AxisType.REDUCE)", "(2, AxisType.GLOBAL)", "(13, AxisType.UNROLL)"],
    }
  ])

  assert records == [{
    "role": "B",
    "lds_buffer_id": 993,
    "nbuf": 2,
    "reduce_epoch": "(0, AxisType.REDUCE)",
    "dbuf_slot_expr": "((0, AxisType.REDUCE)) % 2",
    "tile_count": 1,
    "tile_elems": 256,
    "producer_phase": "prologue_or_body",
    "consumer_phase": "compute",
    "global_ranges": ["(2, AxisType.GLOBAL)"],
    "unroll_ranges": ["(13, AxisType.UNROLL)"],
    "stage_id": 7,
  }]


def test_rotated_lifecycle_plan_requires_ab_dbuf_records():
  plan = audit.rotated_lifecycle_plan([
    {"role": "A", "lds_buffer_id": 990, "nbuf": 2},
    {"role": "B", "lds_buffer_id": 991, "nbuf": 2},
  ])

  assert plan["ok"] is True
  assert plan["late_suppression_allowed"] is False
  assert plan["prologue"][0] == {"op": "produce", "role": "A", "slot": 0, "epoch": "k0", "owner": ("A", 990, 2)}
  assert {"op": "consume", "role": "B", "slot": 1, "epoch": "k+1", "owner": ("B", 991, 2)} in plan["body"]


def test_rotated_lifecycle_plan_fails_closed_without_ab():
  plan = audit.rotated_lifecycle_plan([{"role": "B", "lds_buffer_id": 991, "nbuf": 2}])

  assert plan["ok"] is False
