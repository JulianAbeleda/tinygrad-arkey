from extra.qk.prefill import prefill_stage_owner_audit as audit
from extra.qk.prefill import kernel_lifecycle_trace as life
from tinygrad.schedule import rangeify


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


def test_p4_readiness_blocks_without_owner_aware_stage_lowering():
  summary = {"pre_lowering_ownership_ready": True}
  plan = {"ok": True}

  out = audit.p4_readiness(summary, plan, "postrange")

  assert out["ready"] is False
  assert out["blocked_at"] == "P4"
  assert "STAGE lowering" in out["reason"]
  assert out["forbidden_fallback"].startswith("PREFILL_WMMA_KMAJOR_STAGE_KEY_SUPPRESS")


def test_p4_readiness_rejects_full_lowering_boundary():
  out = audit.p4_readiness({}, {}, "full")

  assert out["ready"] is False
  assert "full lowering" in out["reason"]


def test_rotated_stage_owner_tag_parser_is_fail_closed():
  assert rangeify.prefill_dbuf_rotated_stage_owner_fields(None) == {}
  assert rangeify.prefill_dbuf_rotated_stage_owner_fields(("other", ("role", "B"))) == {}

  fields = rangeify.prefill_dbuf_rotated_stage_owner_fields((
    "wmma_frag_buffer_proof", ("role", "B"), ("lds_buffer_id", 991), ("nbuf", 2),
    ("tile_count", 3), ("tile_elems", 256)
  ))

  assert fields == {
    "kind": "wmma_frag_buffer_proof",
    "role": "B",
    "lds_buffer_id": 991,
    "nbuf": 2,
    "tile_count": 3,
    "tile_elems": 256,
  }


def test_lowering_hook_summary_marks_ab_dbuf_ready():
  rows = [
    {"role": "A", "nbuf": 2, "has_reduce_range": True},
    {"role": "B", "nbuf": 2, "has_reduce_range": True},
  ]

  out = audit.lowering_hook_summary(rows)

  assert out["lowering_hook_owner_ready"] is True
  assert out["lowering_roles"] == ["A", "B"]
  assert out["lowering_count_by_role"] == {"A": 1, "B": 1}


def test_generic_b_stage_contract_empty_fails_closed():
  out = audit.generic_b_stage_contract(audit.UOp.sink())

  assert out["ok"] is False
  assert out["stage_count"] == 0
  assert out["direct_b_stage_consumer_count"] == 0
  assert "WARP x LOCAL" in out["expected_owned_stage_shape"]


def test_p4c_rotation_readiness_blocks_identity_without_lifecycle_split():
  out = audit.p4c_rotation_readiness({
    "ok": True,
    "stages": [{"owned_stage": "B_IDENTITY", "producer_epoch": "same_reduce"}],
    "consumers": [{"carrier_owned_stage": "B_IDENTITY", "carrier_consumer_epoch": "same_reduce"}],
  })

  assert out["ready"] is False
  assert out["blocked_at"] == "P4C.4"
  assert "reduce range" in out["reason"]


def test_owned_b_stage_lifecycle_builds_prologue_body_tail():
  contract = {
    "ok": True,
    "stages": [{
      "owned_stage": "B_IDENTITY", "producer_epoch": "same_reduce",
      "stage_ranges": [{"axis_type": "AxisType.REDUCE", "size": 80}],
    }],
    "consumers": [{"carrier_owned_stage": "B_IDENTITY", "carrier_consumer_epoch": "same_reduce"}],
  }

  plan = audit.owned_b_stage_lifecycle(contract)

  assert plan["ok"] is True
  assert plan["prologue"][0]["epoch"] == "k0"
  assert plan["body"][0] == {"op": "consume", "role": "B", "slot": "k%2", "epoch": "k", "owner": plan["owner"]}
  assert plan["body"][1]["epoch"] == "k+1"
  assert plan["tail"][0]["epoch"] == "last"

  ready = audit.p4c_rotation_readiness(contract, plan)
  assert ready["ready"] is False
  assert "no postrange/codegen emitter" in ready["reason"]


def test_owned_b_stage_emitter_scope_names_identity_and_blocks_rotate():
  out = audit.owned_b_stage_emitter_scope()

  assert "postrange.py::_tc_local_stage_b_src" in out["hook"]
  assert "identity" in out["implemented_modes"]
  assert "object_identity" in out["implemented_modes"]
  assert "rotate" in out["blocked_modes"]
  assert "prologue" in out["required_materializer"]
  assert "silently fall back" in out["forbidden_fallback"]


def test_hand_lifecycle_oracle_extracts_owned_b_emitter_contract():
  out = life._owned_b_emitter_oracle({
    "hand_lifecycle_oracle": {
      "source": "hand",
      "producer_rule": "rule",
      "prologue": [("coop_store", 0)],
      "body": [("compute", 0), ("coop_store", 1), ("compute", 1), ("coop_store", 0)],
      "tail": [("compute", 0), ("compute", 1)],
    }
  }, {
    "store_counts": {"prologue": 8, "body": 24},
    "body_loads_before_first_body_store_count": 8,
    "pipeline_epoch_candidate": True,
    "prologue_body_physical_window_overlap_count": 8,
  })

  assert out is not None
  assert out["prologue_store_slots"] == [0]
  assert out["body_compute_slots"] == [0, 1]
  assert out["body_store_slots"] == [1, 0]
  assert out["owned_b_stage_emitter_contract"]["body"][2] == {"op": "produce", "slot": 1, "epoch": "k+1"}
