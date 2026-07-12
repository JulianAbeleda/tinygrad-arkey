from extra.qk.prefill import prefill_stage_owner_audit as audit
from extra.qk.prefill import kernel_lifecycle_trace as life
from tinygrad.codegen.opt import postrange
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


def test_stage_value_identity_gate_pinpoints_first_loss_not_owner_loss():
  key = {"role": "B", "matrix": "B", "n_tile": 0, "k_tile": 7, "k_inner": [0, 32]}
  snapshots = [
    {"boundary": "anchor_epoch_evidence", "owner_record_count": 1,
     "value_identities": [{"role": "B", "value_key": key}]},
    {"boundary": "postrange", "owner_record_count": 1,
     "value_identities": [{"role": "B", "value_key": key}]},
    {"boundary": "rangeify_add_local_buffers", "owner_record_count": 1, "value_identities": []},
    {"boundary": "amd_pre_isel", "owner_record_count": 0, "value_identities": []},
    {"boundary": "amd_isel", "owner_record_count": 0, "value_identities": []},
  ]

  out = audit.stage_value_identity_survival_gate(snapshots, [key])

  assert out["pass"] is False
  assert out["first_loss_boundary"] == "rangeify_add_local_buffers"
  assert out["boundaries"][2]["owner_record_count"] == 1
  assert "owner/address/window equivalence is insufficient" in out["pass_condition"]


def test_stage_value_identity_gate_requires_final_amd_isel_survival():
  key = {"role": "A", "matrix": "A", "m_tile": 2, "k_tile": 3}
  snapshots = [{"boundary": boundary, "owner_record_count": 1,
                "value_identities": [{"role": "A", "value_key": key}]}
               for boundary in audit.STAGE_VALUE_IDENTITY_BOUNDARIES]

  out = audit.stage_value_identity_survival_gate(snapshots, [key])

  assert out["pass"] is True
  assert out["first_loss_boundary"] is None


def test_stage_value_identity_snapshot_does_not_promote_owner_metadata_to_value_identity():
  owner_only = audit.UOp(audit.Ops.NOOP, audit.dtypes.void, tag=(
    "wmma_frag_buffer_proof", ("role", "B"), ("lds_buffer_id", 991), ("nbuf", 2),
    ("owned_stage", "B_IDENTITY")))

  snap = audit.stage_value_identity_snapshot(audit.UOp.sink(owner_only), "postrange")

  assert snap["owner_record_count"] == 1
  assert snap["value_identity_count"] == 0


def test_single_buffer_value_identity_survives_real_rangeify_local_materialization(monkeypatch):
  import itertools
  from tinygrad.dtype import dtypes
  from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite

  lane = UOp.range(32, 0, AxisType.LOCAL)
  source = UOp(Ops.CONTRACT, dtypes.half.vec(16), (UOp.const(dtypes.half, 1),), ((3, 16),))
  monkeypatch.setattr(postrange, "_tc_local_stage_owned_stage_meta", lambda operand_idx: operand_idx == 1)
  postrange_sink = UOp.sink(postrange._tc_local_stage_src(source, (lane,), 1))
  rangeified_sink = graph_rewrite(postrange_sink, rangeify.pm_add_buffers_local, ctx=itertools.count(0))

  postrange_snapshot = audit.stage_value_identity_snapshot(postrange_sink, "postrange")
  rangeify_snapshot = audit.stage_value_identity_snapshot(rangeified_sink, "rangeify_add_local_buffers")
  postrange_keys = {repr(row["value_key"]) for row in postrange_snapshot["value_identities"]}
  rangeify_keys = {repr(row["value_key"]) for row in rangeify_snapshot["value_identities"]}
  assert len(postrange_keys) == 1
  assert postrange_keys == rangeify_keys
  key = postrange_snapshot["value_identities"][0]["value_key"]
  anchor_snapshot = {**postrange_snapshot, "boundary": "anchor_epoch_evidence"}
  gate = audit.stage_value_identity_survival_gate([anchor_snapshot, postrange_snapshot, rangeify_snapshot], [key])
  assert [row["status"] for row in gate["boundaries"][:3]] == ["pass", "pass", "pass"]
  assert gate["first_loss_boundary"] == "amd_pre_isel"


def test_single_buffer_value_identity_rangeify_parser_is_typed_and_fail_closed():
  from tinygrad.codegen.opt import KernelOptError
  from tinygrad.codegen.opt.prefill_value_key import PrefillSourceValueKey

  legacy = ("wmma_frag_buffer_proof", ("role", "B"), ("lds_buffer_id", 991), ("nbuf", 1))
  assert rangeify.prefill_single_buffer_stage_value_key(legacy) is None
  malformed = legacy + (("value_key", {"role": "B"}),)
  try:
    rangeify.prefill_single_buffer_stage_value_key(malformed)
    assert False, "malformed claimed value identity must fail closed"
  except KernelOptError:
    pass
  mismatched = legacy + (("value_key", PrefillSourceValueKey(
    role="B", output_tile=(0,), k_epoch=0, k_phase=0, vector_offset=0,
    source_id=("uop_key", "source"), buffer_id=("lds", 993, 0))),)
  try:
    rangeify.prefill_single_buffer_stage_value_key(mismatched)
    assert False, "owner/value buffer mismatch must fail closed"
  except KernelOptError:
    pass


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
  assert "destructive stage-key suppression" in out["forbidden_fallback"]


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

  rotated = rangeify.prefill_dbuf_rotated_stage_owner_fields((
    "wmma_frag_buffer_proof", ("role", "B"), ("lds_buffer_id", 991), ("nbuf", 2),
    ("tile_count", 1), ("tile_elems", 256), ("owned_stage", "B_ROTATE"),
    ("lifecycle", "prologue_body_tail"), ("rotation", "kr_mod_nbuf")
  ))

  assert rotated["owned_stage"] == "B_ROTATE"
  assert rotated["lifecycle"] == "prologue_body_tail"
  assert rotated["rotation"] == "kr_mod_nbuf"


def test_ab_owned_stage_metadata_tags_are_opt_in(monkeypatch):
  monkeypatch.setenv("PREFILL_DBUF_OWNED_AB_STAGE_META", "1")
  postrange.getenv.cache_clear()
  try:
    a_tag = postrange._tc_local_stage_buffer_tag(0, 990, 2, 1, 256)
    b_tag = postrange._tc_local_stage_buffer_tag(1, 991, 2, 1, 256)
  finally:
    postrange.getenv.cache_clear()

  a_fields = rangeify.prefill_dbuf_rotated_stage_owner_fields(a_tag)
  b_fields = rangeify.prefill_dbuf_rotated_stage_owner_fields(b_tag)
  assert a_fields["owned_stage"] == "A_IDENTITY"
  assert b_fields["owned_stage"] == "B_IDENTITY"
  assert a_fields["producer_epoch"] == "same_reduce"
  assert b_fields["consumer_epoch"] == "same_reduce"


def test_b_rotate_stage_metadata_is_explicit_and_fail_closed_ready(monkeypatch):
  monkeypatch.setenv("PREFILL_DBUF_OWNED_AB_STAGE_META", "1")
  monkeypatch.setenv("PREFILL_DBUF_OWNED_B_STAGE_EMIT", "rotate")
  postrange.getenv.cache_clear()
  try:
    b_tag = postrange._tc_local_stage_buffer_tag(1, 991, 2, 1, 256)
  finally:
    postrange.getenv.cache_clear()

  fields = rangeify.prefill_dbuf_rotated_stage_owner_fields(b_tag)
  assert fields["owned_stage"] == "B_ROTATE"
  assert fields["lifecycle"] == "prologue_body_tail"
  assert fields["rotation"] == "kr_mod_nbuf"


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


def test_owned_b_stage_lifecycle_accepts_rotate_metadata():
  contract = {
    "ok": True,
    "stages": [{
      "owned_stage": "B_ROTATE", "rotation": "kr_mod_nbuf",
      "stage_ranges": [{"axis_type": "AxisType.REDUCE", "size": 80}],
    }],
    "consumers": [{"carrier_owned_stage": "B_ROTATE", "carrier_rotation": "kr_mod_nbuf"}],
  }

  plan = audit.owned_b_stage_lifecycle(contract)

  assert plan["ok"] is True
  assert plan["source"] == "audit_only_owned_b_rotated_stage_lifecycle"
  assert plan["body"][1]["epoch"] == "k+1"


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
