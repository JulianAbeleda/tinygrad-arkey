import pytest
from extra.qk.mmq_physical_validation import validate_physical_contract

def _ok(**overrides):
  args = dict(local_size=(8, 8, 1), consumed_local_dims=(0, 1),
    lane_map={"m": "lidx0", "n": "lidx1"}, barriers=({"uniform": True},),
    owners=({"m": m, "n": n, "owner": {"lane": m * 2 + n}} for m in range(2) for n in range(2)),
    expected_outputs=((m, n) for m in range(2) for n in range(2)))
  args.update(overrides)
  return validate_physical_contract(**args)

def test_physical_contract_accepts_consumed_uniform_one_to_one_mapping():
  assert _ok()["passed"]

def test_physical_contract_rejects_unused_local_dimension():
  row = _ok(consumed_local_dims=(0,))
  assert not row["passed"] and any("unused non-unit local dimensions" in e for e in row["errors"])

def test_physical_contract_rejects_nonuniform_barrier_and_duplicate_owner():
  row = _ok(barriers=({"uniform": False},), owners=({"m": 0, "n": 0},) * 4)
  assert not row["passed"]
  assert "barrier 0 is not proven uniform" in row["errors"]
  assert "owner map is not exactly one-to-one over outputs" in row["errors"]

def test_physical_contract_requires_explicit_owner_and_consumed_lane_facts():
  row = _ok(lane_map={"m": "lidx0", "n": "lidx2"}, consumed_local_dims=(0, 1),
            owners=({"m": m, "n": n} for m in range(2) for n in range(2)))
  assert not row["passed"]
  assert "lane mapping uses a local dimension not declared consumed" in row["errors"]
  assert "owner map lacks explicit owner facts" in row["errors"]

def test_physical_contract_requires_workgroup_barrier_scope():
  row = _ok(barriers=({"uniform": True, "scope": "subgroup"},))
  assert not row["passed"]
  assert "barrier 0 does not have an explicit workgroup scope" in row["errors"]
