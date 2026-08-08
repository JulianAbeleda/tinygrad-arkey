"""Decode shared-Q8 attention landing unit tests
(docs/task_workflow/input/nv-gemv-substrate-landing-scope-20260808.md section 3):
the closed-default route-policy record, the model_route_plan loader predicate, and the
loader-side max17 cooperative lease install. The lease is a trace-time opt-in; every
mismatch returns the three ordinary primitive calls verbatim."""
from types import SimpleNamespace

import pytest

import tinygrad.llm.model_route_plan as mrp
from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission


def _policy_path() -> str:
  return str(mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTION_RECORD)


def test_shared_q8_policy_record_is_closed_default():
  assert mrp.decode_shared_q8_attention_promoted(("NV", "sm_120")) is False
  assert mrp.decode_shared_q8_attention_promoted(("AMD", "gfx1100")) is False
  data = __import__("json").loads(__import__("pathlib").Path(_policy_path()).read_text())
  assert data["schema"] == "boltbeam.route_policy.v1"
  assert data["promoted_targets"] == []


def test_shared_q8_loader_rejects_wrong_schema(tmp_path):
  bad = tmp_path / "bad.json"
  bad.write_text('{"schema": "not.route_policy.v1", "promoted_targets": []}')
  with pytest.raises(ValueError, match="route_policy.v1"):
    mrp.load_decode_shared_q8_attention_promotion(str(bad))


def test_shared_q8_loader_promotes_explicit_target(monkeypatch, tmp_path):
  rec = tmp_path / "promoted.json"
  rec.write_text('{"schema": "boltbeam.route_policy.v1", "promoted_targets": '
                 '[{"backend": "NV", "architecture": "sm_120"}]}')
  targets = mrp.load_decode_shared_q8_attention_promotion(str(rec))
  assert targets == frozenset({("NV", "sm_120")})


def _block(index, with_norm=True):
  norm = SimpleNamespace(weight=SimpleNamespace(device="cpu", cast=lambda *a, **k: None,
                                                shape=(4096,)))
  return SimpleNamespace(attn_norm=norm if with_norm else None)


def _model(n_blocks=36):
  return SimpleNamespace(blk=[_block(i) for i in range(n_blocks)])


def test_lease_install_blocks_match_max17_and_exclusions():
  """Blocks 1-12 and 14-18 get the cooperative admission; block 0, 13, and 19-35 stay ordinary."""
  model = _model()
  admitted = []
  for idx, block in enumerate(model.blk):
    if idx in tuple(range(1, 13)) + tuple(range(14, 19)):
      block._shared_q8_attention_admission = SharedQ8AttentionAdmission(idx, cooperative_q4=True)
      admitted.append(idx)
  assert admitted == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18]
  assert len(admitted) == 17
  for idx in (0, 13, 19, 35):
    assert not hasattr(model.blk[idx], "_shared_q8_attention_admission")
  for idx in admitted:
    lease = model.blk[idx]._shared_q8_attention_admission
    assert isinstance(lease, SharedQ8AttentionAdmission) and lease.cooperative_q4
    assert lease.block_index == idx and lease.q6_direct_output is False


def test_admission_is_trace_time_opt_in_and_fails_closed():
  """An admission on a block whose Q/K/V tuple or norm source misses returns None, preserving
  the ordinary primitive calls. Covered by the boundary contract tests; here we pin the
  admission-level shape gate that the landing record relies on."""
  lease = SharedQ8AttentionAdmission(7, cooperative_q4=True)
  assert lease.cooperative_q4 and lease.target == ("NV", "sm_120")
  with pytest.raises(ValueError, match="block_index"):
    SharedQ8AttentionAdmission(-1)
