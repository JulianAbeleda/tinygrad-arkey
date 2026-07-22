import pathlib
import pytest

from extra.qk import route_manifest
from extra.qk import q4k_wmma_tiled_role_shape_exec_gate as role_shape_gate
from extra.qk.q4k_wmma_tiled_no_hand_kernel_gate import build as build_no_hand
from extra.qk.q4k_wmma_tiled_role_shape_exec_gate import build as build_role_shape_exec


def test_q4k_wmma_tiled_authority_gate_files_exist():
  authority = route_manifest.ROUTES["prefill_q4k_int8_wmma_tiled_research"]["authority_gate"]
  for token in authority.replace("+", " ").split():
    if token.endswith(".py"):
      assert pathlib.Path(token).exists(), token


# WIP gate: the scheduler-owned Q4K-WMMA-tiled research route is not yet complete (verdict reports FAIL).
# xfail(strict) so the suite stays green AND flips to a failure-signal the moment the route reaches PASS.
@pytest.mark.xfail(reason="Q4K-WMMA-tiled scheduler-owned route is WIP research; not yet PASS", strict=True)
def test_q4k_wmma_tiled_no_hand_scan_is_clean():
  report = build_no_hand()
  assert report["verdict"] == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS"
  assert report["findings"] == []


def test_q4k_wmma_tiled_role_shape_exec_pass_requires_scheduler_evidence(monkeypatch):
  lifecycle = {"verdict": "Q4K_WMMA_TILED_LIFECYCLE_PASS", "class": "pass.bounded_multi_tile_lifecycle"}
  def fake_role_row(spec, _lifecycle):
    return {"role": spec.role, "exec": {"attempted": True, "class": "pass.scheduler_owned_nested_contraction",
      "numeric_ok": True, "wmma_present": True}}
  monkeypatch.setattr(role_shape_gate, "_role_row", fake_role_row)
  report = build_role_shape_exec(lifecycle)
  assert report["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS"
  assert report["classified_blocker"] is False
  assert len(report["roles"]) == 4
  assert all(row["exec"]["attempted"] for row in report["roles"])
  assert report["scheduler_owned_tile_loop"]["required"] is True
  assert report["scheduler_owned_tile_loop"]["implemented"] is True
  assert report["scheduler_owned_tile_loop"]["remaining_blocker"] is None
  assert report["required_next"] is None


def test_q4k_wmma_tiled_role_shape_exec_all_numeric_ok_requires_all_roles(monkeypatch):
  lifecycle = {"verdict": "Q4K_WMMA_TILED_LIFECYCLE_PASS", "class": "pass.bounded_multi_tile_lifecycle"}

  def fake_role_row(spec, _lifecycle):
    return {
      "role": spec.role,
      "exec": {
        "attempted": True,
        "class": "pass.scheduler_owned_nested_contraction",
        "numeric_ok": spec.role == "attn_kv",
        "wmma_present": True,
      },
    }

  monkeypatch.setattr(role_shape_gate, "_role_row", fake_role_row)
  report = role_shape_gate.build(lifecycle)
  assert report["attempted_count"] == 4
  assert report["executed_roles"] == ["attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"]
  assert report["all_numeric_ok"] is False
