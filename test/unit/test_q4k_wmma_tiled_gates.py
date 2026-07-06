import pathlib

from extra.qk import route_manifest
from extra.qk import q4k_wmma_tiled_role_shape_exec_gate as role_shape_gate
from extra.qk.q4k_wmma_tiled_no_hand_kernel_gate import build as build_no_hand
from extra.qk.q4k_wmma_tiled_role_shape_exec_gate import build as build_role_shape_exec


def test_q4k_wmma_tiled_authority_gate_files_exist():
  authority = route_manifest.ROUTES["prefill_q4k_int8_wmma_tiled_research"]["authority_gate"]
  for token in authority.replace("+", " ").split():
    if token.endswith(".py"):
      assert pathlib.Path(token).exists(), token


def test_q4k_wmma_tiled_no_hand_scan_is_clean():
  report = build_no_hand()
  assert report["verdict"] == "Q4K_WMMA_TILED_NO_HAND_KERNEL_PASS"
  assert report["findings"] == []


def test_q4k_wmma_tiled_role_shape_exec_classifier_is_not_a_false_pass():
  lifecycle = {"verdict": "Q4K_WMMA_TILED_LIFECYCLE_PASS", "class": "pass.bounded_multi_tile_lifecycle"}
  report = build_role_shape_exec(lifecycle)
  assert report["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING"
  assert report["classified_blocker"] is True
  assert len(report["roles"]) == 4
  assert all(row["exec"]["attempted"] for row in report["roles"])
  for row in report["roles"]:
    assert row["exec"]["class"] in {
      "pass.generated_tiled_loop",
      "blocked.generated_tiled_loop_execution_failed",
      "blocked.compiler_unavailable",
    }
  assert report["scheduler_owned_tile_loop"]["required"] is True
  assert report["scheduler_owned_tile_loop"]["remaining_blocker"] == "scheduler_owned_tile_loop_missing"
  assert report["required_next"] == "scheduler_owned_tile_loop_missing"


def test_q4k_wmma_tiled_role_shape_exec_all_numeric_ok_requires_all_roles(monkeypatch):
  lifecycle = {"verdict": "Q4K_WMMA_TILED_LIFECYCLE_PASS", "class": "pass.bounded_multi_tile_lifecycle"}

  def fake_role_row(spec, _lifecycle):
    return {
      "role": spec.role,
      "exec": {
        "attempted": True,
        "class": "pass.generated_tiled_loop",
        "numeric_ok": spec.role == "attn_kv",
      },
    }

  monkeypatch.setattr(role_shape_gate, "_role_row", fake_role_row)
  report = role_shape_gate.build(lifecycle)
  assert report["attempted_count"] == 4
  assert report["executed_roles"] == ["attn_kv", "attn_qo", "ffn_down", "ffn_gate_up"]
  assert report["all_numeric_ok"] is False
