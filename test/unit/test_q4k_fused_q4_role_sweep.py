from extra.qk.q4k_fused_q4_role_sweep import ROLES, run

def test_sweep_matrix_is_bounded_tile_then_all_four_14b_roles():
  assert [(r[0], r[1:]) for r in ROLES] == [
    ("attn_kv", (512, 1024, 5120)), ("attn_qo", (512, 5120, 5120)),
    ("ffn_down", (512, 5120, 17408)), ("ffn_gate_up", (512, 17408, 5120))]

def test_stop_rule_stops_after_first_scalable_shape(monkeypatch):
  import extra.qk.q4k_fused_q4_role_sweep as mod
  seen = []
  monkeypatch.setattr(mod, "_case", lambda case, timeout, seed: (seen.append(case) or {"status": "PASS", "shape": {"M": case[1], "N": case[2], "K": case[3]} }))
  report = run(timeout=1)
  assert len(seen) == 1 and report["first_scalable_shape"] == {"M": 16, "N": 16, "K": 256}
