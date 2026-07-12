from extra.qk.prefill import anchor_gemm_regime_timing as timing


def test_matrix_commands_bind_exact_anchor_and_existing_harness():
  pure = timing._matrix_command("pure_scheduler", pin_clock=True, reps=2, iters=3)
  hand = timing._matrix_command("s9_oracle", pin_clock=False, reps=2, iters=3)
  assert pure[1].endswith("hand_vs_generated_shape_matrix.py")
  assert [pure[pure.index(f"--{axis}") + 1] for axis in ("m", "n", "k")] == ["512", "12288", "4096"]
  assert pure[pure.index("--shapes") + 1] == "2,4"
  assert [pure[pure.index(flag) + 1] for flag in ("--loc", "--unr")] == ["0", "8"]
  assert "--skip-hand" in pure and "--pin-clock" in pure
  assert hand[hand.index("--shapes") + 1] == "2,4"
  assert hand[hand.index("--hand-reps") + 1] == "2"
  assert "--skip-generated" in hand
  assert [hand[hand.index(flag) + 1] for flag in ("--waves-m", "--waves-n", "--pad")] == ["4", "2", "16"]


def test_report_labels_ownership_and_requires_all_three_for_complete(monkeypatch):
  monkeypatch.setattr(timing, "_git_revision", lambda: "abc")
  monkeypatch.setattr(timing, "_git_dirty", lambda: False)
  report = timing.build_report(regimes=tuple(timing.REGIMES), pin_clock=True, reps=1, iters=1,
                               runner=lambda name: {"status": "ok", "clock_pin": {"ok": True}, "tflops": 1})
  assert report["shape"] == {"m": 512, "n": 12288, "k": 4096}
  assert report["measurement_scope"] == "role_isolated_dense_fp16_gemm_no_model_load"
  assert report["complete"] is True
  assert report["environment"]["git_dirty"] is False
  rows = {row["regime"]: row for row in report["rows"]}
  assert rows["pure_scheduler"]["strict_pure"] is True
  assert rows["spec_owned"]["provenance"] == "compiler_primitive_spec_owned"
  assert rows["s9_oracle"]["provenance"] == "external_handwritten_kernel"


def test_report_fails_binding_on_missing_measurement():
  report = timing.build_report(regimes=("pure_scheduler",), pin_clock=False, reps=1, iters=1,
                               runner=lambda name: {"status": "no-result"})
  assert report["rows"][0]["binding_pass"] is False
  assert report["complete"] is False


def test_requested_pin_requires_successful_pin_provenance():
  report = timing.build_report(regimes=("pure_scheduler",), pin_clock=True, reps=1, iters=1,
                               runner=lambda name: {"status": "ok", "clock_pin": {"ok": False}})
  assert report["rows"][0]["binding_pass"] is False
