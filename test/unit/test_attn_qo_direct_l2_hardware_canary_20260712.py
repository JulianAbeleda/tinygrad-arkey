from extra.qk.prefill.attn_qo_direct_l2_hardware_canary_20260712 import run_canary
from extra.qk.prefill.register_hardware_promotion import ENABLE_VALUE, EXACT_SHAPE, STAGES, TARGET, TOLERANCES
from test.unit.test_register_hardware_promotion import _authorization, _observation
from test.unit.test_pure_register_evaluation_gate import BINARY, IDENTITY, _compile, _runtime_binding


def _route(storage="direct_l2"):
  return {"role": "attn_qo", "shape": list(EXACT_SHAPE), "target": dict(TARGET),
          "canonical_identity": IDENTITY, "binary_sha256": BINARY, "storage": storage,
          "dispatch_state": "not_attempted"}


def _pair():
  base = {"role": "attn_qo", "shape": {"m": 512, "n": 4096, "k": 4096},
          "environment": {"target": "gfx1100"},
          "pair_key": "attn_qo:512:4096:4096:semantic-v1",
          "artifact": {"status": "pass"}, "correctness": {"status": "pass"},
          "counters": {g: {"status": "live"} for g in ("l2", "memory", "compute")}}
  # P0-3: the LDS row carries its OWN distinct candidate identity.
  return {"direct_l2": base | {"storage": "direct_l2", "canonical_identity": IDENTITY, "binary_sha256": BINARY, "samples_ms": [8.] * 12},
          "lds": base | {"storage": "lds", "canonical_identity": "e" * 64, "binary_sha256": "c" * 64, "samples_ms": [10.] * 12}}


def test_attn_qo_canary_uses_only_fake_callbacks_and_promotes_after_exact_stage():
  seen = []
  def observe(contract):
    seen.append(contract)
    return _observation(len(seen) - 1)
  result = run_canary(candidate={"canonical_identity": IDENTITY},
    compile_artifact=_compile(runtime_binding=_runtime_binding()), route_binding=_route(),
    profile=_runtime_binding()["profile"], enable_value=ENABLE_VALUE,
    stage_artifacts={s["name"]: {"passed": True, "role": "attn_qo", "shape": list(s["shape"]),
      "canonical_identity": IDENTITY, "binary_sha256": BINARY, "target": dict(TARGET)} for s in STAGES},
    observation_callback=observe, benchmark_callback=lambda request: _pair())
  assert result["status"] == "promote_direct_l2"
  assert [x["stage"] for x in seen] == [x["name"] for x in STAGES]
  assert result["dispatch_state"] == "not_attempted"


def test_attn_qo_canary_revokes_on_fake_guard_failure_and_does_not_benchmark():
  calls = []
  def observe(contract):
    row = _observation(0 if not calls else len(calls))
    calls.append(contract)
    return row | {"guards_intact": False}
  result = run_canary(candidate={"canonical_identity": IDENTITY},
    compile_artifact=_compile(runtime_binding=_runtime_binding()), route_binding=_route(),
    profile=_runtime_binding()["profile"], enable_value=ENABLE_VALUE,
    observation_callback=observe, benchmark_callback=lambda _: (_ for _ in ()).throw(AssertionError()))
  assert result["revoked"] is True and result["dispatch_state"] == "not_attempted"


def test_attn_qo_canary_rejects_route_identity_before_callback():
  calls = []
  result = run_canary(candidate={"canonical_identity": IDENTITY},
    compile_artifact=_compile(runtime_binding=_runtime_binding()), route_binding=_route("lds"),
    profile=_runtime_binding()["profile"], enable_value=ENABLE_VALUE,
    stage_artifacts={s["name"]: {"passed": True, "role": "attn_qo", "shape": list(s["shape"]),
      "canonical_identity": IDENTITY, "binary_sha256": BINARY, "target": dict(TARGET)} for s in STAGES},
    observation_callback=lambda _: calls.append(1), benchmark_callback=lambda _: _pair())
  assert result["revoked"] is True and calls == []
