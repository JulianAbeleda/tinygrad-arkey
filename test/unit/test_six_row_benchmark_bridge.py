from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from extra.qk.prefill.six_row_benchmark_bridge import (
  build_exact_research_authority, exact_research_model_scope, parse_identity_assignments,
  research_execution_census_expectations,
)
from extra.qk.prefill.six_row_research_selector import GROUPS, RETAINED_POLICY_IDENTITY, ResearchPolicyBlocked
from extra.qk.prefill_harness import (
  SixRowResearchHarnessConfig, prefill_authority_argv, prefill_run_profile,
)
from extra.qk.prefill_whole_synced import _research_non_jit_prefill_call
from tinygrad.llm.prefill_route_observer import PrefillRouteAttachment
from tinygrad.llm.prefill_routes import prefill_route_override, route_prefill_linear


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "docs/qwen3-14b-prefill-six-row-research-policy-20260718.json"
INVENTORY = ROOT / "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json"
BUNDLE = ROOT / "docs/artifacts/qwen3-14b-prefill-target-accumulate-frozen-20260718"
CANDIDATE = GROUPS[0].expected_binding_identity
FALLBACKS = {group.expected_binding_identity:f"declared:{group.invocation_id}" for group in GROUPS[1:]}


def _authority():
  return build_exact_research_authority(
    policy_path=POLICY, frozen_bundles={CANDIDATE:BUNDLE},
    fallback_program_identities=FALLBACKS, inventory=INVENTORY)


def _fake_model():
  model = SimpleNamespace(blk=[SimpleNamespace() for _ in range(40)])
  rows = json.loads(INVENTORY.read_text())["inventory"]["rows"]
  original = {}
  for row in rows:
    for tensor_identity in row["tensor_identities"]:
      _, block, name, _ = tensor_identity.split(".")
      linear = SimpleNamespace()
      attachment = PrefillRouteAttachment(
        f"runtime:{tensor_identity}", "direct-packed-baseline", tensor_identity,
        {"production": True}, {"target": "scanned"})
      linear._prefill_route_attachment = attachment
      setattr(model.blk[int(block)], name, linear)
      original[tensor_identity] = attachment
  return model, original


def test_authority_is_exact_and_missing_or_duplicate_cli_authority_fails_closed():
  authority = _authority()
  assert authority.policy["artifact_identity"] == RETAINED_POLICY_IDENTITY
  assert authority.frozen_bundles[CANDIDATE] == BUNDLE.resolve()
  with pytest.raises(ResearchPolicyBlocked, match="fallback program"):
    build_exact_research_authority(
      policy_path=POLICY, frozen_bundles={CANDIDATE:BUNDLE},
      fallback_program_identities={}, inventory=INVENTORY)
  with pytest.raises(ResearchPolicyBlocked, match="duplicate"):
    parse_identity_assignments(("a=one", "a=two"), label="test")


def test_whole_model_scope_attaches_all_280_exact_rows_and_restores_production_metadata():
  model, original = _fake_model()
  expectations = research_execution_census_expectations(model, _authority())
  assert len(expectations["required_invocations"]) == 280
  assert sum(expectations["expected_candidate_counts"].values()) == 280
  assert expectations["expected_candidate_counts"][CANDIDATE] == 80
  assert expectations["expected_fallback_count"] == 200
  with exact_research_model_scope(model, _authority()) as config:
    assert config.exact_policy_enabled is True
    routes, identities = [], []
    for tensor_identity, old in original.items():
      _, block, name, _ = tensor_identity.split(".")
      attachment = getattr(model.blk[int(block)], name)._prefill_route_attachment
      assert attachment is not old
      assert attachment.invocation_id == old.invocation_id
      assert attachment.selected_policy["artifact_identity"] == RETAINED_POLICY_IDENTITY
      routes.append(attachment.route_id)
      identities.append(attachment.selected_policy["binding_identity"])
    assert routes.count("q4k_q8_five_buffer_research") == 80
    assert routes.count("direct_packed") == 200
    assert len(identities) == 280
  for tensor_identity, old in original.items():
    _, block, name, _ = tensor_identity.split(".")
    assert getattr(model.blk[int(block)], name)._prefill_route_attachment is old


def test_context_local_override_is_default_off_and_restored():
  sentinel, calls = object(), []
  with prefill_route_override(lambda linear, value: calls.append((linear, value)) or sentinel):
    assert route_prefill_linear("linear", "input") is sentinel
  assert calls == [("linear", "input")]
  with pytest.raises(AttributeError):
    route_prefill_linear("linear", "input")


def test_research_smoke_reenters_direct_forward_and_never_calls_tinyjit_wrapper():
  calls = []
  class Tokens:
    def __init__(self, label): self.label = label
    def contiguous(self): return self
  class Model:
    _q4k_linears = SimpleNamespace(linears=[SimpleNamespace(decode_enabled=True)])
    blk = [SimpleNamespace()]
    def __call__(self, *_args, **_kwargs): raise AssertionError("research smoke entered TinyJit wrapper")
    def forward(self, tokens, start_pos, temperature):
      calls.append((tokens.label, start_pos, temperature))
      return (tokens.label, start_pos)
  model = Model()
  assert _research_non_jit_prefill_call(model, Tokens("first"), 0, "temp", logits_only=False) == ("first", 0)
  assert _research_non_jit_prefill_call(model, Tokens("second"), 512, "temp", logits_only=False) == ("second", 512)
  assert calls == [("first", 0, "temp"), ("second", 512, "temp")]
  assert model._q4k_linears.linears[0].decode_enabled is False


def test_harness_refuses_research_authority_profile_and_passes_explicit_smoke_argv():
  declarations = tuple(f"{identity}={value}" for identity, value in FALLBACKS.items())
  config = SixRowResearchHarnessConfig(str(POLICY), (f"{CANDIDATE}={BUNDLE}",), declarations, str(INVENTORY))
  with pytest.raises(ValueError, match="smoke-only"):
    config.validate(prefill_run_profile("authority"))
  smoke = prefill_run_profile("smoke", warmups=0)
  argv = prefill_authority_argv("/models/qwen3-14b.gguf", smoke,
    model_profile_id="qwen3_14b_q4k_m_gfx1100", six_row_research=config,
    artifact_path="bench/prefill-whole-synced/six-row-smoke.json")
  assert argv.count("--six-row-frozen-bundle") == 1
  assert argv.count("--six-row-fallback-program") == 5
  assert argv[argv.index("--six-row-research-policy") + 1] == str(POLICY)
  assert argv[argv.index("--artifact") + 1].endswith("six-row-smoke.json")
