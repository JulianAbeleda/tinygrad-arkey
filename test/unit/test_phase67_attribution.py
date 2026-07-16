import pytest

from extra.qk.phase67_attribution import bind_matched_run, timing_tax_ledger


D = "a" * 64


def _side(name):
  return {"revision": D, "model_sha256": "b" * 64, "clock_identity": "c" * 64,
          "tok_s_by_context": {str(c): [1000.0, 1001.0, 999.0] for c in (512, 1024, 2048, 4096)},
          "session_order": ["tinygrad", "llama"] * 3,
          **({"route_bindings": [{"invocation_id": "ffn/Q4_K", "candidate_identity": D,
                                  "binary_sha256": "d" * 64}]} if name == "tinygrad" else {})}


def test_matched_run_binds_exact_execution_and_keeps_toks_authority():
  bound = bind_matched_run(_side("tinygrad"), _side("llama"))
  assert len(bound["matched_run_identity"]) == 64
  assert bound["promotion_authority"] == "synchronized_whole_prefill_tok_s"
  assert bound["contexts"] == [512, 1024, 2048, 4096]


@pytest.mark.parametrize("field", ["revision", "model_sha256", "clock_identity"])
def test_matched_run_rejects_identity_drift(field):
  llama = _side("llama"); llama[field] = "e" * 64
  with pytest.raises(ValueError, match="identity differs"): bind_matched_run(_side("tinygrad"), llama)


def test_matched_run_rejects_context_and_binary_gaps():
  tiny = _side("tinygrad"); tiny["tok_s_by_context"].pop("4096")
  with pytest.raises(ValueError, match="context set differs"): bind_matched_run(tiny, _side("llama"))
  tiny = _side("tinygrad"); tiny["route_bindings"][0].pop("binary_sha256")
  with pytest.raises(ValueError, match="binary_sha256"): bind_matched_run(tiny, _side("llama"))


def test_tax_ledger_requires_exact_join_and_ranks_largest_gap():
  bound = bind_matched_run(_side("tinygrad"), _side("llama"))
  tax = {name: {str(c): value for c in bound["contexts"]} for name, value in {
    "candidate_roles": 3.0, "activation_preparation_dequantization": 1.0, "attention": 2.0,
    "launch_synchronization": 0.5, "residual": 0.25}.items()}
  trace = {"matched_run_identity": bound["matched_run_identity"], "contexts": bound["contexts"],
           "route_bindings": [{"invocation_id": "ffn/Q4_K", "candidate_identity": D,
                               "binary_sha256": "d" * 64}], "timing_tax_ms": tax}
  ledger = timing_tax_ledger(bound, trace)
  assert ledger["attribution_only"] is True
  assert [x["tax_class"] for x in ledger["ranked_timing_tax"]][:2] == ["candidate_roles", "attention"]
  trace["matched_run_identity"] = "f" * 64
  with pytest.raises(ValueError, match="not bound"): timing_tax_ledger(bound, trace)
