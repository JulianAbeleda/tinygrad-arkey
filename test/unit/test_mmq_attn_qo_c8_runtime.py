from __future__ import annotations

from pathlib import Path

import pytest

from extra.qk.direct_packed_executable_attestor import DirectPackedAttestationBindings
from extra.qk.mmq_attn_qo_c8_runtime import (
  attestation_bindings, attn_qo_c8_runner_factory,
)


def _composition():
  c6 = {
    "status": "PASS", "family_identity": "family",
    "workload_identity": "workload", "input_identity": "sha256:" + "1"*64,
    "device_identity": "device", "software_identity": "software",
    "queue_comparators": {
      "PM4": "sha256:" + "2"*64, "AQL": "sha256:" + "3"*64},
  }
  return {
    "schema": "tinygrad.mmq_q4k_q8_1.attn_qo_c6_composition.v1",
    "status": "PASS", "family_identity": "family",
    "promotion_eligible_on_candidate_win": False,
    "c6_correctness_evidence": c6,
    "runtime_canary_by_queue": {"PM4": {}, "AQL": {}},
  }


def test_attestation_bindings_are_queue_exact():
  bindings = attestation_bindings(_composition(), clock_identity="clock-policy-0")
  assert set(bindings) == {"PM4", "AQL"}
  assert all(isinstance(row, DirectPackedAttestationBindings)
             for row in bindings.values())
  assert bindings["PM4"].comparator_identity != bindings["AQL"].comparator_identity
  assert bindings["PM4"].required_program_prefix == "q4k_gen_prefill_"


def test_runner_factory_requires_exact_config_before_device_work(tmp_path: Path):
  family = type("Family", (), {"family_identity": "family"})()
  with pytest.raises(ValueError, match="exactly"):
    attn_qo_c8_runner_factory(
      queue_mode="PM4", family=family,
      c6_correctness_evidence=_composition()["c6_correctness_evidence"],
      clock_identity="clock-policy-0", clock_ns=lambda: 0, config={})


def test_runner_factory_requires_both_distinct_qualifications(
    tmp_path: Path, monkeypatch,
    ):
  composition = _composition()
  family = type("Family", (), {"family_identity": "family"})()
  paths = {}
  for name in ("composition", "authority"):
    path = tmp_path / f"{name}.json"
    path.write_text("{}")
    paths[name] = path
  monkeypatch.setattr(
    "extra.qk.mmq_attn_qo_c8_runtime.read_json",
    lambda path, _label: composition if "composition" in str(path) else {})
  monkeypatch.setattr(
    "extra.qk.mmq_attn_qo_c8_runtime.validate_live_software",
    lambda *_args, **_kwargs: {})
  config = {
    "composition": str(paths["composition"]),
    "authority_snapshot": str(paths["authority"]),
    "frozen_bundle": "/bundle", "staged_family_manifest": "/family",
    "qualification_pm4": str(tmp_path / "missing-pm4.json"),
    "qualification_aql": str(tmp_path / "missing-aql.json"),
  }
  with pytest.raises(ValueError, match="must preexist"):
    attn_qo_c8_runner_factory(
      queue_mode="PM4", family=family,
      c6_correctness_evidence=composition["c6_correctness_evidence"],
      clock_identity="clock-policy-0", clock_ns=lambda: 0, config=config)
