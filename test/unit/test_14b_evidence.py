import importlib

import pytest

evidence = importlib.import_module("extra.qk.docs.14b_evidence")


def record():
  roles = {r: {"route_id": f"research.{r}", "prefill_ms": 1.0, "decode_ms": 2.0, "tokens": 4} for r in evidence.ROLES}
  return {"schema": evidence.SCHEMA, "status": "PASS", "research_only": True, "route_promotion": False,
          "run": {"run_id": "r1", "model_id": "14b", "quantization": "mixed_q4_k_m_q6_k_q8_prep", "prompt_tokens": 8, "decode_tokens": 4},
          "hardware": {"device": "gpu", "driver": "d", "runtime": "r", "health_probe_id": "h"}, "roles": roles,
          "prefill_decode": {"prefill": {"elapsed_ms": 3.0}, "decode": {"elapsed_ms": 4.0}},
          "route_census": {"rows": []}, "parity": {"status": "PASS"}, "memory": {"peak_bytes": 1},
          "q8_prep": {"status": "measured"}, "gpu_health": {"status": "PASS"}, "fallbacks": {"count": 0},
          "direct_packed_comparator": {"status": "measured"}, "measurement_definition": {"clock": "monotonic"}}


def test_exact_record_validates(): assert evidence.validate_record(record())["schema"] == evidence.SCHEMA


def test_missing_field_blocks_and_writes_nothing(tmp_path):
  bad = record(); del bad["parity"]; out = tmp_path / "evidence.json"
  result = evidence.run_evidence(collect=lambda: bad, hardware_probe=lambda: record()["hardware"], output=out)
  assert result["status"] == "BLOCKED" and not out.exists()


def test_hardware_failure_is_fail_closed():
  assert evidence.run_evidence(collect=record, hardware_probe=lambda: {})["status"] == "BLOCKED"
