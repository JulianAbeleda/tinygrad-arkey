import json
from pathlib import Path
import pytest
from extra.llm_research.attention_topology_contract import TOPOLOGY_FIELDS, export, validate
from extra.llm_research.decode.attention_topology_search import TARGETS as DECODE, blocked as decode_blocked, search_request as decode_request
from extra.llm_research.prefill.attention_topology_search import TARGETS as PREFILL, blocked as prefill_blocked, search_request as prefill_request

@pytest.mark.parametrize("target", sorted(DECODE))
def test_decode_contract_is_complete_and_blocked_by_class(target, tmp_path):
  rec = decode_blocked(target); validate(rec)
  assert tuple(decode_request(target)["required_topology_fields"]) == TOPOLOGY_FIELDS
  assert decode_request(target)["target"] == {"backend": "AMD", "architecture": "gfx1100", "wave_size": 32}
  assert [w["context"] for w in decode_request(target)["workloads"]] == [512, 4096]
  out = export(rec, tmp_path / f"{target}.json")
  assert json.loads(out.read_text()) == rec

@pytest.mark.parametrize("target", sorted(PREFILL))
def test_prefill_contract_is_complete_and_blocked_by_class(target):
  rec = prefill_blocked(target); validate(rec)
  assert tuple(prefill_request(target)["required_topology_fields"]) == TOPOLOGY_FIELDS
  assert [w["q_len"] for w in prefill_request(target)["workloads"]] == [512, 1024, 2048, 4096]

def test_validator_rejects_collapsed_blockers():
  rec = decode_blocked("G4"); rec["missing"].pop("gpu")
  with pytest.raises(ValueError, match="grammar/search/run/gpu"): validate(rec)

def test_validator_rejects_a_result_claim_in_blocked_record():
  rec = decode_blocked("G4"); rec["selected_plan"] = {"id": "not-a-plan"}
  with pytest.raises(ValueError, match="cannot claim"): validate(rec)

@pytest.mark.parametrize(("path", "factory", "target"), (
  ("extra/llm_research/decode/search_exports/G4.blocked.json", decode_blocked, "G4"),
  ("extra/llm_research/decode/search_exports/G5.blocked.json", decode_blocked, "G5"),
  ("extra/llm_research/prefill/search_exports/Hq32.blocked.json", prefill_blocked, "Hq32"),
  ("extra/llm_research/prefill/search_exports/Hq40.blocked.json", prefill_blocked, "Hq40"),
))
def test_checked_in_blocked_exports_are_deterministic(path, factory, target):
  assert json.loads(Path(path).read_text()) == factory(target)
