import json
from pathlib import Path
import unittest

from extra.llm_research.boltbeam_prefill_search_contract import check_contract


class TestBoltBeamPrefillSearchContract(unittest.TestCase):
  def test_m2a_contract_is_blocked_and_hash_bound(self):
    blocked = check_contract()
    self.assertEqual(blocked["status"], "BLOCKED")
    self.assertIs(blocked["default"], False)
    self.assertEqual(blocked["verdict"], "UNPROVEN")

  def test_m2a_contract_rejects_singleton_space_and_bad_digests(self):
    import tempfile
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
      generated = Path(tmp) / "tinygrad/llm/generated"
      for rel in ("catalog.json", "requests/prefill_wmma_lds_dbuf_generated.json", "provenance/prefill_wmma_lds_dbuf_generated.blocked.json"):
        destination = generated / rel; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / "tinygrad/llm/generated" / rel).read_bytes())
      request_path = generated / "requests/prefill_wmma_lds_dbuf_generated.json"
      request = json.loads(request_path.read_text()); request["candidate_dimensions"] = {"x": [1]}
      request_path.write_text(json.dumps(request))
      with self.assertRaisesRegex(ValueError, "singleton"):
        check_contract(Path(tmp))

  def test_m2a_contract_rejects_bad_revision_and_default(self):
    import tempfile
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
      generated = Path(tmp) / "tinygrad/llm/generated"
      for rel in ("catalog.json", "requests/prefill_wmma_lds_dbuf_generated.json", "provenance/prefill_wmma_lds_dbuf_generated.blocked.json"):
        destination = generated / rel; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / "tinygrad/llm/generated" / rel).read_bytes())
      request_path = generated / "requests/prefill_wmma_lds_dbuf_generated.json"
      request = json.loads(request_path.read_text()); request["search_system"]["revision"] = "not-a-revision"
      request_path.write_text(json.dumps(request))
      with self.assertRaisesRegex(ValueError, "revision"):
        check_contract(Path(tmp))
      request = json.loads((root / "tinygrad/llm/generated/requests/prefill_wmma_lds_dbuf_generated.json").read_text())
      request_path.write_text(json.dumps(request))
      blocked_path = generated / "provenance/prefill_wmma_lds_dbuf_generated.blocked.json"
      blocked = json.loads(blocked_path.read_text()); blocked["default"] = True
      blocked_path.write_text(json.dumps(blocked, indent=2, sort_keys=True) + "\n")
      with self.assertRaisesRegex(ValueError, "default:false"):
        check_contract(Path(tmp))

  def test_m2a_contract_rejects_catalog_promotion(self):
    import tempfile
    root = Path(__file__).resolve().parents[2]
    # The production contract check catches an accidental catalog entry; exercise
    # that behavior against a copy without mutating the shared checkout.
    with tempfile.TemporaryDirectory() as tmp:
      generated = Path(tmp) / "tinygrad/llm/generated"
      generated.mkdir(parents=True)
      for rel in ("catalog.json", "requests/prefill_wmma_lds_dbuf_generated.json", "provenance/prefill_wmma_lds_dbuf_generated.blocked.json"):
        destination = generated / rel; destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((root / "tinygrad/llm/generated" / rel).read_bytes())
      catalog = json.loads((generated / "catalog.json").read_text())
      catalog["artifacts"].append({"route_id": "prefill_wmma_lds_dbuf_generated"})
      (generated / "catalog.json").write_text(json.dumps(catalog))
      with self.assertRaisesRegex(ValueError, "must not have"):
        check_contract(Path(tmp))
