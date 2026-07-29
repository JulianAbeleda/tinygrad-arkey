import hashlib, json, tempfile, unittest
from pathlib import Path

from tinygrad.llm.generated_runtime import GeneratedArtifactError, load_generated_artifact, validate_catalog

def digest(data): return hashlib.sha256(data).hexdigest()
def write(path, content): path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content); return digest(content)

class TestGeneratedArtifactRuntime(unittest.TestCase):
  def _catalog(self, mutate=None):
    tmp = tempfile.TemporaryDirectory(); root = Path(tmp.name)
    plan = b'{"op":"example"}\n'; artifact = b'# @generated\nEXAMPLE = True\n'; correctness = b'ok\n'; performance = b'ok\n'
    plan_hash, artifact_hash = write(root/'plans/example.json', plan), write(root/'artifacts/example.py', artifact)
    correctness_hash, performance_hash = write(root/'evidence/correctness.json', correctness), write(root/'evidence/performance.json', performance)
    provenance = {"schema_version": 1, "route_id": "example", "workload_role": "nonproduction_fixture", "target_backend": "SYNTHETIC", "target_architecture": "example", "shape_quant_guards": {"n": 16}, "search_space_id": "fixture", "search_space_sha256": "0"*64, "search_request_digest": "1"*64, "search_system": "fixture", "search_revision": "rev", "search_run_id": "run", "search_timestamp": "1970-01-01T00:00:00Z", "objective": "latency", "budget": 1, "candidate_count": 1, "selected_candidate_rank": 1, "selected_objective_values": {"latency": 1}, "selected_plan": json.loads(plan), "selected_plan_sha256": plan_hash, "exporter_name": "fixture", "exporter_revision": "rev", "generated_path": "artifacts/example.py", "generated_sha256": artifact_hash, "declared_reusable_primitives": [], "forbidden_fallback_kernel_identities": [], "correctness_evidence": [{"path": "evidence/correctness.json", "sha256": correctness_hash}], "performance_evidence": [{"path": "evidence/performance.json", "sha256": performance_hash}], "runtime_trace_schema": 1, "expected_runtime_identity": "example", "recovery_commit": "none", "manual_post_edit": False}
    provenance_hash = write(root/'provenance/example.json', json.dumps(provenance, sort_keys=True).encode())
    entry = {"route_id": "example", "verdict": "SEARCH_GENERATED_REPRODUCIBLE", "default": False, "target_guard": {"backend": "SYNTHETIC", "architecture": "example"}, "shape_guard": {"n": 16}, "plan_path": "plans/example.json", "plan_sha256": plan_hash, "artifact_path": "artifacts/example.py", "artifact_sha256": artifact_hash, "provenance_path": "provenance/example.json", "provenance_sha256": provenance_hash, "manual_post_edit": False}
    catalog = {"schema": "tinygrad.llm.generated-artifacts/v1", "artifacts": [entry]}
    if mutate: mutate(catalog, root, provenance)
    # Provenance mutations are part of a valid catalog rewrite; individual
    # tests then isolate the intended missing-file/field failure.
    catalog["artifacts"][0]["provenance_sha256"] = write(root/'provenance/example.json', json.dumps(provenance, sort_keys=True).encode())
    (root/'catalog.json').write_text(json.dumps(catalog))
    return tmp, root/'catalog.json'

  def test_reproducible_and_attested_validate(self):
    for verdict in ("SEARCH_GENERATED_REPRODUCIBLE", "SEARCH_GENERATED_ATTESTED"):
      tmp, catalog = self._catalog(lambda c, _r, _p: c['artifacts'][0].update(verdict=verdict))
      with tmp: validate_catalog(catalog)

  def test_valid_load_and_guard_fallback_trace(self):
    tmp, catalog = self._catalog()
    with tmp:
      selected = load_generated_artifact("example", {"backend": "SYNTHETIC", "architecture": "example"}, {"n": 16}, catalog_path=catalog)
      self.assertTrue(selected.uses_generated_artifact); self.assertEqual(selected.trace["verdict"], "SEARCH_GENERATED_REPRODUCIBLE")
      fallback = load_generated_artifact("example", {"backend": "OTHER", "architecture": "example"}, {"n": 16}, catalog_path=catalog)
      self.assertEqual(fallback.trace["verdict"], "TINYGRAD_GENERIC_GENERATED"); self.assertEqual(fallback.trace["fallback_reason"], "guard_mismatch")

  def test_missing_evidence_and_path_escape_rejected(self):
    for mutate, message in ((lambda c, _r, p: p['correctness_evidence'][0].update(path='evidence/missing.json'), 'evidence'), (lambda c, _r, _p: c['artifacts'][0].update(plan_path='../outside.json'), 'escapes')):
      tmp, catalog = self._catalog(mutate)
      with tmp, self.assertRaisesRegex(GeneratedArtifactError, message): validate_catalog(catalog)

  def test_ambiguity_and_artifact_drift_rejected(self):
    def duplicate(c, _r, _p): c['artifacts'].append(dict(c['artifacts'][0]))
    tmp, catalog = self._catalog(duplicate)
    with tmp, self.assertRaisesRegex(GeneratedArtifactError, 'ambiguous'): validate_catalog(catalog)
    def drift(_c, root, _p): (root/'artifacts/example.py').write_text('# @generated\nchanged\n')
    tmp, catalog = self._catalog(drift)
    with tmp, self.assertRaisesRegex(GeneratedArtifactError, 'artifact hash drift'): validate_catalog(catalog)

  def test_manual_post_edit_is_rejected(self):
    tmp, catalog = self._catalog(lambda c, _r, _p: c['artifacts'][0].update(manual_post_edit=True))
    with tmp, self.assertRaisesRegex(GeneratedArtifactError, 'manual_post_edit'): validate_catalog(catalog)

if __name__ == '__main__': unittest.main()
