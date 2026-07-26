import hashlib, json, pathlib, sqlite3, tempfile, unittest
from extra.qk.decode.rocprofv3_index import SCHEMA, index_evidence


class TestRocprofV3Index(unittest.TestCase):
  def _manifest(self, root, trace, output_format="csv", **overrides):
    data = {"schema": SCHEMA, "profiler": "rocprofv3", "profiler_version": "1.1.0", "output_format": output_format,
            "trace_path": trace.name, "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            "expected_kernel_name": "known_control", "positive_control_expected_matches": 2}
    data.update(overrides)
    path = root / "evidence.json"; path.write_text(json.dumps(data)); return path

  def test_csv_indexes_exact_positive_control(self):
    with tempfile.TemporaryDirectory() as temp:
      root = pathlib.Path(temp); trace = root / "trace.csv"
      trace.write_text("Kernel_Name,Dispatch_ID\nknown_control,1\nother,2\nknown_control,3\n")
      result = index_evidence(self._manifest(root, trace))
      self.assertEqual(result["positive_control_observed_matches"], 2)
      self.assertEqual(result["kernel_dispatch_counts"], {"known_control": 2, "other": 1})

  def test_json_and_rocpd_are_indexed_only_with_named_rows(self):
    with tempfile.TemporaryDirectory() as temp:
      root = pathlib.Path(temp); trace = root / "trace.json"
      trace.write_text(json.dumps({"kernel_dispatches": [{"kernel_name": "known_control"}, {"kernel_name": "known_control"}]}))
      self.assertEqual(index_evidence(self._manifest(root, trace, "json"))["positive_control_observed_matches"], 2)
      db = root / "trace.rocpd"
      with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE rocpd_kernel_dispatch (kernel_name TEXT)")
        conn.executemany("INSERT INTO rocpd_kernel_dispatch VALUES (?)", [("known_control",), ("known_control",)])
      self.assertEqual(index_evidence(self._manifest(root, db, "rocpd"))["positive_control_observed_matches"], 2)

  def test_mismatched_or_missing_positive_control_is_rejected(self):
    with tempfile.TemporaryDirectory() as temp:
      root = pathlib.Path(temp); trace = root / "trace.csv"; trace.write_text("kernel_name\nknown_control\n")
      with self.assertRaisesRegex(ValueError, "positive control mismatch"):
        index_evidence(self._manifest(root, trace))
      with self.assertRaisesRegex(ValueError, "required fields"):
        index_evidence(self._manifest(root, trace, expected_kernel_name=None))
