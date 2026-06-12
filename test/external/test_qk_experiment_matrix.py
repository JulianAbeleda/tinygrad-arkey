import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qk_experiment_matrix import make_matrix, matrix_markdown


class TestQKExperimentMatrix(unittest.TestCase):
  def _decision_dir(self, root:pathlib.Path, name:str) -> pathlib.Path:
    out = root / name
    out.mkdir()
    (out / "decision.json").write_text(json.dumps({
      "status": "accept", "reference_mode": "explicit", "gain": 0.10,
      "model_size": "8B", "ab_match": True,
      "explicit": {"avg_tok_s": 50.0}, "generated": {"avg_tok_s": 55.0},
      "storage_policy": {"selected_bytes": 1048576, "cap_bytes": 2097152, "selected_primitive_entries": 1},
      "runtime_storage": {"generated1": {"storage_bytes": 1048576, "runtime_cap_bytes": 2097152, "runtime_cap_used_bytes": 1048576}},
      "reasons": ["test"],
    }))
    return out

  def test_matrix_from_decision_directory(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      out = self._decision_dir(root, "run")
      matrix = make_matrix([out])
      self.assertEqual(matrix["summary"]["accepted"], 1)
      self.assertEqual(matrix["rows"][0]["runtime_storage_bytes"], 1048576)
      self.assertIn("QK Experiment Matrix", matrix_markdown(matrix))

  def test_matrix_expands_experiment_list_json(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      out = self._decision_dir(root, "run")
      spec = root / "experiments.json"
      spec.write_text(json.dumps({"experiments": [{"out": str(out)}]}))
      matrix = make_matrix([spec])
      self.assertEqual(matrix["rows"][0]["path"], str(out))


if __name__ == "__main__":
  unittest.main()
