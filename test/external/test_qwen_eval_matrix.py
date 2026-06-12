import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.qwen_eval_matrix import load_manifest, make_matrix, matrix_markdown


class TestQwenEvalMatrix(unittest.TestCase):
  def _write_run(self, root:pathlib.Path, name:str, status:str="pass", quality_status:str="pass") -> pathlib.Path:
    out = root / name
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({
      "kind": "qwen_eval_summary",
      "status": status,
      "tokens_match": status == "pass",
      "model": "/tmp/model.gguf",
      "policy": "/tmp/policy.json",
      "storage": "shared",
      "prompts": 2,
      "quality": {"status": quality_status, "scored": 2, "passed": 2 if quality_status == "pass" else 1, "pass_rate": 1.0},
      "modes": {
        "explicit": {"generated": 4, "elapsed_s": 1.0, "tok_s": 4.0},
        "generated": {"generated": 4, "elapsed_s": 1.0, "tok_s": 4.0},
      },
      "prompt_rows": [],
    }))
    return out

  def test_matrix_from_manifest(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      out = self._write_run(root, "8b")
      manifest_path = root / "manifest.json"
      manifest_path.write_text(json.dumps({
        "kind": "qwen_eval_manifest",
        "prompts": str(root / "prompts.jsonl"),
        "tokens": 64,
        "rows": [{"id": "8b", "model_size": "8B", "model": "/tmp/model", "policy": "/tmp/policy", "out": str(out)}],
      }))
      manifest = load_manifest(manifest_path)
      matrix = make_matrix(manifest, pathlib.Path.cwd())
      self.assertEqual(matrix["summary"]["parity_passed"], 1)
      self.assertIn("Qwen Eval Matrix", matrix_markdown(matrix))

  def test_manifest_rejects_duplicate_ids(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      manifest_path = root / "manifest.json"
      manifest_path.write_text(json.dumps({
        "kind": "qwen_eval_manifest",
        "rows": [
          {"id": "x", "model_size": "8B", "model": "/tmp/model", "policy": "/tmp/policy", "out": "/tmp/out"},
          {"id": "x", "model_size": "14B", "model": "/tmp/model", "policy": "/tmp/policy", "out": "/tmp/out"},
        ],
      }))
      with self.assertRaisesRegex(ValueError, "duplicate row id"):
        load_manifest(manifest_path)

  def test_committed_qwen_eval_matrix_reproduces(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    manifest = load_manifest(repo / "bench/qwen-eval-20260612/manifest.json")
    matrix = make_matrix(manifest, repo)
    self.assertEqual(json.loads((repo / "bench/qwen-eval-20260612/matrix-summary.json").read_text()), matrix)
    self.assertEqual((repo / "bench/qwen-eval-20260612/matrix-summary.md").read_text(), matrix_markdown(matrix))


if __name__ == "__main__":
  unittest.main()
