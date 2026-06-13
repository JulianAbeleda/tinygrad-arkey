import json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra.llm_sft_smoke_train import build_byte_examples, load_sft_rows, split_rows, summary_markdown


def _row(row_id:str, source_id:str) -> dict:
  return {
    "id": row_id,
    "source_id": source_id,
    "prompt": f"prompt {source_id}",
    "completion": f"answer {source_id}",
  }


class TestLLMSFTSmokeTrain(unittest.TestCase):
  def test_load_sft_rows_requires_prompt_completion_and_source_id(self):
    with TemporaryDirectory() as raw_td:
      path = pathlib.Path(raw_td) / "sft.jsonl"
      path.write_text(json.dumps({"id": "a", "prompt": "p", "completion": "c"}) + "\n")
      with self.assertRaisesRegex(ValueError, "source_id"):
        load_sft_rows(path)

  def test_split_rows_keeps_source_ids_disjoint(self):
    rows = [_row(f"{model}:{idx}", f"item-{idx}") for idx in range(10) for model in ("8b", "14b")]
    train, eval_rows, eval_ids = split_rows(rows, eval_every=5)
    self.assertEqual(len(eval_ids), 2)
    self.assertTrue({row["source_id"] for row in train}.isdisjoint({row["source_id"] for row in eval_rows}))
    self.assertEqual(len(train) + len(eval_rows), len(rows))

  def test_build_byte_examples_shape(self):
    x, y = build_byte_examples([_row("a", "a")], context_bytes=4, vocab_size=128)
    self.assertEqual(x.shape[1], 4 * 128 + 1)
    self.assertEqual(x.shape[0], y.shape[0])
    self.assertGreater(y.shape[0], 0)

  def test_committed_sft_smoke_summary_shape_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    out = repo / "bench/qwen-rollout-20260612/sft-smoke-v1"
    if not out.exists(): return
    summary = json.loads((out / "summary.json").read_text())
    self.assertEqual(summary["kind"], "llm_sft_smoke_train_summary")
    self.assertEqual(summary["status"], "pass")
    self.assertEqual(summary["rows"], 150)
    self.assertEqual(summary["train_rows"] + summary["eval_rows"], 150)
    self.assertGreater(summary["deltas"]["eval_loss"], 0.5)
    self.assertGreater(summary["final"]["eval"]["accuracy"], 0.2)
    self.assertTrue((out / summary["artifacts"]["weights"]).exists())
    self.assertEqual((out / "README.md").read_text(), summary_markdown(summary))


if __name__ == "__main__":
  unittest.main()
