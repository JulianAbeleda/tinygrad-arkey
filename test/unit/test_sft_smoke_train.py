import json, pathlib, tempfile, unittest

from extra.llm.bench.sft_smoke_train import build_byte_examples, load_sft_rows, run_smoke_train, split_rows


FIXTURE = pathlib.Path(__file__).parents[1] / "fixtures/llm/self_training_mvp.jsonl"


class TestSFTSmokeTrain(unittest.TestCase):
  def test_completion_only_targets_and_source_split(self):
    rows = load_sft_rows(FIXTURE)
    train, eval_rows, eval_ids = split_rows(rows, eval_every=5)
    self.assertFalse({row["source_id"] for row in train} & {row["source_id"] for row in eval_rows})
    self.assertEqual(eval_ids, sorted(eval_ids))
    one = rows[0]
    _, targets = build_byte_examples([one], context_bytes=4)
    self.assertEqual(len(targets), len((one["completion"] + "\n").encode()))

  def test_training_improves_heldout_loss_and_reloads_adapter(self):
    rows = load_sft_rows(FIXTURE)
    with tempfile.TemporaryDirectory() as directory:
      adapter = pathlib.Path(directory) / "adapter"
      summary, _ = run_smoke_train(rows, device="CPU", context_bytes=8, vocab_size=128,
                                   steps=60, batch_size=128, lr=0.02, rank=8, alpha=8,
                                   adapter_out=adapter, source=str(FIXTURE))
      self.assertEqual(summary["status"], "pass", json.dumps(summary, indent=2))
      self.assertEqual(summary["base_sha256"]["before"], summary["base_sha256"]["after"])
      self.assertLessEqual(summary["reload_max_abs"], 1e-6)
      self.assertTrue((adapter / "adapter.json").is_file())
      self.assertTrue((adapter / "adapter.npz").is_file())


if __name__ == "__main__": unittest.main()
