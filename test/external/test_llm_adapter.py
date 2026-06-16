import json, pathlib, unittest
from tempfile import TemporaryDirectory

import numpy as np

from tinygrad import Tensor, nn

from extra.llm_adapter import expand_lora_targets, install_lora, load_adapter, save_adapter
from extra.llm_adapter_json_data import write_dataset as write_json_dataset
from extra.llm_adapter_train import _build_examples, split_adapter_rows, summary_markdown
from extra.llm_adapter_signal_data import write_dataset


class ToyTokenizer:
  eos_id = 99
  def prefix(self): return [1]
  def role(self, role): return [2 if role == "user" else 3]
  def end_turn(self): return [4]
  def encode(self, text): return [ord(ch) % 127 for ch in text]


class ToyModel:
  def __init__(self):
    self.output = nn.Linear(3, 5, bias=False)


class ToyBlock:
  def __init__(self):
    self.ffn_gate = nn.Linear(3, 4, bias=False)
    self.ffn_up = nn.Linear(3, 4, bias=False)
    self.ffn_down = nn.Linear(4, 3, bias=False)


class ToyTransformer:
  def __init__(self, blocks:int=3):
    self.output = nn.Linear(3, 5, bias=False)
    self.blk = [ToyBlock() for _ in range(blocks)]


class TestLLMAdapter(unittest.TestCase):
  def test_zero_lora_preserves_base_output(self):
    model = ToyModel()
    x = Tensor.randn(2, 3)
    before = model.output(x).numpy()
    adapters = install_lora(model, ["output"], rank=2, alpha=4.0, seed=1)
    after = model.output(x).numpy()
    np.testing.assert_allclose(before, after)
    self.assertEqual(len(adapters[0].parameters()), 2)
    self.assertTrue(adapters[0].detach_base)

  def test_save_and_load_adapter(self):
    with TemporaryDirectory() as raw_td:
      root = pathlib.Path(raw_td)
      model = ToyModel()
      adapters = install_lora(model, ["output"], rank=2, alpha=4.0, seed=1)
      save_adapter(root, adapters, base_model="toy", source="rows.jsonl", seed=1)
      self.assertEqual(json.loads((root / "adapter.json").read_text())["kind"], "llm_lora_adapter")
      loaded_model = ToyModel()
      loaded = load_adapter(loaded_model, root)
      self.assertEqual(loaded[0].rank, 2)
      np.testing.assert_allclose(adapters[0].lora_a.numpy(), loaded[0].lora_a.numpy())
      np.testing.assert_allclose(adapters[0].lora_b.numpy(), loaded[0].lora_b.numpy())

  def test_expand_last_ffn_target_group(self):
    model = ToyTransformer(blocks=4)
    self.assertEqual(expand_lora_targets(model, ["last2_ffn"]), [
      "blk.2.ffn_gate", "blk.2.ffn_up", "blk.2.ffn_down",
      "blk.3.ffn_gate", "blk.3.ffn_up", "blk.3.ffn_down",
    ])

  def test_install_last_ffn_target_group(self):
    model = ToyTransformer(blocks=3)
    adapters = install_lora(model, ["last1_ffn"], rank=2, alpha=4.0, seed=1)
    self.assertEqual([adapter.target for adapter in adapters], ["blk.2.ffn_gate", "blk.2.ffn_up", "blk.2.ffn_down"])
    self.assertEqual(len(adapters), 3)
    self.assertTrue(all(adapter.detach_base is False for adapter in adapters))

  def test_target_groups_fail_loudly(self):
    with self.assertRaisesRegex(ValueError, "unknown LoRA target"):
      expand_lora_targets(ToyTransformer(), ["everything"])
    with self.assertRaisesRegex(ValueError, "requested 9 blocks"):
      expand_lora_targets(ToyTransformer(blocks=2), ["last9_ffn"])

  def test_build_examples_uses_completion_target(self):
    rows = [{"id": "r1", "source_id": "s1", "prompt": "hi", "completion": "ok", "tags": ["t"]}]
    examples = _build_examples(rows, ToyTokenizer(), "chat", 128)
    self.assertEqual(len(examples), 2)
    self.assertEqual([x["target"] for x in examples], [ord("o") % 127, ord("k") % 127])
    self.assertEqual(examples[0]["input_tokens"], examples[0]["prompt_tokens"])
    self.assertGreater(examples[1]["input_tokens"], examples[1]["prompt_tokens"])

  def test_build_examples_can_append_eos_target(self):
    rows = [{"id": "r1", "source_id": "s1", "prompt": "hi", "completion": "ok", "tags": ["t"]}]
    examples = _build_examples(rows, ToyTokenizer(), "chat", 128, append_eos=True)
    self.assertEqual([x["target"] for x in examples], [ord("o") % 127, ord("k") % 127, ToyTokenizer.eos_id])

  def test_split_adapter_rows_honors_explicit_split(self):
    rows = [
      {"id": "a", "source_id": "a", "split": "train", "prompt": "p", "completion": "OK"},
      {"id": "b", "source_id": "b", "split": "eval", "prompt": "p", "completion": "OK"},
    ]
    train_rows, eval_rows, eval_ids = split_adapter_rows(rows)
    self.assertEqual([row["id"] for row in train_rows], ["a"])
    self.assertEqual([row["id"] for row in eval_rows], ["b"])
    self.assertEqual(eval_ids, ["b"])

  def test_signal_dataset_writes_sft_and_eval_prompts(self):
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      summary = write_dataset(out, target="OK", eval_every=2, limit=4)
      self.assertEqual(summary["train_rows"], 2)
      self.assertEqual(summary["eval_rows"], 2)
      sft_rows = [(json.loads(line)) for line in (out / "sft.jsonl").read_text().splitlines()]
      eval_rows = [(json.loads(line)) for line in (out / "eval-prompts.jsonl").read_text().splitlines()]
      self.assertEqual({row["split"] for row in sft_rows}, {"train", "eval"})
      self.assertTrue(all(row["completion"] == "OK" for row in sft_rows))
      self.assertTrue(all(row["expected_exact"] == "OK" and row["max_tokens"] == 1 for row in eval_rows))

  def test_json_dataset_writes_matching_sft_and_eval_prompts(self):
    with TemporaryDirectory() as raw_td:
      out = pathlib.Path(raw_td)
      summary = write_json_dataset(out, eval_every=2, limit=4)
      self.assertEqual(summary["train_rows"], 2)
      self.assertEqual(summary["eval_rows"], 2)
      sft_rows = [(json.loads(line)) for line in (out / "sft.jsonl").read_text().splitlines()]
      eval_rows = [(json.loads(line)) for line in (out / "eval-prompts.jsonl").read_text().splitlines()]
      by_id = {row["id"]: row for row in sft_rows}
      self.assertEqual({row["split"] for row in sft_rows}, {"train", "eval"})
      for row in eval_rows:
        self.assertEqual(json.loads(by_id[row["id"]]["completion"]), row["expected_json"])
        self.assertEqual(row["max_tokens"], 24)

  def test_committed_qwen_adapter_artifact_shape_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / "bench/qwen-adapter-20260613/8b-output-lora-r4"
    if not (root / "train-summary.json").exists():
      self.skipTest("committed bench artifact absent (gitignored post-prune); regenerate to re-lock")
    summary = json.loads((root / "train-summary.json").read_text())
    self.assertEqual(summary["kind"], "llm_adapter_train_summary")
    self.assertEqual(summary["status"], "pass")
    self.assertEqual(summary["adapter_kind"], "output_lora")
    self.assertEqual(summary["targets"], ["output"])
    self.assertGreater(summary["deltas"]["adapter_l2"], 0.0)
    self.assertTrue((root / summary["artifacts"]["config"]).exists())
    self.assertTrue((root / summary["artifacts"]["weights"]).exists())
    self.assertEqual((root / "README.md").read_text(), summary_markdown(summary))


if __name__ == "__main__":
  unittest.main()
