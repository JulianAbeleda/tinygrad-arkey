import json, pathlib, unittest
from tempfile import TemporaryDirectory

import numpy as np

from tinygrad import Tensor, nn

from extra.llm_adapter import install_lora, load_adapter, save_adapter
from extra.llm_adapter_train import _build_examples, summary_markdown


class ToyTokenizer:
  def prefix(self): return [1]
  def role(self, role): return [2 if role == "user" else 3]
  def end_turn(self): return [4]
  def encode(self, text): return [ord(ch) % 127 for ch in text]


class ToyModel:
  def __init__(self):
    self.output = nn.Linear(3, 5, bias=False)


class TestLLMAdapter(unittest.TestCase):
  def test_zero_lora_preserves_base_output(self):
    model = ToyModel()
    x = Tensor.randn(2, 3)
    before = model.output(x).numpy()
    adapters = install_lora(model, ["output"], rank=2, alpha=4.0, seed=1)
    after = model.output(x).numpy()
    np.testing.assert_allclose(before, after)
    self.assertEqual(len(adapters[0].parameters()), 2)

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

  def test_build_examples_uses_completion_target(self):
    rows = [{"id": "r1", "source_id": "s1", "prompt": "hi", "completion": "ok", "tags": ["t"]}]
    examples = _build_examples(rows, ToyTokenizer(), "chat", 128)
    self.assertEqual(len(examples), 2)
    self.assertEqual([x["target"] for x in examples], [ord("o") % 127, ord("k") % 127])
    self.assertEqual(examples[0]["input_tokens"], examples[0]["prompt_tokens"])
    self.assertGreater(examples[1]["input_tokens"], examples[1]["prompt_tokens"])

  def test_committed_qwen_adapter_artifact_shape_if_present(self):
    repo = pathlib.Path(__file__).resolve().parents[2]
    root = repo / "bench/qwen-adapter-20260613/8b-output-lora-r4"
    if not root.exists(): return
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
