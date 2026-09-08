import unittest
from extra.llm.bench.qwen_lora_smoke_train import build_first_completion_examples


class FakeTokenizer:
  eos_id = 0
  def prefix(self): return [1]
  def role(self, role): return [2] if role == "user" else [3]
  def encode(self, text): return [ord(char) for char in text]
  def end_turn(self): return [4]


class TestQwenLoRASmokeTrain(unittest.TestCase):
  def test_only_first_completion_token_is_labeled(self):
    rows = [
      {"id":"a", "source_id":"turn-a", "prompt":"short", "completion":"READ"},
      {"id":"b", "source_id":"turn-b", "prompt":"a longer prompt", "completion":"OPEN"},
    ]
    examples = build_first_completion_examples(rows, FakeTokenizer())
    self.assertEqual([example["target"] for example in examples], [ord("R"), ord("O")])
    self.assertEqual(len(examples[0]["tokens"]), len(examples[1]["tokens"]))
    self.assertNotIn(ord("R"), examples[0]["tokens"][-1:])


if __name__ == "__main__": unittest.main()
