import unittest
from extra.llm.bench.qwen_lora_smoke_train import build_first_completion_examples


class FakeTokenizer:
  eos_id = 0
  def prefix(self): return [1]
  def role(self, role): return [2] if role == "user" else [3]
  def encode(self, text): return [ord(char) for char in text]
  def decode(self, tokens): return "".join(chr(token) for token in tokens)
  def end_turn(self): return [4]


class TestQwenLoRASmokeTrain(unittest.TestCase):
  def test_only_first_completion_token_is_labeled(self):
    rows = [
      {"id":"a", "source_id":"turn-a", "prompt":"short", "completion":"READ"},
      {"id":"b", "source_id":"turn-b", "prompt":"a longer prompt", "completion":"OPEN"},
    ]
    examples = build_first_completion_examples(rows, FakeTokenizer())
    self.assertEqual([example["target"] for example in examples], [ord("R"), ord("O")])
    self.assertEqual([example["target_text"] for example in examples], ["R", "O"])
    self.assertLess(len(examples[0]["tokens"]), len(examples[1]["tokens"]))
    self.assertNotIn(ord("R"), examples[0]["tokens"][-1:])

  def test_system_prompt_is_present_before_user_prompt(self):
    rows = [{"id":"a", "source_id":"turn-a", "prompt":"user", "completion":"READ"}]
    example = build_first_completion_examples(rows, FakeTokenizer(), system_prompt="system")[0]
    self.assertLess(example["tokens"].index(ord("s")), example["tokens"].index(ord("u")))


if __name__ == "__main__": unittest.main()
