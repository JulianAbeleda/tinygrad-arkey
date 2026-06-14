import json, unittest

from extra.llm_eval_common import score_prompt
from extra.llm_json_rejection_sample import build_rejection_dataset

def _source(row_id:str, split:str, category:str, answer):
  return {
    "id": row_id,
    "source_id": row_id,
    "split": split,
    "category": category,
    "prompt": f"Return JSON. Question {row_id}?",
    "completion": json.dumps({"answer": answer}, separators=(",", ":")),
    "expected_json": {"answer": answer},
    "normalized_answer": answer,
    "tags": ["json_answer", category, split],
    "max_tokens": 24,
  }

def _sample(source:dict, sample_idx:int, text:str):
  prompt = {"expected_json": source["expected_json"]}
  return {
    "id": f"{source['id']}:sample{sample_idx:02d}",
    "source_id": source["id"],
    "category": source["category"],
    "prompt": source["prompt"],
    "tags": source["tags"],
    "max_tokens": 24,
    "expected_json": source["expected_json"],
    "normalized_answer": source["normalized_answer"],
    "sample_idx": sample_idx,
    "temperature": 0.0,
    "seed": 1 + sample_idx,
    "text": text,
    "tokens": [1, 2],
    "generated": 2,
    "elapsed_s": 1.0,
    "tok_s": 2.0,
    "score": score_prompt(prompt, text),
  }

class TestLLMJsonRejectionSample(unittest.TestCase):
  def test_build_rejection_dataset_selects_strict_passes_only(self):
    train = [_source("train_a", "train", "math", "42"), _source("train_b", "train", "code", "len")]
    eval_rows = [_source("eval_a", "eval", "math", "99")]
    samples = [
      _sample(train[0], 0, '{"answer":"41"}'),
      _sample(train[0], 1, '{"answer":"42"}'),
      _sample(train[1], 0, 'Answer: {"answer":"len"}'),
      _sample(train[1], 1, '{"answer":"len"}'),
    ]
    accepted, near_miss, sft_rows, summary = build_rejection_dataset(samples, train, eval_rows)
    self.assertEqual([row["id"] for row in accepted], ["train_a:sample01", "train_b:sample01"])
    self.assertEqual([row["id"] for row in near_miss], ["train_a:sample00"])
    self.assertEqual(summary["selected_train_rows"], 2)
    self.assertEqual(summary["integrity"]["train_eval_source_overlap"], 0)
    self.assertEqual([row["split"] for row in sft_rows], ["train", "train", "eval"])
    self.assertEqual(sft_rows[0]["completion"], '{"answer":"42"}')

  def test_build_rejection_dataset_rejects_eval_sourced_sample(self):
    train = [_source("train_a", "train", "math", "42")]
    eval_rows = [_source("eval_a", "eval", "math", "99")]
    with self.assertRaisesRegex(ValueError, "non-train source"):
      build_rejection_dataset([_sample(eval_rows[0], 0, '{"answer":"99"}')], train, eval_rows)

if __name__ == "__main__":
  unittest.main()
