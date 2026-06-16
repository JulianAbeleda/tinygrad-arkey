"""Tests for the bounded machine-search schema authority (`extra/qk_search_spec.py`).

Synthetic data only — no hardware, no model load, no runtime behaviour. The one
real-artifact assertion golden-skips if the committed policy.json is absent,
matching the repo's golden-test convention (`test_qk_config.py`).
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

from extra.llm_eval_common import load_json
from extra.qk_search_spec import (
  AcceptedPolicy, Constraints, Model, Objective, Phase, SearchRow,
  assemble_search_row, backend_choices, baseline, from_generated_policy,
  load_accepted_policy, load_search_rows, model_choices, model_size_key,
  objective_choices, op_scope_choices, phase_choices, save_accepted_policy,
  save_search_rows, search_space_choices, validate_backend, validate_model,
  validate_objective, validate_op_scope, validate_phase, validate_search_space,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = REPO / "test/external/fixtures/qk_policy_min.json"
REAL_8B_POLICY = REPO / "bench/qk-shared-storage-20260612/8b/policy.json"


def _valid_row(**over) -> dict:
  base = dict(row_id="r0", phase="decode", model="qwen3_8b", op_scope="q4k_gemv",
              backend="AMD", search_space="primitive_policy", objective="tok_s")
  base.update(over)
  return base


class TestEnums(unittest.TestCase):
  def test_choices_match_enum_members(self):
    self.assertEqual(set(phase_choices()), {p.value for p in Phase})
    self.assertEqual(set(model_choices()), {m.value for m in Model})
    self.assertEqual(set(op_scope_choices()), {"q4k_gemv", "q6k_gemv", "attention", "ffn_down", "lm_head", "scheduler"})
    self.assertEqual(set(search_space_choices()), {"primitive_policy", "demotion", "flash_threshold", "storage", "schedule", "lds_blocking"})
    self.assertEqual(set(objective_choices()), {o.value for o in Objective})
    self.assertEqual(backend_choices(), ("AMD",))

  def test_validate_roundtrip(self):
    self.assertEqual(validate_phase("decode"), "decode")
    self.assertEqual(validate_model("qwen3_14b"), "qwen3_14b")
    self.assertEqual(validate_op_scope("attention"), "attention")
    self.assertEqual(validate_search_space("flash_threshold"), "flash_threshold")
    self.assertEqual(validate_objective("hbm_pct"), "hbm_pct")
    self.assertEqual(validate_backend("AMD"), "AMD")

  def test_validate_raises_on_unknown(self):
    for fn, bad in ((validate_phase, "encode"), (validate_model, "llama_7b"), (validate_op_scope, "gemm"),
                    (validate_search_space, "magic"), (validate_objective, "throughput"), (validate_backend, "CUDA")):
      with self.assertRaises(ValueError):
        fn(bad)

  def test_model_size_key(self):
    self.assertEqual(model_size_key("qwen3_8b"), "8B")
    self.assertEqual(model_size_key("qwen3_32b"), "32B")
    with self.assertRaises(ValueError):
      model_size_key("qwen3_70b")


class TestConstraints(unittest.TestCase):
  def test_defaults_valid(self):
    c = Constraints()
    self.assertTrue(c.exact_required)
    self.assertEqual(c.ctx_range, (1, 4096))

  def test_rejects_bad_values(self):
    with self.assertRaises(ValueError): Constraints(dnll_epsilon=-0.1)
    with self.assertRaises(ValueError): Constraints(max_storage_mb=0)
    with self.assertRaises(ValueError): Constraints(max_storage_mb=-5)
    with self.assertRaises(ValueError): Constraints(ctx_range=(0, 10))
    with self.assertRaises(ValueError): Constraints(ctx_range=(100, 50))
    with self.assertRaises(ValueError): Constraints(ctx_range=(1, 2, 3))

  def test_roundtrip(self):
    c = Constraints(exact_required=False, dnll_epsilon=0.01, max_storage_mb=512, ctx_range=(1, 399), no_beam_remote=True)
    self.assertEqual(Constraints.from_dict(c.to_dict()), c)


class TestAssembleSearchRow(unittest.TestCase):
  def test_valid_returns_canonical_shape(self):
    row = assemble_search_row(**_valid_row())
    self.assertEqual(row["id"], "r0")
    self.assertEqual(set(row), {"id", "phase", "model", "op_scope", "backend", "search_space", "objective", "constraints"})
    self.assertEqual(row["constraints"]["ctx_range"], [1, 4096])

  def test_raises_on_bad_enum(self):
    with self.assertRaises(ValueError): assemble_search_row(**_valid_row(phase="encode"))
    with self.assertRaises(ValueError): assemble_search_row(**_valid_row(backend="CUDA"))
    with self.assertRaises(ValueError): assemble_search_row(**_valid_row(op_scope="gemm"))

  def test_raises_on_empty_row_id(self):
    with self.assertRaises(ValueError): assemble_search_row(**_valid_row(row_id=""))

  def test_custom_constraints_propagate(self):
    row = assemble_search_row(**_valid_row(constraints=Constraints(ctx_range=(1, 399), exact_required=False)))
    self.assertEqual(row["constraints"]["ctx_range"], [1, 399])
    self.assertFalse(row["constraints"]["exact_required"])


class TestSearchRowIO(unittest.TestCase):
  def test_row_roundtrip(self):
    r = SearchRow.from_dict(assemble_search_row(**_valid_row(constraints=Constraints(ctx_range=(1, 399)))))
    self.assertEqual(SearchRow.from_dict(r.to_dict()), r)

  def test_jsonl_table_roundtrip(self):
    rows = [SearchRow.from_dict(assemble_search_row(**_valid_row(row_id=f"r{i}", op_scope=op)))
            for i, op in enumerate(("q4k_gemv", "attention", "lm_head"))]
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "spec.jsonl"
      save_search_rows(path, rows)
      self.assertEqual(load_search_rows(path), rows)

  def test_jsonl_rejects_duplicate_ids(self):
    rows = [SearchRow.from_dict(assemble_search_row(**_valid_row(row_id="dup"))),
            SearchRow.from_dict(assemble_search_row(**_valid_row(row_id="dup")))]
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "spec.jsonl"
      save_search_rows(path, rows)
      with self.assertRaises(ValueError):
        load_search_rows(path)


class TestAcceptedPolicy(unittest.TestCase):
  def _valid(self, **over) -> AcceptedPolicy:
    base = dict(model="qwen3_8b", phase="decode", backend="AMD", ctx_range=(1, 399), objective="tok_s",
                baseline_tok_s=55.0, accepted_tok_s=60.9, quality_gate="dNLL <= baseline + epsilon",
                exactness="byte-identical", commit="abc123")
    base.update(over)
    return AcceptedPolicy(**base)

  def test_valid_roundtrip(self):
    ap = self._valid(memory_cap_mb=512)
    self.assertEqual(AcceptedPolicy.from_dict(ap.to_dict()), ap)

  def test_rejects_bad_values(self):
    with self.assertRaises(ValueError): self._valid(phase="encode")
    with self.assertRaises(ValueError): self._valid(backend="CUDA")
    with self.assertRaises(ValueError): self._valid(baseline_tok_s=-1.0)
    with self.assertRaises(ValueError): self._valid(commit="")
    with self.assertRaises(ValueError): self._valid(ctx_range=(10, 1))
    with self.assertRaises(ValueError): self._valid(memory_cap_mb=0)

  def test_json_roundtrip(self):
    ap = self._valid()
    with tempfile.TemporaryDirectory() as d:
      path = pathlib.Path(d) / "accepted.json"
      save_accepted_policy(path, ap)
      self.assertEqual(load_accepted_policy(path), ap)


class TestBaseline(unittest.TestCase):
  def test_returns_numbers(self):
    b = baseline("qwen3_8b")
    self.assertEqual(b["size"], "8B")
    self.assertGreater(b["llama_tok_s"], 0)
    self.assertGreater(b["model_bytes"], 0)
    self.assertGreater(b["hbm_peak_gbs"], 0)

  def test_unknown_model_raises(self):
    with self.assertRaises(ValueError):
      baseline("qwen3_70b")


class TestFromGeneratedPolicy(unittest.TestCase):
  def test_adapter_on_fixture(self):
    policy = load_json(FIXTURE)
    ap = from_generated_policy(policy, model="qwen3_8b", baseline_tok_s=50.4, accepted_tok_s=60.9, ctx_range=(1, 399))
    self.assertEqual(ap.model, "qwen3_8b")
    self.assertEqual(ap.phase, "decode")
    self.assertEqual(ap.backend, "AMD")
    self.assertEqual(ap.commit, "deadbeef0")
    self.assertIsNone(ap.memory_cap_mb)
    self.assertEqual(ap.exactness, "byte-identical")
    # round-trips through the durable record shape
    self.assertEqual(AcceptedPolicy.from_dict(ap.to_dict()), ap)

  def test_adapter_rejects_non_policy(self):
    with self.assertRaises(ValueError):
      from_generated_policy({"kind": "something_else"}, model="qwen3_8b", baseline_tok_s=1.0, accepted_tok_s=1.0)
    with self.assertRaises(ValueError):
      from_generated_policy({"kind": "qk_generated_policy", "generator_version": 99, "commit": "x"},
                            model="qwen3_8b", baseline_tok_s=1.0, accepted_tok_s=1.0)

  def test_adapter_on_real_artifact(self):
    if not REAL_8B_POLICY.exists():
      self.skipTest("real 8B policy artifact not present")
    policy = load_json(REAL_8B_POLICY)
    ap = from_generated_policy(policy, model="qwen3_8b", baseline_tok_s=50.41, accepted_tok_s=52.07, ctx_range=(1, 399))
    self.assertEqual(ap.backend, "AMD")
    self.assertTrue(ap.commit)
    self.assertEqual(AcceptedPolicy.from_dict(ap.to_dict()), ap)


if __name__ == "__main__":
  unittest.main()
