import json, pathlib, tempfile, unittest

from tinygrad import Tensor, dtypes
from tinygrad.llm.model import QKPrimitiveBudget, _load_qk_generated_policy, _qk_storage_summary, _shared_packed_view

def _policy(entries):
  return {"kind": "qk_generated_policy", "generator_version": 0, "commit": "test", "entries": entries}

def _entry(typ:int, rows:int, cols:int, winner:str, parts:int=1, opts=("LOCAL:0:64",), family:str="q4_k_packed_u32",
           reduction:str="split_k_partial"):
  return {
    "winner": winner,
    "descriptor": {"ggml_type": typ, "rows": rows, "cols": cols},
    "candidate": {"parts": parts, "opts": list(opts), "family": family, "reduction": reduction},
  }

class TestQKGeneratedPolicyRuntime(unittest.TestCase):
  def _write(self, obj) -> pathlib.Path:
    td = tempfile.TemporaryDirectory()
    self.addCleanup(td.cleanup)
    path = pathlib.Path(td.name) / "policy.json"
    path.write_text(json.dumps(obj))
    return path

  def test_loads_shape_format_policy(self):
    path = self._write(_policy([_entry(12, 12288, 4096, "v1_q4_packed", 1), _entry(12, 1024, 4096, "fused_graph", 0, ())]))
    table = _load_qk_generated_policy(str(path))
    self.assertEqual(table["by_shape"][(12, 12288, 4096)]["winner"], "v1_q4_packed")
    self.assertEqual(table["by_shape"][(12, 1024, 4096)]["winner"], "fused_graph")

  def test_loads_tensor_scoped_policy(self):
    entry = _entry(12, 12288, 4096, "fused_graph", 0, ())
    entry["scope"] = "tensor"
    entry["descriptor"]["tensor"] = "blk.0.ffn_gate.weight"
    entry["policy_reason"] = "memory_cap_fused_over_budget"
    path = self._write(_policy([entry]))
    table = _load_qk_generated_policy(str(path))
    key = ("blk.0.ffn_gate.weight", 12, 12288, 4096)
    self.assertEqual(table["by_tensor"][key]["winner"], "fused_graph")
    self.assertEqual(table["by_tensor"][key]["policy_reason"], "memory_cap_fused_over_budget")

  def test_loads_direct_output_family_and_reduction(self):
    entry = _entry(12, 1024, 4096, "direct_out", 1, ("LOCAL:0:64",),
                   family="q4_k_packed_u32_direct", reduction="direct_out")
    entry["scope"] = "tensor"
    entry["descriptor"]["tensor"] = "blk.0.attn_q.weight"
    path = self._write(_policy([entry]))
    table = _load_qk_generated_policy(str(path))
    loaded = table["by_tensor"][("blk.0.attn_q.weight", 12, 1024, 4096)]
    self.assertEqual(loaded["family"], "q4_k_packed_u32_direct")
    self.assertEqual(loaded["reduction"], "direct_out")

  def test_rejects_conflicting_shape_policy(self):
    path = self._write(_policy([
      _entry(12, 12288, 4096, "v1_q4_packed", 1),
      _entry(12, 12288, 4096, "fused_graph", 0, ()),
    ]))
    with self.assertRaisesRegex(ValueError, "conflicting"):
      _load_qk_generated_policy(str(path))

  def test_rejects_wrong_kind_or_version(self):
    with self.assertRaisesRegex(ValueError, "not a QK"):
      _load_qk_generated_policy(str(self._write({"kind": "other", "entries": []})))
    with self.assertRaisesRegex(ValueError, "generator_version"):
      _load_qk_generated_policy(str(self._write({"kind": "qk_generated_policy", "generator_version": 99, "entries": []})))

  def test_budget_reserves_and_caps_storage(self):
    budget = QKPrimitiveBudget(1024)
    self.assertTrue(budget.reserve("a", 512, "Q4_K"))
    self.assertEqual(budget.used_bytes, 512)
    self.assertFalse(budget.reserve("b", 600, "Q6_K"))
    self.assertEqual(budget.used_bytes, 512)

  def test_budget_strict_raises_on_cap(self):
    budget = QKPrimitiveBudget(1024, strict=True)
    self.assertTrue(budget.reserve("a", 900, "Q4_K"))
    with self.assertRaisesRegex(MemoryError, "storage cap exceeded"):
      budget.reserve("b", 200, "Q4_K")

  def test_storage_summary_counts_modes_and_bytes(self):
    from tinygrad.llm.model import Q4KPrimitiveLinear
    q4 = object.__new__(Q4KPrimitiveLinear)
    q4.q4k_storage = type("S", (), {
      "source_bytes": 144, "persistent_bytes": 0, "shared_bytes": 0,
      "nonpersistent_bytes": 144, "mode": "q4_ondemand",
    })()
    summary = _qk_storage_summary([q4])
    self.assertEqual(summary["source_bytes"], 144)
    self.assertEqual(summary["persistent_bytes"], 0)
    self.assertEqual(summary["nonpersistent_bytes"], 144)
    self.assertEqual(summary["by_mode"], {"q4_ondemand": 1})

  def test_storage_summary_counts_shared_bytes(self):
    from tinygrad.llm.model import Q4KPrimitiveLinear, Q6KPrimitiveLinear
    q4 = object.__new__(Q4KPrimitiveLinear)
    q4.q4k_storage = type("S", (), {
      "source_bytes": 144, "persistent_bytes": 0, "shared_bytes": 144,
      "nonpersistent_bytes": 0, "mode": "shared",
    })()
    q6 = object.__new__(Q6KPrimitiveLinear)
    q6.q6k_storage = type("S", (), {
      "source_bytes": 210, "persistent_bytes": 210, "shared_bytes": 0,
      "nonpersistent_bytes": 0, "mode": "sidecar",
    })()
    summary = _qk_storage_summary([q4, q6])
    self.assertEqual(summary["source_bytes"], 354)
    self.assertEqual(summary["persistent_bytes"], 210)
    self.assertEqual(summary["shared_bytes"], 144)
    self.assertEqual(summary["by_kind"], {"Q4K": 0, "Q6K": 210})
    self.assertEqual(summary["by_mode"], {"shared": 1, "sidecar": 1})

  def test_shared_packed_view_reinterprets_raw_buffer(self):
    raw = Tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=dtypes.uint8).realize()
    view = _shared_packed_view({"raw_tensor": raw}, 0, 8, dtypes.uint32)
    self.assertEqual(view.shape, (2,))
    self.assertEqual(view.dtype, dtypes.uint32)
    self.assertEqual(view.numpy().tolist(), [0x04030201, 0x08070605])

if __name__ == "__main__":
  unittest.main()
