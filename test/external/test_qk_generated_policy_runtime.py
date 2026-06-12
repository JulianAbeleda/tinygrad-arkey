import json, pathlib, tempfile, unittest

from tinygrad.llm.model import _load_qk_generated_policy

def _policy(entries):
  return {"kind": "qk_generated_policy", "generator_version": 0, "commit": "test", "entries": entries}

def _entry(typ:int, rows:int, cols:int, winner:str, parts:int=1, opts=("LOCAL:0:64",)):
  return {
    "winner": winner,
    "descriptor": {"ggml_type": typ, "rows": rows, "cols": cols},
    "candidate": {"parts": parts, "opts": list(opts), "family": "q4_k_packed_u32"},
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

if __name__ == "__main__":
  unittest.main()
