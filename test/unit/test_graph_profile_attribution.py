import json, tempfile, os, unittest
from extra.qk.graph_profile_attribution import summarize, PROVEN_NAMES

def _write_capture(path, entries):
  with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps({"entries": entries}) + "\n")

def _entry(name, device="AMD0", start=0.0, end=1.0, metadata=None):
  return {"device": device, "name": name, "metadata": metadata, "start": str(start), "end": str(end),
          "duration": str(end - start)}

class TestGraphProfileAttributionSummarize(unittest.TestCase):
  def test_proven_candidate_identity_wins_over_semantic_op(self):
    # a proven GEMM kernel name should resolve via PROVEN_NAMES even if (hypothetically) tagged with a
    # semantic_op metadata -- candidate identity is the exact join and takes priority.
    proven_name = next(iter(PROVEN_NAMES))
    entries = [_entry(proven_name, metadata={"semantic_op": "rms_norm"})]
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "cap.jsonl")
      _write_capture(p, entries)
      out = summarize([p])
    self.assertIn(PROVEN_NAMES[proven_name], out["roles"])
    self.assertEqual(out["roles"][PROVEN_NAMES[proven_name]]["count"], 1)
    self.assertNotIn("rms_norm", out["roles"])

  def test_semantic_op_metadata_buckets_non_candidate_roles(self):
    roles = ["rms_norm", "rope", "attn_score", "attn_av", "attn_mask", "softmax", "residual"]
    entries = [_entry(f"__generic_op_{i}__", metadata={"semantic_op": role}) for i, role in enumerate(roles)]
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "cap.jsonl")
      _write_capture(p, entries)
      out = summarize([p])
    for role in roles:
      self.assertIn(role, out["roles"], f"missing role bucket: {role}")
      self.assertEqual(out["roles"][role]["count"], 1)
    self.assertNotIn("unknown", out["roles"])

  def test_untagged_dispatch_stays_unknown(self):
    entries = [_entry("__add__", metadata=None), _entry("matmul", metadata={})]
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "cap.jsonl")
      _write_capture(p, entries)
      out = summarize([p])
    self.assertIn("unknown", out["roles"])
    self.assertEqual(out["roles"]["unknown"]["count"], 2)

  def test_device_busy_union_output_intact(self):
    entries = [_entry("__add__", device="AMD0", start=0.0, end=5.0),
               _entry("__mul__", device="AMD0", start=3.0, end=8.0, metadata={"semantic_op": "residual"})]
    with tempfile.TemporaryDirectory() as d:
      p = os.path.join(d, "cap.jsonl")
      _write_capture(p, entries)
      out = summarize([p])
    self.assertIn("AMD0", out["device_busy"])
    self.assertAlmostEqual(out["device_busy"]["AMD0"]["sum_ticks"], 10.0)
    self.assertAlmostEqual(out["device_busy"]["AMD0"]["union_ticks"], 8.0)

if __name__ == "__main__":
  unittest.main()
