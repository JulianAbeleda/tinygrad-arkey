import pathlib, tempfile, unittest

from extra.qk_decode_summary import _md, parse_log


class TestQKDecodeSummary(unittest.TestCase):
  def test_parse_storage_debug_line(self):
    text = "\n".join([
      "QK_GENERATED_POLICY_DEBUG loaded=policy.json entries=449",
      "Q4K_PRIMITIVE_DEBUG installed=112 skipped_total=3 runtime_storage_cap=3 source_bytes=1462763520 storage_bytes=0 runtime_cap_bytes=1610612736 runtime_cap_used_bytes=0 storage_mode=q4_ondemand",
      "Q6K_PRIMITIVE_DEBUG installed=32 skipped_total=4 source_bytes=137625600 storage_bytes=137625600 runtime_cap_bytes=1610612736 runtime_cap_used_bytes=137625600 storage_mode=sidecar",
      "QK_PRIMITIVE_STORAGE_DEBUG installed=144 source_bytes=1600389120 storage_bytes=137625600 runtime_cap_bytes=1610612736 runtime_cap_used_bytes=137625600 by_kind=Q4K:0,Q6K:137625600 by_mode=q4_ondemand:112,sidecar:32",
      "250.00 ms,   4.00 tok/s,   80.00 GB/s, 1000/2000 MB  -- sample",
    ])
    with tempfile.TemporaryDirectory() as td:
      path = pathlib.Path(td) / "decode.log"
      path.write_text(text)
      row = parse_log("generated", path)
    self.assertEqual(row["installs"]["Q4K"]["storage_bytes"], 0)
    self.assertEqual(row["installs"]["Q4K"]["runtime_storage_cap"], 3)
    self.assertEqual(row["storage"]["storage_bytes"], 137625600)
    self.assertIn("storage MB", _md([row]))


if __name__ == "__main__":
  unittest.main()
