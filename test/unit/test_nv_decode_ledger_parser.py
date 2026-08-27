import unittest

from extra.llm_research.decode.nv_q6k_down_packed_lanemap_profile import _current_decode_replays


def _row(size:int, name:str) -> dict:
  return {"entries": [{"name": name, "duration": 1.0, "start": float(i), "end": float(i+1)} for i in range(size)]}


class TestNVDecodeLedgerParser(unittest.TestCase):
  def test_current_decode_signature_excludes_prefill(self):
    prefill = [_row(n, "E_toks_symbolic") for n in (32, 64, 128, 256)]
    decode = [_row(n, "decode_kernel") for n in (33, 66, 132, 185)]
    replays = _current_decode_replays(prefill + decode + prefill + decode)
    self.assertEqual(len(replays), 2)
    self.assertTrue(all(len(replay) == 416 for replay in replays))
    self.assertTrue(all("_toks" not in entry["name"] for replay in replays for entry in replay))

  def test_toks_contamination_fails_closed(self):
    rows = [_row(n, "decode_kernel") for n in (33, 66, 132, 185)]
    rows[2]["entries"][0]["name"] = "E_toks_symbolic"
    self.assertEqual(_current_decode_replays(rows), [])


if __name__ == "__main__":
  unittest.main()
