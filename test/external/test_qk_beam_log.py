import json, pathlib, unittest

REPO = pathlib.Path(__file__).resolve().parents[2]
N0 = REPO / "bench/amd-decode-flywheel-proof-20260614/native-matmul-N0"


def _key(opts):
  return tuple((o["op"], o["axis"], tuple(o["arg"]) if isinstance(o["arg"], list) else o["arg"]) for o in opts)


class TestQKBeamLog(unittest.TestCase):
  def test_committed_n0b_substrate_dataset(self):
    log = N0 / "beam_log.jsonl"
    if not log.exists():
      self.skipTest("N0b not run yet")
    recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    self.assertGreater(len(recs), 500)
    shapes = sorted({(r["M"], r["K"], r["N"]) for r in recs})
    self.assertGreaterEqual(len(shapes), 4)
    byshape = {s: [r for r in recs if (r["M"], r["K"], r["N"]) == s and r["valid"]] for s in shapes}

    # (1) rugged: large spread between best and worst valid config (config choice matters)
    for s in shapes:
      tfs = [r["tflops"] for r in byshape[s] if r["tflops"]]
      self.assertGreater(max(tfs) / min(tfs), 10.0, s)

    # (2) NO universal winner: no config is in the top-5 of every shape (a lookup fails -> learnable)
    top5 = {}
    for s in shapes:
      for r in sorted(byshape[s], key=lambda r: -r["tflops"])[:5]:
        top5[_key(r["opts"])] = top5.get(_key(r["opts"]), 0) + 1
    self.assertEqual(sum(1 for n in top5.values() if n == len(shapes)), 0)

  def test_committed_n0b_summary(self):
    f = N0 / "n0b_summary.json"
    if not f.exists():
      self.skipTest("N0b not run yet")
    d = json.loads(f.read_text())
    self.assertEqual(d["phase"], "Phase N0b")
    for s in d["shapes"]:
      self.assertIsNotNone(s["best_tflops"])


if __name__ == "__main__":
  unittest.main()
