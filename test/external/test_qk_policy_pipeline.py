import pathlib, unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from extra.qk_policy_pipeline import _runtime_storage_summary, _validate_or_init_manifest, _write_stage_status, _profile_specs


class TestQKPolicyPipeline(unittest.TestCase):
  def _args(self, td:pathlib.Path, **overrides):
    model = overrides.pop("model", td / "Qwen3-8B-Q4_K_M.gguf")
    if not model.exists(): model.write_bytes(b"fake-model")
    out = overrides.pop("out", td / "out")
    out.mkdir(exist_ok=True)
    values = {
      "model": model, "repo": pathlib.Path.cwd(), "out": out, "device": "AMD", "level": 2, "iters": 2,
      "benchmark": 128, "reference_mode": "explicit", "repeats": 3, "max_extra_repeats": 2,
      "ab_tokens": 32, "profile": "auto", "profile_tokens": 8, "accept_gain": 0.03,
      "tie_band": 0.03, "profile_gain": 0.20, "candidate_timeout": 120.0,
      "policy_max_storage_mb": None, "reuse": False, "force": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)

  def test_generic_reference_profile_does_not_enable_primitives(self):
    args = SimpleNamespace(reference_mode="generic", device="AMD", model_size="32B")
    specs = _profile_specs(args, pathlib.Path("policy.json"))
    reference = [x for x in specs if x[0].startswith("reference")]
    self.assertEqual([x[1] for x in reference], ["32b-baseline-batched-debug2.log", "32b-baseline-named-debug2.log"])
    for _, _, env in reference:
      self.assertNotIn("Q4K_PRIMITIVE", env)
      self.assertNotIn("Q6K_PRIMITIVE", env)

  def test_explicit_reference_profile_enables_primitives(self):
    args = SimpleNamespace(reference_mode="explicit", device="AMD", model_size="14B")
    specs = _profile_specs(args, pathlib.Path("policy.json"))
    reference = [x for x in specs if x[0].startswith("reference")]
    self.assertEqual([x[1] for x in reference], ["14b-q4q6-primitive-batched-debug2.log", "14b-q4q6-primitive-named-debug2.log"])
    for _, _, env in reference:
      self.assertEqual(env["Q4K_PRIMITIVE"], "1")
      self.assertEqual(env["Q6K_PRIMITIVE"], "1")

  def test_manifest_validates_reuse_identity(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      args = self._args(td)
      _validate_or_init_manifest(args)
      _validate_or_init_manifest(self._args(td, model=args.model, out=args.out, reuse=True))
      with self.assertRaisesRegex(ValueError, "manifest does not match"):
        _validate_or_init_manifest(self._args(td, model=args.model, out=args.out, benchmark=256, reuse=True))
      forced = self._args(td, model=args.model, out=args.out, benchmark=256, reuse=True, force=True)
      _validate_or_init_manifest(forced)
      self.assertFalse(forced.reuse)

  def test_manifest_allows_profile_only_reuse_change(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      args = self._args(td, profile="never")
      _validate_or_init_manifest(args)
      _validate_or_init_manifest(self._args(td, model=args.model, out=args.out, reuse=True, profile="auto"))

  def test_stage_status_updates_manifest(self):
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      args = self._args(td)
      _validate_or_init_manifest(args)
      _write_stage_status(args, "search", "passed", outputs=[args.out / "search.json"], metadata={"x": 1})
      manifest = (args.out / "manifest.json").read_text()
      self.assertIn('"search"', manifest)
      self.assertIn('"status": "passed"', manifest)
      self.assertEqual((args.out / "search.status.json").exists(), True)

  def test_runtime_storage_summary_keeps_storage_rows(self):
    rows = [
      {"label": "explicit1", "storage": None},
      {"label": "generated1", "storage": {"storage_bytes": 123}},
    ]
    self.assertEqual(_runtime_storage_summary(rows), {"generated1": {"storage_bytes": 123}})


if __name__ == "__main__":
  unittest.main()
