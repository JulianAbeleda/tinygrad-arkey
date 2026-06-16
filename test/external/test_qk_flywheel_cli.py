import importlib, json, pathlib, unittest
from tempfile import TemporaryDirectory

from extra import qk_flywheel_cli as cli

REPO = pathlib.Path(__file__).resolve().parents[2]
PROOF = REPO / "bench/amd-decode-flywheel-proof-20260614"


class TestQKFlywheelCli(unittest.TestCase):
  def test_every_command_resolves_to_an_importable_main(self):
    for name, module_path in cli.COMMANDS.items():
      module = importlib.import_module(module_path)
      self.assertTrue(callable(getattr(module, "main", None)), f"{name} -> {module_path} has no main()")

  def test_no_args_and_unknown_command_are_misuse(self):
    self.assertEqual(cli.main([]), 2)
    self.assertEqual(cli.main(["definitely-not-a-command"]), 2)

  def test_help_and_list_exit_clean(self):
    self.assertEqual(cli.main(["--list"]), 0)
    self.assertEqual(cli.main(["--help"]), 0)

  def test_dispatch_matches_direct_module_invocation(self):
    """The cost-model subcommand routed through the dispatcher must produce the
    same artifact as invoking the module's own main() with the same args."""
    cost_model = importlib.import_module("extra.qk_flywheel_cost_model")
    examples = str(PROOF / "kernel-triage-v1-featured-plus/examples.jsonl")
    with TemporaryDirectory() as raw_td:
      td = pathlib.Path(raw_td)
      via_cli = td / "cli"
      via_direct = td / "direct"
      common = ["--examples", examples, "--backend", "centroid", "--seed", "20260614"]
      self.assertEqual(cli.main(["cost-model", *common, "--out", str(via_cli)]), 0)
      import sys
      saved = sys.argv
      sys.argv = ["qk_flywheel_cost_model", *common, "--out", str(via_direct)]
      try:
        self.assertEqual(cost_model.main(), 0)
      finally:
        sys.argv = saved
      self.assertEqual(
        (via_cli / "predictions.jsonl").read_text(),
        (via_direct / "predictions.jsonl").read_text(),
      )
      self.assertEqual(
        json.loads((via_cli / "summary.json").read_text())["conclusion"],
        json.loads((via_direct / "summary.json").read_text())["conclusion"],
      )


if __name__ == "__main__":
  unittest.main()
