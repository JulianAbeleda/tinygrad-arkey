import hashlib
import json
import os
import pathlib
import subprocess
import sys
import unittest

from tinygrad.llm import bench


class TestLLMBench(unittest.TestCase):
  def test_sha256_file_and_missing(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / "model.gguf"
      path.write_bytes(b"benchmark-model")
      self.assertEqual(bench.sha256_file(path), hashlib.sha256(b"benchmark-model").hexdigest())
      self.assertIsNone(bench.sha256_file(path.parent / "absent"))

  def test_schema_fails_closed_for_throughput(self):
    args = bench.parser().parse_args(["--metadata-only"])
    record = bench.build_record(args, ["--metadata-only"])
    self.assertFalse(bench.validate_record(record)["authority"]["throughput_authoritative"])
    record["measurement"]["throughput_tokens_per_s"] = 12.5
    with self.assertRaisesRegex(ValueError, "throughput"): bench.validate_record(record)

  def test_git_dirty_state_is_explicit(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
      self.assertEqual(bench.git_state(directory), {"commit": None, "dirty": None, "state": "unknown"})

  def test_git_dirty_state_detects_uncommitted_change(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
      def git(*args): subprocess.run(["git", "-C", directory, *args], check=True, stdout=subprocess.DEVNULL)
      git("init", "-q")
      git("config", "user.email", "bench@example.invalid")
      git("config", "user.name", "Benchmark Test")
      path = pathlib.Path(directory) / "tracked"
      path.write_text("clean")
      git("add", "tracked")
      git("commit", "-qm", "initial")
      path.write_text("dirty")
      state = bench.git_state(directory)
      self.assertEqual(state["state"], "ok")
      self.assertTrue(state["dirty"])

  def test_cli_metadata_schema_cpu_only(self):
    proc = subprocess.run([sys.executable, "-m", "tinygrad.llm.bench", "--dry-run", "--route-id", "decode"],
                          check=True, text=True, capture_output=True)
    record = json.loads(proc.stdout)
    self.assertEqual(record["schema_version"], bench.SCHEMA_VERSION)
    self.assertEqual(record["target"], {"requested":"CPU", "effective":"CPU", "kind":"cpu_smoke"})
    self.assertEqual(record["device"]["probe"], "disabled_no_accelerator_probe")
    self.assertEqual(record["routes"], [{"route_id": "decode", "status": "unproven", "plan_hash": None, "artifact_hash": None}])
    self.assertIsNone(record["measurement"]["throughput_tokens_per_s"])
    self.assertFalse(record["authority"]["throughput_authoritative"])

  def test_cli_refuses_to_imply_measurement_without_explicit_metadata_mode(self):
    proc = subprocess.run([sys.executable, "-m", "tinygrad.llm.bench"], text=True, capture_output=True)
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("choose --metadata-only or --execute", proc.stderr)

  def test_module_does_not_depend_on_research_tree(self):
    with open(bench.__file__) as source: self.assertNotIn("extra.llm_research", source.read())

  def test_execute_control_uses_generic_fallback_and_records_no_throughput(self):
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
      path = pathlib.Path(directory) / "model.gguf"; path.write_bytes(b"gguf")
      args = bench.parser().parse_args(["--execute", "--model", str(path), "--phase", "prefill", "--context", "3"])
      class Output:
        def realize(self): return self
      seen = {}
      def loader(model_path):
        from tinygrad.llm import model as model_module
        seen["route"] = os.environ.get("TINYGRAD_PREFILL_ROUTE")
        seen["decode_route"] = os.environ.get("TINYGRAD_DECODE_ROUTE")
        seen["generic_control"] = model_module._GENERIC_LLM_CONTROL.get()
        return (lambda tokens, start, temperature: Output()), {}
      record = bench.run_control(args, bench.build_record(args), loader=loader, tensor_factory=lambda tokens: (tokens, 0.0))
      self.assertEqual(seen["route"], "fp16")
      self.assertEqual(seen["decode_route"], "fp16")
      self.assertTrue(seen["generic_control"])
      self.assertEqual(record["execution"]["status"], "completed")
      self.assertEqual(record["routes"], [{"route_id":"generic_fp16", "status":"generic", "plan_hash":None, "artifact_hash":None}])
      self.assertIsNone(record["measurement"]["throughput_tokens_per_s"])
      self.assertEqual(record["target"]["effective"], "CPU")
