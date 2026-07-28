import importlib.util
import io
import pathlib
import sys
import types
import unittest
from contextlib import redirect_stderr


ROOT = pathlib.Path(__file__).parents[2]
SCRIPT = ROOT / "extra/usbgpu/tests/minimal_amd_compute.py"


def load_module():
  spec = importlib.util.spec_from_file_location("minimal_amd_compute", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class TestEGPUMinimalCompute(unittest.TestCase):
 def test_validate_result_accepts_exact_float_values_and_rejects_any_difference(self):
  module = load_module()
  module.validate_result([2, 5, 10, 17])
  with self.assertRaisesRegex(ValueError, "AMD result mismatch"):
   module.validate_result([2, 5, 10, 18])

 def test_run_uses_amd_device_and_realizes_before_transfer(self):
  module, events = load_module(), []

  class FakeValue:
    def realize(self): events.append("realize"); return self
    def tolist(self): events.append("tolist"); return [2, 5, 10, 17]

  class FakeTensor:
    def __init__(self, values, device):
      assert values == [1, 2, 3, 4]
      assert device == "AMD-device"
    def __mul__(self, other): assert other is self; return self
    def __add__(self, other): assert other == 1; return FakeValue()

  fake = types.SimpleNamespace(Device={"AMD": "AMD-device"}, Tensor=FakeTensor)
  prior = sys.modules.get("tinygrad")
  sys.modules["tinygrad"] = fake
  try: module.run()
  finally:
   if prior is None: del sys.modules["tinygrad"]
   else: sys.modules["tinygrad"] = prior
  self.assertEqual(events, ["realize", "tolist"])

 def test_main_reports_runtime_failure_to_stderr(self):
  module, stderr = load_module(), io.StringIO()
  prior = module.run
  module.run = lambda: (_ for _ in ()).throw(RuntimeError("no AMD"))
  try:
   with redirect_stderr(stderr): self.assertEqual(module.main(), 1)
  finally: module.run = prior
  self.assertIn("minimal AMD compute failed: no AMD", stderr.getvalue())
