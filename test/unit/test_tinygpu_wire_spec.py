import copy, json, pathlib, re, struct, unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "extra/usbgpu/protocol"
FIXTURES = PROTOCOL / "fixtures"

def load(name):
  with open(FIXTURES / name) as f: return json.load(f)

def validate_status(status):
  required = {"schema", "provider_generation", "state", "enabled", "policy_id", "interval_ms", "maximum_timer_leeway_ms",
              "expected_identity", "last_identity_dword", "attempts", "successes", "failures", "consecutive_failures",
              "last_attempt_monotonic_ns", "last_success_monotonic_ns", "success_gap_over_leeway_count", "max_success_gap_ms",
              "timer_error", "counter_saturated", "active_workload_leases", "active_bar_mappings", "active_dma_allocations"}
  if set(status) != required: return False
  if status["schema"] != "tinygpu.keepalive.v1" or status["state"] not in {"unsupported", "inactive", "active_healthy", "active_degraded", "quiescing", "stopped"}: return False
  if status["expected_identity"] != "1002:744c" or not re.fullmatch(r"0x[0-9a-f]{8}", status["last_identity_dword"]): return False
  if status["policy_id"] != "usb4_amd_744c_v1" or status["interval_ms"] != 1000 or status["maximum_timer_leeway_ms"] != 100: return False
  u64 = ("provider_generation", "attempts", "successes", "failures", "consecutive_failures", "last_attempt_monotonic_ns", "last_success_monotonic_ns", "success_gap_over_leeway_count", "max_success_gap_ms")
  u32 = ("interval_ms", "maximum_timer_leeway_ms", "active_workload_leases", "active_bar_mappings", "active_dma_allocations")
  if any(type(status[k]) is not int or not 0 <= status[k] <= (1 << 64)-1 for k in u64): return False
  if any(type(status[k]) is not int or not 0 <= status[k] <= (1 << 32)-1 for k in u32): return False
  if type(status["timer_error"]) is not int or not -(1 << 31) <= status["timer_error"] < (1 << 31): return False
  if type(status["enabled"]) is not bool or type(status["counter_saturated"]) is not bool: return False
  return status["attempts"] == status["successes"] + status["failures"]

def validate_error(value, expected_code):
  return set(value) == {"schema", "code", "message"} and value["schema"] == "tinygpu.error.v1" and value["code"] == expected_code and \
    type(value["message"]) is str and 1 <= len(value["message"].encode()) <= 512

class TestTinyGPUWireSpec(unittest.TestCase):
  def test_authority_mentions_all_fixture_contracts(self):
    text = (PROTOCOL / "tinygpu-wire-v1.md").read_text()
    for name in ("legacy-rpc-v1.json", "handshake-v1.json", "error-v1.json", "keepalive-status-v1.json"):
      self.assertIn(name, text)
    self.assertIn("independent codec", text)
    self.assertIn("33 bytes", text)
    self.assertIn("17 bytes", text)
    self.assertIn("Legacy success responses retain their command-specific", text)

  def test_legacy_wire_examples_have_exact_layout(self):
    fixture = load("legacy-rpc-v1.json")
    self.assertEqual((fixture["request_format"], fixture["request_bytes"]), ("<BIIQQQ", struct.calcsize("<BIIQQQ")))
    self.assertEqual((fixture["response_format"], fixture["response_bytes"]), ("<BQQ", struct.calcsize("<BQQ")))
    self.assertEqual(list(fixture["common_commands"].values()), list(range(12)))
    self.assertEqual(list(fixture["python_reserved_commands"].values()), [12, 13, 14])
    for item in fixture["examples"]:
      fmt = "<BIIQQQ" if len(item["fields"]) == 6 else "<BQQ"
      self.assertEqual(struct.pack(fmt, *item["fields"]).hex(), item["hex"])

  def test_negotiation_is_above_reserved_ids_and_probe_is_exact(self):
    fixture = load("handshake-v1.json")
    self.assertEqual(list(fixture["commands"].values()), [15, 16, 17, 18, 19])
    self.assertEqual(struct.pack("<BIIQQQ", *fixture["legacy_probe"]["fields"]).hex(), fixture["legacy_probe"]["hex"])
    self.assertEqual(len(bytes.fromhex(fixture["legacy_probe"]["hex"])), 33)
    self.assertEqual(fixture["legacy_outcomes"]["maps_to"], "unsupported_protocol")
    self.assertEqual(len(bytes.fromhex(fixture["legacy_outcomes"]["generic_error_response_hex"])), 17)
    self.assertEqual(fixture["success_payload"]["schema"], "tinygpu.handshake.v1")
    self.assertEqual(fixture["success_payload"]["capabilities"], 3)

  def test_response_statuses_and_typed_error_payloads(self):
    fixture = load("error-v1.json")
    statuses = fixture["response_status"]
    self.assertEqual(list(statuses.values()), list(range(11)))
    self.assertEqual(struct.pack("<BQQ", fixture["legacy_error"]["status"], 0, 0).hex(), fixture["legacy_error"]["response_hex"])
    for item in fixture["typed_errors"]:
      payload = json.dumps(item["payload"], separators=(",", ":")).encode()
      self.assertLessEqual(len(payload), fixture["maximum_payload_bytes"])
      self.assertTrue(validate_error(item["payload"], item["payload"]["code"]))
      self.assertEqual(payload.hex(), item["payload_hex"])
      self.assertEqual(struct.pack("<BQQ", item["status"], len(payload), 0).hex(), item["response_hex"])
      self.assertEqual(struct.unpack("<BQQ", struct.pack("<BQQ", item["status"], len(payload), 0)), (item["status"], len(payload), 0))

  def test_status_fixture_and_negative_cases(self):
    fixture = load("keepalive-status-v1.json")
    valid = fixture["valid"]
    self.assertTrue(validate_status(valid))
    self.assertLessEqual(len(json.dumps(valid, separators=(",", ":")).encode()), fixture["maximum_payload_bytes"])
    for case in fixture["invalid"]:
      value = copy.deepcopy(valid)
      value.update(case["patch"])
      self.assertFalse(validate_status(value), case["name"])

if __name__ == "__main__": unittest.main()
