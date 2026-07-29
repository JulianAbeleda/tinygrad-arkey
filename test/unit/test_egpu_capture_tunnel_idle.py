import copy, importlib.util, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "extra/usbgpu/protocol/fixtures"
SPEC = importlib.util.spec_from_file_location("capture_tunnel_idle", ROOT / "extra/usbgpu/tests/capture_tunnel_idle.py")
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


def fixtures():
  keepalive = json.loads((FIXTURES / "keepalive-status-v1.json").read_text())["valid"]
  power = json.loads((FIXTURES / "power-residency-status-v4.json").read_text())["valid"]
  keepalive["attempts"] = keepalive["successes"] = 1
  keepalive["last_attempt_monotonic_ns"] = keepalive["last_success_monotonic_ns"] = 2_000_000_000
  power["last_canary_success_monotonic_ns"] = 2_000_000_000
  return keepalive, power


def handshake():
  return {"schema":"tinygpu.handshake.v1", "protocol_major":1, "protocol_minor":0,
          "capabilities":11, "server_build_id":"tinygrad-arkey-native-v13"}


def test_capture_idle_preserves_paired_early_samples_and_registry():
  keepalive, power = fixtures(); now = [0.0]; calls = [0]
  def status_reader():
    calls[0] += 1; value = copy.deepcopy(keepalive)
    value["attempts"] = value["successes"] = calls[0]
    value["last_attempt_monotonic_ns"] = value["last_success_monotonic_ns"] = (calls[0] + 1) * 1_000_000_000
    return value
  def power_reader():
    value = copy.deepcopy(power); value["last_canary_success_monotonic_ns"] = (calls[0] + 1) * 1_000_000_000
    return value
  evidence = capture.capture_idle(handshake_reader=handshake, status_reader=status_reader, power_reader=power_reader,
    registry_reader=lambda:{"pci_devices":"IOPCITunnelL1Enable"}, duration_s=2, interval_s=1,
    clock=lambda:now[0], sleeper=lambda seconds:now.__setitem__(0, now[0] + seconds), wall_clock_ns=lambda:123)
  assert evidence["status"] == "passed"
  assert len(evidence["samples"]) == 3
  assert evidence["registry"] == {"pci_devices":"IOPCITunnelL1Enable"}
  assert evidence["samples"][0]["keepalive"]["successes"] == 1
  assert evidence["samples"][-1]["power"]["last_canary_success_monotonic_ns"] == 4_000_000_000


def test_capture_idle_stops_and_preserves_first_failure():
  keepalive, power = fixtures(); now = [0.0]; calls = [0]; checkpoints = []
  def status_reader():
    calls[0] += 1
    if calls[0] == 2: raise capture.QualificationError("keepalive status failed: 3")
    return copy.deepcopy(keepalive)
  evidence = capture.capture_idle(handshake_reader=handshake, status_reader=status_reader,
    power_reader=lambda:copy.deepcopy(power), duration_s=5, interval_s=1,
    clock=lambda:now[0], sleeper=lambda seconds:now.__setitem__(0, now[0] + seconds),
    checkpoint=lambda value:checkpoints.append(value))
  assert evidence["status"] == "failed"
  assert evidence["first_failure"]["type"] == "QualificationError"
  assert evidence["first_failure"]["message"] == "keepalive status failed: 3"
  assert len(evidence["samples"]) == 1
  assert checkpoints[-1]["status"] == "failed"


def test_validate_handshake_rejects_wrong_native_identity_shape():
  value = handshake(); value["server_build_id"] = "contains spaces"
  try: capture.validate_handshake(value)
  except capture.QualificationError as exc: assert str(exc) == "invalid diagnostic handshake"
  else: raise AssertionError("invalid handshake was accepted")


def test_v13_registration_is_exact_and_active():
  capture.validate_v13_registration("* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/13) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]\n")
  for invalid in (
    "* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/11) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]\n",
    "  * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/13) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]\n",
    "* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/13) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]\n"
    "* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/13) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]\n",
  ):
    try: capture.validate_v13_registration(invalid)
    except capture.QualificationError: pass
    else: raise AssertionError("invalid registration was accepted")
