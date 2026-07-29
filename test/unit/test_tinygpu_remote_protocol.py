import json
import inspect
import os
import pathlib
import socket
import struct
import subprocess
import threading
import types

import pytest

from tinygrad.runtime.support.system import APLRemotePCIDevice, RemoteCmd, RemotePCIDevice, TinyGPUWireError, \
  _tinygpu_validate_status, _tinygpu_validate_power_status
from tinygrad.runtime.ops_amd import PCIIface


ROOT = pathlib.Path(__file__).resolve().parents[2]
STATUS = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/keepalive-status-v1.json").read_text())["valid"]
POWER_STATUS = json.loads((ROOT / "extra/usbgpu/protocol/fixtures/power-residency-status-v4.json").read_text())["valid"]


def _device(sock):
  dev = object.__new__(APLRemotePCIDevice)
  dev.sock = sock
  return dev


def _serve(sock, responder):
  def run():
    try: responder(sock, sock.recv(33))
    finally: sock.close()
  thread = threading.Thread(target=run)
  thread.start()
  return thread


def _response(status=0, payload=b"", resp1=0): return struct.pack("<BQQ", status, len(payload), resp1) + payload


def test_handshake_and_status_use_frozen_extension_ids():
  client, server = socket.socketpair()
  dev = _device(client)
  seen = []
  handshake = {"schema": "tinygpu.handshake.v1", "protocol_major": 1, "protocol_minor": 0,
               "capabilities": 11, "server_build_id": "test-build-1"}
  def serve(sock, first):
    seen.append(struct.unpack("<BIIQQQ", first))
    payload = json.dumps(handshake, separators=(",", ":")).encode()
    sock.sendall(_response(payload=payload, resp1=1))
    second = sock.recv(33)
    seen.append(struct.unpack("<BIIQQQ", second))
    payload = json.dumps(STATUS, separators=(",", ":")).encode()
    sock.sendall(_response(payload=payload))
  thread = _serve(server, serve)
  dev._negotiate_tinygpu()
  assert dev.keepalive_status() == STATUS
  thread.join(timeout=2)
  assert seen == [(15, 0, 0, 1, 0, 0), (18, 0, 0, 0, 0, 0)]
  client.close()


def test_power_residency_status_uses_separate_command_and_schema():
  client, server = socket.socketpair(); dev = _device(client)
  dev.tinygpu_handshake, dev.tinygpu_capabilities = {"schema":"tinygpu.handshake.v1"}, 11
  seen=[]
  def serve(sock, request):
    seen.append(struct.unpack("<BIIQQQ", request))
    sock.sendall(_response(payload=json.dumps(POWER_STATUS, separators=(",", ":")).encode()))
  thread = _serve(server, serve)
  assert dev.power_residency_status() == POWER_STATUS
  assert seen == [(20, 0, 0, 0, 0, 0)]
  thread.join(timeout=2); client.close()


def test_exact_legacy_generic_error_maps_to_unsupported_protocol_only():
  client, server = socket.socketpair()
  dev = _device(client)
  thread = _serve(server, lambda sock, request: sock.sendall(_response(status=1)))
  with pytest.raises(TinyGPUWireError, match="unsupported_protocol") as exc:
    dev._negotiate_tinygpu()
  assert exc.value.kind == "unsupported_protocol"
  thread.join(timeout=2)


def test_partial_header_is_a_protocol_error_not_unsupported():
  client, server = socket.socketpair()
  dev = _device(client)
  thread = _serve(server, lambda sock, request: sock.sendall(b"\x01"))
  with pytest.raises(TinyGPUWireError) as exc:
    dev._negotiate_tinygpu()
  assert exc.value.kind == "partial_read"
  thread.join(timeout=2)
  client.close()


def test_clean_eof_maps_to_unsupported_but_oversize_does_not():
  client, server = socket.socketpair(); dev = _device(client)
  thread = _serve(server, lambda sock, request: None)
  with pytest.raises(TinyGPUWireError) as exc: dev._negotiate_tinygpu()
  assert exc.value.kind == "unsupported_protocol"; thread.join(timeout=2)
  client, server = socket.socketpair(); dev = _device(client)
  thread = _serve(server, lambda sock, request: sock.sendall(struct.pack("<BQQ", 0, 65537, 0)))
  with pytest.raises(TinyGPUWireError) as exc: dev._negotiate_tinygpu()
  assert exc.value.kind == "payload_too_large"; thread.join(timeout=2); client.close()


def test_status_validation_rejects_fixture_negative_cases():
  for patch in ({"attempts": 1, "successes": 0, "failures": 0}, {"last_identity_dword": "0X744C1002"}):
    value = STATUS | patch
    with pytest.raises(TinyGPUWireError):
      _tinygpu_validate_status(json.dumps(value).encode())


def test_typed_error_requires_matching_schema_and_code():
  client, server = socket.socketpair()
  dev = _device(client)
  bad = json.dumps({"schema": "tinygpu.error.v1", "code": "busy", "message": "no"}).encode()
  thread = _serve(server, lambda sock, request: sock.sendall(_response(status=2, payload=bad)))
  with pytest.raises(TinyGPUWireError) as exc:
    dev._tinygpu_rpc(RemoteCmd.HANDSHAKE, arg0=1)
  assert exc.value.kind == "malformed_payload"
  thread.join(timeout=2)
  client.close()


def test_typed_error_length_is_rejected_before_payload_read():
  client, server = socket.socketpair(); dev = _device(client)
  thread = _serve(server, lambda sock, request: sock.sendall(struct.pack("<BQQ", 7, 1025, 0)))
  with pytest.raises(TinyGPUWireError) as exc: dev._tinygpu_rpc(RemoteCmd.KEEPALIVE_STATUS)
  assert exc.value.kind == "payload_too_large"; thread.join(timeout=2); client.close()


def test_duplicate_status_json_and_bad_handshake_build_id_are_rejected():
  duplicate = b'{"schema":"tinygpu.keepalive.v1","schema":"tinygpu.keepalive.v1"}'
  with pytest.raises(TinyGPUWireError) as exc: _tinygpu_validate_status(duplicate)
  assert exc.value.kind == "malformed_payload"
  client, server = socket.socketpair(); dev = _device(client)
  bad = {"schema":"tinygpu.handshake.v1", "protocol_major":1, "protocol_minor":0, "capabilities":11, "server_build_id":"bad id"}
  thread = _serve(server, lambda sock, request: sock.sendall(_response(payload=json.dumps(bad).encode(), resp1=1)))
  with pytest.raises(TinyGPUWireError) as exc: dev._negotiate_tinygpu()
  assert exc.value.kind == "malformed_payload"; thread.join(timeout=2); client.close()


def test_handshake_requires_keepalive_lease_and_power_capabilities_locally():
  client, server = socket.socketpair(); dev = _device(client)
  weak = {"schema":"tinygpu.handshake.v1", "protocol_major":1, "protocol_minor":0, "capabilities":3, "server_build_id":"test"}
  thread = _serve(server, lambda sock, request: sock.sendall(_response(payload=json.dumps(weak).encode(), resp1=1)))
  with pytest.raises(TinyGPUWireError) as exc: dev._negotiate_tinygpu()
  assert exc.value.kind == "unsupported_capability"; thread.join(timeout=2); client.close()


def test_power_status_validation_preserves_unhealthy_diagnostic_state():
  for patch in ({"power_request_confirmed":False}, {"last_observed_power_flags":65536}, {"unexpected_downgrade_count":1},
                {"pci_command_confirmed":False}, {"pci_command_after":3}):
    value = POWER_STATUS | patch
    decoded = _tinygpu_validate_power_status(json.dumps(value).encode())
    assert decoded == value


def test_fd_response_requires_exactly_one_rights_ancillary_fd():
  client, server = socket.socketpair()
  def serve(sock, request): sock.sendall(_response())
  thread = _serve(server, serve)
  with pytest.raises(TinyGPUWireError) as exc: RemotePCIDevice._rpc(client, 0, RemoteCmd.MAP_SYSMEM_FD, has_fd=True)
  assert exc.value.kind == "malformed_payload"; thread.join(timeout=2); client.close()


def test_fd_rpc_accepts_typed_error_without_ancillary_fd():
  client, server = socket.socketpair()
  error = json.dumps({"schema":"tinygpu.error.v1", "code":"busy", "message":"lease busy"}, separators=(",", ":")).encode()
  thread = _serve(server, lambda sock, request: sock.sendall(_response(status=6, payload=error)))
  with pytest.raises(TinyGPUWireError) as exc:
    RemotePCIDevice._rpc(client, 0, RemoteCmd.MAP_SYSMEM_FD, has_fd=True)
  assert exc.value.kind == "busy"; thread.join(timeout=2); client.close()


def test_fd_is_closed_if_payload_read_fails(monkeypatch):
  client, server = socket.socketpair(); read_fd, write_fd = os.pipe(); closed=[]; real_close=os.close
  def serve(sock, request):
    sock.sendmsg([_response()], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", read_fd))])
  thread = _serve(server, serve)
  monkeypatch.setattr("tinygrad.runtime.support.system.os.close", lambda fd:(closed.append(fd), real_close(fd))[1])
  with pytest.raises(TinyGPUWireError) as exc:
    RemotePCIDevice._rpc(client, 0, RemoteCmd.MAP_SYSMEM_FD, has_fd=True, readout_size=1)
  assert exc.value.kind == "disconnect" and closed
  thread.join(timeout=2); client.close(); real_close(read_fd); real_close(write_fd)


def test_lease_release_is_sent_once_and_clears_local_lease():
  client, server = socket.socketpair(); dev = _device(client); dev.tinygpu_lease = 42
  seen=[]
  def serve(sock, request):
    seen.append(struct.unpack("<BIIQQQ", request)); sock.sendall(_response())
  thread = _serve(server, serve)
  dev._release_workload_lease(); dev._release_workload_lease()
  assert dev.tinygpu_lease is None and seen == [(17, 0, 0, 42, 0, 0)]
  thread.join(timeout=2); client.close()


def test_failed_lease_release_remains_retryable():
  client, server = socket.socketpair(); dev = _device(client); dev.tinygpu_lease = 42
  error = json.dumps({"schema":"tinygpu.error.v1", "code":"internal_error", "message":"try again"}, separators=(",", ":")).encode()
  thread = _serve(server, lambda sock, request: sock.sendall(_response(status=7, payload=error)))
  with pytest.raises(TinyGPUWireError): dev._release_workload_lease()
  assert dev.tinygpu_lease == 42
  thread.join(timeout=2); client.close()


def test_amd_finalization_closes_remote_transport_after_device_fini_even_on_failure():
  events = []
  iface = object.__new__(PCIIface)
  iface.pci_dev = types.SimpleNamespace(close=lambda: events.append("close"))
  def fini():
    events.append("fini")
    raise RuntimeError("fini failed")
  iface.dev_impl = types.SimpleNamespace(fini=fini)
  with pytest.raises(RuntimeError, match="fini failed"): iface.device_fini()
  assert events == ["fini", "close"]
  assert "atexit.register(self.close)" not in inspect.getsource(APLRemotePCIDevice.__init__)


def test_ensure_app_never_downloads_or_replaces(monkeypatch):
  monkeypatch.setattr("tinygrad.runtime.support.system.os.path.isfile", lambda _path: False)
  with pytest.raises(RuntimeError, match="explicit approval"):
    APLRemotePCIDevice.ensure_app()

  monkeypatch.setattr("tinygrad.runtime.support.system.os.path.isfile", lambda _path: True)
  monkeypatch.setattr("tinygrad.runtime.support.system.subprocess.run",
                      lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", "Identifier=org.tinygrad.arkey.tinygpu.installer\n"))
  APLRemotePCIDevice.ensure_app()
