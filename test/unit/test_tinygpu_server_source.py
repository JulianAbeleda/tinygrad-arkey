from pathlib import Path

ROOT = Path(__file__).parents[2]
SERVER = (ROOT / "extra/usbgpu/tbgpu/installer/Shared/server.c").read_text()
CLI = (ROOT / "extra/usbgpu/tbgpu/installer/Shared/TinyGPUCLIRunner.swift").read_text()

def test_wire_has_independent_fixed_width_codec_and_bounds():
  for token in ("uint8_t b[33]", "uint8_t wire[17]", "get_le32", "get_le64", "put_le64", "recvall", "sendall"):
    assert token in SERVER
  assert "__attribute__((packed))" not in SERVER

def test_handshake_precedes_driver_open_and_status_cli_is_direct():
  client = SERVER[SERVER.index("static void handle_client"):]
  handshake = client[client.index("if (req.cmd == CMD_HANDSHAKE)"):client.index("if(!session.handshaken)")]
  assert "open_tinygpu" not in handshake and "dext_rpc" not in handshake
  assert "CMD_KEEPALIVE_STATUS" in SERVER and "status_rpc(&session" in SERVER
  assert "int tinygpu_keepalive_status" in SERVER and "int tinygpu_keepalive_handshake" in SERVER
  assert "case \"keepalive\":" in CLI and "tinygpu_keepalive_status" in CLI and "tinygpu_keepalive_handshake" in CLI
  assert "run_server(args[2])" not in CLI[CLI.index('case "keepalive"'):CLI.index('case "status"')]

def test_leases_gate_hardware_and_cleanup_is_idempotent():
  assert "req.cmd <= CMD_RESIZE_BAR && req.cmd != CMD_RESET && !session.lease" in SERVER
  assert "dext_rpc(&session,7,&session.lease" in SERVER
  assert "cleanup(&session)" in SERVER

def test_typed_errors_and_fd_short_send_are_bounded():
  assert "tinygpu.error.v1" in SERVER and "n>1024" in SERVER
  assert "sendall(fd, wire + n" in SERVER
  assert "request_valid(&req)" in SERVER
  invalid = SERVER[SERVER.index("if(!request_valid(&req))"):SERVER.index("if (req.cmd == CMD_HANDSHAKE)")]
  assert "break;" in invalid and "continue;" not in invalid

def test_shared_memory_name_is_unlinked_immediately_after_mapping():
  mapping = SERVER[SERVER.index("static int map_sysmem_fd"):SERVER.index("static int validate_bar")]
  mmap_at = mapping.index("mmap(")
  assert mmap_at < mapping.index("shm_unlink(shm_name)", mmap_at) < mapping.index("IOConnectCallStructMethod")
