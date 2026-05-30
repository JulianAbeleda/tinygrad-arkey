import os, socket, struct, threading, unittest

from tinygrad.runtime.support.system import RemoteCmd, RemotePCIDevice

REQ = '<BIIQQQ'

def remote_dev(sock:socket.socket, dev_id:int=3) -> RemotePCIDevice:
  dev = object.__new__(RemotePCIDevice)
  dev.sock, dev.dev_id = sock, dev_id
  return dev

def response(status:int=0, resp0:int=0, resp1:int=0, payload:bytes=b'') -> bytes:
  return struct.pack('<BQQ', status, resp0, resp1) + payload

class RPCServer(threading.Thread):
  def __init__(self, sock:socket.socket, responses:list[bytes]):
    super().__init__()
    self.sock, self.responses, self.requests = sock, responses, []

  def run(self):
    try:
      for resp in self.responses:
        hdr = self.sock.recv(struct.calcsize(REQ), socket.MSG_WAITALL)
        if len(hdr) < struct.calcsize(REQ): return
        cmd, dev_id, bar, arg0, arg1, arg2 = struct.unpack(REQ, hdr)
        payload = self.sock.recv(arg1, socket.MSG_WAITALL) if cmd in (RemoteCmd.MMIO_WRITE, RemoteCmd.SYSMEM_WRITE) and arg1 else b''
        self.requests.append((cmd, dev_id, bar, arg0, arg1, arg2, payload))
        if resp: self.sock.sendall(resp)
    finally:
      self.sock.close()

class TestRemotePCIRPC(unittest.TestCase):
  def setUp(self):
    RemotePCIDevice.reset_stats()

  def rpc_pair(self, *responses:bytes) -> tuple[socket.socket, RPCServer]:
    client, server_sock = socket.socketpair()
    server = RPCServer(server_sock, list(responses))
    server.start()
    self.addCleanup(server.join, 1)
    self.addCleanup(client.close)
    return client, server

  def test_rpc_success_records_stats(self):
    sock, server = self.rpc_pair(response(resp0=4, resp1=7, payload=b'abcd'))
    got = RemotePCIDevice._rpc(sock, 5, RemoteCmd.MMIO_READ, 0x10, 4, bar=2, readout_size=4)

    self.assertEqual(got, (4, 7, b'abcd', None))
    self.assertEqual(server.requests, [(RemoteCmd.MMIO_READ, 5, 2, 0x10, 4, 0, b'')])
    self.assertEqual(RemotePCIDevice.stats()["roundtrips"], 1)
    self.assertEqual(RemotePCIDevice.command_stats()["MMIO_READ"]["count"], 1)
    self.assertEqual(RemotePCIDevice.command_stats()["MMIO_READ"].get("failures", 0), 0)

  def test_rpc_error_records_failed_command_without_roundtrip(self):
    sock, _ = self.rpc_pair(response(status=1, resp0=len(b'bad remote'), payload=b'bad remote'))
    with self.assertRaisesRegex(RuntimeError, "RPC failed: bad remote"):
      RemotePCIDevice._rpc(sock, 0, RemoteCmd.CFG_READ)

    stats = RemotePCIDevice.command_stats()["CFG_READ"]
    self.assertEqual(RemotePCIDevice.stats()["roundtrips"], 0)
    self.assertEqual(stats["count"], 1)
    self.assertEqual(stats["failures"], 1)

  def test_closed_response_records_failed_command(self):
    sock, _ = self.rpc_pair(b'')
    with self.assertRaisesRegex(RuntimeError, "Connection closed"):
      RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)

    stats = RemotePCIDevice.command_stats()["PING"]
    self.assertEqual(RemotePCIDevice.stats()["roundtrips"], 0)
    self.assertEqual(stats["failures"], 1)

  def test_health_parses_healthy_and_dirty(self):
    sock, _ = self.rpc_pair(
      response(resp0=len(b'healthy'), resp1=0, payload=b'healthy'),
      response(resp0=len(b'dirty: wedged'), resp1=1, payload=b'dirty: wedged'),
    )
    dev = remote_dev(sock)

    self.assertEqual(dev.health(), (True, "healthy"))
    self.assertEqual(dev.health(), (False, "dirty: wedged"))

  def test_alloc_contiguous_sysmem_uses_mode_two(self):
    sock, server = self.rpc_pair(response(resp0=16, resp1=4, payload=struct.pack("<QQ", 0x1000, 0x2000)))
    dev = remote_dev(sock, dev_id=8)

    _, paddrs = dev.alloc_contiguous_sysmem(0x2000)

    self.assertEqual(paddrs, [0x1000, 0x2000])
    self.assertEqual(server.requests, [(RemoteCmd.MAP_SYSMEM, 8, 0, 0x2000, 2, 0, b'')])

  def test_alloc_sysmem_contiguous_uses_mode_one(self):
    sock, server = self.rpc_pair(response(resp0=16, resp1=4, payload=struct.pack("<QQ", 0x1000, 0x2000)))
    dev = remote_dev(sock, dev_id=8)

    _, paddrs = dev.alloc_sysmem(0x2000, contiguous=True)

    self.assertEqual(paddrs, [0x1000, 0x2000])
    self.assertEqual(server.requests, [(RemoteCmd.MAP_SYSMEM, 8, 0, 0x2000, 1, 0, b'')])

  def test_bulk_read_and_write_account_bytes(self):
    sock, server = self.rpc_pair(response(resp0=3, payload=b'xyz'), response())
    dev = remote_dev(sock, dev_id=9)

    self.assertEqual(dev._bulk_read(RemoteCmd.MMIO_READ, 5, 0x20, 3), b'xyz')
    dev._bulk_write(RemoteCmd.MMIO_WRITE, 6, 0x30, b'abcde')

    stats = RemotePCIDevice.stats()
    self.assertEqual(stats["recv_bytes"], 3)
    self.assertEqual(stats["sent_bytes"], 5)
    self.assertEqual(server.requests[0], (RemoteCmd.MMIO_READ, 9, 5, 0x20, 3, 0, b''))
    self.assertEqual(server.requests[1], (RemoteCmd.MMIO_WRITE, 9, 6, 0x30, 5, 0, b'abcde'))

  def test_rpc_timeout_is_restored_on_success_and_error(self):
    sock, _ = self.rpc_pair(response(), response(status=1, resp0=len(b'nope'), payload=b'nope'))
    sock.settimeout(2.5)

    old_timeout = os.environ.get("REMOTE_RPC_TIMEOUT")
    os.environ["REMOTE_RPC_TIMEOUT"] = "0.25"
    try:
      RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
      self.assertEqual(sock.gettimeout(), 2.5)
      with self.assertRaisesRegex(RuntimeError, "RPC failed: nope"):
        RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
      self.assertEqual(sock.gettimeout(), 2.5)
    finally:
      if old_timeout is None: os.environ.pop("REMOTE_RPC_TIMEOUT", None)
      else: os.environ["REMOTE_RPC_TIMEOUT"] = old_timeout

if __name__ == "__main__":
  unittest.main()
