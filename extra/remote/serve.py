#!/usr/bin/env python3
import collections, socket, struct, sys, time
from tinygrad.runtime.support.system import PCIDevice, RemoteCmd, System
from tinygrad.helpers import DEBUG, OSX

def resp(resp0=0, resp1=0, status=0): return struct.pack('<BQQ', status, resp0, resp1)
def resp_err(msg): return struct.pack('<BQQ', 1, len(err:=msg.encode()), 0) + err

discovered_devices: list[tuple[type[PCIDevice], str]] = []
opened_devices: dict[int, PCIDevice] = {}
opened_buses: dict[str, int] = {}
mapped_bars: dict[tuple[int, int], object] = {}
sysmem_allocs: list[tuple] = []
stats = collections.Counter()
last_error = ""
dirty_error = ""

def stat(cmd:RemoteCmd, delta:float): stats[f"{cmd.name}_count"] += 1; stats[f"{cmd.name}_ms"] += int(delta * 1000)
def log(msg:str, level:int=1):
  if DEBUG >= level: print(f"remote: {msg}", flush=True)
def mark_dirty(msg:str):
  global dirty_error, last_error
  dirty_error, last_error = msg, msg
  log(f"DIRTY {msg}")
def clear_dirty():
  global dirty_error
  if dirty_error: log("HEALTH restored by RESET")
  dirty_error = ""
def device_index(dev:tuple[type[PCIDevice], str]) -> int:
  for i, (_, pcibus) in enumerate(discovered_devices):
    if pcibus == dev[1]: return i
  discovered_devices.append(dev)
  return len(discovered_devices) - 1
def device_open_key(cl:type[PCIDevice], pcibus:str) -> str:
  # APLRemotePCIDevice routes all PCI ids through the same TinyGPU usb4 endpoint.
  return "aplremote:usb4" if cl.__name__ == "APLRemotePCIDevice" else pcibus

def handle(conn, cmd, dev_id, bar, arg0, arg1, arg2):
  if cmd == RemoteCmd.PING:
    return conn.sendall(resp())

  if cmd == RemoteCmd.HEALTH:
    state = f"dirty: {dirty_error}" if dirty_error else "healthy"
    data = state.encode()
    return conn.sendall(resp(len(data), int(bool(dirty_error))) + data)

  if cmd == RemoteCmd.PROBE:
    payload = conn.recv(arg1, socket.MSG_WAITALL) if arg1 > 0 else b""
    filter_devices: dict[int, list[int]] = {}
    for i in range(0, len(payload), 8):
      mask, dev = struct.unpack('<II', payload[i:i+8])
      filter_devices.setdefault(mask, []).append(dev)
    base_class = None if arg0 == 0 else int(arg0)
    devs = System.list_devices(arg2, tuple([(x, tuple(y)) for x,y in filter_devices.items()]), base_class)
    data = "\n".join(f"{p[1]}:{device_index(p)}" for p in devs).encode()
    log(f"PROBE vendor={arg2:#x} base_class={base_class} devices={len(devs)}", 2)
    return conn.sendall(resp(len(data), len(devs)) + data)

  if dirty_error and cmd != RemoteCmd.RESET:
    raise RuntimeError(f"bridge dirty: {dirty_error}")

  # lazy device open
  if dev_id not in opened_devices:
    if dev_id >= len(discovered_devices): raise RuntimeError(f"device {dev_id} not probed")
    cl, pcibus = discovered_devices[dev_id]
    open_key = device_open_key(cl, pcibus)
    if open_key in opened_buses:
      opened_devices[dev_id] = opened_devices[opened_buses[open_key]]
      log(f"OPEN dev={dev_id} pcibus={pcibus} reused={opened_buses[open_key]}")
    else:
      log(f"OPEN dev={dev_id} pcibus={pcibus}")
      opened_devices[dev_id] = cl("SV", pcibus)
      opened_buses[open_key] = dev_id
  pci_dev = opened_devices[dev_id]

  if cmd == RemoteCmd.MAP_BAR:
    if (dev_id, bar) not in mapped_bars:
      mapped_bars[(dev_id, bar)] = pci_dev.map_bar(bar)
      base, size = pci_dev.bar_info(bar)
      log(f"MAP_BAR dev={dev_id} bar={bar} base={base:#x} size={size:#x}")
    conn.sendall(resp(*pci_dev.bar_info(bar)))
  elif cmd == RemoteCmd.CFG_READ:
    conn.sendall(resp(pci_dev.read_config(arg0, arg1)))
  elif cmd == RemoteCmd.CFG_WRITE:
    pci_dev.write_config(arg0, arg2, arg1)
    conn.sendall(resp())
  elif cmd == RemoteCmd.RESIZE_BAR:
    pci_dev.resize_bar(bar)
    conn.sendall(resp())
  elif cmd == RemoteCmd.RESET:
    pci_dev.reset()
    clear_dirty()
    conn.sendall(resp())
  elif cmd == RemoteCmd.MMIO_READ:
    bar_view = mapped_bars[(dev_id, bar)]
    if arg0 % 4 == 0 and arg1 == 4: conn.sendmsg([resp(arg1), struct.pack('<I', bar_view.view(fmt='I')[arg0 // 4])])
    else: conn.sendmsg([resp(arg1), bar_view[arg0:arg0+arg1]])
  elif cmd == RemoteCmd.MMIO_WRITE:
    data = conn.recv(arg1, socket.MSG_WAITALL)
    bar_view = mapped_bars[(dev_id, bar)]
    if arg0 % 4 == 0 and arg1 == 4: bar_view.view(fmt='I')[arg0 // 4] = struct.unpack('<I', data)[0]
    else: bar_view[arg0:arg0+arg1] = data
    conn.sendall(resp())
  elif cmd == RemoteCmd.MAP_SYSMEM:
    st = time.perf_counter()
    memview, paddrs = pci_dev.alloc_sysmem(arg0, contiguous=bool(arg1))
    sysmem_allocs.append((memview, paddrs))
    paddrs_bytes = struct.pack(f'<{len(paddrs)}Q', *paddrs)
    log(f"MAP_SYSMEM dev={dev_id} size={arg0:#x} contiguous={bool(arg1)} paddrs={len(paddrs)} "
        f"handle={len(sysmem_allocs) - 1} ms={(time.perf_counter()-st)*1000:.2f}")
    conn.sendall(resp(len(paddrs_bytes), len(sysmem_allocs) - 1) + paddrs_bytes)
  elif cmd == RemoteCmd.SYSMEM_READ:
    if bar >= len(sysmem_allocs): raise RuntimeError(f"invalid sysmem handle {bar} (count={len(sysmem_allocs)})")
    conn.sendmsg([resp(arg1), sysmem_allocs[bar][0][arg0:arg0+arg1]])
  elif cmd == RemoteCmd.SYSMEM_WRITE:
    if bar >= len(sysmem_allocs): raise RuntimeError(f"invalid sysmem handle {bar} (count={len(sysmem_allocs)})")
    sysmem_allocs[bar][0][arg0:arg0+arg1] = conn.recv(arg1, socket.MSG_WAITALL)
    conn.sendall(resp())
  else: raise RuntimeError(f"unknown command {cmd}")

def serve(conn:socket.socket):
  global last_error
  REQ = '<BIIQQQ'
  while True:
    hdr = conn.recv(struct.calcsize(REQ), socket.MSG_WAITALL)
    if len(hdr) < struct.calcsize(REQ): raise ConnectionError("client disconnected")
    cmd, dev_id, bar, arg0, arg1, arg2 = struct.unpack(REQ, hdr)
    cmd_name = RemoteCmd(cmd).name if cmd in RemoteCmd._value2member_map_ else str(cmd)
    if DEBUG >= 4: print(f"cmd={cmd_name} dev={dev_id} bar={bar} arg0={arg0:#x} arg1={arg1:#x} arg2={arg2:#x}")
    st = time.perf_counter()
    try:
      handle(conn, RemoteCmd(cmd), dev_id, bar, arg0, arg1, arg2)
      stat(RemoteCmd(cmd), time.perf_counter() - st)
    except ConnectionError: raise
    except Exception as e:
      if cmd in RemoteCmd._value2member_map_ and RemoteCmd(cmd) not in {RemoteCmd.PING, RemoteCmd.HEALTH, RemoteCmd.PROBE}:
        mark_dirty(str(e))
      else:
        last_error = str(e)
      stats["errors"] += 1
      print(f"ERROR: {e}")
      conn.sendall(resp_err(str(e)))

if __name__ == "__main__":
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 6667
  server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  server.bind(("0.0.0.0", port))
  server.listen(1)
  s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  try: s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
  finally: s.close()
  print(f"listening on {ip}:{port}")
  while True:
    conn, addr = server.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for bt in [socket.SO_SNDBUF, socket.SO_RCVBUF]: conn.setsockopt(socket.SOL_SOCKET, bt, 64 << 20)
    try: serve(conn)
    except ConnectionError:
      if DEBUG >= 1: print(f"disconnected stats={dict(stats)} last_error={last_error}", flush=True)
      else: print("disconnected")
