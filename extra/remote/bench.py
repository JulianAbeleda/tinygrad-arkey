#!/usr/bin/env python3
import argparse, os, sys, time
from tinygrad.runtime.support.amd import AMD_RUNTIME_DEVICES
from tinygrad.runtime.support.system import RemoteCmd, RemotePCIDevice

LAT_N_RUNS = 500
THROUGHPUT_N_RUNS = 8
SIZES = [4, 1 << 10, 8 << 20]

def fmt_bytes(n:int) -> str:
  for suffix, div in [('G',1<<30),('M',1<<20),('K',1<<10)]:
    if n >= div: return f"{n/div:.4g}{suffix}"
  return f"{n}B"

def find_device(vendor:str):
  if vendor in ("amd", "any"):
    if (devs:=RemotePCIDevice.remote_list(0x1002, ((0xffff, AMD_RUNTIME_DEVICES),), 0)): return "AMD", devs[0]
  if vendor in ("nvidia", "any"):
    if (devs:=RemotePCIDevice.remote_list(0x10de, ((0, (0,)),), 0x03)): return "NVIDIA", devs[0]
  return None, None

def run_tensor_sanity(remote:str) -> str:
  os.environ["REMOTE"], os.environ["DEV"] = remote, "PCI+AMD"
  from tinygrad import Tensor
  got = (Tensor([1, 2, 3], device="AMD") + 1).numpy().tolist()
  return "ok" if got == [2, 3, 4] else f"bad_result={got}"

def print_stats(prefix:str):
  stats = RemotePCIDevice.stats()
  elapsed = max(float(stats["elapsed"]), 1e-9)
  sent_mb, recv_mb = int(stats["sent_bytes"]) / 1e6, int(stats["recv_bytes"]) / 1e6
  print(f"{prefix}: roundtrips={stats['roundtrips']} sent={sent_mb:.2f}MB ({sent_mb/elapsed:.2f}MB/s) "
        f"recv={recv_mb:.2f}MB ({recv_mb/elapsed:.2f}MB/s) elapsed={elapsed:.2f}s")

def print_command_stats():
  cmd_stats = RemotePCIDevice.command_stats()
  if not cmd_stats: return
  print(f"\n{'cmd':>14s}  {'count':>7s}  {'avg ms':>9s}  {'fail':>5s}  {'sent':>10s}  {'recv':>10s}")
  for name, st in sorted(cmd_stats.items()):
    count = int(st.get("count", 0))
    avg_ms = float(st.get("ms", 0.0)) / max(count, 1)
    print(f"{name:>14s}  {count:>7d}  {avg_ms:>9.3f}  {int(st.get('failures', 0)):>5d}  "
          f"{fmt_bytes(int(st.get('sent_bytes', 0))):>10s}  {fmt_bytes(int(st.get('recv_bytes', 0))):>10s}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Remote PCI bridge health and throughput benchmark")
  parser.add_argument("remote", nargs="?", default=os.environ.get("REMOTE", "127.0.0.1:6667"))
  parser.add_argument("--vendor", choices=("amd", "nvidia", "any"), default="amd")
  parser.add_argument("--skip-tensor", action="store_true", help="skip tinygrad AMD tensor sanity check")
  args = parser.parse_args()

  os.environ["REMOTE"] = args.remote
  os.environ.setdefault("REMOTE_RPC_TIMEOUT", os.environ.get("REMOTE_TIMEOUT", "3"))
  RemotePCIDevice.reset_stats()

  print(f"remote target: {args.remote}", flush=True)
  try:
    kind, selected = find_device(args.vendor)
  except Exception as e:
    print(f"health: dead (probe failed: {e})")
    sys.exit(2)

  if selected is None:
    print(f"health: dirty (no {args.vendor.upper()} GPU found on remote)")
    sys.exit(1)

  sock, name = selected
  pci = RemotePCIDevice("BN", name, sock=sock)
  print(f"device: {kind} {name}")

  health = "healthy"

  # ping (minimal server round-trip, no device I/O)
  sock = pci.sock
  try:
    for _ in range(10): RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
    st = time.perf_counter()
    for _ in range(LAT_N_RUNS): RemotePCIDevice._rpc(sock, 0, RemoteCmd.PING)
    ping_lat = (time.perf_counter() - st) / LAT_N_RUNS
    print(f"PING latency: {ping_lat*1e6:.1f} us ({1/ping_lat:,.0f} ops/sec)")

    ok, msg = pci.health()
    print(f"bridge health: {msg}")
    if not ok: health = "dirty"

    cfg_vendor = pci.read_config(0, 2)
    print(f"config vendor: {cfg_vendor:#06x}")

    bar0_base, bar0_size = pci.bar_info(0)
    print(f"BAR0: base={bar0_base:#x} size={bar0_size:#x}")

    st = time.perf_counter()
    sysmem, paddrs = pci.alloc_sysmem(max(SIZES))
    print(f"MAP_SYSMEM: size={fmt_bytes(max(SIZES))} paddrs={len(paddrs)} ms={(time.perf_counter()-st)*1000:.2f}")
  except Exception as e:
    print(f"health: dirty (runtime check failed: {e})")
    print_stats("remote stats")
    sys.exit(1)

  # throughput
  print(f"\n{'size':>10s}  {'write MB/s':>10s}  {'read MB/s':>10s}")
  for sz in SIZES:
    data = b'\x01' * sz

    try:
      for _ in range(5): sysmem[0:sz] = data
      st = time.perf_counter()
      for _ in range(THROUGHPUT_N_RUNS): sysmem[0:sz] = data
      pci.read_config(0, 4) # flush, since writes are posted
      w = (time.perf_counter() - st) / THROUGHPUT_N_RUNS

      for _ in range(5): sysmem[0:sz]
      st = time.perf_counter()
      for _ in range(THROUGHPUT_N_RUNS): sysmem[0:sz]
      r = (time.perf_counter() - st) / THROUGHPUT_N_RUNS
    except Exception as e:
      health = "dirty"
      print(f"{fmt_bytes(sz):>10s}  failed: {e}")
      continue

    print(f"{fmt_bytes(sz):>10s}  {sz/w/1e6:>10.1f}  {sz/r/1e6:>10.1f}")

  if not args.skip_tensor and kind == "AMD":
    try: print(f"\ntensor sanity: {run_tensor_sanity(args.remote)}")
    except Exception as e:
      health = "dirty"
      print(f"\ntensor sanity: failed ({e})")

  print_stats("\nremote stats")
  print_command_stats()
  print(f"health: {health}")
  sys.exit(0 if health == "healthy" else 1)
