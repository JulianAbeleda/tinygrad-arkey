#!/usr/bin/env python3
"""Offline fault-to-allocation/dispatch attribution.

Takes (1) a dump written by tinygrad.device.alloc_trace_dump() (ALLOC_TRACE ring, see tinygrad/device.py)
and (2) a list of GPU fault addresses -- either bare addresses, or a raw `journalctl -k` / `dmesg` log this
script parses itself -- and for each fault address reports:

  - whether it falls inside a live allocation interval, and which
  - nearest lower/upper allocation VA boundaries and the distance to each
  - that allocation's lifetime (freed before the fault? still live? unknown?)
  - the dispatch and kernarg pointer argument most likely responsible (containing, or nearest-below/above)
  - PASID / VMID / client ID / read-write status, when the address came from a parsed kernel-log fault event

Context: ~140 fault addresses in 0x00007xxx_xxxxx000 have been observed in kernel logs on this box. That is
exactly the range KFDIface.alloc's anon_mmap(0, ...) places tinygrad's own GPU buffers in (tinygrad/runtime/
ops_amd.py:795-823) -- tinygrad never picks its own GPU VA, so every one of those addresses should be
attributable to *something* tinygrad allocated or dispatched, if ALLOC_TRACE was on for the run that faulted.

USAGE
-----
  # fault addresses from a file (one hex address per line, `0x...` optional), or given directly:
  python3 analyze_faults.py --dump alloc_trace.json --faults faults.txt
  python3 analyze_faults.py --dump alloc_trace.json 0x7c7a20b4a000 0x7c7a20b7a000

  # against a real kernel log for the PREVIOUS boot (this machine rebooted 2026-07-26 07:03, so `-b -1`
  # is that boot). `-o short-unix` is IMPORTANT: it is what lets this script align kernel-log wall-clock
  # timestamps against ALLOC_TRACE's alloc_ts_ns / free_ts_ns / submit_ts_ns (also wall-clock, epoch ns).
  # Without a machine-parseable timestamp, liveness is reported "at dump time" instead of "at fault time"
  # (see --dump-time-fallback note in the per-fault report).
  sudo journalctl -k -b -1 -o short-unix > faultlog.txt
  python3 analyze_faults.py --dump alloc_trace.json --kernel-log faultlog.txt

  # log only, no ALLOC_TRACE dump yet (e.g. to see what's in the log before a run) -- still parses and
  # reports PASID/VMID/client/RW/status per fault, just skips the allocation/dispatch correlation:
  python3 analyze_faults.py --kernel-log faultlog.txt

If both --kernel-log and --faults/positional addresses are given, the log is parsed for metadata and the
explicit address list is what gets analyzed (addresses not seen in the log get no PASID/VMID/etc context).
"""
from __future__ import annotations
import argparse, json, re, sys
from dataclasses import dataclass, field

# ============================================================================================
# GCVM_L2_PROTECTION_FAULT_STATUS bitfield layout, gfx11 (verified against
# /usr/src/amdgpu-6.16.13-2341068.24.04/amd/include/asic_reg/gc/gc_11_0_0_sh_mask.h on this box,
# and cross-checked against docs/fault-scope-for-review-20260726.md's manual decode of 0x008012B1:
# MORE_FAULTS=1, WALKER_ERROR=0, PERMISSION_FAULTS=0xb, MAPPING_ERROR=0, CID=9, RW=0, VMID=8 -- exact match).
# ============================================================================================
_STATUS_FIELDS = [  # (name, shift, mask)
  ("MORE_FAULTS", 0x0, 0x00000001), ("WALKER_ERROR", 0x1, 0x0000000E), ("PERMISSION_FAULTS", 0x4, 0x000000F0),
  ("MAPPING_ERROR", 0x8, 0x00000100), ("CID", 0x9, 0x0003FE00), ("RW", 0x12, 0x00040000), ("ATOMIC", 0x13, 0x00080000),
  ("VMID", 0x14, 0x00F00000), ("VF", 0x18, 0x01000000), ("VFID", 0x19, 0x1E000000), ("PRT", 0x1d, 0x20000000),
]

def decode_protection_fault_status(status:int) -> dict[str,int]:
  return {name: (status & mask) >> shift for name, shift, mask in _STATUS_FIELDS}

# ============================================================================================
# Kernel-log parsing. journalctl -k -b -1 / dmesg -T records the fault as several consecutive lines; group
# them by proximity into one FaultEvent per `[gfxhub] page fault` (or `[mmhub]`, seen on some ASICs) line.
# ============================================================================================

_RE_PAGE_FAULT = re.compile(r"\[(?P<hub>\w+hub)\]\s+page fault\s+\(src_id:(?P<src>\d+)\s+ring:(?P<ring>\d+)\s+"
                             r"vmid:(?P<vmid>\d+)\s+pasid:(?P<pasid>\d+)\)")
_RE_ADDR       = re.compile(r"in page starting at address\s+(?P<addr>0x[0-9a-fA-F]+)\s+from client\s+(?P<client>\d+)")
_RE_STATUS     = re.compile(r"GCVM_L2_PROTECTION_FAULT_STATUS:\s*(?P<status>0x[0-9a-fA-F]+)")
# name can itself contain parens (e.g. "SQC (inst)"), so anchor on the trailing "(0x..)" at end of line
# instead of stopping at the first "(" -- .+ is greedy, so it backtracks to the LAST "(0x...)" group.
_RE_UTCL2      = re.compile(r"Faulty UTCL2 client ID:\s*(?P<name>.+?)\s*\((?P<hex>0x[0-9a-fA-F]+)\)\s*$")
_RE_SHORT_UNIX = re.compile(r"^(?P<sec>\d+)\.(?P<subsec>\d+)\s")  # `journalctl -o short-unix` line prefix
_RE_ANY_HEX    = re.compile(r"0x[0-9a-fA-F]+")

@dataclass
class FaultEvent:
  addr: int
  line_no: int
  ts_ns: int|None = None     # wall-clock epoch ns, if the log line carried a parseable timestamp
  hub: str|None = None
  vmid: int|None = None
  pasid: int|None = None
  ih_client: int|None = None      # the IH ring "from client N" -- different numbering from UTCL2 CID below
  utcl2_client_name: str|None = None
  utcl2_client_id: int|None = None
  status_raw: int|None = None
  status_fields: dict[str,int] = field(default_factory=dict)

def _line_ts_ns(line:str) -> int|None:
  m = _RE_SHORT_UNIX.match(line)
  if m is None: return None
  # integer arithmetic, not float(...)*1e9 -- a float64 can't hold seconds-since-epoch (~1.8e9, needs 31
  # bits) AND nanosecond-scale fractional precision (another ~30 bits) at the same time without rounding.
  sec, subsec = m.group("sec"), m.group("subsec")
  return int(sec) * 1_000_000_000 + int(subsec.ljust(9, "0")[:9])

def parse_kernel_log(text:str) -> list[FaultEvent]:
  """Best-effort multi-line grouping: a `[*hub] page fault` line opens an event; addr/status/UTCL2 lines
  within the next few lines attach to it. Lines belonging to an unrelated fault population (BUG 1's
  0xFFFFFFBFE000 / 0x0 / 0x100000000 -- see docs/gpu-page-fault-population-analysis-20260725.md) are still
  parsed and returned; the caller decides what's in scope."""
  events: list[FaultEvent] = []
  cur: FaultEvent|None = None
  cur_open_at = -1
  ATTACH_WINDOW = 8  # lines within which a status/addr/UTCL2 line is assumed to belong to the open event
  for i, line in enumerate(text.splitlines()):
    if (m := _RE_PAGE_FAULT.search(line)) is not None:
      if cur is not None: events.append(cur)
      cur = FaultEvent(addr=0, line_no=i, ts_ns=_line_ts_ns(line), hub=m.group("hub"),
                        vmid=int(m.group("vmid")), pasid=int(m.group("pasid")))
      cur_open_at = i
      continue
    if cur is None or i - cur_open_at > ATTACH_WINDOW: continue
    if (m := _RE_ADDR.search(line)) is not None:
      cur.addr = int(m.group("addr"), 16); cur.ih_client = int(m.group("client"))
      if cur.ts_ns is None: cur.ts_ns = _line_ts_ns(line)
    elif (m := _RE_STATUS.search(line)) is not None:
      cur.status_raw = int(m.group("status"), 16)
      cur.status_fields = decode_protection_fault_status(cur.status_raw)
    elif (m := _RE_UTCL2.search(line)) is not None:
      cur.utcl2_client_name = m.group("name").strip()
      cur.utcl2_client_id = int(m.group("hex"), 16)
  if cur is not None and cur.addr != 0: events.append(cur)
  return [e for e in events if e.addr != 0]

def parse_addr_list(text:str) -> list[int]:
  """One address per line (bare hex, `0x...` optional); also tolerates comma/space-separated addresses
  and ignores blank lines / `#` comments."""
  out = []
  for line in text.splitlines():
    line = line.split("#", 1)[0].strip()
    if not line: continue
    for tok in re.split(r"[,\s]+", line):
      if not tok: continue
      try: out.append(int(tok, 16) if tok.lower().startswith("0x") else int(tok, 16))
      except ValueError: pass
  return out

# ============================================================================================
# ALLOC_TRACE dump model
# ============================================================================================

@dataclass
class AllocRec:
  alloc_id: int; device: str; va_start: int; va_end: int; req_size: int; mapped_size: int
  alloc_seq: int; free_seq: int|None; alloc_ts_ns: int; free_ts_ns: int|None
  def contains(self, addr:int) -> bool: return self.va_start <= addr < self.va_end
  def live_at(self, ts_ns:int|None) -> bool|None:
    """True/False if we can tell, None if unknown (no timestamp to compare against, or free is unresolved)."""
    if ts_ns is None: return None if self.free_seq is None else False  # can only assert "freed by dump time"
    if ts_ns < self.alloc_ts_ns: return False
    if self.free_ts_ns is not None and ts_ns >= self.free_ts_ns: return False
    return True

@dataclass
class DispatchRec:
  dispatch_id: int; device: str; kernel: str; global_size: list[int]; local_size: list[int]
  submit_seq: int; submit_ts_ns: int; signal_target: int; completed: bool|None
  args: list[dict]  # [{"va":..,"size":..}, ...]

def load_dump(path:str) -> tuple[list[AllocRec], list[DispatchRec], dict]:
  d = json.load(open(path))
  if d.get("format") != "tinygrad-alloc-trace-v1":
    print(f"warning: unrecognized dump format {d.get('format')!r}, proceeding anyway", file=sys.stderr)
  allocs = [AllocRec(a["alloc_id"], a["device"], a["va_start"], a["va_end"], a["req_size"], a["mapped_size"],
                      a["alloc_seq"], a.get("free_seq"), a["alloc_ts_ns"], a.get("free_ts_ns"))
            for a in d.get("allocs", [])]
  dispatches = [DispatchRec(x["dispatch_id"], x["device"], x["kernel"], x["global_size"], x["local_size"],
                             x["submit_seq"], x.get("submit_ts_ns", 0), x["signal_target"], x.get("completed"), x["args"])
                for x in d.get("dispatches", [])]
  return allocs, dispatches, d

# ============================================================================================
# Attribution
# ============================================================================================

def attribute(addr:int, ts_ns:int|None, allocs:list[AllocRec], dispatches:list[DispatchRec]) -> dict:
  containing = [a for a in allocs if a.contains(addr)]
  if ts_ns is not None:
    live_containing = [a for a in containing if a.live_at(ts_ns)]
  else:
    live_containing = None  # unknown without a timestamp

  # nearest lower / upper VA boundaries across ALL recorded allocations (any point in their lifetime)
  below = [a for a in allocs if a.va_end <= addr]
  above = [a for a in allocs if a.va_start > addr]
  nearest_below = max(below, key=lambda a: a.va_end) if below else None
  nearest_above = min(above, key=lambda a: a.va_start) if above else None

  # dispatch/pointer-arg attribution: containing args first (exact), else nearest arg extent by distance.
  # Prefer dispatches close in time to the fault when we have a timestamp; else most-recently-submitted.
  arg_hits, arg_near = [], []
  for d in dispatches:
    if ts_ns is not None and d.submit_ts_ns and d.submit_ts_ns > ts_ns: continue  # can't be responsible for a fault before it ran
    for i, a in enumerate(d.args):
      va, size = a["va"], a["size"]
      if va == 0 and size == 0: continue  # non-HCQ / unrecorded arg slot
      if va <= addr < va + size:
        arg_hits.append((d, i, 0))
      else:
        dist = (va - addr) if addr < va else (addr - (va + size))
        arg_near.append((d, i, dist))
  arg_hits.sort(key=lambda t: -(t[0].submit_ts_ns or t[0].submit_seq))
  arg_near.sort(key=lambda t: (t[2], -(t[0].submit_ts_ns or t[0].submit_seq)))

  return {"containing": containing, "live_containing": live_containing, "nearest_below": nearest_below,
          "nearest_above": nearest_above, "arg_hits": arg_hits[:3], "arg_near": arg_near[:3]}

def fmt_alloc(a:AllocRec|None) -> str:
  if a is None: return "none recorded"
  lifetime = "still live at dump" if a.free_seq is None else f"freed (free_seq={a.free_seq})"
  return (f"alloc_id={a.alloc_id} device={a.device} [0x{a.va_start:x}, 0x{a.va_end:x}) "
          f"req={a.req_size}B mapped={a.mapped_size}B alloc_seq={a.alloc_seq} {lifetime}")

def report_one(addr:int, ev:FaultEvent|None, allocs:list[AllocRec], dispatches:list[DispatchRec]) -> None:
  ts_ns = ev.ts_ns if ev is not None else None
  r = attribute(addr, ts_ns, allocs, dispatches)
  print(f"\n{'='*90}\nfault address 0x{addr:016x}")
  if ev is not None:
    parts = []
    if ev.hub: parts.append(f"hub={ev.hub}")
    if ev.vmid is not None: parts.append(f"vmid={ev.vmid}")
    if ev.pasid is not None: parts.append(f"pasid={ev.pasid}")
    if ev.ih_client is not None: parts.append(f"ih_client={ev.ih_client}")
    if ev.utcl2_client_name: parts.append(f"UTCL2_client={ev.utcl2_client_name}(0x{ev.utcl2_client_id:x})")
    if ev.status_raw is not None:
      f_ = ev.status_fields
      parts.append(f"status=0x{ev.status_raw:08x} [RW={f_['RW']} CID=0x{f_['CID']:x} PERMISSION_FAULTS=0x{f_['PERMISSION_FAULTS']:x} "
                    f"MAPPING_ERROR={f_['MAPPING_ERROR']} VMID={f_['VMID']} MORE_FAULTS={f_['MORE_FAULTS']}]")
    if ev.ts_ns is not None: parts.append(f"ts_ns={ev.ts_ns}")
    print("  kernel log: " + ", ".join(parts) if parts else "  kernel log: (address only, no fields parsed)")
  else:
    print("  kernel log: no matching parsed event (address given directly)")

  if not allocs and not dispatches:
    print("  no ALLOC_TRACE dump provided -- structural fields above are all we have for this address.")
    return

  print(f"  live interval: ", end="")
  if r["live_containing"] is not None:
    if r["live_containing"]: print(f"YES, inside {len(r['live_containing'])} live allocation(s) at fault time:")
    else: print("NO (not inside any allocation that was live at the fault's timestamp)")
    for a in r["live_containing"] or []: print(f"    {fmt_alloc(a)}")
  elif r["containing"]:
    print(f"UNKNOWN at fault time (no usable timestamp) -- but inside {len(r['containing'])} allocation(s) "
          f"at SOME point in the recorded run:")
    for a in r["containing"]: print(f"    {fmt_alloc(a)}  (dump-time snapshot: {'freed' if a.free_seq is not None else 'still live'})")
  else:
    print("NO allocation ever recorded contains this address.")

  b, a = r["nearest_below"], r["nearest_above"]
  print(f"  nearest lower mapping: {fmt_alloc(b)}" + (f"  (distance {addr - b.va_end} bytes below fault)" if b else ""))
  print(f"  nearest upper mapping: {fmt_alloc(a)}" + (f"  (distance {a.va_start - addr} bytes above fault)" if a else ""))

  if r["arg_hits"]:
    print("  dispatch/pointer arg CONTAINING the fault address (most recent first):")
    for d, i, _ in r["arg_hits"]:
      arg = d.args[i]
      print(f"    dispatch_id={d.dispatch_id} kernel={d.kernel!r} arg[{i}] va=0x{arg['va']:x} size={arg['size']}B "
            f"global={d.global_size} local={d.local_size} submit_seq={d.submit_seq}")
  elif r["arg_near"]:
    print("  no dispatch arg contains the address; nearest kernarg pointer extents (out-of-bounds candidates):")
    for d, i, dist in r["arg_near"]:
      arg = d.args[i]
      print(f"    dispatch_id={d.dispatch_id} kernel={d.kernel!r} arg[{i}] va=0x{arg['va']:x} size={arg['size']}B "
            f"distance={dist}B global={d.global_size} local={d.local_size} submit_seq={d.submit_seq}")
  else:
    print("  no dispatch in the dump has a recorded pointer arg near this address.")

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--dump", help="path to a JSON dump from tinygrad.device.alloc_trace_dump()")
  ap.add_argument("--faults", help="file with one fault address per line")
  ap.add_argument("--kernel-log", help="raw journalctl/dmesg text to parse for fault events (recommended: "
                                        "`sudo journalctl -k -b -1 -o short-unix > faultlog.txt`)")
  ap.add_argument("addrs", nargs="*", help="fault addresses given directly on the command line")
  args = ap.parse_args()

  allocs: list[AllocRec] = []
  dispatches: list[DispatchRec] = []
  if args.dump:
    allocs, dispatches, meta = load_dump(args.dump)
    print(f"loaded dump: {len(allocs)} allocation(s), {len(dispatches)} dispatch(es) "
          f"(recorded totals: {meta.get('total_allocs_recorded')}/{meta.get('total_dispatches_recorded')}, "
          f"ring capacity {meta.get('alloc_ring_capacity')}/{meta.get('dispatch_ring_capacity')})")

  events_by_addr: dict[int, list[FaultEvent]] = {}
  if args.kernel_log:
    text = open(args.kernel_log).read()
    events = parse_kernel_log(text)
    print(f"parsed {len(events)} fault event(s) from {args.kernel_log}")
    for e in events: events_by_addr.setdefault(e.addr, []).append(e)

  target_addrs: list[int] = []
  if args.faults: target_addrs += parse_addr_list(open(args.faults).read())
  target_addrs += [int(a, 16) if a.lower().startswith("0x") else int(a, 16) for a in args.addrs]
  if not target_addrs and events_by_addr:
    target_addrs = sorted(events_by_addr.keys())  # analyze everything the log gave us

  if not target_addrs:
    print("no fault addresses given (use --faults, positional args, or --kernel-log). Nothing to do.", file=sys.stderr)
    return 1

  print(f"\nanalyzing {len(target_addrs)} fault address(es)")
  for addr in target_addrs:
    evs = events_by_addr.get(addr)
    if evs:
      for ev in evs: report_one(addr, ev, allocs, dispatches)
    else:
      report_one(addr, None, allocs, dispatches)
  return 0

if __name__ == "__main__":
  sys.exit(main())
