#!/usr/bin/env python3
import argparse, ctypes, hashlib, pathlib, re, sys, tarfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from tinygrad.helpers import fetch_fw, mv_address, pad_bytes
from tinygrad.runtime.autogen.am import am, fw

HEX = r"0x[0-9a-fA-F]+"

LINUX_BL_RE = re.compile(
  rf"^(?P<t>\d+) bl_load enter .* cmd=(?P<cmd>{HEX}) fw_pri_mc=(?P<fw_pri_mc>{HEX}) c2p36=(?P<c2p36>{HEX}) size=(?P<size>{HEX})"
)
LINUX_WAIT_ENTER_RE = re.compile(r"^(?P<t>\d+) wait_bl enter")
LINUX_WAIT_RET_RE = re.compile(r"^(?P<t>\d+) wait_bl ret=(?P<ret>-?\d+)(?: duration_ns=(?P<duration_ns>\d+) reads=(?P<reads>\d+))?")
LINUX_WAIT_RREG_RE = re.compile(rf"^(?P<t>\d+) wait_bl_rreg dt_ns=(?P<dt_ns>\d+) read=(?P<read>\d+) reg=(?P<reg>{HEX}) val=(?P<val>{HEX})")
LINUX_GART_ENTER_RE = re.compile(
  rf"gart_map enter .* offset=(?P<offset>{HEX}) pages=(?P<pages>\d+) .* dma0=(?P<dma0>{HEX}) dma_last=(?P<dma_last>{HEX}) flags=(?P<flags>{HEX})"
)
LINUX_GART_RET_RE = re.compile(
  rf"gart_map ret .* first_idx=(?P<first_idx>{HEX}) last_idx=(?P<last_idx>{HEX}) pte0=(?P<pte0>{HEX}) pte_last=(?P<pte_last>{HEX})"
)
LINUX_REG_RE = re.compile(rf"^(?P<t>\d+) (?P<op>[rw]reg) .*reg=(?P<reg>{HEX})(?: val=(?P<val>{HEX}))?")

TG_RE = re.compile(r"PSP (?P<msg>.*)")
TG_GART_RE = re.compile(
  rf"gart pte table_paddr=(?P<table_paddr>{HEX}) pt_base=(?P<pt_base>{HEX}) msg1_off=(?P<msg1_off>{HEX}) "
  rf"gart_page=(?P<gart_page>{HEX}) paddr0=(?P<paddr0>{HEX}) paddr_last=(?P<paddr_last>{HEX}) "
  rf"pte0=(?P<pte0>{HEX}) pte_last=(?P<pte_last>{HEX})"
)
TG_PRE_BL_RE = re.compile(rf"pre-bl msg1 kind=(?P<kind>\S+) addr=(?P<addr>{HEX}) c2p36=(?P<c2p36>{HEX}) size=(?P<size>{HEX})")
TG_SKIP_RE = re.compile(rf"KDB skip prefix bytes=(?P<skip>{HEX}) old_size=(?P<old>{HEX}) new_size=(?P<new>{HEX})")
TG_LOAD_RE = re.compile(r"load component fw=(?P<fw>\S+) compid=(?P<compid>0x[0-9a-fA-F]+) bytes=(?P<bytes>\d+)")
TG_WRITE_MSG1_RE = re.compile(rf"write msg1 kind=(?P<kind>\S+) reg36={HEX} val=(?P<val>{HEX}) msg1_addr=(?P<addr>{HEX})")
TG_WRITE_COMPID_RE = re.compile(rf"write compid reg35={HEX} val=(?P<val>{HEX})")
TG_WAIT_RE = re.compile(rf"wait BL reg35={HEX} val=(?P<val>{HEX})")
TG_REG_RE = re.compile(rf"reg (?P<name>reg\S+?)(?:\[(?P<inst>\d+)\])?=(?P<val>{HEX})")
TG_MSG1_READBACK_RE = re.compile(r"msg1 readback ok bytes=(?P<bytes>\d+) first=(?P<first>[0-9a-fA-F]+) last=(?P<last>[0-9a-fA-F]+)")
TG_SNAPSHOT_BEGIN_RE = re.compile(r"parity snapshot (?P<label>\S+) begin")
TG_SNAPSHOT_END_RE = re.compile(r"parity snapshot (?P<label>\S+) end")
C2PMSG_BASE = 0x16040
C2PMSG_COUNT = 128
FOCUS_C2PMSG = (35, 36, 64, 67, 69, 70, 71, 81)

REG_NAMES = [
  "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR", "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR",
  "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_LSB", "regMMMC_VM_SYSTEM_APERTURE_DEFAULT_ADDR_MSB",
  "regMMVM_L2_PROTECTION_FAULT_STATUS", "regMMVM_CONTEXT0_CNTL",
  "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_HI32",
  "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_HI32",
  "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32", "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_HI32",
  "regMP0_SMN_C2PMSG_33", "regMP0_SMN_C2PMSG_35", "regMP0_SMN_C2PMSG_36",
  "regMP0_SMN_C2PMSG_64", "regMP0_SMN_C2PMSG_67", "regMP0_SMN_C2PMSG_81",
  "regMP0_SMN_C2PMSG_90", "regMP0_SMN_C2PMSG_92",
]

def as_int(s:str|None) -> int|None:
  return None if s is None else int(s, 0)

def hexv(v:int|None) -> str:
  return "missing" if v is None else f"{v:#x}"

def pte_paddr(pte:int|None) -> int|None:
  return None if pte is None else pte & 0x0000FFFFFFFFF000

def pte_flags(pte:int|None) -> int|None:
  return None if pte is None else pte & ~0x0000FFFFFFFFF000

def linux_c2pmsg_idx(reg:int|None) -> int|None:
  if reg is None or reg < C2PMSG_BASE or reg >= C2PMSG_BASE + C2PMSG_COUNT: return None
  return reg - C2PMSG_BASE

def c2pmsg_name(idx:int, pref:str="regMP0_SMN_C2PMSG") -> str:
  return f"{pref}_{idx}"

def linux_reg_name(reg:int|None) -> str:
  if (idx := linux_c2pmsg_idx(reg)) is not None: return f"C2PMSG{idx}"
  return hexv(reg)

def read_texts(path:pathlib.Path) -> dict[str, str]:
  if path.is_dir():
    return {str(p.relative_to(path)): p.read_text(errors="replace") for p in path.rglob("*") if p.is_file()}
  if path.suffixes[-2:] == [".tar", ".gz"] or path.name.endswith((".tar.gz", ".tgz")):
    texts = {}
    with tarfile.open(path, "r:gz") as tf:
      for member in tf.getmembers():
        if member.isfile():
          f = tf.extractfile(member)
          if f is not None: texts[member.name] = f.read().decode("utf-8", errors="replace")
    return texts
  return {path.name: path.read_text(errors="replace")}

def lines_for(path:pathlib.Path, prefer:list[str]) -> list[str]:
  texts = read_texts(path)
  selected = []
  for name, text in texts.items():
    if any(name.endswith(suffix) for suffix in prefer): selected.append(text)
  if not selected: selected = list(texts.values())
  return "\n".join(selected).splitlines()

def tinygrad_lines(path:pathlib.Path) -> list[str]:
  texts = read_texts(path)
  selected = []
  for primary in ("real.log", "audit.log", "key-grep.txt"):
    matches = [text for name, text in texts.items() if name.endswith(primary)]
    if matches:
      selected.extend(matches)
      break
  selected.extend(text for name, text in texts.items() if name.endswith("post-psp-status.log"))
  if not selected: selected = list(texts.values())
  return "\n".join(selected).splitlines()

def psp_fw_components(fname:str) -> dict[int, dict]:
  blob = memoryview(bytearray(fetch_fw("amdgpu", fname, fw.hashes[fname])))
  chdr = am.struct_common_firmware_header.from_address(mv_address(blob))
  hdr_t = getattr(am, f"struct_psp_firmware_header_v{chdr.header_version_major}_{chdr.header_version_minor}")
  hdr = hdr_t.from_address(mv_address(blob))
  comps = {}
  for fw_i in range(hdr.psp_fw_bin_count):
    desc = am.struct_psp_fw_bin_desc.from_address(ctypes.addressof(hdr.psp_fw_bin) + fw_i * ctypes.sizeof(am.struct_psp_fw_bin_desc))
    start = hdr.header.ucode_array_offset_bytes + desc.offset_bytes
    data = bytes(blob[start:start + desc.size_bytes])
    comps[desc.fw_type] = {
      "index": fw_i, "name": am.enum_psp_fw_type.get(desc.fw_type, f"UNKNOWN_{desc.fw_type}"),
      "offset": start, "desc_offset": desc.offset_bytes, "size": desc.size_bytes,
      "sha256": hashlib.sha256(data).hexdigest(), "data": data,
    }
  return comps

def firmware_report(fname:str, skip:int|None, readback:dict) -> list[str]:
  comps = psp_fw_components(fname)
  kdb = comps.get(am.PSP_FW_TYPE_PSP_KDB)
  if kdb is None: return [f"firmware {fname}: PSP_KDB missing"]
  skip = skip or 0
  skipped = kdb["data"][skip:]
  padded = pad_bytes(skipped + b"\x00" * 4, 16)
  lines = [
    f"firmware file: {fname}",
    f"kdb full: index={kdb['index']} offset={kdb['offset']:#x} size={kdb['size']:#x} sha256={kdb['sha256']}",
    f"kdb full bytes: first32={kdb['data'][:32].hex()} last32={kdb['data'][-32:].hex()}",
    f"kdb skipped: skip={skip:#x} size={len(skipped):#x} sha256={hashlib.sha256(skipped).hexdigest()}",
    f"kdb skipped bytes: first32={skipped[:32].hex()} last32={skipped[-32:].hex()}",
    f"kdb padded msg1: size={len(padded):#x} first16={padded[:16].hex()} last16={padded[-16:].hex()}",
  ]
  if readback:
    lines.append(row("readback byte count", str(len(padded)), str(readback.get("bytes"))))
    lines.append(row("readback first16", padded[:16].hex(), readback.get("first")))
    lines.append(row("readback last16", padded[-16:].hex(), readback.get("last")))
  return lines

def parse_linux(path:pathlib.Path) -> dict:
  out = {"bl": [], "waits": [], "wait_rregs": [], "gart": {}, "reg_events": [], "c2pmsg_events": [], "source": str(path)}
  wait_start = None
  seen_bl = set()
  seen_reg_events = set()
  seen_events = set()
  for line in lines_for(path, ["psp-linux-good.trace", "psp-linux-good-deep.trace", "linux-pre-kdb-key-events.txt", "linux-c2pmsg-events.txt"]):
    if m := LINUX_BL_RE.search(line):
      item = tuple((k, as_int(v)) for k, v in m.groupdict().items())
      if item not in seen_bl:
        seen_bl.add(item)
        out["bl"].append({k: v for k, v in item})
    elif m := LINUX_WAIT_ENTER_RE.search(line):
      wait_start = as_int(m.group("t"))
    elif m := LINUX_WAIT_RET_RE.search(line):
      end = as_int(m.group("t"))
      out["waits"].append({"start": wait_start, "end": end, "ret": int(m.group("ret")),
                           "duration_ns": as_int(m.group("duration_ns")) if m.group("duration_ns") else (None if wait_start is None else end - wait_start),
                           "reads": as_int(m.group("reads"))})
      wait_start = None
    elif m := LINUX_WAIT_RREG_RE.search(line):
      out["wait_rregs"].append({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := LINUX_GART_ENTER_RE.search(line):
      out["gart"].update({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := LINUX_GART_RET_RE.search(line):
      out["gart"].update({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := LINUX_REG_RE.search(line):
      reg, val = as_int(m.group("reg")), as_int(m.group("val"))
      reg_item = (as_int(m.group("t")), m.group("op"), reg, val)
      if reg_item not in seen_reg_events:
        seen_reg_events.add(reg_item)
        out["reg_events"].append({"t": reg_item[0], "op": reg_item[1], "reg": reg_item[2], "val": reg_item[3]})
      if (idx := linux_c2pmsg_idx(reg)) is not None:
        item = (as_int(m.group("t")), m.group("op"), idx, reg, val)
        if item not in seen_events:
          seen_events.add(item)
          out["c2pmsg_events"].append({"t": item[0], "op": item[1], "idx": item[2], "reg": item[3], "val": item[4]})
  return out

def parse_tinygrad(path:pathlib.Path) -> dict:
  out = {"source": str(path), "gart": {}, "pre_bl": {}, "skip": {}, "load": [], "regs": {}, "wait_vals": [],
         "write_msg1": {}, "write_compid": None, "readback": {}, "timeout": False, "snapshots": [], "post_status": {}}
  snapshot = None
  for line in tinygrad_lines(path):
    if "BL not ready" in line: out["timeout"] = True
    if not (m := TG_RE.search(line)): continue
    msg = m.group("msg")
    if m := TG_SNAPSHOT_BEGIN_RE.search(msg):
      snapshot = {"label": m.group("label"), "regs": {}}
      out["snapshots"].append(snapshot)
      continue
    if m := TG_SNAPSHOT_END_RE.search(msg):
      snapshot = None
      continue
    if m := TG_GART_RE.search(msg): out["gart"].update({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := TG_PRE_BL_RE.search(msg): out["pre_bl"].update({k: (as_int(v) if k != "kind" else v) for k, v in m.groupdict().items()})
    elif m := TG_SKIP_RE.search(msg): out["skip"].update({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := TG_LOAD_RE.search(msg):
      item = m.groupdict()
      item["compid"] = as_int(item["compid"])
      item["bytes"] = int(item["bytes"])
      out["load"].append(item)
    elif m := TG_WRITE_MSG1_RE.search(msg): out["write_msg1"].update({k: (as_int(v) if k != "kind" else v) for k, v in m.groupdict().items()})
    elif m := TG_WRITE_COMPID_RE.search(msg): out["write_compid"] = as_int(m.group("val"))
    elif m := TG_WAIT_RE.search(msg): out["wait_vals"].append(as_int(m.group("val")))
    elif m := TG_MSG1_READBACK_RE.search(msg):
      out["readback"] = {"bytes": int(m.group("bytes")), "first": m.group("first"), "last": m.group("last")}
    elif m := TG_REG_RE.search(msg):
      name = m.group("name")
      if m.group("inst") is not None: name = f"{name}[{m.group('inst')}]"
      out["regs"][name] = as_int(m.group("val"))
      if snapshot is not None: snapshot["regs"][name] = as_int(m.group("val"))
  return out

def row(label:str, linux, tinygrad, note:str="") -> str:
  verdict = "same" if linux == tinygrad and linux is not None else ("missing" if linux is None or tinygrad is None else "diff")
  return f"{label:48} linux={linux!s:18} tinygrad={tinygrad!s:18} {verdict:7} {note}".rstrip()

def reg_lookup(regs:dict, name:str) -> int|None:
  return next((regs[c] for c in (name, f"{name}[0]") if c in regs), None)

def c2pmsg_regs(regs:dict) -> dict[int, int]:
  out = {}
  for idx in range(C2PMSG_COUNT):
    val = next((regs[name] for name in (c2pmsg_name(idx), c2pmsg_name(idx, "regMPASP_SMN_C2PMSG")) if name in regs), None)
    if val is not None: out[idx] = val
  return out

def linux_c2pmsg_reads(linux:dict, limit:int=48) -> list[str]:
  events = [e for e in linux["c2pmsg_events"] if e["val"] is not None and (e["op"] == "wreg" or e["val"] != 0)]
  lines = []
  for e in events[:limit]:
    lines.append(f"{e['t']} {e['op']} C2PMSG{e['idx']}={hexv(e['val'])}")
  if len(events) > limit: lines.append(f"... {len(events) - limit} more nonzero Linux C2PMSG events")
  return lines

def c2pmsg_delta_report(linux:dict, tiny:dict) -> list[str]:
  lines = ["C2PMSG Delta"]
  linux_nonzero = {}
  for e in linux["c2pmsg_events"]:
    if e["val"] not in (None, 0): linux_nonzero[e["idx"]] = e["val"]
  if linux_nonzero:
    lines.append("Linux nonzero C2PMSG values seen in rreg/wreg trace:")
    lines.extend(f"  C2PMSG{idx}={hexv(linux_nonzero[idx])}" for idx in sorted(linux_nonzero))
  else:
    lines.append("Linux nonzero C2PMSG values seen in rreg/wreg trace: missing")

  if tiny["snapshots"]:
    lines.append("Tinygrad snapshot nonzero/different C2PMSG values:")
    prev = {}
    for snap in tiny["snapshots"]:
      cur = c2pmsg_regs(snap["regs"])
      interesting = sorted(idx for idx, val in cur.items() if val != 0 or prev.get(idx) not in (None, val))
      shown = ", ".join(f"{idx}={hexv(cur[idx])}" for idx in interesting[:64])
      if len(interesting) > 64: shown += f", ... {len(interesting) - 64} more"
      lines.append(f"  {snap['label']}: {shown or 'all zero/missing'}")
      prev = cur
  else:
    lines.append("Tinygrad dense snapshots: missing")

  linux_only = sorted(idx for idx, val in linux_nonzero.items() if all(c2pmsg_regs(s["regs"]).get(idx) != val for s in tiny["snapshots"]))
  if linux_only:
    lines.append("Linux nonzero values not matched in any tinygrad snapshot:")
    lines.append("  " + ", ".join(f"C2PMSG{idx}={hexv(linux_nonzero[idx])}" for idx in linux_only[:64]))
  lines.append("")
  return lines

def linux_kdb_register_window_report(linux:dict, before:int=48, after:int=96, after_ms:float=2.0) -> list[str]:
  lines = ["Linux register window around KDB"]
  kdb = next((x for x in linux["bl"] if x.get("cmd") == 0x80000), None)
  kdb_t = kdb.get("t") if kdb else None
  if kdb_t is None:
    lines.append("  missing KDB bl_load timestamp")
    lines.append("")
    return lines

  before_events = [e for e in linux["reg_events"] if e["t"] is not None and e["t"] < kdb_t]
  after_events = [e for e in linux["reg_events"] if e["t"] is not None and e["t"] >= kdb_t and (e["t"] - kdb_t) <= after_ms * 1_000_000]

  def changed_events(events:list[dict]) -> list[dict]:
    out, last = [], {}
    for e in events:
      key = (e["op"], e["reg"])
      if e["op"] == "wreg" or last.get(key) != e["val"]:
        out.append(e)
      last[key] = e["val"]
    return out

  before_changed, after_changed = changed_events(before_events), changed_events(after_events)

  lines.append(f"Last {min(before, len(before_changed))} changed/written register events before KDB bl_load:")
  for e in before_changed[-before:]:
    dt_ms = (e["t"] - kdb_t) / 1_000_000
    lines.append(f"  {dt_ms:9.3f} ms {e['op']:4} {linux_reg_name(e['reg']):>12}={hexv(e['val'])}")
  if not before_changed: lines.append("  none")

  lines.append(f"First changed/written register events from KDB through +{after_ms:g} ms:")
  for e in after_changed[:after]:
    dt_ms = (e["t"] - kdb_t) / 1_000_000
    lines.append(f"  +{dt_ms:8.3f} ms {e['op']:4} {linux_reg_name(e['reg']):>12}={hexv(e['val'])}")
  if len(after_changed) > after: lines.append(f"  ... {len(after_changed) - after} more changed/written events through +{after_ms:g} ms")
  if not after_changed: lines.append("  none")
  lines.append("")
  return lines

def c2pmsg_focus_timeline_report(linux:dict, tiny:dict, limit:int=80) -> list[str]:
  lines = ["Focused C2PMSG timeline"]
  kdb = next((x for x in linux["bl"] if x.get("cmd") == 0x80000), None)
  kdb_t = kdb.get("t") if kdb else None

  lines.append("Linux bootloader load order after KDB:")
  if kdb_t is None:
    lines.append("  missing KDB bl_load timestamp")
  else:
    for item in [x for x in linux["bl"] if x.get("t") is not None and x["t"] >= kdb_t][:8]:
      dt_ms = (item["t"] - kdb_t) / 1_000_000
      lines.append(f"  +{dt_ms:9.3f} ms cmd={hexv(item.get('cmd'))} c2p36={hexv(item.get('c2p36'))} size={hexv(item.get('size'))}")

  lines.append("Linux focused C2PMSG changes after KDB bl_load enter:")
  if kdb_t is None:
    lines.append("  missing KDB bl_load timestamp")
  else:
    last:dict[int, int|None] = {}
    shown = 0
    for e in linux["c2pmsg_events"]:
      if e["t"] is None or e["t"] < kdb_t or e["idx"] not in FOCUS_C2PMSG: continue
      val = e["val"]
      changed = last.get(e["idx"]) != val
      if e["op"] != "wreg" and not changed: continue
      last[e["idx"]] = val
      dt_ms = (e["t"] - kdb_t) / 1_000_000
      lines.append(f"  +{dt_ms:9.3f} ms {e['op']:4} C2PMSG{e['idx']:<2}={hexv(val)}")
      shown += 1
      if shown >= limit:
        lines.append(f"  ... truncated at {limit} focused events")
        break
    if shown == 0: lines.append("  no focused events found")

  lines.append("Tinygrad focused C2PMSG snapshots:")
  if tiny["snapshots"]:
    for snap in tiny["snapshots"]:
      regs = snap["regs"]
      vals = []
      for idx in FOCUS_C2PMSG:
        vals.append(f"{idx}={hexv(reg_lookup(regs, c2pmsg_name(idx)))}")
      lines.append(f"  {snap['label']}: " + ", ".join(vals))
  else:
    lines.append("  missing")
  lines.append("")
  return lines

def report(linux:dict, tiny:dict, firmware:str|None) -> str:
  lines = ["PSP trace comparison", f"linux source: {linux['source']}", f"tinygrad source: {tiny['source']}", ""]
  kdb = next((x for x in linux["bl"] if x.get("cmd") == 0x80000), None)
  tg_kdb = next((x for x in tiny["load"] if x.get("compid") == 0x80000), None)
  lines.append("KDB mailbox")
  lines.append(row("cmd/compid", hexv(kdb.get("cmd") if kdb else None), hexv(tg_kdb.get("compid") if tg_kdb else None)))
  lines.append(row("c2p36", hexv(kdb.get("c2p36") if kdb else None), hexv(tiny["write_msg1"].get("val"))))
  lines.append(row("payload bytes", str(kdb.get("size") if kdb else None), str(tg_kdb.get("bytes") if tg_kdb else None)))
  lines.append(row("msg1 addr", hexv(kdb.get("fw_pri_mc") if kdb else None), hexv(tiny["write_msg1"].get("addr"))))
  lines.append(row("post-write reg35 first", "not captured", hexv(tiny["wait_vals"][1] if len(tiny["wait_vals"]) > 1 else None),
                   "linux deep trace shows transient 0 before ready; exact write not captured"))
  ready_waits = [w for w in linux["waits"] if w.get("duration_ns") is not None]
  if ready_waits:
    lines.append(f"linux wait durations ns: {', '.join(str(w['duration_ns']) for w in ready_waits[:8])}")
    if any(w.get("reads") is not None for w in ready_waits):
      lines.append(f"linux wait read counts: {', '.join(str(w.get('reads')) for w in ready_waits[:8])}")
  if linux["wait_rregs"]:
    lines.append("linux wait C2PMSG35 reads:")
    # Keep the report readable when deep tracing captures a long poll loop.
    for e in linux["wait_rregs"][:64]:
      lines.append(f"  dt_ns={e['dt_ns']} read={e['read']} val={hexv(e['val'])}")
    if len(linux["wait_rregs"]) > 64: lines.append(f"  ... {len(linux['wait_rregs']) - 64} more wait reads")
  if tiny["wait_vals"]:
    lines.append(f"tinygrad observed wait BL values: {', '.join(hexv(v) for v in tiny['wait_vals'])}")
  lines.append(f"tinygrad timed out waiting BL: {tiny['timeout']}")
  lines.append("")

  if firmware:
    lines.append("KDB payload bytes")
    lines.extend(firmware_report(firmware, tiny["skip"].get("skip"), tiny["readback"]))
    lines.append("")

  lines.append("GART")
  lines.append(row("offset/msg1_off", hexv(linux["gart"].get("offset")), hexv(tiny["gart"].get("msg1_off"))))
  tiny_pages = None
  if tiny["pre_bl"].get("size") is not None: tiny_pages = tiny["pre_bl"]["size"] // 0x1000
  elif tiny["gart"].get("paddr_last") is not None and tiny["gart"].get("paddr0") is not None: tiny_pages = (tiny["gart"]["paddr_last"] - tiny["gart"]["paddr0"]) // 0x1000 + 1
  lines.append(row("pages", str(linux["gart"].get("pages")), str(tiny_pages)))
  lines.append(row("first_idx/gart_page", hexv(linux["gart"].get("first_idx")), hexv(tiny["gart"].get("gart_page"))))
  linux_last_idx = linux["gart"].get("last_idx")
  tiny_last_idx = tiny["gart"].get("gart_page") + tiny_pages - 1 if tiny["gart"].get("gart_page") is not None and tiny_pages is not None else None
  lines.append(row("last_idx", hexv(linux_last_idx), hexv(tiny_last_idx)))
  lines.append(row("pte flags", hexv(pte_flags(linux["gart"].get("pte0"))), hexv(pte_flags(tiny["gart"].get("pte0")))))
  linux_pte_delta = None if linux["gart"].get("pte_last") is None or linux["gart"].get("pte0") is None else pte_paddr(linux["gart"]["pte_last"]) - pte_paddr(linux["gart"]["pte0"])
  tiny_pte_delta = None if tiny["gart"].get("pte_last") is None or tiny["gart"].get("pte0") is None else pte_paddr(tiny["gart"]["pte_last"]) - pte_paddr(tiny["gart"]["pte0"])
  lines.append(row("pte paddr delta", hexv(linux_pte_delta), hexv(tiny_pte_delta)))
  lines.append(row("backing paddr first", hexv(linux["gart"].get("dma0")), hexv(tiny["gart"].get("paddr0")),
                   "physical address need not match across boots"))
  lines.append(row("backing paddr last", hexv(linux["gart"].get("dma_last")), hexv(tiny["gart"].get("paddr_last")),
                   "physical address need not match across boots"))
  lines.append(row("tiny table_paddr", "not captured", hexv(tiny["gart"].get("table_paddr"))))
  lines.append(row("tiny pt_base", "not captured", hexv(tiny["gart"].get("pt_base"))))
  lines.append("")

  lines.append("Tinygrad register snapshot")
  for name in REG_NAMES:
    val = reg_lookup(tiny["regs"], name)
    lines.append(f"{name:48} {hexv(val)}")
  lines.append("")

  if tiny["snapshots"]:
    lines.append("Tinygrad PSP register timeline")
    for snap in tiny["snapshots"]:
      regs = snap["regs"]
      lines.append(f"{snap['label']}: C2PMSG35={hexv(reg_lookup(regs, 'regMP0_SMN_C2PMSG_35'))} "
                   f"C2PMSG36={hexv(reg_lookup(regs, 'regMP0_SMN_C2PMSG_36'))} "
                   f"C2PMSG81={hexv(reg_lookup(regs, 'regMP0_SMN_C2PMSG_81'))} "
                   f"FAULT={hexv(reg_lookup(regs, 'regMMVM_L2_PROTECTION_FAULT_STATUS'))}")
    lines.append("")

  lines.extend(c2pmsg_delta_report(linux, tiny))
  lines.extend(linux_kdb_register_window_report(linux))
  lines.extend(c2pmsg_focus_timeline_report(linux, tiny))

  lines.append("Observations")
  if kdb and tg_kdb and kdb.get("size") == tg_kdb.get("bytes"):
    lines.append("- KDB payload size matches Linux-good: 0x1700. The 0x640 skip is not the leading suspect.")
  if firmware and tiny["readback"]:
    lines.append("- Tinygrad msg1 readback can now be checked against the local skipped firmware payload above.")
  if tiny_pte_delta is not None:
    lines.append(f"- Tinygrad PTE physical delta: {tiny_pte_delta:#x}.")
  if tiny["regs"].get("regMMVM_L2_PROTECTION_FAULT_STATUS[0]", tiny["regs"].get("regMMVM_L2_PROTECTION_FAULT_STATUS")) == 0:
    lines.append("- MMHUB fault status stayed zero during the failing attempt.")
  if tiny["timeout"]:
    lines.append("- The remaining visible difference is behavioral: Linux returns ready after transient 0, tinygrad does not.")
  else:
    lines.append("- Tinygrad did not time out in this capture.")
  return "\n".join(lines) + "\n"

def main():
  parser = argparse.ArgumentParser(description="Compare Linux-good AMD PSP trace data with a tinygrad PSP boot attempt")
  parser.add_argument("--linux", required=True, type=pathlib.Path, help="Linux-good trace directory or .tar.gz")
  parser.add_argument("--tinygrad", required=True, type=pathlib.Path, help="tinygrad attempt directory or .tar.gz")
  parser.add_argument("--firmware", default="psp_13_0_10_sos.bin", help="PSP SOS firmware name for KDB byte comparison; empty disables")
  args = parser.parse_args()
  sys.stdout.write(report(parse_linux(args.linux), parse_tinygrad(args.tinygrad), args.firmware or None))

if __name__ == "__main__":
  main()
