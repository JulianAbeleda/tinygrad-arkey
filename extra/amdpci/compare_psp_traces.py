#!/usr/bin/env python3
import argparse, pathlib, re, sys, tarfile

HEX = r"0x[0-9a-fA-F]+"

LINUX_BL_RE = re.compile(
  rf"^(?P<t>\d+) bl_load enter .* cmd=(?P<cmd>{HEX}) fw_pri_mc=(?P<fw_pri_mc>{HEX}) c2p36=(?P<c2p36>{HEX}) size=(?P<size>{HEX})"
)
LINUX_WAIT_ENTER_RE = re.compile(r"^(?P<t>\d+) wait_bl enter")
LINUX_WAIT_RET_RE = re.compile(r"^(?P<t>\d+) wait_bl ret=(?P<ret>-?\d+)")
LINUX_GART_ENTER_RE = re.compile(
  rf"gart_map enter .* offset=(?P<offset>{HEX}) pages=(?P<pages>\d+) .* dma0=(?P<dma0>{HEX}) dma_last=(?P<dma_last>{HEX}) flags=(?P<flags>{HEX})"
)
LINUX_GART_RET_RE = re.compile(
  rf"gart_map ret .* first_idx=(?P<first_idx>{HEX}) last_idx=(?P<last_idx>{HEX}) pte0=(?P<pte0>{HEX}) pte_last=(?P<pte_last>{HEX})"
)

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

def read_texts(path:pathlib.Path) -> dict[str, str]:
  if path.is_dir():
    return {str(p.relative_to(path)): p.read_text(errors="replace") for p in path.rglob("*") if p.is_file()}
  if path.suffixes[-2:] == [".tar", ".gz"] or path.name.endswith(".tar.gz"):
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

def parse_linux(path:pathlib.Path) -> dict:
  out = {"bl": [], "waits": [], "gart": {}, "source": str(path)}
  wait_start = None
  for line in lines_for(path, ["psp-linux-good.trace", "psp-linux-good-deep.trace", "linux-pre-kdb-key-events.txt"]):
    if m := LINUX_BL_RE.search(line):
      out["bl"].append({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := LINUX_WAIT_ENTER_RE.search(line):
      wait_start = as_int(m.group("t"))
    elif m := LINUX_WAIT_RET_RE.search(line):
      end = as_int(m.group("t"))
      out["waits"].append({"start": wait_start, "end": end, "ret": int(m.group("ret")),
                           "duration_ns": None if wait_start is None else end - wait_start})
      wait_start = None
    elif m := LINUX_GART_ENTER_RE.search(line):
      out["gart"].update({k: as_int(v) for k, v in m.groupdict().items()})
    elif m := LINUX_GART_RET_RE.search(line):
      out["gart"].update({k: as_int(v) for k, v in m.groupdict().items()})
  return out

def parse_tinygrad(path:pathlib.Path) -> dict:
  out = {"source": str(path), "gart": {}, "pre_bl": {}, "skip": {}, "load": [], "regs": {}, "wait_vals": [],
         "write_msg1": {}, "write_compid": None, "readback": {}, "timeout": False}
  for line in lines_for(path, ["real.log", "audit.log", "key-grep.txt", "post-psp-status.log"]):
    if "BL not ready" in line: out["timeout"] = True
    if not (m := TG_RE.search(line)): continue
    msg = m.group("msg")
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
  return out

def row(label:str, linux, tinygrad, note:str="") -> str:
  verdict = "same" if linux == tinygrad and linux is not None else ("missing" if linux is None or tinygrad is None else "diff")
  return f"{label:48} linux={linux!s:18} tinygrad={tinygrad!s:18} {verdict:7} {note}".rstrip()

def report(linux:dict, tiny:dict) -> str:
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
  lines.append(f"tinygrad timed out waiting BL: {tiny['timeout']}")
  lines.append("")

  lines.append("GART")
  lines.append(row("offset/msg1_off", hexv(linux["gart"].get("offset")), hexv(tiny["gart"].get("msg1_off"))))
  lines.append(row("pages", str(linux["gart"].get("pages")), "256"))
  lines.append(row("first_idx/gart_page", hexv(linux["gart"].get("first_idx")), hexv(tiny["gart"].get("gart_page"))))
  linux_last_idx = linux["gart"].get("last_idx")
  tiny_last_idx = tiny["gart"].get("gart_page") + 255 if tiny["gart"].get("gart_page") is not None else None
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
    candidates = [name, f"{name}[0]"]
    val = next((tiny["regs"][c] for c in candidates if c in tiny["regs"]), None)
    lines.append(f"{name:48} {hexv(val)}")
  lines.append("")

  lines.append("Observations")
  if kdb and tg_kdb and kdb.get("size") == tg_kdb.get("bytes"):
    lines.append("- KDB payload size matches Linux-good: 0x1700. The 0x640 skip is not the leading suspect.")
  if tiny_pte_delta is not None:
    lines.append(f"- Tinygrad PTE physical delta: {tiny_pte_delta:#x}.")
  if tiny["regs"].get("regMMVM_L2_PROTECTION_FAULT_STATUS[0]", tiny["regs"].get("regMMVM_L2_PROTECTION_FAULT_STATUS")) == 0:
    lines.append("- MMHUB fault status stayed zero during the failing attempt.")
  lines.append("- The main remaining difference in these captures is behavioral: Linux returns ready after the transient 0, tinygrad never does.")
  return "\n".join(lines) + "\n"

def main():
  parser = argparse.ArgumentParser(description="Compare Linux-good AMD PSP trace data with a tinygrad PSP boot attempt")
  parser.add_argument("--linux", required=True, type=pathlib.Path, help="Linux-good trace directory or .tar.gz")
  parser.add_argument("--tinygrad", required=True, type=pathlib.Path, help="tinygrad attempt directory or .tar.gz")
  args = parser.parse_args()
  sys.stdout.write(report(parse_linux(args.linux), parse_tinygrad(args.tinygrad)))

if __name__ == "__main__":
  main()
