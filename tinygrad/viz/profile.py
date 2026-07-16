import itertools, json, re, struct, traceback
from contextlib import contextmanager
from decimal import Decimal
from typing import Any, Callable, Generator
from tinygrad.device import ProfileDeviceEvent, ProfileGraphEntry, ProfileGraphEvent, ProfileProgramEvent
from tinygrad.helpers import Context, getenv, ProfileEvent, ProfilePointEvent, ProfileRangeEvent, TracingKey, unwrap
from tinygrad.uop.ops import sym_infer
from tinygrad.viz.graph import VizData, create_step
from tinygrad.viz.amd import amd_decode
# encoder helpers

def enum_str(s, cache:dict[str, int]) -> int:
  if (cret:=cache.get(s)) is not None: return cret
  cache[s] = ret = len(cache)
  return ret

def rel_ts(ts:int|Decimal, start_ts:int, ctx:str="") -> int:
  val = int(ts) - start_ts
  if val < 0 or val > 0xFFFFFFFF: raise ValueError(f"timestamp out of range: {ctx} diff={val} (ts={ts} start={start_ts})")
  return val

# Profiler API

DevEvent = ProfileRangeEvent|ProfileGraphEntry|ProfilePointEvent
def flatten_events(profile:list[ProfileEvent], device_ts_diffs:dict[str, Decimal]) -> Generator[tuple[Decimal, Decimal, DevEvent], None, None]:
  for e in profile:
    if isinstance(e, ProfileRangeEvent): yield (e.st+(diff:=device_ts_diffs.get(e.device, Decimal(0))), (e.en if e.en is not None else e.st)+diff, e)
    elif isinstance(e, ProfilePointEvent): yield (e.ts, e.ts, e)
    elif isinstance(e, ProfileGraphEvent):
      cpu_ts = []
      for ent in e.ents: cpu_ts += [e.sigs[ent.st_id]+(diff:=device_ts_diffs.get(ent.device, Decimal(0))), e.sigs[ent.en_id]+diff]
      yield (st:=min(cpu_ts)), (et:=max(cpu_ts)), ProfileRangeEvent(f"{e.ents[0].device.split(':')[0]} Graph", f"batched {len(e.ents)}", st, et)
      for i,ent in enumerate(e.ents): yield (cpu_ts[i*2], cpu_ts[i*2+1], ent)

# normalize event timestamps and attach kernel metadata
def timeline_layout(data:VizData, dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, scache:dict[str, int]) -> bytes|None:
  events:list[bytes] = []
  ei:ProfilePointEvent|None = None
  for st,et,dur,e in dev_events:
    if isinstance(e, ProfilePointEvent) and e.name == "exec": ei = e
    if dur == 0: continue
    name, key = e.name, None
    fmt:dict = {}
    if (ref:=data.ref_map.get(name)) is not None and ref < len(data.ctxs):
      name = data.ctxs[ref]["name"]
      if (p:=data.ctxs[ref].get("prg")) is not None and (ki:=p.src[0].arg).estimates is not None and ei is not None:
        fmt["FLOPS"] = int(sym_infer(ki.estimates.ops, var_vals:=ei.arg['var_vals'])/(t:=dur*1e-6))
        fmt["B/s mem"], fmt["B/s lds"] = int(sym_infer(ki.estimates.mem, var_vals)/t), int(sym_infer(ki.estimates.lds, var_vals)/t)
        if ei.arg["metadata"]: fmt["metadata"] = ",".join([str(m) for m in ei.arg['metadata']+["batched" if isinstance(e,ProfileGraphEntry) else ""]])
        key = ei.key
    elif isinstance(e.name, TracingKey):
      name = e.name.display_name
      ref = next((v for k in e.name.keys if (v:=data.ref_map.get(k)) is not None), None)
      if isinstance(e.name.ret, str): fmt.update(json.loads(e.name.ret[4:]) if e.name.ret.startswith("JSON") else {"metadata":e.name.ret})
      elif isinstance(e.name.ret, int): fmt["B/s"], fmt["B"] = int(e.name.ret/(dur*1e-6)), e.name.ret
      elif e.name.tb: fmt["tb"] = e.name.tb
    events.append(struct.pack("<IIIIfI", enum_str(name, scache), 0 if ref is None else ref+1, 0 if key is None else key+1, rel_ts(st,start_ts, f"'{name}' on {e.device}"),
                              dur, enum_str(json.dumps(fmt),scache)))
  return struct.pack("<BI", 0, len(events))+b"".join(events) if events else None

def encode_mem_free(key:int, ts:int, execs:list[ProfilePointEvent], scache:dict) -> bytes:
  ei_encoding:list[tuple[int, int, int, int]] = [] # <[u32, u32, u32, u8] [run id, display name, buffer number and mode (2 = r/w, 1 = w, 0 = r)]
  for e in execs:
    num = next(i for i,k in enumerate(e.arg["bufs"]) if k == key)
    mode = 2 if (num in e.arg["inputs"] and num in e.arg["outputs"]) else 1 if (num in e.arg["outputs"]) else 0
    ei_encoding.append((e.key, enum_str(e.arg["name"], scache), num, mode))
  return struct.pack("<BIII", 0, ts, key, len(ei_encoding))+b"".join(struct.pack("<IIIB", *t) for t in ei_encoding)

def graph_layout(k:str, dev_events:list[tuple[int, int, float, DevEvent]], start_ts:int, end_ts:int, peaks:list[int], dtype_size:dict[str, int],
                 scache:dict[str, int]) -> tuple[str, bytes|None]:
  if k.startswith("LINE:"):
    xy = [(rel_ts(e.ts, start_ts, f"line '{k}' on {e.device}"), e.key) for st,_,_,e in dev_events if isinstance(e, ProfilePointEvent)]
    peaks.append(peak:=max([y for _,y in xy]))
    return k.replace("LINE:", ""), struct.pack("<BIBQ", 1, len(xy), 1, peak)+b"".join(struct.pack("<IQ", x, y) for x,y in xy)
  peak, mem = 0, 0
  temp:dict[int, int] = {}
  events:list[bytes] = []
  buf_ei:dict[int, list[ProfilePointEvent]] = {}
  for st,_,_,e in dev_events:
    if not isinstance(e, ProfilePointEvent): continue
    if e.name == "alloc":
      safe_sz = min(1_000_000_000_000, e.arg["sz"])
      events.append(struct.pack("<BIIIQ", 1, rel_ts(e.ts, start_ts, f"alloc on {e.device}"), e.key, enum_str(e.arg["dtype"].name, scache), safe_sz))
      dtype_size.setdefault(e.arg["dtype"].name, e.arg["dtype"].itemsize)
      temp[e.key] = nbytes = safe_sz*e.arg["dtype"].itemsize
      mem += nbytes
      if mem > peak: peak = mem
    if e.name == "exec" and e.arg["bufs"]:
      for b in e.arg["bufs"]: buf_ei.setdefault(b, []).append(e)
    if e.name == "free":
      events.append(encode_mem_free(e.key, rel_ts(e.ts, start_ts, f"free on {e.device}"), buf_ei.pop(e.key, []), scache))
      mem -= temp.pop(e.key)
  for t in temp: events.append(encode_mem_free(t, rel_ts(end_ts, start_ts, f"end_ts for {k}"), buf_ei.pop(t, []), scache))
  peaks.append(peak)
  return f"{k} Memory", struct.pack("<BIBQ", 1, len(events), 0, peak)+b"".join(events) if events else None

# by default, VIZ does not start when there is an error
# use this to instead display the traceback to the user
@contextmanager
def soft_err(fn:Callable):
  try: yield
  except Exception: fn({"src":traceback.format_exc()})

def row_tuple(row:str) -> tuple[tuple[int, int], ...]:
  return ((0, 0),) if "Clock" in row else tuple((ord(ss[0][0]), int(ss[1])) if len(ss:=x.split(":"))>1 else (999,999) for x in row.split())

# *** Performance counters

metrics:dict[str, Callable[[dict[str, tuple[int, int, int]]], str]] = {
  "VALU utilization": lambda s: f"{100 * (s['SQ_INSTS_VALU'][0] / s['SQ_INSTS_VALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
  "SALU utilization": lambda s: f"{100 * (s['SQ_INSTS_SALU'][0] / s['SQ_INSTS_SALU'][2]) / (s['GRBM_GUI_ACTIVE'][1] * 4):.1f}%",
}

def unpack_pmc(e) -> dict:
  rows:list[list] = []
  stats:dict[str, tuple[int, int, int]] = {}  # name -> (sum, max, count)
  view, ptr = memoryview(e.blob).cast('Q'), 0
  for s in e.sched:
    sample_cols = ["XCC", "INST", "SE", "SA"] + [f"WGP:{i}" for i in range(s.wgp)]
    row:list = [s.name, 0, {"cols":sample_cols, "rows":[]}]
    max_val, cnt = 0, 0
    for sample in itertools.product(range(s.xcc), range(s.inst), range(s.se), range(s.sa)):
      vals:list[int] = []
      # pack work group processors on the same se
      for _ in range(s.wgp):
        row[1] += (val:=int(view[ptr]))
        max_val, cnt = max(max_val, val), cnt + 1
        vals.append(val)
        ptr += 1
      row[2]["rows"].append(sample+tuple(vals))
    stats[s.name] = (row[1], max_val, cnt)
    rows.append(row)
  for name, fn in metrics.items():
    try: rows.append([name, fn(stats)])
    except KeyError: pass
  return {"rows":rows, "cols":["Name", "Sum"]}

# ** on startup, list all the performance counter traces

def load_amd_counters(data:VizData, profile:list) -> None:
  counter_events:dict[tuple[int, int], dict] = {}
  durations:dict[str, list[float]] = {}
  prg_events:dict[int, ProfileProgramEvent] = {}
  arch = ""
  for e in profile:
    if type(e).__name__ in {"ProfilePMCEvent", "ProfileSQTTEvent"}:
      counter_events.setdefault((e.kern, e.exec_tag), {}).setdefault(type(e).__name__, []).append(e)
    if isinstance(e, ProfileRangeEvent) and e.device.startswith("AMD") and e.en is not None:
      durations.setdefault(str(e.name), []).append(float(e.en-e.st))
    if isinstance(e, ProfileProgramEvent) and e.tag is not None: prg_events[e.tag] = e
    if isinstance(e, ProfileDeviceEvent) and e.device.startswith("AMD"): arch = f"gfx{unwrap(e.props)['gfx_target_version']//1000}"
  if len(counter_events) == 0: return None
  data.ctxs.append({"name":"All Counters", "steps":[create_step("PMC", ("/all-pmc", len(data.ctxs), 0), (durations, all_counters:={}))]})
  run_number = {n:0 for n,_ in counter_events}
  for (k, tag),v in counter_events.items():
    # use the colored name if it exists
    name = data.ctxs[r]["prg"].src[0].arg.name if (r:=data.ref_map.get(pname:=prg_events[k].name)) is not None else pname
    run_number[k] += 1
    steps:list[dict] = []
    if (pmc:=v.get("ProfilePMCEvent")):
      steps.append(create_step("PMC", ("/prg-pmc", len(data.ctxs), len(steps)), pmc[0]))
      all_counters[(name, run_number[k], pname)] = pmc[0]
    # to decode a SQTT trace, we need the raw stream, program binary and device properties
    if (sqtt:=v.get("ProfileSQTTEvent")):
      for e in sqtt:
        if e.itrace: steps.append(create_step(f"SE:{e.se} PKTS", (f"/sqtt-{e.se}",len(data.ctxs),len(steps)), data=(e.blob,prg_events[k].lib,arch)))
      try:
        with Context(DEBUG=0): from extra.hardware.sqtt.roc import unpack_occ
        steps.append(create_step("OCC", ("/amd-sqtt-occ", len(data.ctxs), len(steps)),
                                 data={"fxn":unpack_occ, "args":((k, tag), sqtt, prg_events[k], arch)}))
      except Exception: pass
    data.ctxs.append({"name":f"SQTT {name}"+(f" n{run_number[k]}" if run_number[k] > 1 else ""), "steps":steps})

wave_colors = {"WMMA": "#1F7857", **{x:"#ffffc0" for x in ["VALU", "VINTERP"]}, "SALU": "#cef263", "SMEM": "#ffc0c0", "STORE": "#4fa3cc",
               **{x:"#b2b7c9" for x in ["VMEM", "SGMEM"]}, "LDS": "#9fb4a6", "IMMEDIATE": "#f3b44a", "BARRIER": "#d00000",
               "JUMP_NO": "#fb8500", "JUMP": "#ffb703", "WAVERDY": "#1a2a2a"}

def sqtt_timeline(data:bytes, lib:bytes, target:str) -> Generator[ProfileEvent, None, None]:
  from tinygrad.renderer.amd.sqtt import (map_insts, InstructionInfo, PacketType, INST, InstOp, VALUINST, IMMEDIATE, IMMEDIATE_MASK, VMEMEXEC,
                                          ALUEXEC, INST_RDNA4, InstOpRDNA4, TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4, CDNA_INST, InstOpCDNA,
                                          WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND, WAVERDY)
  pc_map = {addr:str(inst) for addr,inst in amd_decode(lib, target).items()}
  row_ends:dict[str, Decimal] = {}
  row_counts:dict[str, itertools.count] = {}
  curr_barrier:dict[int, ProfileRangeEvent] = {}
  exec_pending:dict[str, list[tuple[str, str]]] = {}
  dispatch_to_exec = {"WMMA":"VALU", "VALU":"VALU", "VALU1":"VALU", "VALUT":"VALU", "VALUB":"VALU", "VALUINST":"VALU", "VINTERP":"VALU",
                      "SGMEM":"VMEM", "FLAT":"VMEM", "LDS":"LDS", "SALU":"SALU", "SMEM":"SALU", "VMEM":"VMEM"}
  def add(name:str, p:PacketType, wave:int|None=None, info:InstructionInfo|None=None) -> Generator[ProfileEvent, None, None]:
    row = f"WAVE:{wave}" if (wave:=getattr(p, "wave", wave)) is not None else f"{p.__class__.__name__}:0 {name.replace('_ALT', '')}"
    # by default we extend the packet to one cycle after timestamp
    start_time, end_time = p._time, p._time+1
    # exec links to dispatch, dispatch links to PC
    link:dict|None = {"pc":info.pc} if info else None
    if isinstance(p, (ALUEXEC, VMEMEXEC)):
      dispatch_id, op_type = exec_pending[name].pop(0)
      # wmma exec gets its own color and its own row on rdna4
      if op_type.startswith("WMMA"):
        name = name+"_WMMA"
        if not op_type.startswith("WMMA_VALU"): row = "ALUEXEC:0 WMMA"
      # transcendental valu gets its own row
      if op_type.startswith("VALUT"): row = "ALUEXEC:0 TFU"
      # extend execs by the op type's known duration, p._time marks the first or last cycle based on the op type
      duration = int(dur_match.group(1)) if (dur_match:=re.match(r".*_(\d+)$", op_type)) else 1
      if any(ss in row for ss in ("SALU", "TFU", "VMEM", "LDS")): start_time, end_time = p._time, p._time+duration
      else: start_time, end_time = p._time-duration, p._time
      link = {"link":dispatch_id}
    # queue inst dispatches
    idx = next(row_counts.setdefault(row, itertools.count(0)))
    if isinstance(p, (VALUINST, INST, INST_RDNA4)) and (exec_type:=dispatch_to_exec.get(name.replace("OTHER_", "").split("_")[0])) is not None:
      if name.startswith("OTHER_"): exec_type = f"{exec_type}_ALT"
      # detect rdna3 wmma from the asm, only rdna4 has an op type for it
      if isinstance(p, VALUINST) and (asm:=getattr(unwrap(info).inst, "op_name", "")).startswith("V_WMMA"):
        name = f"WMMA_VALU_{16 if 'IU4' in asm else 32}"
      exec_pending.setdefault(exec_type, []).append((f"{row}-{idx}", name))
    # construct and yield the event for this packet
    if row not in row_ends: yield ProfilePointEvent(row, "JSON", "pcMap", pc_map, ts=Decimal(0))
    yield (e:=ProfileRangeEvent(row, TracingKey(name, ret="JSON"+json.dumps(link) if link else None), Decimal(start_time), Decimal(end_time)))
    row_ends[row] = unwrap(e.en)
    # barrier on this wave extends to fill the time it was waiting
    if wave is not None:
      if (barrier:=curr_barrier.pop(wave, None)) is not None: barrier.en = Decimal(p._time)
      if name in {"BARRIER", "BARRIER_SIGNAL"}: curr_barrier[wave] = e
  NS_PER_TICK = 10  # 100MHz
  prev_pair:tuple[int, int]|None = None # (shader, realtime)
  yield ProfilePointEvent("", "JSON", "waveColors", list(wave_colors.items()), ts=Decimal(0))
  for p, info in map_insts(data, lib, target):
    if isinstance(p, (TS_DELTA_OR_MARK, TS_DELTA_OR_MARK_RDNA4)) and p.is_marker:
      pair = (p._time, p.delta)
      if prev_pair is None: prev_pair = pair
      else:
        (s0, r0), (s1, r1) = prev_pair, pair
        freq_hz = (s1 - s0) * 1_000_000_000 // ((r1 - r0) * NS_PER_TICK)
        yield ProfilePointEvent("LINE:Shader Clock", "freq_hz", freq_hz, ts=Decimal(p._time))
        prev_pair = pair
    if isinstance(p, (INST, INST_RDNA4, CDNA_INST)):
      name = p.op.name if isinstance(p.op, (InstOp, InstOpRDNA4, InstOpCDNA)) else f"0x{p.op:02x}"
      yield from add(name, p, info=info)
    if isinstance(p, (VALUINST, IMMEDIATE, WAVEEND, WAVEEND_RDNA4, CDNA_WAVEEND)): yield from add(p.__class__.__name__, p, info=info)
    if isinstance(p, IMMEDIATE_MASK): yield from add("IMMEDIATE", p, wave=unwrap(info).wave, info=info)
    if isinstance(p, WAVERDY):
      for wave in range(16):
        if p.mask & (1 << wave):
          if wave in curr_barrier: yield from add("WAVERDY", p, wave=wave)
    if isinstance(p, (VMEMEXEC, ALUEXEC)):
      name = str(p.src).split('.')[1]
      if name == "VALU_SALU":
        yield from add("VALU", p)
        yield from add("SALU", p)
      else:
        yield from add(name, p)

def device_sort_fn(k:str) -> tuple:
  special = {"GC": 0, "USER": 1, "TINY": 2, "ALLDEVS":100, "DISK": 999}
  is_memory = k.endswith(" Memory")
  p = k.split(" ")[0].split(":")
  dev_base = p[0] if len(p) < 2 or not p[1].isdigit() else f"{p[0]}:{p[1]}"
  return (is_memory, special.get(p[0], special['ALLDEVS']), dev_base, k)

def get_profile(data:VizData, profile:list[ProfileEvent], sort_fn:Callable[[str], Any]=device_sort_fn) -> bytes|None:
  # start by getting the time diffs
  device_ts_diffs:dict[str, Decimal] = {}
  device_decoders:dict[str, Callable[[VizData, list[ProfileEvent]], None]] = {}
  for ev in profile:
    if isinstance(ev, ProfileDeviceEvent):
      device_ts_diffs[ev.device] = ev.tdiff
      if (d:=ev.device.split(":")[0]) == "AMD": device_decoders[d] = load_amd_counters
      if d == "NV": device_decoders[d] = load_nv_counters
  # load device specific counters
  for fxn in device_decoders.values(): fxn(data, profile)
  # map events per device
  dev_events:dict[str, list[tuple[int, int, float, DevEvent]]] = {}
  markers:list[ProfilePointEvent] = []
  ext_data:dict[str, Any] = {}
  start_ts:int|None = None
  end_ts:int|None = None
  for ts,en,e in flatten_events(profile, device_ts_diffs):
    dev_events.setdefault(e.device,[]).append((st:=int(ts), et:=int(en), float(en-ts), e))
    if start_ts is None or st < start_ts: start_ts = st
    if end_ts is None or et > end_ts: end_ts = et
    if isinstance(e, ProfilePointEvent) and e.name == "marker": markers.append(e)
    if isinstance(e, ProfilePointEvent) and e.name == "JSON": ext_data[e.key] = e.arg
  if start_ts is None: return None
  # return layout of per device events
  layout:dict[str, bytes|None] = {}
  scache:dict[str, int] = {}
  peaks:list[int] = []
  dtype_size:dict[str, int] = {}
  for k,v in dev_events.items():
    v.sort(key=lambda e:e[0])
    layout[k] = timeline_layout(data, v, start_ts, scache)
    layout.update([graph_layout(k, v, start_ts, unwrap(end_ts), peaks, dtype_size, scache)])
  sorted_layout = sorted([k for k,v in layout.items() if v is not None], key=sort_fn)
  ret = [b"".join([struct.pack("<B", len(k)), k.encode(), unwrap(layout[k])]) for k in sorted_layout]
  index = json.dumps({"strings":list(scache), "dtypeSize":dtype_size,
                      "markers":[{"ts":rel_ts(e.ts, start_ts, f"marker '{e.arg.get('name','?')}'"), **e.arg} for e in markers],
                      **ext_data}).encode()
  return struct.pack("<IQII", rel_ts(unwrap(end_ts), start_ts, "end_ts"), max(peaks,default=0), len(index), len(ret))+index+b"".join(ret)

# ** PMA counters

def load_nv_counters(data:VizData, profile:list) -> None:
  steps:list[dict] = []
  sm_version = {e.device:e.props.get("sm_version", 0x800) for e in profile if isinstance(e, ProfileDeviceEvent) and e.props is not None}
  run_number:dict[str, int] = {}
  for e in profile:
    if type(e).__name__ == "ProfilePMAEvent":
      run_number[e.kern] = run_num = run_number.get(e.kern, 0)+1
      steps.append(create_step(f"PMA {e.kern}"+(f"n{run_num}" if run_num>1 else ""), ("/prg-pma-pkts", len(data.ctxs), len(steps)),
                               data=(e.blob, sm_version[e.device])))
  if steps: data.ctxs.append({"name":"All Counters", "steps":steps})
