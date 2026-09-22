#!/usr/bin/env python3
"""Research-only replay of CUDA's Blackwell persisting-L2 reservation MME call."""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys

ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))

CUDA_RESERVE_WORDS=(
  0x20032D00, 0x00000000, 0x000A0000, 0x001F0000,
  0x20012144, 0x0017E2AC,
)
CUDA_RESET_WORDS=CUDA_RESERVE_WORDS[:2]+(0x00000000,)+CUDA_RESERVE_WORDS[3:]

def submit(dev, reserve: bool) -> None:
  from tinygrad.runtime.ops_nv import NVComputeQueue
  q=NVComputeQueue();q.q(*(CUDA_RESERVE_WORDS if reserve else CUDA_RESET_WORDS))
  q.signal(dev.timeline_signal,dev.next_timeline()).submit(dev);dev.synchronize()

def main() -> int:
  ap=argparse.ArgumentParser();ap.add_argument("arm",choices=("control","reserve"));ap.add_argument("--primitive",action="store_true");ap.add_argument("--n",type=int,default=18);args=ap.parse_args()
  from tinygrad import Device
  dev=Device["NV"];submit(dev,args.arm=="reserve")
  out={"arm":args.arm,"status":"PASS","words":[hex(x) for x in (CUDA_RESERVE_WORDS if args.arm=="reserve" else CUDA_RESET_WORDS)]}
  if args.primitive:
    from tinygrad.runtime.ops_nv import NVProgram
    from nv_flash_l2_priority_turnability import _alloc,_compile,_run_sample,FOOTPRINT_BYTES,DISTURB_BYTES,GRID
    footprint=_alloc(dev,FOOTPRINT_BYTES);disturb=_alloc(dev,DISTURB_BYTES);sink=_alloc(dev,GRID[0]*4)
    for b in (footprint,disturb,sink):dev.allocator._copyin(b,memoryview(bytearray(b.size)))
    specs={"footprint_normal":(FOOTPRINT_BYTES,"normal",footprint),"footprint_evict_last":(FOOTPRINT_BYTES,"evict_last",footprint),"disturb_normal":(DISTURB_BYTES,"normal",disturb)}
    programs={};buffers={"out":sink}
    for name,(size,policy,buf) in specs.items():programs[name]=NVProgram(dev,name,_compile(dev,name,size//4,policy));buffers[name]=buf
    rows={}
    for name,prime,stream in (("hot","normal","none"),("cold","normal","normal"),("persist","evict_last","normal")):
      xs=[_run_sample(dev,programs,buffers,prime,stream,180) for _ in range(args.n)];rows[name]=statistics.median(xs[4:])
    rows["recovery_us"]=rows["cold"]-rows["persist"];rows["recovery_share"]=(rows["recovery_us"]/(rows["cold"]-rows["hot"]))
    out["primitive_us"]=rows
  print(json.dumps(out))
  return 0
if __name__=="__main__":raise SystemExit(main())
