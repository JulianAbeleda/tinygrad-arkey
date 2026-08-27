#!/usr/bin/env python3
"""Reduce nv_ioctl_trace.c output to stable RM allocation/mapping facts."""
from __future__ import annotations

import argparse, collections, json, re, struct
from pathlib import Path

FIELD = re.compile(r"([a-z_]+)=([0-9a-f]*)")

def u32(buf: bytes, off: int) -> int: return struct.unpack_from("<I", buf, off)[0]
def u64(buf: bytes, off: int) -> int: return struct.unpack_from("<Q", buf, off)[0]

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("trace", type=Path)
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  classes: collections.Counter[str] = collections.Counter()
  ioctls: collections.Counter[str] = collections.Counter()
  channels, usermodes, maps, dma_maps = [], [], [], []
  for line in args.trace.read_text().splitlines():
    fields = dict(FIELD.findall(line))
    reqm = re.search(r"\breq=([0-9a-f]+)", line)
    if not reqm: continue
    req = int(reqm.group(1), 16); nr = req & 0xff
    ioctls[f"0x{nr:02x}"] += 1
    outer = bytes.fromhex(fields.get("outer_post", ""))
    if nr == 0x2b and len(outer) >= 16:
      cls = u32(outer, 12); classes[f"0x{cls:04x}"] += 1
      nested = bytes.fromhex(fields.get("post", ""))
      rec = {"h_root": u32(outer, 0), "h_parent": u32(outer, 4), "h_object": u32(outer, 8), "class": f"0x{cls:04x}"}
      if cls == 0xc661:
        if len(nested) >= 8: rec["nested_u64"] = u64(nested, 0)
        usermodes.append(rec)
      elif cls == 0xc96f and len(nested) >= 144:
        rec.update(h_object_buffer=u32(nested, 4), gp_fifo_offset=u64(nested, 8), gp_fifo_entries=u32(nested, 16),
                   flags=u32(nested, 20), h_userd_memory=[u32(nested, 32+i*4) for i in range(8)],
                   userd_offset=[u64(nested, 64+i*8) for i in range(8)], cid=u32(nested, 132))
        channels.append(rec)
    elif nr == 0x4e and len(outer) >= 48:
      # Both observed NVOS33 encodings retain these common offsets.
      maps.append({"h_client":u32(outer,0), "h_device":u32(outer,4), "h_memory":u32(outer,8),
                   "offset":u64(outer,16), "length":u64(outer,24), "linear":u64(outer,32)})
    elif nr == 0x57 and len(outer) >= 56:
      dma_maps.append({"h_client":u32(outer,0), "h_device":u32(outer,4), "h_dma":u32(outer,8), "h_memory":u32(outer,12),
                       "offset":u64(outer,16), "length":u64(outer,24), "flags":u32(outer,32), "dma_offset":u64(outer,40)})
  result = {"trace":str(args.trace), "line_count":sum(ioctls.values()), "ioctl_counts":dict(sorted(ioctls.items())),
            "class_counts":dict(sorted(classes.items())), "usermodes":usermodes, "channels":channels,
            "map_memory":maps, "map_memory_dma":dma_maps}
  text = json.dumps(result, indent=2, sort_keys=True)
  if args.out: args.out.write_text(text+"\n")
  print(text)

if __name__ == "__main__": main()
