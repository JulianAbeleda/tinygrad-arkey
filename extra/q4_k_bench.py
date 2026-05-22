#!/usr/bin/env python3
import argparse, pathlib, struct, time
from dataclasses import dataclass
from math import prod

from tinygrad import Tensor
from tinygrad.helpers import round_up
from tinygrad.llm.gguf import ggml_data_to_tensor

GGML_Q4_K = 12

@dataclass(frozen=True)
class GGUFInfo:
  name: str
  dims: tuple[int, ...]
  typ: int
  off: int

def read(fmt:str, f):
  return struct.unpack("<"+fmt, f.read(struct.calcsize(fmt)))[0]

def read_str(f) -> str:
  return f.read(read("Q", f)).decode("utf-8")

def skip_value(f, typ:int):
  if typ == 8:
    f.seek(read("Q", f), 1)
  elif typ == 9:
    item_typ, n = read("i", f), read("Q", f)
    for _ in range(n): skip_value(f, item_typ)
  else:
    f.seek({0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}[typ], 1)

def read_metadata(path:pathlib.Path) -> tuple[int, list[GGUFInfo]]:
  with path.open("rb") as f:
    magic, version, n_tensors, n_kv = f.read(4), read("i", f), read("q", f), read("q", f)
    if magic != b"GGUF" or version not in (2, 3): raise ValueError(f"{path} is not a supported GGUF file")
    alignment = 32
    for _ in range(n_kv):
      key, typ = read_str(f), read("i", f)
      if key == "general.alignment" and typ == 4: alignment = read("I", f)
      else: skip_value(f, typ)
    infos = [GGUFInfo(read_str(f), tuple(read("Q", f) for _ in range(read("I", f))), read("i", f), read("Q", f)) for _ in range(n_tensors)]
    return round_up(f.tell(), alignment), infos

def pick_tensor(infos:list[GGUFInfo], name:str|None) -> GGUFInfo:
  if name is not None:
    for info in infos:
      if info.name == name: return info
    raise ValueError(f"tensor {name!r} not found")
  for info in infos:
    if info.typ == GGML_Q4_K and len(info.dims) == 2 and any(x in info.name for x in ("ffn_down", "ffn_up", "attn_q", "attn_output")):
      return info
  for info in infos:
    if info.typ == GGML_Q4_K and len(info.dims) == 2: return info
  raise ValueError("no 2D Q4_K tensor found")

def bench(label:str, iters:int, fn):
  fn()
  st = time.perf_counter()
  for _ in range(iters): fn()
  dt = (time.perf_counter() - st) / iters
  print(f"{label}: {dt*1000:.3f} ms ({1/dt:.2f}/s)")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Baseline GGUF Q4_K decode/matvec benchmark")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", help="exact tensor name to benchmark")
  parser.add_argument("--device", default=None, help="tinygrad device, for example AMD or CPU")
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--list", action="store_true", help="list Q4_K tensors and exit")
  args = parser.parse_args()

  data_start, infos = read_metadata(args.gguf)
  if args.list:
    for info in infos:
      if info.typ == GGML_Q4_K: print(f"{info.name} dims={tuple(reversed(info.dims))} off={info.off}")
    raise SystemExit(0)

  info = pick_tensor(infos, args.tensor)
  print(f"tensor: {info.name} dims={tuple(reversed(info.dims))} ggml_type={info.typ} device={args.device or 'default'}")

  raw = Tensor(args.gguf, device=args.device)
  raw_slice = raw[data_start + info.off:]
  n = prod(info.dims)
  shape = tuple(reversed(info.dims))

  def decode():
    ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).contiguous().realize()

  decoded = ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).contiguous().realize()
  x = Tensor.ones((1, decoded.shape[-1]), device=args.device, dtype=decoded.dtype).realize()

  def matvec():
    x.matmul(decoded.transpose()).realize()

  bench("decode_q4_k", args.iters, decode)
  bench("matvec_decoded", args.iters, matvec)
