#!/usr/bin/env python3
import argparse, csv, json, os, pathlib, struct, sys, time
from dataclasses import dataclass
from math import prod

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import GlobalCounters, round_up
from tinygrad.llm.gguf import ggml_data_to_tensor

GGML_Q4_K = 12
Q4_K_BLOCK_ELEMS = 256
Q4_K_BLOCK_BYTES = 144

@dataclass(frozen=True)
class GGUFInfo:
  name: str
  dims: tuple[int, ...]
  typ: int
  off: int

@dataclass(frozen=True)
class GGUFMetadata:
  data_start: int
  infos: list[GGUFInfo]
  kv: dict[str, int|float|str|bool|list]

def read(fmt:str, f):
  return struct.unpack("<"+fmt, f.read(struct.calcsize(fmt)))[0]

def read_str(f) -> str:
  return f.read(read("Q", f)).decode("utf-8")

def read_value(f, typ:int):
  if typ == 8: return read_str(f)
  if typ == 9:
    item_typ, n = read("i", f), read("Q", f)
    return [read_value(f, item_typ) for _ in range(n)]
  if typ == 0: return read("?", f)
  if typ == 1: return read("b", f)
  if typ == 2: return read("h", f)
  if typ == 3: return read("H", f)
  if typ == 4: return read("i", f)
  if typ == 5: return read("I", f)
  if typ == 6: return read("f", f)
  if typ == 7: return read("?", f)
  if typ == 10: return read("q", f)
  if typ == 11: return read("Q", f)
  if typ == 12: return read("d", f)
  raise ValueError(f"unsupported GGUF value type {typ}")

def read_metadata(path:pathlib.Path) -> GGUFMetadata:
  with path.open("rb") as f:
    magic, version, n_tensors, n_kv = f.read(4), read("i", f), read("q", f), read("q", f)
    if magic != b"GGUF" or version not in (2, 3): raise ValueError(f"{path} is not a supported GGUF file")
    alignment, kv = 32, {}
    for _ in range(n_kv):
      key, typ = read_str(f), read("i", f)
      kv[key] = read_value(f, typ)
      if key == "general.alignment": alignment = int(kv[key])
    infos = [GGUFInfo(read_str(f), tuple(read("Q", f) for _ in range(read("I", f))), read("i", f), read("Q", f)) for _ in range(n_tensors)]
    return GGUFMetadata(round_up(f.tell(), alignment), infos, kv)

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

def tensor_shape(info:GGUFInfo) -> tuple[int, ...]:
  return tuple(reversed(info.dims))

def q4_k_weight_bytes(info:GGUFInfo) -> int:
  return prod(info.dims) // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES

def q4_k_reference(t:Tensor, n:int) -> Tensor:
  blocks = t[:(n//Q4_K_BLOCK_ELEMS)*Q4_K_BLOCK_BYTES].reshape((-1, Q4_K_BLOCK_BYTES)).contiguous()
  d, dmin = (blocks[:,i:i+2].bitcast(dtypes.float16).cast(dtypes.float32).unsqueeze(-1) for i in [0, 2])
  s = blocks[:,4:16]
  sc = s[:,0:4].bitwise_and(63).cat(s[:,8:12].bitwise_and(0xF).bitwise_or(s[:,0:4].rshift(6).lshift(4)), dim=-1)
  mn = s[:,4:8].bitwise_and(63).cat(s[:,8:12].rshift(4).bitwise_or(s[:,4:8].rshift(6).lshift(4)), dim=-1)
  q = Tensor.stack((qs:=blocks[:,16:144].reshape(-1,4,32)).bitwise_and(0xF), qs.rshift(4), dim=2).reshape(-1,8,32)
  return (d * sc.unsqueeze(-1) * q - dmin * mn.unsqueeze(-1)).flatten(-2)

def correctness_gate(raw_slice:Tensor, n:int, info:GGUFInfo) -> None:
  ref = q4_k_reference(raw_slice, n).reshape(*tensor_shape(info)).contiguous().realize()
  got = ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*tensor_shape(info)).contiguous().realize()
  # Tensor equality is enough here: the gate is bit-exact relative to the frozen current expression.
  ok = (got == ref).all().numpy().item()
  if not ok: raise AssertionError(f"Q4_K correctness gate failed for {info.name}")

def bench(label:str, iters:int, q4_bytes:int, fn) -> dict[str, float|int|str]:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  dt = (time.perf_counter() - st) / iters
  return {"name": label, "iters": iters, "ms": dt*1000, "per_s": 1/dt, "kernels": GlobalCounters.kernel_count / iters,
          "global_mem_mb": GlobalCounters.global_mem / iters / 1e6, "q4_weight_mb": q4_bytes / 1e6,
          "q4_eff_gbs": q4_bytes / dt / 1e9}

def model_shape_targets(infos:list[GGUFInfo], kv:dict, max_shapes:int|None=None) -> list[GGUFInfo]:
  arch = kv.get("general.architecture", "")
  dim = int(kv.get(f"{arch}.embedding_length", 0)) if arch else 0
  hidden = int(kv.get(f"{arch}.feed_forward_length", 0)) if arch else 0
  targets = []
  # Prefer real model tensors from early dense blocks, grouped by distinct decode GEMV shape.
  preferred = ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_output", "attn_k", "attn_v")
  seen: set[tuple[int, int]] = set()
  for kind in preferred:
    for info in infos:
      if info.typ != GGML_Q4_K or len(info.dims) != 2 or f".{kind}.weight" not in info.name: continue
      shape = tensor_shape(info)
      if len(shape) != 2: continue
      if dim and shape[1] not in (dim, hidden) and shape[0] not in (dim, hidden): continue
      if shape in seen: continue
      seen.add(shape)
      targets.append(info)
      break
  return targets[:max_shapes] if max_shapes is not None else targets

def emit(results:list[dict], fmt:str):
  if fmt == "json":
    print(json.dumps(results, indent=2, sort_keys=True))
  elif fmt == "csv":
    writer = csv.DictWriter(sys.stdout, fieldnames=sorted({k for r in results for k in r.keys()}))
    writer.writeheader()
    writer.writerows(results)
  else:
    for r in results:
      print(f"{r['tensor']} {r['shape']} {r['name']}: {r['ms']:.3f} ms ({r['per_s']:.2f}/s) "
            f"q4_eff={r['q4_eff_gbs']:.2f} GB/s kernels={r['kernels']:.1f} mem={r['global_mem_mb']:.2f} MB")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Baseline GGUF Q4_K decode/matvec benchmark")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", help="exact tensor name to benchmark")
  parser.add_argument("--device", default=None, help="tinygrad device, for example AMD or CPU")
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--seq-len", type=int, default=1, help="input rows for matmul; 1 is decode, >1 is prefill-shaped")
  parser.add_argument("--all-shapes", action="store_true", help="benchmark representative Q4_K decode GEMV shapes from model metadata")
  parser.add_argument("--max-shapes", type=int, default=None, help="cap representative shapes")
  parser.add_argument("--no-correctness", action="store_true", help="skip the mandatory correctness gate, for debugging only")
  parser.add_argument("--format", choices=("text", "json", "csv"), default="text")
  parser.add_argument("--list", action="store_true", help="list Q4_K tensors and exit")
  args = parser.parse_args()
  if args.seq_len < 1: raise ValueError("--seq-len must be >= 1")

  meta = read_metadata(args.gguf)
  if args.list:
    for info in meta.infos:
      if info.typ == GGML_Q4_K: print(f"{info.name} dims={tensor_shape(info)} off={info.off}")
    raise SystemExit(0)

  targets = model_shape_targets(meta.infos, meta.kv, args.max_shapes) if args.all_shapes else [pick_tensor(meta.infos, args.tensor)]
  if args.format == "text":
    arch = meta.kv.get("general.architecture", "")
    cfg = {k: meta.kv.get(f"{arch}.{k}") for k in ("embedding_length", "feed_forward_length", "block_count", "attention.head_count",
                                                   "attention.head_count_kv") if f"{arch}.{k}" in meta.kv}
    print(f"model_config: arch={arch} {cfg}")
    print(f"variant: GGUF_Q4K_WIDE={os.getenv('GGUF_Q4K_WIDE', '0')} device={args.device or 'default'} seq_len={args.seq_len}")

  raw = Tensor(args.gguf, device=args.device)
  results = []
  for info in targets:
    raw_slice = raw[meta.data_start + info.off:]
    n, shape, q4_bytes = prod(info.dims), tensor_shape(info), q4_k_weight_bytes(info)
    if not args.no_correctness:
      correctness_gate(raw_slice, n, info)
      if args.format == "text": print(f"correctness: PASS {info.name}")

    decoded = ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).cast(dtypes.float16).contiguous().realize()
    x = Tensor.ones((args.seq_len, decoded.shape[-1]), device=args.device, dtype=decoded.dtype).realize()

    @TinyJit
    def matvec():
      return x.matmul(decoded.transpose())

    @TinyJit
    def decode_matvec():
      return x.matmul(ggml_data_to_tensor(raw_slice, n, info.typ).reshape(*shape).cast(dtypes.float16).transpose())

    base = {"tensor": info.name, "shape": "x".join(map(str, shape)), "ggml_type": info.typ,
            "device": args.device or "default", "seq_len": args.seq_len, "q4_bytes": q4_bytes}
    results += [{**base, **bench("matmul_decoded", args.iters, q4_bytes, matvec)},
                {**base, **bench("decode_q4_k_plus_matmul", args.iters, q4_bytes, decode_matvec)}]
  emit(results, args.format)
