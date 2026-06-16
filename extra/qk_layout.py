#!/usr/bin/env python3
from __future__ import annotations

import pathlib, struct
from dataclasses import dataclass
from math import prod

from tinygrad import Tensor, dtypes
from tinygrad.helpers import round_up
from tinygrad.llm.gguf import ggml_data_to_tensor

GGML_Q4_K = 12
GGML_Q6_K = 14

QK_BLOCK_ELEMS = 256
Q4_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q4_K_BLOCK_BYTES = 144
Q4K_WORDS_PER_BLOCK = Q4_K_BLOCK_BYTES // 4
Q6_K_BLOCK_ELEMS = QK_BLOCK_ELEMS
Q6_K_BLOCK_BYTES = 210
Q6K_HALFWORDS_PER_BLOCK = Q6_K_BLOCK_BYTES // 2
Q8_1_BLOCK_ELEMS = 32

_QUANT_BYTES = {GGML_Q4_K: Q4_K_BLOCK_BYTES, GGML_Q6_K: Q6_K_BLOCK_BYTES}
_FORMAT_NAMES = {GGML_Q4_K: "Q4_K", GGML_Q6_K: "Q6_K"}

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

def format_name(ggml_type:int) -> str:
  return _FORMAT_NAMES.get(ggml_type, f"ggml_type_{ggml_type}")

def tensor_shape(info:GGUFInfo) -> tuple[int, ...]:
  return tuple(reversed(info.dims))

def quant_weight_bytes(info:GGUFInfo) -> int:
  if info.typ not in _QUANT_BYTES: raise ValueError(f"{info.name} has unsupported quant type {info.typ}")
  if prod(info.dims) % QK_BLOCK_ELEMS != 0: raise ValueError(f"{info.name} element count is not QK block aligned")
  return prod(info.dims) // QK_BLOCK_ELEMS * _QUANT_BYTES[info.typ]

def q4_k_weight_bytes(info:GGUFInfo) -> int:
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  return quant_weight_bytes(info)

def q6_k_weight_bytes(info:GGUFInfo) -> int:
  if info.typ != GGML_Q6_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q6_K")
  return quant_weight_bytes(info)

def packed_byte_range(meta:GGUFMetadata, info:GGUFInfo) -> tuple[int, int]:
  return meta.data_start + info.off, quant_weight_bytes(info)

def require_packed_alignment(meta:GGUFMetadata, info:GGUFInfo, itemsize:int) -> None:
  byte_start, nbytes = packed_byte_range(meta, info)
  if byte_start % itemsize != 0 or nbytes % itemsize != 0:
    raise ValueError(f"{info.name} byte range is not uint{itemsize*8} aligned: start={byte_start} nbytes={nbytes}")

def packed_u8_slice(path:pathlib.Path, meta:GGUFMetadata, info:GGUFInfo, device:str|None=None) -> Tensor:
  byte_start, nbytes = packed_byte_range(meta, info)
  return Tensor(path)[byte_start:byte_start+nbytes].to(device).contiguous().realize()

def packed_u32_slice(path:pathlib.Path, meta:GGUFMetadata, info:GGUFInfo, device:str|None=None) -> Tensor:
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  require_packed_alignment(meta, info, 4)
  byte_start, nbytes = packed_byte_range(meta, info)
  return Tensor(path, dtype=dtypes.uint32)[byte_start//4:byte_start//4+nbytes//4].to(device).contiguous().realize()

def packed_u16_slice(path:pathlib.Path, meta:GGUFMetadata, info:GGUFInfo, device:str|None=None) -> Tensor:
  if info.typ != GGML_Q6_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q6_K")
  require_packed_alignment(meta, info, 2)
  byte_start, nbytes = packed_byte_range(meta, info)
  return Tensor(path, dtype=dtypes.uint16)[byte_start//2:byte_start//2+nbytes//2].to(device).contiguous().realize()

def pick_tensor(infos:list[GGUFInfo], name:str|None, ggml_type:int=GGML_Q4_K) -> GGUFInfo:
  if name is not None:
    for info in infos:
      if info.name == name: return info
    raise ValueError(f"tensor {name!r} not found")
  preferred = ("ffn_down", "ffn_up", "attn_q", "attn_output")
  for info in infos:
    if info.typ == ggml_type and len(info.dims) == 2 and any(x in info.name for x in preferred):
      return info
  for info in infos:
    if info.typ == ggml_type and len(info.dims) == 2: return info
  raise ValueError(f"no 2D {format_name(ggml_type)} tensor found")

def role_from_name(name:str) -> str:
  roles = ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_k", "attn_v", "attn_output", "output", "token_embd")
  for role in roles:
    if name == f"{role}.weight" or f".{role}.weight" in name: return role
  return "unknown"

def model_shape_targets(infos:list[GGUFInfo], kv:dict, max_shapes:int|None=None, ggml_type:int=GGML_Q4_K) -> list[GGUFInfo]:
  arch = kv.get("general.architecture", "")
  dim = int(kv.get(f"{arch}.embedding_length", 0)) if arch else 0
  hidden = int(kv.get(f"{arch}.feed_forward_length", 0)) if arch else 0
  targets = []
  preferred = ("ffn_gate", "ffn_up", "ffn_down", "attn_q", "attn_output", "attn_k", "attn_v")
  seen: set[tuple[int, int]] = set()
  for kind in preferred:
    for info in infos:
      if info.typ != ggml_type or len(info.dims) != 2 or f".{kind}.weight" not in info.name: continue
      shape = tensor_shape(info)
      if len(shape) != 2: continue
      if dim and shape[1] not in (dim, hidden) and shape[0] not in (dim, hidden): continue
      if shape in seen: continue
      seen.add(shape)
      targets.append(info)
      break
  return targets[:max_shapes] if max_shapes is not None else targets

# The GGML Q4_K / Q6_K block format is owned by tinygrad/llm/gguf.py (the live
# decode loader). These references delegate to it so the block layout lives in
# one place; the q4_k/q6_k dequant math previously copied here was byte-identical
# (pinned by test_qk_layout.test_q*_k_reference_matches_current_gguf_expression).
def q4_k_reference(t:Tensor, n:int) -> Tensor:
  return ggml_data_to_tensor(t, n, GGML_Q4_K)

def q6_k_reference(t:Tensor, n:int) -> Tensor:
  return ggml_data_to_tensor(t, n, GGML_Q6_K)

def quant_reference(t:Tensor, n:int, ggml_type:int) -> Tensor:
  if ggml_type == GGML_Q4_K: return q4_k_reference(t, n)
  if ggml_type == GGML_Q6_K: return q6_k_reference(t, n)
  raise ValueError(f"unsupported quant reference type {ggml_type}")

def q8_1_quantize(x:Tensor, block_elems:int=Q8_1_BLOCK_ELEMS) -> tuple[Tensor, Tensor]:
  if x.shape[-1] % block_elems != 0: raise ValueError(f"last dimension {x.shape[-1]} is not q8_1 block aligned")
  blocks = x.cast(dtypes.float32).reshape(-1, block_elems)
  scales = blocks.abs().max(axis=1, keepdim=True) / 127.0
  scales = (scales == 0).where(1.0, scales)
  qs = (blocks / scales).round().clip(-128, 127).cast(dtypes.int8)
  return qs.flatten().contiguous(), scales.reshape(-1).contiguous()

def q8_1_dequantize(qs:Tensor, scales:Tensor, block_elems:int=Q8_1_BLOCK_ELEMS) -> Tensor:
  if qs.shape[-1] % block_elems != 0: raise ValueError(f"last dimension {qs.shape[-1]} is not q8_1 block aligned")
  return (qs.reshape(-1, block_elems).cast(dtypes.float32) * scales.reshape(-1, 1).cast(dtypes.float32)).flatten()
