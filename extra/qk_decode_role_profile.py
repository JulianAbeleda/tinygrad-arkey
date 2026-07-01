#!/usr/bin/env python3
"""Model-driven decode kernel role attribution helpers.

The older weight-path attribution tools classified kernels with Qwen3-8B constants
like 4096/12288/151936. This module builds the same facts from a GGUF tensor
table, then classifies kernel names by matching their dimensions against the
profile. It is intentionally tinygrad-free so it can run before loading a model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import collections, pathlib, re, struct
from typing import Any

GGML_TYPE_NAMES = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
  12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S",
  23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
  39: "MXFP4", 41: "Q1_0",
}

# Effective bits per weight from tinygrad.llm.gguf._GGML_QUANT block sizes.
GGML_BITS_PER_WEIGHT = {
  0: 32.0, 1: 16.0, 2: 4.5, 3: 5.0, 6: 5.5, 7: 6.0, 8: 8.5,
  12: 4.5, 13: 5.5, 14: 6.5625, 18: 3.0625, 21: 3.4375, 22: 2.5625,
  23: 4.25, 24: 8.0, 25: 16.0, 26: 32.0, 27: 64.0, 28: 64.0, 30: 16.0,
  39: 4.25, 41: 1.125,
}

_KERNEL_INT_RE = re.compile(r"_(\d+)")


@dataclass(frozen=True)
class WeightRole:
  role: str
  tensor_name: str
  rows: int
  cols: int
  ggml_type: int
  quant: str
  count: int


@dataclass(frozen=True)
class DecodeRoleProfile:
  model_id: str
  model_path: str
  arch: str | None
  hidden: int | None
  ffn: int | None
  vocab: int | None
  layers: int | None
  weights: tuple[WeightRole, ...]

  def to_json(self) -> dict[str, Any]:
    return {**asdict(self), "weights": [asdict(w) for w in self.weights]}


class _GGUFReader:
  def __init__(self, path:pathlib.Path):
    self.f = path.open("rb")

  def read(self, n:int) -> bytes:
    data = self.f.read(n)
    if len(data) != n: raise EOFError("truncated GGUF")
    return data

  def u32(self) -> int: return struct.unpack("<I", self.read(4))[0]
  def i32(self) -> int: return struct.unpack("<i", self.read(4))[0]
  def u64(self) -> int: return struct.unpack("<Q", self.read(8))[0]
  def i64(self) -> int: return struct.unpack("<q", self.read(8))[0]
  def f32(self) -> float: return struct.unpack("<f", self.read(4))[0]
  def f64(self) -> float: return struct.unpack("<d", self.read(8))[0]
  def string(self) -> str: return self.read(self.u64()).decode("utf-8")

  def value(self, typ:int) -> Any:
    if typ == 0: return self.read(1)
    if typ == 1: return struct.unpack("<b", self.read(1))[0]
    if typ == 2: return struct.unpack("<H", self.read(2))[0]
    if typ == 3: return struct.unpack("<h", self.read(2))[0]
    if typ == 4: return self.u32()
    if typ == 5: return self.i32()
    if typ == 6: return self.f32()
    if typ == 7: return struct.unpack("<?", self.read(1))[0]
    if typ == 8: return self.string()
    if typ == 9:
      elem_typ, n = self.i32(), self.u64()
      # Token arrays are large but still tiny next to model load; reading them
      # keeps parsing simple and gives exact vocab when needed.
      return [self.value(elem_typ) for _ in range(n)]
    if typ == 10: return self.u64()
    if typ == 11: return self.i64()
    if typ == 12: return self.f64()
    raise ValueError(f"unsupported GGUF metadata type {typ}")


def read_gguf_metadata(path:str | pathlib.Path) -> tuple[dict[str, Any], list[tuple[str, tuple[int, ...], int, int]]]:
  p = pathlib.Path(path).expanduser()
  r = _GGUFReader(p)
  if r.read(4) != b"GGUF": raise ValueError(f"{p} is not a GGUF file")
  version = r.i32()
  if version not in (2, 3): raise ValueError(f"unsupported GGUF version {version}")
  n_tensors, n_kv = r.i64(), r.i64()
  kv: dict[str, Any] = {}
  for _ in range(n_kv):
    key, typ = r.string(), r.i32()
    kv[key] = r.value(typ)
  infos = []
  for _ in range(n_tensors):
    name = r.string()
    dims = tuple(r.u64() for _ in range(r.u32()))
    infos.append((name, dims, r.i32(), r.u64()))
  return kv, infos


def _role_from_tensor_name(name:str) -> str:
  if name == "output.weight": return "lm_head"
  if "ffn_gate" in name or "ffn_up" in name: return "ffn_gate_up"
  if "ffn_down" in name: return "ffn_down"
  if "attn_output" in name or "attn_q.weight" in name: return "attn_qo"
  if "attn_k.weight" in name or "attn_v.weight" in name: return "attn_kv"
  if "token_embd" in name: return "embedding"
  return "other"


def profile_from_gguf(path:str | pathlib.Path, model_id:str | None=None) -> DecodeRoleProfile:
  p = pathlib.Path(path).expanduser()
  kv, infos = read_gguf_metadata(p)
  arch = kv.get("general.architecture")
  hidden = kv.get(f"{arch}.embedding_length") if arch else None
  ffn = kv.get(f"{arch}.feed_forward_length") if arch else None
  layers = kv.get(f"{arch}.block_count") if arch else None
  vocab = len(kv["tokenizer.ggml.tokens"]) if "tokenizer.ggml.tokens" in kv else None

  grouped: dict[tuple[str, int, int, int], list[str]] = collections.defaultdict(list)
  for name, dims, typ, _off in infos:
    if not name.endswith(".weight") or len(dims) != 2: continue
    role = _role_from_tensor_name(name)
    if role == "embedding": continue
    rows, cols = tuple(reversed(dims))
    grouped[(role, rows, cols, typ)].append(name)
    if role == "lm_head": vocab = vocab or rows

  weights = tuple(WeightRole(role=role, tensor_name=names[0], rows=rows, cols=cols, ggml_type=typ,
                             quant=GGML_TYPE_NAMES.get(typ, f"GGML_{typ}"), count=len(names))
                  for (role, rows, cols, typ), names in sorted(grouped.items()))
  return DecodeRoleProfile(model_id=model_id or p.stem, model_path=str(p), arch=arch, hidden=hidden, ffn=ffn,
                           vocab=vocab, layers=layers, weights=weights)


def _shape_candidates(profile:DecodeRoleProfile) -> dict[tuple[int, int], list[WeightRole]]:
  out: dict[tuple[int, int], list[WeightRole]] = collections.defaultdict(list)
  for w in profile.weights:
    out[(w.rows, w.cols)].append(w)
  return out


def _extract_kernel_dims(name:str, profile:DecodeRoleProfile) -> tuple[int, int] | None:
  ints = [int(x) for x in _KERNEL_INT_RE.findall(name)]
  shapes = _shape_candidates(profile)
  for i in range(len(ints) - 1):
    pair = (ints[i], ints[i + 1])
    if pair in shapes: return pair
  return None


def _quant_from_name(name:str, matched:list[WeightRole]) -> str:
  nm = name.lower()
  if nm.startswith("q4k"): return "Q4_K"
  if nm.startswith("q5k"): return "Q5_K"
  if nm.startswith("q6k"): return "Q6_K"
  if nm.startswith("q8") or "q8_0" in nm: return "Q8_0"
  if matched: return matched[0].quant
  if "half" in nm or "f16" in nm: return "F16"
  return "unknown"


def classify_kernel(name:str, profile:DecodeRoleProfile) -> dict[str, Any]:
  nm = name.lower()
  matdims = _extract_kernel_dims(nm, profile)
  matched = _shape_candidates(profile).get(matdims, []) if matdims else []
  roles = sorted({m.role for m in matched})
  quant = _quant_from_name(nm, matched)
  route_class = (
    "generated_g3" if ("lanemap" in nm or "_g3" in nm or "futuresight" in nm) else
    "owned_warp" if "warp" in nm else
    "coop_partial" if "coop" in nm or "partial" in nm else
    "gemv" if "gemv" in nm or "mmvq" in nm else
    "fallback_graph"
  )
  role = roles[0] if len(roles) == 1 else ("ambiguous:" + ",".join(roles) if roles else "other")
  is_weight = quant != "unknown" and matdims is not None and ("gemv" in nm or "mmvq" in nm or "coop" in nm or "lanemap" in nm)
  bpw = GGML_BITS_PER_WEIGHT.get(matched[0].ggml_type, 16.0) if matched else {
    "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625, "Q8_0": 8.5, "F16": 16.0,
  }.get(quant, 16.0)
  bytes_per_call = int(matdims[0] * matdims[1] * bpw / 8) if is_weight and matdims else 0

  reduce_product = None
  reduce_class = None
  if nm.startswith("r_"):
    vals = [int(x) for x in _KERNEL_INT_RE.findall("_" + nm[2:])]
    reduce_product = 1
    for v in vals: reduce_product *= v
    outs = collections.defaultdict(list)
    for w in profile.weights: outs[w.rows].append(w)
    candidates = outs.get(reduce_product, [])
    if reduce_product == profile.vocab:
      reduce_class = "vocab_reduce_or_sampling"
    elif candidates:
      rs = sorted({c.role for c in candidates})
      reduce_class = "reduce_for_" + ("ambiguous:" + ",".join(rs) if len(rs) > 1 else rs[0])
    else:
      reduce_class = "reduce_other"

  bucket = "other"
  if nm.startswith("r_"): bucket = "reduce_partial"
  elif is_weight and quant == "Q4_K": bucket = "q4k_gemv"
  elif is_weight and quant == "Q6_K": bucket = "lm_head" if role == "lm_head" else "q6k_gemv"
  elif is_weight: bucket = f"{quant.lower()}_gemv"
  elif "flash" in nm or "attention" in nm: bucket = "attention"
  elif any(x in nm for x in ("cast", "copy", "where", "rope", "norm")): bucket = "norm_rope_elementwise"

  return {
    "kernel": name, "bucket": bucket, "role": role, "roles": roles, "quant": quant,
    "route_class": route_class, "is_weight": is_weight,
    "matdims": list(matdims) if matdims else [], "bytes_per_call": bytes_per_call,
    "reduce_product": reduce_product, "reduce_class": reduce_class,
  }


def summarize_profile(profile:DecodeRoleProfile) -> dict[str, Any]:
  by_role: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
  for w in profile.weights:
    by_role[w.role].append({"shape": [w.rows, w.cols], "quant": w.quant, "count": w.count,
                            "example": w.tensor_name})
  return {**profile.to_json(), "by_role": dict(sorted(by_role.items()))}
