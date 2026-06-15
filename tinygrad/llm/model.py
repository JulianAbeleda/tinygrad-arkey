from __future__ import annotations
import collections, functools, itertools, json, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, dtypes, getenv, function
from tinygrad.helpers import prod
from tinygrad.llm.gguf import gguf_load, gguf_load_with_metadata
from tinygrad.uop.ops import resolve

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

class Q4KPrimitiveStorage:
  __slots__ = ("words", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, words:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.words, self.source_bytes, self.persistent_bytes = words, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class Q6KPrimitiveStorage:
  __slots__ = ("halfs", "source_bytes", "persistent_bytes", "shared_bytes", "nonpersistent_bytes", "mode")
  def __init__(self, halfs:Tensor, source_bytes:int, persistent_bytes:int, mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.halfs, self.source_bytes, self.persistent_bytes = halfs, source_bytes, persistent_bytes
    self.shared_bytes, self.nonpersistent_bytes, self.mode = shared_bytes, nonpersistent_bytes, mode

class QKPrimitiveBudget:
  __slots__ = ("cap_bytes", "used_bytes", "strict")
  def __init__(self, cap_bytes:int|None=None, strict:bool=False):
    self.cap_bytes, self.used_bytes, self.strict = cap_bytes, 0, strict

  def reserve(self, name:str, bytes_needed:int, kind:str) -> bool:
    if bytes_needed < 0: raise ValueError(f"{kind} storage bytes must be non-negative for {name}: {bytes_needed}")
    if self.cap_bytes is not None and self.used_bytes + bytes_needed > self.cap_bytes:
      if self.strict:
        raise MemoryError(f"{kind} primitive storage cap exceeded for {name}: used={self.used_bytes} "
                          f"need={bytes_needed} cap={self.cap_bytes}")
      return False
    self.used_bytes += bytes_needed
    return True

class Q4KPrimitiveRegistry:
  __slots__ = ("linears",)
  def __init__(self, linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]|None=None): self.linears = linears or []

_VDOT_QUANT_CACHE: dict = {}  # E0: per-token q8 quant cache keyed by x.uop.key (q/k/v + gate/up share)

class Q4KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, words:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0, kernel_mode:str="partial"):
    if kernel_mode not in ("partial", "direct_out"): raise ValueError(f"unsupported Q4_K primitive kernel mode {kernel_mode!r}")
    if kernel_mode == "direct_out" and parts != 1: raise ValueError("Q4_K direct_out primitive requires parts=1")
    self.weight, self.bias = weight, bias
    self.q4k_storage = Q4KPrimitiveStorage(words, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.kernel_mode = kernel_mode
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def __call__(self, x:Tensor) -> Tensor:
    # This primitive is a decode GEMV path. Prefill, batching, and unsupported bias cases use the normal tinygrad graph.
    if not self.decode_enabled or self.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != self.in_features:
      return self._fallback(x)
    from extra.q4_k_gemv_primitive import q4k_gemv_kernel, q4k_gemv_partial_kernel
    x_vec = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
    words = self.q4k_storage.words.to(x.device).contiguous() if self.q4k_storage.mode == "q4_ondemand" else self.q4k_storage.words.to(x.device)
    if self.kernel_mode == "direct_out":
      out = Tensor.empty(self.out_features, dtype=dtypes.float32, device=x.device)
      got = out.custom_kernel(words, x_vec, fxn=q4k_gemv_kernel(self.out_features, self.in_features, "none", self.opts))[0]
      return got.reshape(1, 1, self.out_features)
    partials = Tensor.empty(self.out_features, self.parts, dtype=dtypes.float32, device=x.device)
    if getenv("Q4K_VDOT") and self.parts == 1:  # D1/E0: schedulable builtin v_dot4 (udot4) decode GEMV
      from extra.q4_k_gemv_primitive import q4k_q8_1_vdot_builtin_partial_kernel, q8_1_bias_pack_u32_kernel
      from extra.qk_layout import q8_1_quantize
      amort = bool(getenv("Q4K_VDOT_AMORT"))  # E0: quantize x ONCE/token, shared across q/k/v and gate/up
      ck = x.uop.key if amort else None
      cached = _VDOT_QUANT_CACHE.get(ck) if amort else None
      if cached is None:
        q, scales = q8_1_quantize(x_vec.cast(dtypes.float32))
        q_bias_words = Tensor.empty(self.in_features // 4, dtype=dtypes.uint32, device=x.device).custom_kernel(
          q, fxn=q8_1_bias_pack_u32_kernel(self.in_features))[0]
        if amort: _VDOT_QUANT_CACHE[ck] = (q_bias_words, scales); _VDOT_QUANT_CACHE["m"] = _VDOT_QUANT_CACHE.get("m", 0)+1
      else:
        q_bias_words, scales = cached; _VDOT_QUANT_CACHE["h"] = _VDOT_QUANT_CACHE.get("h", 0)+1
      partial = partials.custom_kernel(words, q_bias_words, scales,
        fxn=q4k_q8_1_vdot_builtin_partial_kernel(self.out_features, self.in_features, 1, "none", ()))[0]
      return partial.sum(axis=1).reshape(1, 1, self.out_features)
    partial = partials.custom_kernel(words, x_vec, fxn=q4k_gemv_partial_kernel(self.out_features, self.in_features, self.parts, "none", self.opts))[0]
    return partial.sum(axis=1).reshape(1, 1, self.out_features)

class Q4KFusedLinear:
  # B1 horizontal-fusion probe: one Q4_K GEMV over concatenated sibling weight rows (q/k/v or gate/up),
  # then split. Decode-only (uses the fused primitive when decode_enabled); prefill falls back to the
  # separate originals (whose own decode_enabled=False routes them to the dense path).
  def __init__(self, fused:Q4KPrimitiveLinear, originals:list[Q4KPrimitiveLinear], splits:list[int]):
    self.fused, self.originals, self.splits = fused, originals, splits
  def __call__(self, x:Tensor) -> list[Tensor]:
    if self.fused.decode_enabled:
      out = self.fused(x)  # (1,1,sum)
      res, c = [], 0
      for s in self.splits: res.append(out[..., c:c+s]); c += s
      return res
    return [l(x) for l in self.originals]

def _build_fused_q4k(linears:list[Q4KPrimitiveLinear], tag:str) -> Q4KFusedLinear:
  words = linears[0].q4k_storage.words.cat(*[l.q4k_storage.words for l in linears[1:]], dim=0).contiguous().realize()
  out_features, in_features, q4_bytes = sum(l.out_features for l in linears), linears[0].in_features, words.numel()*4
  fused = Q4KPrimitiveLinear(None, None, words, out_features, in_features, 1, linears[0].opts,
                             f"fused_{tag}", q4_bytes, q4_bytes, "sidecar", 0, 0)
  return Q4KFusedLinear(fused, linears, [l.out_features for l in linears])

def _install_q4k_fusions(model) -> None:
  # gated by Q4K_FUSE: fuse q/k/v->attn_qkv and gate/up->ffn_gateup on each dense block; register the
  # fused primitives so decode_enabled gets toggled per step.
  for block in getattr(model, "blk", []):
    if all(isinstance(getattr(block, n, None), Q4KPrimitiveLinear) for n in ("attn_q", "attn_k", "attn_v")):
      block.attn_qkv = _build_fused_q4k([block.attn_q, block.attn_k, block.attn_v], "qkv")
      model._q4k_linears.linears.append(block.attn_qkv.fused)
    if all(isinstance(getattr(block, n, None), Q4KPrimitiveLinear) for n in ("ffn_gate", "ffn_up")):
      block.ffn_gateup = _build_fused_q4k([block.ffn_gate, block.ffn_up], "gateup")
      model._q4k_linears.linears.append(block.ffn_gateup.fused)

class Q6KPrimitiveLinear:
  def __init__(self, weight:Tensor, bias:Tensor|None, halfs:Tensor, out_features:int, in_features:int, parts:int, opts:tuple,
               name:str, source_bytes:int, persistent_bytes:int, storage_mode:str,
               shared_bytes:int=0, nonpersistent_bytes:int=0):
    self.weight, self.bias = weight, bias
    self.q6k_storage = Q6KPrimitiveStorage(halfs, source_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes)
    self.out_features, self.in_features, self.parts, self.opts, self.name = out_features, in_features, parts, opts, name
    self.decode_enabled = False

  def _fallback(self, x:Tensor) -> Tensor:
    return x.linear(self.weight.transpose(), self.bias)

  def __call__(self, x:Tensor) -> Tensor:
    # Q6_K is currently only a measured win for the large decode GEMV ffn_down shape.
    if not self.decode_enabled or self.bias is not None or len(x.shape) != 3 or x.shape[0] != 1 or x.shape[-1] != self.in_features:
      return self._fallback(x)
    from extra.q6_k_gemv_primitive import q6k_gemv_partial_kernel
    x_vec = x[:, 0, :].reshape(self.in_features).cast(dtypes.float16).contiguous()
    partials = Tensor.empty(self.out_features, self.parts, dtype=dtypes.float32, device=x.device)
    partial = partials.custom_kernel(self.q6k_storage.halfs.to(x.device), x_vec,
                                     fxn=q6k_gemv_partial_kernel(self.out_features, self.in_features, self.parts, self.opts))[0]
    return partial.sum(axis=1).reshape(1, 1, self.out_features)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def _q4k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  if ".ffn_gate.weight" in name or ".ffn_up.weight" in name: return 1, ("LOCAL:0:64",)
  if ".ffn_down.weight" in name: return 4, ("LOCAL:0:32",)
  if ".attn_q.weight" in name or ".attn_output.weight" in name: return 1, ("LOCAL:0:64",)
  return None

def _q6k_policy(name:str) -> tuple[int, tuple[str, ...]]|None:
  # ffn_down wins decisively (the dominant Q6_K decode cost). attn_v/output were historically left to the
  # fused graph; re-measured 2026-06-15 on RX 7900 XTX they now also win (+5%, 50.8->53.1 tok/s, identical
  # output) -- kept behind Q6K_COVER_MORE pending broader (14B/32B) validation.
  if ".ffn_down.weight" in name: return 1, ("LOCAL:0:64",)
  if getenv("Q6K_COVER_MORE"):
    if ".attn_v.weight" in name: return 4, ("LOCAL:0:32",)
    if name == "output.weight": return 1, ("LOCAL:0:64",)
  return None

def _qk_policy_value(entry:dict) -> dict:
  cand = entry.get("candidate") or {}
  return {
    "winner": entry.get("winner"), "parts": int(cand.get("parts", 0)),
    "opts": tuple(cand.get("opts", ())), "family": cand.get("family", ""),
    "reduction": cand.get("reduction", ""),
    "policy_reason": entry.get("policy_reason", ""), "storage": entry.get("storage", {}),
  }

def _load_qk_generated_policy(path:str) -> dict:
  policy_path = pathlib.Path(path).expanduser()
  data = json.loads(policy_path.read_text())
  if data.get("kind") != "qk_generated_policy": raise ValueError(f"{policy_path} is not a QK generated policy cache")
  if data.get("generator_version") not in (0, 1):
    raise ValueError(f"{policy_path} has unsupported generator_version={data.get('generator_version')}")
  by_shape: dict[tuple[int, int, int], dict] = {}
  by_tensor: dict[tuple[str, int, int, int], dict] = {}
  for entry in data.get("entries", []):
    desc, cand = entry.get("descriptor", {}), entry.get("candidate") or {}
    key = (int(desc["ggml_type"]), int(desc["rows"]), int(desc["cols"]))
    value = _qk_policy_value(entry)
    if entry.get("scope") == "tensor":
      tensor = str(desc.get("tensor", ""))
      if not tensor: raise ValueError(f"{policy_path} has tensor-scoped entry without descriptor.tensor")
      tensor_key = (tensor, *key)
      if tensor_key in by_tensor and by_tensor[tensor_key] != value:
        raise ValueError(f"{policy_path} has conflicting tensor generated policy entries for key={tensor_key}: "
                         f"{by_tensor[tensor_key]} vs {value}")
      by_tensor[tensor_key] = value
    else:
      if key in by_shape and by_shape[key] != value:
        raise ValueError(f"{policy_path} has conflicting generated policy entries for key={key}: {by_shape[key]} vs {value}")
      by_shape[key] = value
  if not by_shape and not by_tensor: raise ValueError(f"{policy_path} contains no generated policy entries")
  return {"by_shape": by_shape, "by_tensor": by_tensor}

def _qk_generated_policy_len(policy:dict|None) -> int:
  if policy is None: return 0
  return len(policy.get("by_shape", {})) + len(policy.get("by_tensor", {}))

def _qk_generated_policy_entry(policy:dict|None, typ:int, rows:int, cols:int, name:str|None=None) -> dict|None:
  if policy is None: return None
  if name is not None and (entry:=policy.get("by_tensor", {}).get((name, typ, rows, cols))) is not None: return entry
  return policy.get("by_shape", {}).get((typ, rows, cols))

def _qk_storage_cap_from_env() -> int|None:
  raw = getenv("QK_PRIMITIVE_MAX_STORAGE_MB", "")
  if raw == "": return None
  cap = int(float(raw) * 1024 * 1024)
  if cap < 0: raise ValueError(f"QK_PRIMITIVE_MAX_STORAGE_MB must be non-negative, got {raw!r}")
  return cap

def _qk_storage_mode_from_env() -> str:
  mode = getenv("QK_PRIMITIVE_STORAGE", "sidecar")
  if mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"QK_PRIMITIVE_STORAGE must be sidecar, q4_ondemand, or shared, got {mode!r}")
  return mode

def _q6k_effective_storage_mode(requested_mode:str) -> str:
  # q4_ondemand is a Q4_K-only experiment. Q6_K stays persistent unless storage is shared.
  if requested_mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"unsupported QK primitive storage mode {requested_mode!r}")
  return "shared" if requested_mode == "shared" else "sidecar"

def _qk_storage_summary(linears:list[Q4KPrimitiveLinear|Q6KPrimitiveLinear]) -> dict:
  by_kind: collections.Counter[str] = collections.Counter()
  by_mode: collections.Counter[str] = collections.Counter()
  source_bytes = persistent_bytes = shared_bytes = nonpersistent_bytes = 0
  for linear in linears:
    storage = linear.q4k_storage if isinstance(linear, Q4KPrimitiveLinear) else linear.q6k_storage
    kind = "Q4K" if isinstance(linear, Q4KPrimitiveLinear) else "Q6K"
    by_kind[kind] += storage.persistent_bytes
    by_mode[storage.mode] += 1
    source_bytes += storage.source_bytes
    persistent_bytes += storage.persistent_bytes
    shared_bytes += getattr(storage, "shared_bytes", 0)
    nonpersistent_bytes += getattr(storage, "nonpersistent_bytes", 0)
  return {
    "source_bytes": source_bytes, "persistent_bytes": persistent_bytes,
    "shared_bytes": shared_bytes, "nonpersistent_bytes": nonpersistent_bytes,
    "by_kind": dict(sorted(by_kind.items())), "by_mode": dict(sorted(by_mode.items())),
  }

def _shared_packed_view(meta:dict, byte_start:int, nbytes:int, dtype) -> Tensor:
  raw = meta.get("raw_tensor")
  if raw is None: raise ValueError("shared QK primitive storage requires gguf_load_with_metadata raw_tensor")
  if byte_start % dtype.itemsize != 0 or nbytes % dtype.itemsize != 0:
    raise ValueError(f"shared QK primitive storage requires uint{dtype.itemsize*8}-aligned range: "
                     f"byte_start={byte_start} nbytes={nbytes}")
  raw_view = raw[byte_start:byte_start+nbytes].uop.buffer
  return Tensor(UOp.from_buffer(raw_view.view(nbytes//dtype.itemsize, dtype, 0), raw.device))

def _module_at(root, path:str):
  obj = root
  for part in path.split("."):
    obj = obj[int(part)] if isinstance(obj, list) and part.isdigit() else getattr(obj, part)
  return obj

def _set_module_at(root, path:str, value) -> None:
  if "." not in path: setattr(root, path, value); return  # top-level module (e.g. output)
  parent_path, attr = path.rsplit(".", 1)
  parent = _module_at(root, parent_path)
  if isinstance(parent, list) and attr.isdigit(): parent[int(attr)] = value
  else: setattr(parent, attr, value)

def _install_q4k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar") -> list[Q4KPrimitiveLinear]:
  from extra.q4_k_gemv_primitive import parse_opt
  supported_generated_families = {"q4_k_packed_u32", "q4_k_packed_u32_direct"}
  raw_words = Tensor(gguf, dtype=dtypes.uint32)
  installed: list[Q4KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  debug = bool(getenv("Q4K_PRIMITIVE_DEBUG", getenv("QK_GENERATED_POLICY_DEBUG", 0)))
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 12:
      skipped["not_q4_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if (policy := _q4k_policy(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      parts, opt_specs = policy
      kernel_mode = "partial"
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] not in supported_generated_families:
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
      kernel_mode = "direct_out" if policy_entry["family"] == "q4_k_packed_u32_direct" or policy_entry.get("reduction") == "direct_out" else "partial"
      if kernel_mode == "direct_out" and parts != 1:
        skipped["policy_invalid_direct_parts"] += 1
        continue
    byte_start = meta["data_start"] + off
    if byte_start % 4 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")]
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q4_bytes = prod(dims) // 256 * 144
    persistent_bytes = q4_bytes if storage_mode == "sidecar" else 0
    if not budget.reserve(name, persistent_bytes, "Q4_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    source = raw_words[byte_start//4:byte_start//4+q4_bytes//4]
    if storage_mode == "shared":
      words, shared_bytes, nonpersistent_bytes = _shared_packed_view(meta, byte_start, q4_bytes, dtypes.uint32), q4_bytes, 0
    else:
      words = source.contiguous() if storage_mode == "q4_ondemand" else source.to(None).contiguous().realize()
      shared_bytes, nonpersistent_bytes = 0, q4_bytes if storage_mode == "q4_ondemand" else 0
    q4k_linear = Q4KPrimitiveLinear(module.weight, module.bias, words, rows, cols, parts, tuple(parse_opt(x) for x in opt_specs), name,
                                    q4_bytes, persistent_bytes, storage_mode, shared_bytes, nonpersistent_bytes, kernel_mode=kernel_mode)
    _set_module_at(model, module_path, q4k_linear)
    installed.append(q4k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:mode={x.kernel_mode}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q4K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q4K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed

def _install_q6k_primitives(model, gguf:pathlib.Path, meta:dict, generated_policy:dict|None=None,
                            budget:QKPrimitiveBudget|None=None, storage_mode:str="sidecar") -> list[Q6KPrimitiveLinear]:
  from extra.q6_k_gemv_primitive import parse_opt
  raw_halfs = Tensor(gguf, dtype=dtypes.uint16)
  installed: list[Q6KPrimitiveLinear] = []
  skipped: collections.Counter[str] = collections.Counter()
  budget = budget or QKPrimitiveBudget()
  debug = bool(getenv("Q6K_PRIMITIVE_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("QK_GENERATED_POLICY_DEBUG", 0))))
  for name, dims, typ, off in meta["tensor_infos"]:
    if typ != 14:
      skipped["not_q6_k"] += 1
      continue
    if len(dims) != 2:
      skipped["not_2d"] += 1
      continue
    if not name.endswith(".weight"):
      skipped["not_weight"] += 1
      continue
    rows, cols = tuple(reversed(dims))
    if generated_policy is None:
      if (policy := _q6k_policy(name)) is None:
        skipped["policy_fallback"] += 1
        continue
      parts, opt_specs = policy
    else:
      if (policy_entry := _qk_generated_policy_entry(generated_policy, typ, rows, cols, name)) is None:
        skipped["policy_missing"] += 1
        continue
      if policy_entry["winner"] == "fused_graph":
        skipped["policy_fused"] += 1
        continue
      if policy_entry["family"] != "q6_k_packed_u16":
        skipped["policy_unsupported"] += 1
        continue
      parts, opt_specs = policy_entry["parts"], policy_entry["opts"]
    byte_start = meta["data_start"] + off
    if byte_start % 2 != 0:
      skipped["misaligned"] += 1
      continue
    module_path = name[:-len(".weight")]
    try: module = _module_at(model, module_path)
    except (AttributeError, IndexError, ValueError):
      skipped["missing_module"] += 1
      continue
    if not hasattr(module, "weight"):
      skipped["missing_weight"] += 1
      continue
    if getattr(module, "bias", None) is not None:
      skipped["bias"] += 1
      continue
    q6_bytes = prod(dims) // 256 * 210
    persistent_bytes = 0 if storage_mode == "shared" else q6_bytes
    if not budget.reserve(name, persistent_bytes, "Q6_K"):
      skipped["runtime_storage_cap"] += 1
      continue
    if storage_mode == "shared":
      halfs, shared_bytes = _shared_packed_view(meta, byte_start, q6_bytes, dtypes.uint16), q6_bytes
    else:
      halfs, shared_bytes = raw_halfs[byte_start//2:byte_start//2+q6_bytes//2].to(None).contiguous().realize(), 0
    q6k_linear = Q6KPrimitiveLinear(module.weight, module.bias, halfs, rows, cols, parts, tuple(parse_opt(x) for x in opt_specs), name,
                                    q6_bytes, persistent_bytes, storage_mode, shared_bytes, 0)
    _set_module_at(model, module_path, q6k_linear)
    installed.append(q6k_linear)
  if debug:
    skipped_s = " ".join(f"{k}={v}" for k, v in sorted(skipped.items()))
    installed_s = " ".join(f"{x.name}:parts={x.parts}:opts={[str(o) for o in x.opts]}" for x in installed[:8])
    more_s = f" ...+{len(installed)-8}" if len(installed) > 8 else ""
    summary = _qk_storage_summary(installed)
    cap = -1 if budget.cap_bytes is None else budget.cap_bytes
    print(f"Q6K_PRIMITIVE_DEBUG installed={len(installed)} skipped_total={sum(skipped.values())} {skipped_s} "
          f"source_bytes={summary['source_bytes']} storage_bytes={summary['persistent_bytes']} "
          f"shared_bytes={summary['shared_bytes']} nonpersistent_bytes={summary['nonpersistent_bytes']} "
          f"runtime_cap_bytes={cap} runtime_cap_used_bytes={budget.used_bytes} storage_mode={storage_mode}")
    if installed: print(f"Q6K_PRIMITIVE_DEBUG installed_linears {installed_s}{more_s}")
  return installed

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    if hasattr(self, "ffn_gateup"):  # B1 fused gate/up
      gate, up = self.ffn_gateup(x)
      return self.ffn_down(gate.silu().contiguous() * up)
    if getenv("Q4K_UNFUSE"):  # run FFN matmuls in fp16 so RDNA3 WMMA tensor cores can apply (minimal-overhead)
      xh = x.cast(dtypes.float16)
      return self.ffn_down((self.ffn_gate(xh).silu().contiguous() * self.ffn_up(xh)).cast(dtypes.float16))
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp):
      h =     x + self._attention(self.attn_norm(x), start_pos)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    if hasattr(self, "attn_qkv"): q, k, v = self.attn_qkv(x)  # B1 fused q/k/v
    else: q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    q = apply_rope(q[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
    k = assigned_kv[0, :, :, 0:start_pos+T, :]
    v = assigned_kv[1, :, :, 0:start_pos+T, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)     # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    return self.attn_output(attn if not self.config.attn_output_gate else (attn * gate.sigmoid()))

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # TODO: how is the dtype of this determined?
      self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._q4k_linears = Q4KPrimitiveRegistry()
    # we specialize the JIT for prefill and rollout
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    return self.output(self.output_norm(x))

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    logits = self.logits(tokens, start_pos)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    is_prefill = resolve(tokens.shape[1] != 1)
    if getenv("Q4K_VDOT_AMORT"): _VDOT_QUANT_CACHE.clear()  # E0: fresh quant cache per forward/trace
    for q4k_linear in self._q4k_linears.linears: q4k_linear.decode_enabled = not is_prefill
    return (self.prefill_jit if is_prefill else self.rollout_jit)(tokens.contiguous(), start_pos, temperature)

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:int|None=None,
                realize=bool(getenv("REALIZE", 0))) -> tuple[Transformer, dict]:
    # TODO: remove the need for copy to default device
    use_q4k_primitive = bool(getenv("Q4K_PRIMITIVE", 0))
    # Q6_K (ffn_down etc. in mixed-quant Q4_K_M) defaults ON with Q4K_PRIMITIVE: it's the decode bottleneck
    # otherwise (the slow fp-dequant fallback ~= 59% of GPU work), Q6_K dequant is exact (identical output),
    # and enabling it is a ~2.2x decode win. Set Q6K_PRIMITIVE=0 to opt out.
    use_q6k_primitive = bool(getenv("Q6K_PRIMITIVE", 1 if use_q4k_primitive else 0))
    qk_generated_policy_path = getenv("QK_GENERATED_POLICY", "")
    use_qk_generated_policy = bool(qk_generated_policy_path)
    if (use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy) and isinstance(gguf, Tensor):
      raise ValueError("quant primitive paths require a GGUF path, not a preloaded Tensor")
    if use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy:
      kv, state_dict, q4k_meta = gguf_load_with_metadata(gguf)
    else:
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
      q4k_meta = None

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    max_context = min(max_context, kv[f'{arch}.context_length']) if max_context is not None else kv[f'{arch}.context_length']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict)
    model = Transformer(config)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if q4k_meta is not None:
      primitive_linears = []
      primitive_budget = QKPrimitiveBudget(_qk_storage_cap_from_env(), bool(getenv("QK_GENERATED_POLICY_STRICT", 0)))
      q4_storage_mode = _qk_storage_mode_from_env()
      q6_storage_mode = _q6k_effective_storage_mode(q4_storage_mode)
      generated_policy = _load_qk_generated_policy(qk_generated_policy_path) if use_qk_generated_policy else None
      if generated_policy is not None:
        if bool(getenv("QK_GENERATED_POLICY_DEBUG", 0)):
          print(f"QK_GENERATED_POLICY_DEBUG loaded={qk_generated_policy_path} entries={_qk_generated_policy_len(generated_policy)}")
        primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q4_storage_mode)
        primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q6_storage_mode)
      else:
        if use_q4k_primitive: primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q4_storage_mode)
        if use_q6k_primitive: primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q6_storage_mode)
      if bool(getenv("QK_GENERATED_POLICY_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("Q6K_PRIMITIVE_DEBUG", 0)))):
        summary = _qk_storage_summary(primitive_linears)
        cap = -1 if primitive_budget.cap_bytes is None else primitive_budget.cap_bytes
        by_kind_s = ",".join(f"{k}:{v}" for k, v in summary["by_kind"].items()) or "none"
        by_mode_s = ",".join(f"{k}:{v}" for k, v in summary["by_mode"].items()) or "none"
        print(f"QK_PRIMITIVE_STORAGE_DEBUG installed={len(primitive_linears)} source_bytes={summary['source_bytes']} "
              f"storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
              f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} "
              f"runtime_cap_used_bytes={primitive_budget.used_bytes} by_kind={by_kind_s} by_mode={by_mode_s} "
              f"requested_storage_mode={q4_storage_mode} q4_effective_storage_mode={q4_storage_mode} "
              f"q6_effective_storage_mode={q6_storage_mode}")
      if primitive_linears: model._q4k_linears = Q4KPrimitiveRegistry(primitive_linears)
      if primitive_linears and getenv("Q4K_FUSE"): _install_q4k_fusions(model)  # B1 horizontal-fusion probe
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    out, prompt_len = None, len(tokens)
    while len(tokens) < self.max_context:
      sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
      out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp).realize()
      start_pos += nt.val
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
