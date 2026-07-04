from __future__ import annotations
import collections, pathlib
from dataclasses import dataclass
from tinygrad import Tensor, UOp, dtypes, getenv, Device
from tinygrad.helpers import prod
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.decode_routes import q4k_primitive_linear_call, q6k_primitive_linear_call
from tinygrad.llm.route_policy import _q4k_policy, _q6k_policy, _qk_generated_policy_entry


def _qk_amd_gfx1100_arch_ok() -> bool:
  try: return Device.DEFAULT == "AMD" and "gfx1100" in str(getattr(Device["AMD"], "arch", ""))
  except Exception: return False
QK_AMD_GFX1100_ARCH_OK = _qk_amd_gfx1100_arch_ok()

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

  def prefill_packed_weight(self) -> Tensor:
    if self.q4k_storage.mode == "sidecar" or bool(getenv("PREFILL_PACKED_STREAM", 0)): return self.q4k_storage.words
    if not hasattr(self, "_prefill_q4k_words"):
      self._prefill_q4k_words = self.q4k_storage.words.clone().realize()
    return self._prefill_q4k_words

  def prefill_fp16_weight(self) -> Tensor:
    raw = self.q4k_storage.words.bitcast(dtypes.uint8).reshape(-1)
    return ggml_data_to_tensor(raw, self.out_features * self.in_features, 12).reshape(
      self.out_features, self.in_features).cast(dtypes.float16).contiguous()

  def __call__(self, x:Tensor) -> Tensor:
    return q4k_primitive_linear_call(self, x, self._fallback, QK_AMD_GFX1100_ARCH_OK)

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

  def prefill_packed_weight(self) -> Tensor:
    if self.q6k_storage.mode == "sidecar" or bool(getenv("PREFILL_PACKED_STREAM", 0)): return self.q6k_storage.halfs
    if not hasattr(self, "_prefill_q6k_halfs"):
      self._prefill_q6k_halfs = self.q6k_storage.halfs.clone().realize()
    return self._prefill_q6k_halfs

  def prefill_fp16_weight(self) -> Tensor:
    raw = self.q6k_storage.halfs.bitcast(dtypes.uint8).reshape(-1)
    return ggml_data_to_tensor(raw, self.out_features * self.in_features, 14).reshape(
      self.out_features, self.in_features).cast(dtypes.float16).contiguous()

  def __call__(self, x:Tensor) -> Tensor:
    return q6k_primitive_linear_call(self, x, self._fallback, QK_AMD_GFX1100_ARCH_OK)

def _qk_storage_cap_from_env() -> int|None:
  raw = getenv("QK_PRIMITIVE_MAX_STORAGE_MB", "")
  if raw == "": return None
  cap = int(float(raw) * 1024 * 1024)
  if cap < 0: raise ValueError(f"QK_PRIMITIVE_MAX_STORAGE_MB must be non-negative, got {raw!r}")
  return cap

def _qk_storage_mode_from_env(default:str="sidecar") -> str:
  mode = getenv("QK_PRIMITIVE_STORAGE", default)
  if mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"QK_PRIMITIVE_STORAGE must be sidecar, q4_ondemand, or shared, got {mode!r}")
  return mode

def _q6k_effective_storage_mode(requested_mode:str) -> str:
  # q4_ondemand is a Q4_K-only experiment. Q6_K stays persistent unless storage is shared.
  if requested_mode not in ("sidecar", "q4_ondemand", "shared"):
    raise ValueError(f"unsupported QK primitive storage mode {requested_mode!r}")
  return "shared" if requested_mode == "shared" else "sidecar"

@dataclass(frozen=True)
class QKConfig:
  """Single authority for the QK primitive *install* config read from the environment
  inside `Transformer.from_gguf` once primitives are active (i.e. the flags consumed
  under `if q4k_meta is not None`). Centralizes the reads + validation that were
  scattered as `getenv` calls so invalid QK runtime config is rejected in one place.

  Scope is deliberately the install config, not everything QK-shaped:
  - Activation gating (`Q4K_PRIMITIVE`/`Q6K_PRIMITIVE`/`QK_GENERATED_POLICY`) stays at
    the from_gguf gate -- its auto-default is coupled to the gguf source + device, a
    separate (runtime, not env-only) concern.
  - Forward-pass probe flags (`Q4K_VDOT`/`Q4K_VDOT_AMORT`/`Q6K_COVER_MORE`/`Q4K_UNFUSE`/
    `FLASH_DECODE`/`FLASH_L`) are read per-call at their own sites; folding them here
    would change when they are read.

  Built via `from_env(storage_default=...)` at the top of the active-primitive block,
  so every field is read exactly when the original scattered reads were (when the block
  is entered) -- a behaviour-preserving centralization."""
  generated_policy_strict: bool
  max_storage_bytes: int | None
  storage_mode: str
  q6_storage_mode: str
  policy_debug: bool
  storage_debug: bool
  demote_q6k_ffndown: bool
  demote_targets: tuple[str, ...]
  fuse_q4k: bool

  @staticmethod
  def from_env(*, storage_default:str) -> "QKConfig":
    # Read order mirrors the original sites: cap + strict (QKPrimitiveBudget), then the
    # validated storage mode, then the debug/probe flags -- so a first-raise on invalid
    # input lands on the same variable as before.
    max_storage_bytes = _qk_storage_cap_from_env()
    generated_policy_strict = bool(getenv("QK_GENERATED_POLICY_STRICT", 0))
    storage_mode = _qk_storage_mode_from_env(storage_default)
    # B3: per-tensor Q6->Q4 demotion. QK_DEMOTE_TENSORS (comma-sep name substrings) generalizes the
    # single-tensor Q6K_DEMOTE_FFNDOWN flag; the flag stays as the ffn_down shortcut for back-compat.
    demote_q6k_ffndown = bool(getenv("Q6K_DEMOTE_FFNDOWN"))
    explicit = tuple(t for t in getenv("QK_DEMOTE_TENSORS", "").replace(" ", "").split(",") if t)
    demote_targets = explicit or (("ffn_down",) if demote_q6k_ffndown else ())
    return QKConfig(
      generated_policy_strict=generated_policy_strict,
      max_storage_bytes=max_storage_bytes,
      storage_mode=storage_mode,
      q6_storage_mode=_q6k_effective_storage_mode(storage_mode),
      policy_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", 0)),
      storage_debug=bool(getenv("QK_GENERATED_POLICY_DEBUG", getenv("Q4K_PRIMITIVE_DEBUG", getenv("Q6K_PRIMITIVE_DEBUG", 0)))),
      demote_q6k_ffndown=demote_q6k_ffndown,
      demote_targets=demote_targets,
      fuse_q4k=bool(getenv("Q4K_FUSE")))

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
    q4k_linear = Q4KPrimitiveLinear(module.weight, module.bias, words, rows, cols, parts, tuple(qk_ops.q4k_parse_opt(x) for x in opt_specs), name,
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
    q6k_linear = Q6KPrimitiveLinear(module.weight, module.bias, halfs, rows, cols, parts, tuple(qk_ops.q6k_parse_opt(x) for x in opt_specs), name,
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

def _demote_q6k_to_q4(model, linears:list, targets:tuple[str, ...]) -> list:
  # B3: re-quantize over-provisioned Q6_K tensors to Q4_K (offline quantizer; ffn_down measured ~free
  # quality, fewer per-token bytes -> an operating point llama.cpp's fixed Q4_K_M doesn't offer). `targets`
  # is a tuple of tensor-name substrings (e.g. ("ffn_down","attn_v")) selected by the demotion search;
  # each demoted tensor's (parts, opts) reuse _q4k_policy, with a shape-based fallback for roles it omits.
  out = []
  for lin in linears:
    if isinstance(lin, Q6KPrimitiveLinear) and any(t in lin.name for t in targets):
      pol = _q4k_policy(lin.name) or ((4, ("LOCAL:0:32",)) if lin.out_features > 8192 else (1, ("LOCAL:0:64",)))
      parts, opt_strs = pol
      opts = tuple(qk_ops.q4k_parse_opt(x) for x in opt_strs)
      words = Tensor(qk_ops.quantize_q4_k(lin.weight.numpy())).to(None).contiguous().realize()
      q4_bytes = lin.out_features * lin.in_features // 256 * 144
      q4 = Q4KPrimitiveLinear(lin.weight, lin.bias, words, lin.out_features, lin.in_features, parts, opts,
                              lin.name, q4_bytes, q4_bytes, "sidecar")
      _set_module_at(model, lin.name[:-len(".weight")], q4)
      out.append(q4)
    else:
      out.append(lin)
  return out
