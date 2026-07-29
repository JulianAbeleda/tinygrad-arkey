from __future__ import annotations
# Process-wide LLM runtime lifecycle and registry state. This module deliberately
# depends on the model domain but never on the HTTP/CLI presentation layer.
import os, sys, codecs, typing, re, unicodedata, json, time, pathlib, threading, gc
from tinygrad import Device
from tinygrad.helpers import DEBUG, Context, fetch, partition, stderr_log
from tinygrad.llm.model import Transformer

class SimpleTokenizer:
  def __init__(self, normal_tokens:dict[str, int], special_tokens:dict[str, int], preset:str="llama3",
               bos_id:int|None=None, eos_id:int=0, eot_id:int|None=None):
    preset = {"qwen35":"qwen2","qwen35moe":"qwen2"}.get(preset, preset)
    if preset not in ("llama3","llama-v3","llama-bpe","qwen2","olmo","kimi-k2","tekken","glm4"):
      raise ValueError(f"Invalid tokenizer preset '{preset}'")
    # https://github.com/openai/gpt-2/blob/9b63575ef42771a015060c964af2c3da4cf7c8ab/src/encoder.py#L9
    bs = [*range(33, 127), *range(161, 173), *range(174, 256)]  # bytes that map to themselves
    self._byte_decoder = {chr(b): b for b in bs} | {chr(256+i): b for i,b in enumerate(b for b in range(256) if b not in bs)}

    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L286
    # 0x323b0 is one past the max codepoint in unicode categories L/N/Z (0x323af is max L)
    def ucat_range(pre: str): return "".join(re.escape(chr(cp)) for cp in range(0x323b0) if unicodedata.category(chr(cp)).startswith(pre))
    r_ws, r_p_N, r_p_L = r"\t\n\x0b\x0c\r\x85" + ucat_range("Z"), ucat_range("N"), ucat_range("L")
    self._split_to_word = re.compile("(?i:'s|'t|'re|'ve|'m|'ll|'d)|" + \
      f"[^\\r\\n{r_p_N}{r_p_L}]?[{r_p_L}]+|[{r_p_N}]{{1,3}}| ?[^{r_ws}{r_p_N}{r_p_L}]+[\\r\\n]*|[{r_ws}]*[\\r\\n]+|[{r_ws}]+(?![^{r_ws}])|[{r_ws}]+")
    self._split_to_sentence = re.compile("|".join(re.escape(tok) for tok in special_tokens.keys()) if special_tokens else r"(?!)")

    self._normal_tokens = {bytes(self._byte_decoder[c] for c in tok): tid for tok, tid in normal_tokens.items()}
    self._special_tokens = special_tokens
    self._tok2bytes = {tid: tok for tok, tid in self._normal_tokens.items()} | {tid: tok.encode() for tok, tid in self._special_tokens.items()}
    self.preset = preset
    self.bos_id, self.eos_id, self.eot_id = bos_id, eos_id, eot_id

  @staticmethod
  def from_gguf_kv(kv:dict):
    # https://github.com/ggml-org/llama.cpp/blob/94933c8c2eeaa9a7983e3f6c08af76bd86724094/src/llama-vocab.cpp#L1818-L1820
    vocab: typing.Iterable[tuple[str, int]] = ((tok, idx) for idx, tok in enumerate(kv["tokenizer.ggml.tokens"]))
    normal_tokens, special_tokens = partition(vocab, lambda e: kv["tokenizer.ggml.token_type"][e[1]] == 1)
    return SimpleTokenizer(dict(normal_tokens), dict(special_tokens), kv["tokenizer.ggml.pre"],
      bos_id=kv.get('tokenizer.ggml.bos_token_id') if kv.get('tokenizer.ggml.add_bos_token', True) else None,
      eos_id=kv.get('tokenizer.ggml.eos_token_id', 0), eot_id=kv.get('tokenizer.ggml.eot_token_id'))

  def _encode_word(self, word:bytes) -> list[int]:
    if (early_token:=self._normal_tokens.get(word)) is not None: return [early_token]
    parts = [bytes([b]) for b in word]
    # greedily merge any parts that we can
    while True:
      i = min([(sys.maxsize, -1)] + [(self._normal_tokens.get(parts[j]+parts[j+1], sys.maxsize), j) for j in range(len(parts)-1)])[1]
      if i == -1: break
      parts[i:i+2] = [parts[i] + parts[i+1]]
    try: return [self._normal_tokens[p] for p in parts]
    except KeyError: raise RuntimeError("token not found")
  def _encode_sentence(self, chunk:str) -> list[int]:
    return [tok for word in self._split_to_word.findall(chunk) for tok in self._encode_word(word.encode())]
  def encode(self, text:str) -> list[int]:
    tokens: list[int] = []
    pos = 0
    for match in self._split_to_sentence.finditer(text):
      tokens.extend(self._encode_sentence(text[pos:match.start(0)]) + [self._special_tokens[text[match.start(0):match.end(0)]]])
      pos = match.end(0)
    return tokens + self._encode_sentence(text[pos:])

  def decode(self, ids:list[int]) -> str: return b''.join(self._tok2bytes[tid] for tid in ids).decode(errors='replace')
  def stream_decoder(self) -> typing.Callable[..., str]:
    dec = codecs.getincrementaldecoder('utf-8')('replace')
    def _decode(tid:int|None=None) -> str: return dec.decode(self._tok2bytes[tid]) if tid is not None else dec.decode(b'', final=True)
    return _decode
  def role(self, role:str):
    if self.preset == 'olmo': return self.encode("<|" + role + "|>\n")  # OLMoE Instruct format
    if self.preset == 'kimi-k2': return self.encode("<|im_" + role + "|>" + role + "<|im_middle|>")
    if self.preset == 'qwen2': return self.encode("<|im_start|>" + role + "\n")
    if self.preset == 'glm4': return self.encode("<|" + role + "|>")
    if self.preset == 'tekken':
      if role == 'user': return self.encode("[INST]")
      if role == 'assistant': return []
      raise ValueError(f"Unsupported role '{role}' for tokenizer preset '{self.preset}'")
    return self.encode("<|start_header_id|>" + role + "<|end_header_id|>\n\n")
  def end_turn(self):
    if self.preset == 'olmo': return self.encode("\n")
    if self.preset == 'kimi-k2': return [self.eos_id]
    if self.preset == 'qwen2': return [self.eos_id] + self.encode("\n")
    if self.preset == 'glm4': return []
    if self.preset == 'tekken': return self.encode("[/INST]")
    return [self.eos_id]
  def prefix(self) -> list[int]:
    return ([] if self.bos_id is None else [self.bos_id]) + (self.encode("<sop>") if self.preset == 'glm4' else [])
  def is_end(self, token_id:int) -> bool: return token_id in (self.eos_id, self.eot_id)

models = {
  "llama3.2:1b": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf",
  "llama3.2:1b-q4": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
  "llama3.2:3b": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q6_K.gguf",
  "llama3.2:3b-f16": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-f16.gguf",
  "llama3.1:8b": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
  "qwen3:0.6b": "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
  "qwen3:1.7b": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
  "qwen3:8b": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
  "qwen3:30b-a3b": "https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf",
  "qwen3.5:0.8b": "https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf",
  "qwen3.5:4b": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf",
  "qwen3.5:9b": "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf",
  "qwen3.5:27b": "https://huggingface.co/unsloth/Qwen3.5-27B-GGUF/resolve/main/Qwen3.5-27B-Q4_K_M.gguf",
  "qwen3.5:35b-a3b": "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf",
  "olmoe": "https://huggingface.co/allenai/OLMoE-1B-7B-0924-Instruct-GGUF/resolve/main/olmoe-1b-7b-0924-instruct-q4_k_m.gguf",
  "moonlight": "https://huggingface.co/gabriellarson/Moonlight-16B-A3B-Instruct-GGUF/resolve/main/Moonlight-16B-A3B-Instruct-Q4_K_M.gguf",
  "glm-4.7-flash": "https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/resolve/main/GLM-4.7-Flash-Q4_K_M.gguf",
}

# *** tinygrad runtime / client separation server ***
#
# Two HTTP surfaces on one process (see docs/tinygrad-runtime-client-separation-roadmap-20260630.md):
#   /v1/*       OpenAI-compatible inference surface for clients (OpenCode, AI-SDK, llama.cpp-style tooling)
#   /runtime/*  lifecycle + introspection controls for the proprietary app / local operator tooling
# The runtime owns model load, tokenizer, KV cache, prefill/decode, sampling, GPU memory. It does NOT own
# sessions, repo context, tools, or prompt packing -- those live above this boundary in the client.

# Phase R2 structured error contract: error type -> default HTTP status. Every failure returns
# {"error": {"message", "type", "code", "request_id"}} so OpenAI-compatible clients can parse it.
RUNTIME_ERROR_STATUS = {
  "model_not_loaded": 409,
  "unknown_model": 404,
  "context_length_exceeded": 400,
  "generation_cancelled": 499,
  "runtime_busy": 429,
  "invalid_request": 400,
  "internal_runtime_error": 500,
}

class RuntimeFault(Exception):
  def __init__(self, err_type:str, message:str):
    self.err_type, self.message = err_type, message
    super().__init__(message)

def _quant_from_name(name:str|None) -> str|None:
  if not name: return None
  m = re.search(r'(IQ\d+[A-Z0-9_]*|Q\d+_K(?:_[A-Z]+)?|Q\d+_\d+|BF16|F16|F32)', name, re.I)
  return m.group(1).upper() if m else None

def _device_target() -> str|None:
  # best-effort GPU target (e.g. gfx1100); may be None on backends that don't expose it
  try:
    dev = Device[Device.DEFAULT]
    for obj in (getattr(dev, 'renderer', None), dev):
      if obj is None: continue
      for attr in ('arch', 'target', 'agent_name'):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v: return v
  except Exception: pass
  return None

DEFAULT_REGISTRY_PATH = pathlib.Path(os.environ.get("TINYGRAD_RUNTIME_MODELS",
  os.path.expanduser("~/.config/tinygrad/runtime_models.json")))

def build_registry(builtin:dict[str, str], registry_path:pathlib.Path=DEFAULT_REGISTRY_PATH) -> dict[str, dict]:
  # Phase R3: the registry is the source of truth for /v1/models and /runtime/load. Built-in aliases seed it
  # (fallback compat); an optional JSON file at registry_path extends/overrides rows.
  target = _device_target()
  reg: dict[str, dict] = {}
  for mid, src in builtin.items():
    reg[mid] = {"id": mid, "path": src, "architecture": None, "quant": _quant_from_name(src),
                "default_context": 4096, "backend": Device.DEFAULT, "target": target,
                "status": "available", "tags": ["builtin"], "enabled": True}
  if registry_path.exists():
    try:
      data = json.loads(registry_path.read_text())
      rows = data.get("models", data) if isinstance(data, dict) else data
      for row in rows:
        mid = row["id"]
        base = reg.get(mid, {"id": mid, "path": None, "architecture": None, "quant": None, "default_context": 4096,
                             "backend": Device.DEFAULT, "target": target, "status": "available", "tags": [], "enabled": True})
        base.update({k: v for k, v in row.items() if v is not None})
        if base.get("path") and not base.get("quant"): base["quant"] = _quant_from_name(base["path"])
        reg[mid] = base
      stderr_log(f"runtime: loaded {len(rows)} model row(s) from {registry_path}\n")
    except Exception as e:
      stderr_log(f"runtime: failed to read registry {registry_path}: {e}\n")
  return reg

class RuntimeState:
  """Central, process-wide runtime state. One loaded model per process (first-cut policy, see roadmap R4/R10)."""
  def __init__(self, registry:dict[str, dict], remote_metrics:bool=False):
    self.registry = registry
    self.remote_metrics = remote_metrics
    self.model: Transformer|None = None
    self.model_id: str|None = None
    self.model_name: str|None = None
    self.tok: SimpleTokenizer|None = None
    self.path: str|None = None
    self.max_context: int|None = None
    self.architecture: str|None = None
    self.quant: str|None = None
    self.backend = Device.DEFAULT
    self.target = _device_target()
    self.warmup_done = False
    self.last_warmup_s: float|None = None
    self.last_warmup_compiles: int|None = None
    self.load_count = 0
    self.request_count = 0
    self.last_error: str|None = None
    # gen_lock serializes generation AND lifecycle mutation: the server is threaded (so /runtime/status and
    # /runtime/cancel stay responsive during generation), and a single shared model/KV cache must never be
    # touched by two requests at once. Non-blocking acquire -> runtime_busy.
    self.gen_lock = threading.Lock()
    self.cancel_event = threading.Event()
    self.metrics = {"last_prompt_tokens": None, "last_completion_tokens": None, "last_cached_prefix_tokens": None,
                    "last_prefill_tok_s": None, "last_decode_tok_s": None, "last_finish_reason": None}

  @property
  def loaded(self) -> bool: return self.model is not None

  def _resolve_source(self, model_ref:str|None, path:str|None) -> tuple[str|None, dict|None]:
    if path: return path, (self.registry.get(model_ref) if model_ref else None)
    if model_ref is None: return None, None
    row = self.registry.get(model_ref)
    if row is not None and row.get("path"): return row["path"], row
    if model_ref in models: return models[model_ref], None
    return None, None

  def _register_loaded(self, kv:dict, source:str):
    self.architecture = kv.get('general.architecture')
    self.quant = _quant_from_name(source)
    row = self.registry.get(self.model_id)
    if row is None:
      self.registry[self.model_id] = {"id": self.model_id, "path": source, "architecture": self.architecture,
        "quant": self.quant, "default_context": self.max_context, "backend": self.backend, "target": self.target,
        "status": "loaded", "tags": [], "enabled": True}
    else:
      row["status"] = "loaded"
      if row.get("architecture") is None: row["architecture"] = self.architecture
      if row.get("quant") is None: row["quant"] = self.quant

  def adopt(self, model:Transformer, kv:dict, tok:SimpleTokenizer, model_id:str, model_name:str, source:str,
            warmup_done:bool=False, warmup_s:float|None=None, warmup_compiles:int|None=None):
    """Take ownership of an already-loaded model (used by startup preload and by load())."""
    self.model, self.tok = model, tok
    self.model_id, self.model_name, self.path = model_id, model_name, source
    self.max_context = model.max_context
    self.warmup_done, self.last_warmup_s, self.last_warmup_compiles = warmup_done, warmup_s, warmup_compiles
    self.last_error = None
    self.load_count += 1
    self.metrics = {k: None for k in self.metrics}
    self._register_loaded(kv, source)

  def load(self, model_ref:str|None, path:str|None=None, max_context:int|None=None, warmup:bool=True) -> dict:
    source, row = self._resolve_source(model_ref, path)
    if source is None:
      raise RuntimeFault("unknown_model", f"model '{model_ref}' is not in the registry and no path was given")
    mc = max_context or (row.get("default_context") if row else None)
    if self.loaded: self.unload()   # single-model policy: free the old model first
    if row is not None: row["status"] = "loading"
    try:
      src = fetch(source)
      model, kv = Transformer.from_gguf(src, mc)
      model_name = kv.get('general.name') or kv.get('general.basename') or (model_ref or pathlib.Path(source).stem)
      tok = SimpleTokenizer.from_gguf_kv(kv)
      warm_s, warm_compiles = self._do_warmup(model) if warmup else (None, None)
      mid = model_ref if (model_ref and model_ref in self.registry) else (model_ref or pathlib.Path(source).stem)
      self.adopt(model, kv, tok, mid, model_name, str(source), warmup_done=warmup, warmup_s=warm_s, warmup_compiles=warm_compiles)
    except RuntimeFault: raise
    except Exception as e:
      self.last_error = str(e)
      if row is not None: row["status"] = "error"
      raise RuntimeFault("internal_runtime_error", f"failed to load '{model_ref or path}': {e}")
    return self.status()

  @staticmethod
  def _do_warmup(model:Transformer) -> tuple[float, int]:
    from tinygrad.device import Compiler
    compiles_before = Compiler.cache_misses
    st = time.perf_counter()
    # run 2 tokens through the model twice to capture the SDPA (short-ctx) decode jit before serving requests,
    # then pre-capture the flash (ctx>=threshold) decode jit so the in-generation crossover doesn't stall inline.
    with Context(DEBUG=max(DEBUG.value, 1)):
      for _ in range(2): list(zip(range(2), model.generate([0])))
      model.warmup_flash_decode()
    return time.perf_counter() - st, Compiler.cache_misses - compiles_before

  def warmup(self) -> dict:
    if not self.loaded: raise RuntimeFault("model_not_loaded", "no model loaded")
    self.last_warmup_s, self.last_warmup_compiles = self._do_warmup(self.model)
    self.warmup_done = True
    return {"warmup_done": True, "warmup_s": self.last_warmup_s, "warmup_compiles": self.last_warmup_compiles,
            "model": self.model_id}

  def unload(self) -> dict:
    mid = self.model_id
    if mid in self.registry and self.registry[mid].get("status") == "loaded": self.registry[mid]["status"] = "available"
    self.model = self.tok = None
    self.model_id = self.model_name = self.path = None
    self.max_context = self.architecture = self.quant = None
    self.warmup_done, self.last_warmup_s, self.last_warmup_compiles = False, None, None
    self.metrics = {k: None for k in self.metrics}
    gc.collect()   # best-effort: drop tinygrad buffer refs so GPU memory can be released
    return {"loaded": False, "unloaded": mid}

  def clear_prefix_cache(self) -> dict:
    if self.loaded and getattr(self.model, "_cached_tokens", None):
      n = len(self.model._cached_tokens)
      self.model.reset_generation_state()
      return {"prefix_cache_cleared": True, "cleared_tokens": n}
    return {"prefix_cache_cleared": False, "cleared_tokens": 0}

  def status(self) -> dict:
    m = self.metrics
    return {
      "loaded": self.loaded, "model": self.model_id, "model_name": self.model_name, "path": self.path,
      "architecture": self.architecture, "quant": self.quant,
      "max_context": self.max_context, "kv_cache_tokens": self.max_context,
      "cached_prefix_tokens": m["last_cached_prefix_tokens"],
      "backend": self.backend, "target": self.target,
      "prefill_v2": bool(getattr(getattr(self.model, "config", None), "prefill_v2", False)),
      "prefill_concrete_kv": bool(getattr(getattr(self.model, "config", None), "prefill_concrete_kv", False)),
      "warmup_done": self.warmup_done, "last_warmup_s": self.last_warmup_s,
      "last_warmup_compiles": self.last_warmup_compiles,
      "busy": self.gen_lock.locked(), "load_count": self.load_count, "request_count": self.request_count,
      "last_prefill_tok_s": m["last_prefill_tok_s"], "last_decode_tok_s": m["last_decode_tok_s"],
      "last_finish_reason": m["last_finish_reason"], "last_error": self.last_error,
    }

  def metrics_dict(self) -> dict:
    return {"loaded": self.loaded, "model": self.model_id, "max_context": self.max_context,
            "load_count": self.load_count, "request_count": self.request_count, **self.metrics}

  def cache_dict(self) -> dict:
    # Phase R8: report the runtime-owned caches so the client can avoid needless reload/recompile.
    from tinygrad.helpers import CACHEDB
    from tinygrad.device import Compiler
    cdb = pathlib.Path(CACHEDB)
    model_file = None
    if self.path and (p := pathlib.Path(self.path)).exists():
      model_file = {"path": str(p), "size_bytes": p.stat().st_size}
    cached_toks = len(self.model._cached_tokens) if self.loaded and getattr(self.model, "_cached_tokens", None) else 0
    hits, misses = Compiler.cache_hits, Compiler.cache_misses
    return {
      # on-disk compiled-kernel cache (sqlite); size grows as new kernels are compiled across runs
      "kernel_cache": {"path": str(cdb), "exists": cdb.exists(), "size_bytes": cdb.stat().st_size if cdb.exists() else 0},
      # live process compile counters: a hit reused a cached compiled kernel, a miss compiled a fresh one.
      # kernels_compiled (== misses) is the compiled-kernel count proxy; last_warmup_compiles tells the client
      # whether the last warmup actually did compile work (0 -> kernels were already cached).
      "compile_cache": {"hits": hits, "misses": misses, "total": hits + misses,
                        "kernels_compiled": misses, "last_warmup_compiles": self.last_warmup_compiles},
      "model_file_cache": model_file,
      "prefix_cache": {"cached_tokens": cached_toks, "last_cached_prefix_tokens": self.metrics["last_cached_prefix_tokens"]},
      "warmup_done": self.warmup_done, "last_warmup_s": self.last_warmup_s,
    }
