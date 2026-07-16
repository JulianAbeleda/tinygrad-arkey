from __future__ import annotations
import os, sys, argparse, codecs, typing, re, unicodedata, json, uuid, time, pathlib, threading, gc, socketserver
from tinygrad import nn, Device
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import partition, DEBUG, Timing, GlobalCounters, stderr_log, colored, Context, fetch, profile_marker
from tinygrad.runtime.support.system import RemotePCIDevice
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer

def remote_pressure_snapshot() -> dict[str, typing.Any]:
  return {"stats": RemotePCIDevice.stats(), "commands": RemotePCIDevice.command_stats()}

def format_remote_pressure(label:str, tokens:int, snap:dict[str, typing.Any]) -> str:
  stats = snap["stats"]
  sent_mb, recv_mb = stats["sent_bytes"] / 1e6, stats["recv_bytes"] / 1e6
  tok = max(tokens, 1)
  cmds = sorted(snap["commands"].items(), key=lambda x: x[1].get("count", 0), reverse=True)[:4]
  cmd_s = " ".join(f"{name}:{int(st.get('count', 0))}" for name, st in cmds)
  return (f"{label}: tokens={tokens} roundtrips={stats['roundtrips']} rt/tok={stats['roundtrips']/tok:.2f} "
          f"sent={sent_mb:.2f}MB sent/tok={sent_mb/tok:.2f}MB recv={recv_mb:.2f}MB recv/tok={recv_mb/tok:.2f}MB "
          f"elapsed={stats['elapsed']:.2f}s cmds=[{cmd_s}]")

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

class Handler(HTTPRequestHandler):
  server: LLMServer
  def log_request(self, code='-', size='-'): pass

  def _send_json(self, obj:dict, status_code:int=200):
    return self.send_data(json.dumps(obj).encode(), status_code=status_code)

  def _send_error(self, err_type:str, message:str, request_id:str|None=None, status_code:int|None=None):
    status = status_code if status_code is not None else RUNTIME_ERROR_STATUS.get(err_type, 400)
    body = {"error": {"message": message, "type": err_type, "code": err_type,
                      "request_id": request_id or f"req-{uuid.uuid4().hex[:16]}"}}
    return self.send_data(json.dumps(body).encode(), status_code=status)

  def do_GET(self):
    s = self.server.state
    if self.path == "/v1/models":
      data = [{"id": r["id"], "object": "model", "created": 0, "owned_by": "tinygrad",
               "quant": r.get("quant"), "architecture": r.get("architecture"),
               "max_context": r.get("default_context"), "status": r.get("status")}
              for r in s.registry.values() if r.get("enabled", True)]
      return self._send_json({"object": "list", "data": data})
    if self.path == "/runtime/status": return self._send_json(s.status())
    if self.path == "/runtime/models": return self._send_json({"object": "list", "data": list(s.registry.values())})
    if self.path == "/runtime/metrics": return self._send_json(s.metrics_dict())
    if self.path == "/runtime/cache": return self._send_json(s.cache_dict())
    return self.send_data((pathlib.Path(__file__).parent / "chat.html").read_bytes(), content_type="text/html")

  def _guard_context(self, ids:list[int]):
    # Phase R2: explicit prompt-token overflow guard BEFORE Transformer.generate (which would otherwise pad to
    # max_context and fail / produce nothing). Leave room for at least one completion token.
    mc = self.server.state.max_context
    if len(ids) >= mc:
      raise RuntimeFault("context_length_exceeded",
        f"prompt is {len(ids)} tokens but max_context is {mc}; the client must truncate to leave room for output")

  def _stream_tokens(self, ids:list[int], max_tokens:int|None, temperature:float):
    """Core token generator shared by chat and completions. Yields ('delta', text) then ('finish', reason).
    Updates runtime metrics and honors the cancel_event. The caller holds gen_lock for the whole iteration."""
    s = self.server.state
    model, tok = s.model, s.tok
    cache_start_pos = model.get_start_pos(ids)
    prefill_tokens = len(ids) - cache_start_pos
    stderr_log(f"{self.path}  {colored('--', 'BLACK')}  "
               f"in:{colored(f'{cache_start_pos:5d}', 'green')} +{prefill_tokens:5d}  {colored('--', 'BLACK')}  ")
    out: list[int] = []
    finish_reason = "stop"
    st = pt = time.perf_counter()
    dec = tok.stream_decoder()
    if s.remote_metrics: RemotePCIDevice.reset_stats()
    prefill_snap = None
    for next_id in model.generate(ids, temperature=temperature):
      if len(out) == 0:
        pt = time.perf_counter()
        pf = prefill_tokens / (pt - st) if pt > st else 0.0
        s.metrics["last_prefill_tok_s"] = round(pf, 2)
        stderr_log(f"prefill:{pf:4.0f} tok/s  {colored('--', 'BLACK')}  ")
        if s.remote_metrics:
          prefill_snap = remote_pressure_snapshot()
          RemotePCIDevice.reset_stats()
      if s.cancel_event.is_set(): finish_reason = "cancelled"; break
      if tok.is_end(next_id): break
      out.append(next_id)
      yield ("delta", dec(next_id))
      if max_tokens is not None and len(out) >= max_tokens: finish_reason = "length"; break
    if (tail := dec()): yield ("delta", tail)
    et = time.perf_counter()
    dec_tps = len(out) / (et - pt) if len(out) > 1 and et > pt else 0.0
    s.metrics.update({"last_prompt_tokens": len(ids), "last_completion_tokens": len(out),
                      "last_cached_prefix_tokens": cache_start_pos, "last_decode_tok_s": round(dec_tps, 2),
                      "last_finish_reason": finish_reason})
    stderr_log(f"gen:{dec_tps:4.0f} tok/s  {colored('--', 'BLACK')}  "
               f"out:{len(out):5d}  {colored('--', 'BLACK')}  total:{et-st:6.2f}s\n")
    if s.remote_metrics:
      if prefill_snap is not None: stderr_log(format_remote_pressure("remote prefill", prefill_tokens, prefill_snap) + "\n")
      stderr_log(format_remote_pressure("remote decode", max(len(out) - 1, 0), remote_pressure_snapshot()) + "\n")
    # 'cancelled' is reported in metrics; over the wire we use the standard OpenAI 'stop' finish_reason
    yield ("finish", "stop" if finish_reason == "cancelled" else finish_reason)

  def _usage(self, ids:list[int]) -> dict:
    comp = self.server.state.metrics["last_completion_tokens"] or 0
    return {"prompt_tokens": len(ids), "completion_tokens": comp, "total_tokens": len(ids) + comp}

  def _acquire_gen(self):
    s = self.server.state
    if not s.gen_lock.acquire(blocking=False):
      raise RuntimeFault("runtime_busy", "a generation is already in progress (one request at a time)")
    s.cancel_event.clear()
    s.request_count += 1

  def handle_chat(self, body:dict):
    s = self.server.state
    if not s.loaded: raise RuntimeFault("model_not_loaded", "no model loaded; POST /runtime/load first")
    tok = s.tok
    ids: list[int] = tok.prefix()
    for i, msg in enumerate(body["messages"]):
      ids += tok.role(msg["role"])
      content = msg["content"]
      if isinstance(content, str): ids += tok.encode(content)
      elif isinstance(content, list):
        for c in content:
          if c["type"] == "text": ids += tok.encode(c["text"])
          else: raise RuntimeFault("invalid_request", f"unhandled content part type: {c['type']}")
      else: raise RuntimeFault("invalid_request", f"unknown content type: {type(content)}")
      if msg["role"] == "assistant" and i == len(body["messages"]) - 1: break
      ids += tok.end_turn()
    else: ids += tok.role("assistant")

    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    temperature = float(body.get("temperature", 0.0))
    model_name = body.get("model") or s.model_id
    self._guard_context(ids)
    stream = bool(body.get("stream"))
    include_usage = (not stream) or body.get("stream_options", {}).get("include_usage", False)
    tmpl = {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model_name}
    def chunks():
      yield {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}], **tmpl}
      for kind, payload in self._stream_tokens(ids, max_tokens, temperature):
        if kind == "delta": yield {"choices": [{"index": 0, "delta": {"content": payload}, "finish_reason": None}], **tmpl}
        else:
          yield {"choices": [{"index": 0, "delta": {}, "finish_reason": payload}], **tmpl}
          if include_usage: yield {"choices": [], "usage": self._usage(ids), **tmpl}

    self._acquire_gen()
    try:
      if stream: return self.stream_json(chunks())
      out, finish = [], "stop"
      for c in chunks():
        if c["choices"] and c["choices"][0].get("delta", {}).get("content"): out.append(c["choices"][0]["delta"]["content"])
        if c["choices"] and c["choices"][0].get("finish_reason"): finish = c["choices"][0]["finish_reason"]
      return self._send_json({**tmpl, "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(out)}, "finish_reason": finish}],
        "usage": self._usage(ids)})
    finally:
      s.gen_lock.release()

  def handle_completions(self, body:dict):
    s = self.server.state
    if not s.loaded: raise RuntimeFault("model_not_loaded", "no model loaded; POST /runtime/load first")
    tok = s.tok
    prompt = body.get("prompt", "")
    if isinstance(prompt, list):
      if prompt and not isinstance(prompt[0], str):
        raise RuntimeFault("invalid_request", "token-array prompts are not supported; send a string prompt")
      prompt = "".join(prompt)
    ids: list[int] = tok.prefix() + tok.encode(prompt)
    max_tokens = body.get("max_tokens")
    temperature = float(body.get("temperature", 0.0))
    model_name = body.get("model") or s.model_id
    self._guard_context(ids)
    stream = bool(body.get("stream"))
    tmpl = {"id": f"cmpl-{uuid.uuid4().hex[:24]}", "object": "text_completion", "created": int(time.time()),
            "model": model_name}
    def chunks():
      for kind, payload in self._stream_tokens(ids, max_tokens, temperature):
        if kind == "delta": yield {"choices": [{"index": 0, "text": payload, "finish_reason": None}], **tmpl}
        else: yield {"choices": [{"index": 0, "text": "", "finish_reason": payload}], **tmpl}

    self._acquire_gen()
    try:
      if stream: return self.stream_json(chunks())
      out, finish = [], "stop"
      for c in chunks():
        if c["choices"][0].get("text"): out.append(c["choices"][0]["text"])
        if c["choices"][0].get("finish_reason"): finish = c["choices"][0]["finish_reason"]
      return self._send_json({**tmpl, "choices": [{"index": 0, "text": "".join(out), "finish_reason": finish, "logprobs": None}],
        "usage": self._usage(ids)})
    finally:
      s.gen_lock.release()

  def _runtime_mutation(self, body:dict):
    # load/unload/warmup/cache-clear mutate the loaded model -> must not race a generation. Hold gen_lock.
    s = self.server.state
    if not s.gen_lock.acquire(blocking=False):
      raise RuntimeFault("runtime_busy", "a generation is in progress; retry after it completes")
    try:
      if self.path == "/runtime/load":
        return self._send_json(s.load(body.get("model"), body.get("path"), body.get("max_context"), body.get("warmup", True)))
      if self.path == "/runtime/unload": return self._send_json(s.unload())
      if self.path == "/runtime/warmup": return self._send_json(s.warmup())
      if self.path == "/runtime/cache/clear": return self._send_json(s.clear_prefix_cache())
    finally:
      s.gen_lock.release()

  def do_POST(self):
    s = self.server.state
    raw_body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
    try:
      body: dict[str, typing.Any] = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    except json.JSONDecodeError as e:
      return self._send_error("invalid_request", f"invalid JSON body: {e}")
    if DEBUG >= 1 and body: print(json.dumps(body, indent=2))
    try:
      if self.path == "/v1/chat/completions": return self.handle_chat(body)
      if self.path == "/v1/completions": return self.handle_completions(body)
      if self.path in ("/runtime/load", "/runtime/unload", "/runtime/warmup", "/runtime/cache/clear"):
        return self._runtime_mutation(body)
      if self.path == "/runtime/cancel":
        # threaded server: this arrives while a generation is still running and signals it to stop next token
        s.cancel_event.set()
        return self._send_json({"cancelled": True, "busy": s.gen_lock.locked()})
      return self._send_error("invalid_request", f"unhandled path {self.path}", status_code=404)
    except RuntimeFault as f:
      return self._send_error(f.err_type, f.message)
    except BrokenPipeError:
      return   # client disconnected mid-response; nothing to send
    except Exception as e:
      if DEBUG >= 1:
        import traceback; traceback.print_exc()
      return self._send_error("internal_runtime_error", str(e))

class LLMServer(socketserver.ThreadingMixIn, TCPServerWithReuse):
  # Threaded so /runtime/status and /runtime/cancel stay responsive during a generation. Concurrency safety
  # comes from RuntimeState.gen_lock (one generation/mutation at a time), not from serializing the whole server.
  daemon_threads = True
  def __init__(self, server_address:tuple, state:RuntimeState):
    self.state = state
    super().__init__(server_address, Handler)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", "-m", default=list(models.keys())[0], help=f"Model choice ({', '.join(models.keys())}) or path to a local GGUF file")
  parser.add_argument("--max_context", type=lambda v: v if v == "auto" else int(v), default="auto",
                      help="Max context length: 'auto' (default) auto-scans free VRAM and admits the largest safe "
                           "context (refuses loud if the model can't fit a useful fp16-KV context, e.g. 32B); an "
                           "explicit int is still admission-checked and fails loud rather than OOMing.")
  parser.add_argument("--stream", choices=["auto", "on", "off"], default="auto",
                      help="StreamingLLM streaming tier (lossy, unbounded logical context in an N-token window; content "
                           "older than the window + 4 sinks is forgotten). 'auto' (default): use it only as the last "
                           "admission rung when no lossless tier fits. 'on': force it for deliberately-unbounded "
                           "generation even when a lossless tier fits. 'off': exact-context semantics (refuse rather "
                           "than stream).")
  parser.add_argument("--serve", nargs='?', type=int, const=8000, metavar="PORT", help="Run OpenAI compatible API (optional port, default 8000)")
  parser.add_argument("--registry", type=str, default=None, help="Path to a runtime_models.json registry (Phase R3)")
  parser.add_argument("--no-preload", action="store_true", help="Start the server without loading a model (load later via /runtime/load)")
  parser.add_argument("--warmup", action="store_true", help="warmup the JIT")
  parser.add_argument("--benchmark", nargs='?', type=int, const=20, metavar="COUNT", help="Benchmark tok/s (optional count, default 20)")
  parser.add_argument("--remote-metrics", action="store_true", help="Print remote roundtrip and byte metrics for benchmark/server generations")
  args = parser.parse_args()

  registry = build_registry(models, pathlib.Path(args.registry) if args.registry else DEFAULT_REGISTRY_PATH)
  state = RuntimeState(registry, remote_metrics=args.remote_metrics)

  # serve without a model when explicitly requested: the client drives load via /runtime/load
  if args.serve and args.no_preload:
    print(f"serving with no preloaded model; POST /runtime/load to select one. registry has {len(registry)} model(s).")
    LLMServer(('', args.serve), state).serve_forever()
    return

  # load the model
  source = models.get(args.model, args.model)
  model, kv = Transformer.from_gguf(fetch(source), args.max_context, stream=args.stream)
  model_name = kv.get('general.name') or kv.get('general.basename') or args.model
  model_id = args.model if args.model in models else pathlib.Path(args.model).stem
  file_sizes = [y.nbytes() for y in UOp.sink(*[x.uop for x in nn.state.get_parameters(model)]).toposort() if y.op is Ops.BUFFER]
  print(f"using model \"{model_name}\" with {sum(file_sizes):,} bytes and {sum(x.numel() for x in nn.state.get_parameters(model)):,} params")

  # get tokenizer
  tok = SimpleTokenizer.from_gguf_kv(kv)

  # warmup the JIT
  warm_s, warm_compiles = None, None
  if args.warmup or args.serve:
    warm_s, warm_compiles = RuntimeState._do_warmup(model)

  # adopt the preloaded model into the runtime state (centralized for /runtime/* and /v1/*)
  state.adopt(model, kv, tok, model_id, model_name, str(source), warmup_done=bool(args.warmup or args.serve),
              warmup_s=warm_s, warmup_compiles=warm_compiles)

  # start server
  if args.serve: LLMServer(('', args.serve), state).serve_forever()

  # do benchmark
  if args.benchmark is not None:
    gen = model.generate(toks:=[tok.bos_id or 0])
    for i in range(args.benchmark):
      profile_marker(f"decode @ {i}")
      GlobalCounters.reset()
      if args.remote_metrics: RemotePCIDevice.reset_stats()
      with Timing(on_exit=lambda x: f", {1e9/x:6.2f} tok/s, {GlobalCounters.global_mem/x:7.2f} GB/s,"
                  f" {GlobalCounters.global_mem//1000000}/{GlobalCounters.mem_used//1000000} MB  --  "+\
                  tok.decode(toks).replace("\n", "\\n")): next(gen)
      if args.remote_metrics: print(format_remote_pressure(f"remote decode @ {i}", 1, remote_pressure_snapshot()))
    exit(0)

  # interactive chat
  ids: list[int] = tok.prefix()
  while 1:
    try:
      ids += tok.role("user") + tok.encode(input('>>> ')) + tok.end_turn() + tok.role("assistant")
    except EOFError:
      break
    dec = tok.stream_decoder()
    for next_id in model.generate(ids):
      sys.stdout.write(dec(next_id) if not tok.is_end(next_id) else dec() + "\n\n")
      sys.stdout.flush()
      if tok.is_end(next_id): break

if __name__ == "__main__": main()
