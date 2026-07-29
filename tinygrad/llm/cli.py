from __future__ import annotations
# Production LLM CLI owner; kept beside the tinygrad runtime model domain.
import sys, argparse, typing, json, uuid, time, pathlib, socketserver
from tinygrad import nn
from tinygrad.uop.ops import UOp, Ops
from tinygrad.helpers import DEBUG, Timing, GlobalCounters, stderr_log, colored, fetch, profile_marker
from tinygrad.runtime.support.system import RemotePCIDevice
from tinygrad.viz.serve import TCPServerWithReuse, HTTPRequestHandler
from tinygrad.llm.model import Transformer
from tinygrad.llm.runtime_state import (SimpleTokenizer, models, _quant_from_name, _device_target,
                                        DEFAULT_REGISTRY_PATH, build_registry, RuntimeFault, RuntimeState)

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
  parser.add_argument("--benchmark-context", type=int, metavar="TOKENS",
                      help="Prefill exactly TOKENS synthetic tokens before --benchmark decode samples")
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
    if args.benchmark <= 0: parser.error("--benchmark COUNT must be positive")
    if args.benchmark_context is not None and not 1 <= args.benchmark_context < model.max_context:
      parser.error(f"--benchmark-context must be in 1..{model.max_context-1}")
    prompt_tokens = args.benchmark_context or 1
    if prompt_tokens + args.benchmark > model.max_context and not model.config.ring:
      parser.error("benchmark context plus decode samples exceeds the admitted max context")
    gen = model.generate(toks:=[tok.bos_id or 0] * prompt_tokens)
    if args.benchmark_context is not None:
      GlobalCounters.reset()
      started = time.perf_counter_ns()
      first = next(gen)
      elapsed = time.perf_counter_ns() - started
      identities = sorted({str(getattr(lin, "_prefill_full_kernel_candidate_identity"))
                           for lin in model._q4k_linears.linears
                           if getattr(lin, "_prefill_full_kernel_candidate_identity", None)})
      print(f"prefill@{prompt_tokens}: {prompt_tokens * 1e9 / elapsed:.2f} tok/s, "
            f"elapsed={elapsed/1e6:.2f}ms, selected_candidate_identities={identities}")
      toks = [first]
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
