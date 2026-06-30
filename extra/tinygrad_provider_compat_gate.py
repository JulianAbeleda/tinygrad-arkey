#!/usr/bin/env python3
"""Phase R9 provider-compatibility gate (see docs/tinygrad-runtime-client-separation-roadmap-20260630.md).

Proves a running tinygrad runtime server can be driven by both an OpenAI-compatible client (OpenCode, AI-SDK,
llama.cpp-style tooling) over /v1/* and the proprietary app over /runtime/*. Uses only the Python stdlib so it
can run as a plain provider-compat check with no extra deps.

Usage:
  # against an already-running server (started with: python -m tinygrad.llm.cli --serve 8000 -m <model>)
  python extra/tinygrad_provider_compat_gate.py --base-url http://127.0.0.1:8000 --model qwen3:8b

  # load/unload checks require a model the registry can resolve (id or --load-path)
  python extra/tinygrad_provider_compat_gate.py --model qwen3:0.6b --load-path /home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf

Verdicts: R9_PASS_PROVIDER_COMPAT / R9_BLOCKED_OPENAI_COMPAT_SURFACE
"""
from __future__ import annotations
import sys, json, time, argparse, urllib.request, urllib.error

class Gate:
  def __init__(self, base_url:str): self.base = base_url.rstrip("/"); self.results: list[tuple[str,bool,str]] = []
  def _req(self, method:str, path:str, body:dict|None=None, stream:bool=False, timeout:float=120.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(self.base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=timeout)
    if stream: return resp
    raw = resp.read().decode("utf-8")
    return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else raw)
  def check(self, name:str, fn):
    try:
      ok, detail = fn()
      self.results.append((name, bool(ok), detail))
      print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    except Exception as e:
      self.results.append((name, False, f"exception: {e}"))
      print(f"  [FAIL] {name}: exception: {e}")

  # ---- individual checks ----
  def c_models(self):
    st, j = self._req("GET", "/v1/models")
    ids = [m["id"] for m in j.get("data", [])]
    return (st == 200 and j.get("object") == "list" and len(ids) > 0, f"{len(ids)} models, e.g. {ids[:3]}")
  def c_status(self):
    st, j = self._req("GET", "/runtime/status")
    return (st == 200 and "loaded" in j and "max_context" in j, f"loaded={j.get('loaded')} max_context={j.get('max_context')}")
  def c_chat_nonstream(self, model):
    st, j = self._req("POST", "/v1/chat/completions",
                      {"model": model, "messages": [{"role": "user", "content": "Say hi in one word."}],
                       "max_tokens": 8, "temperature": 0.0})
    txt = j["choices"][0]["message"]["content"] if isinstance(j, dict) and j.get("choices") else ""
    usage = j.get("usage", {}) if isinstance(j, dict) else {}
    return (st == 200 and j.get("object") == "chat.completion" and len(txt) > 0 and "completion_tokens" in usage,
            f"text={txt!r} usage={usage}")
  def c_chat_stream(self, model):
    resp = self._req("POST", "/v1/chat/completions",
                     {"model": model, "messages": [{"role": "user", "content": "Count: one two three"}],
                      "max_tokens": 12, "temperature": 0.0, "stream": True}, stream=True)
    chunks, got_done, got_content, got_finish = 0, False, False, None
    for line in resp:
      line = line.decode("utf-8").strip()
      if not line.startswith("data:"): continue
      payload = line[len("data:"):].strip()
      if payload == "[DONE]": got_done = True; break
      d = json.loads(payload); chunks += 1
      ch = d.get("choices") or []
      if ch and ch[0].get("delta", {}).get("content"): got_content = True
      if ch and ch[0].get("finish_reason"): got_finish = ch[0]["finish_reason"]
    return (chunks > 0 and got_done and got_content and got_finish is not None,
            f"chunks={chunks} done={got_done} content={got_content} finish={got_finish}")
  def c_max_tokens_stop(self, model):
    st, j = self._req("POST", "/v1/chat/completions",
                      {"model": model, "messages": [{"role": "user", "content": "Write a long story about a robot."}],
                       "max_tokens": 5, "temperature": 0.0})
    fr = j["choices"][0].get("finish_reason")
    comp = j.get("usage", {}).get("completion_tokens")
    return (fr == "length" and comp is not None and comp <= 5, f"finish_reason={fr} completion_tokens={comp}")
  def c_context_overflow(self, model):
    # huge prompt -> structured context_length_exceeded, NOT a tensor shape crash
    big = "word " * 20000
    try:
      st, j = self._req("POST", "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": big}], "max_tokens": 4})
    except urllib.error.HTTPError as e:
      j = json.loads(e.read().decode("utf-8")); st = e.code
    err = j.get("error", {}) if isinstance(j, dict) else {}
    return (st == 400 and err.get("type") == "context_length_exceeded", f"status={st} error={err}")
  def c_completions(self, model):
    st, j = self._req("POST", "/v1/completions",
                      {"model": model, "prompt": "The capital of France is", "max_tokens": 4, "temperature": 0.0})
    txt = j["choices"][0].get("text") if isinstance(j, dict) and j.get("choices") else None
    return (st == 200 and j.get("object") == "text_completion" and txt is not None, f"text={txt!r}")
  def c_metrics(self):
    st, j = self._req("GET", "/runtime/metrics")
    return (st == 200 and "last_decode_tok_s" in j, f"decode_tok_s={j.get('last_decode_tok_s')} prefill={j.get('last_prefill_tok_s')}")
  def c_cache(self):
    st, j = self._req("GET", "/runtime/cache")
    cc = j.get("compile_cache", {}) if isinstance(j, dict) else {}
    ok = st == 200 and "compile_cache" in j and "hits" in cc and "kernels_compiled" in cc and "kernel_cache" in j
    return (ok, f"compile hits={cc.get('hits')} misses={cc.get('misses')} warmup_compiles={cc.get('last_warmup_compiles')}")
  def c_load(self, model, load_path):
    st, j = self._req("POST", "/runtime/load", {"model": model, "path": load_path, "warmup": True}, timeout=600)
    return (st == 200 and j.get("loaded") and j.get("model") == model, f"loaded={j.get('loaded')} model={j.get('model')} warm={j.get('last_warmup_s')}")
  def c_chat_after_load(self, model):
    st, j = self._req("POST", "/v1/chat/completions",
                      {"model": model, "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 4})
    return (st == 200 and j.get("choices"), f"ok after load, finish={j['choices'][0].get('finish_reason') if j.get('choices') else None}")
  def c_unload(self):
    st, j = self._req("POST", "/runtime/unload", {})
    st2, status = self._req("GET", "/runtime/status")
    return (st == 200 and status.get("loaded") is False, f"unloaded={j.get('unloaded')} now loaded={status.get('loaded')}")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", default="http://127.0.0.1:8000")
  ap.add_argument("--model", default="qwen3:8b", help="model id the server has loaded / can load")
  ap.add_argument("--load-path", default=None, help="optional explicit GGUF path for the /runtime/load lifecycle check")
  ap.add_argument("--skip-lifecycle", action="store_true", help="skip /runtime/load+unload (e.g. to keep a preloaded model)")
  args = ap.parse_args()

  print(f"== R9 provider-compat gate -> {args.base_url} (model={args.model}) ==")
  g = Gate(args.base_url)
  g.check("GET /v1/models", g.c_models)
  g.check("GET /runtime/status", g.c_status)
  g.check("POST /v1/chat/completions (non-stream)", lambda: g.c_chat_nonstream(args.model))
  g.check("POST /v1/chat/completions (stream)", lambda: g.c_chat_stream(args.model))
  g.check("max_tokens -> finish_reason=length", lambda: g.c_max_tokens_stop(args.model))
  g.check("context overflow -> context_length_exceeded", lambda: g.c_context_overflow(args.model))
  g.check("POST /v1/completions", lambda: g.c_completions(args.model))
  g.check("GET /runtime/metrics", g.c_metrics)
  g.check("GET /runtime/cache (compile hit/miss observability)", g.c_cache)
  if not args.skip_lifecycle:
    g.check("POST /runtime/load", lambda: g.c_load(args.model, args.load_path))
    g.check("POST /v1/chat/completions (after load)", lambda: g.c_chat_after_load(args.model))
    g.check("POST /runtime/unload", g.c_unload)

  passed = sum(1 for _, ok, _ in g.results if ok)
  total = len(g.results)
  verdict = "R9_PASS_PROVIDER_COMPAT" if passed == total else "R9_BLOCKED_OPENAI_COMPAT_SURFACE"
  print(f"\n== {passed}/{total} checks passed -> {verdict} ==")
  sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
  main()
