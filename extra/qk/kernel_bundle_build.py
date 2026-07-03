#!/usr/bin/env python3
"""Offline AOT build + load-side golden probe (GPU-run; import-safe). Keeps model.py untouched -- the bundle installs
into cache.db standalone, so a subsequent normal load is warm. Auto-hooking from_gguf is a later nicety.

BUILD (on the build/CI box, GPU):
  DEV=AMD JIT=1 python extra/qk/kernel_bundle_build.py build --model <gguf> --out qwen3-8b.gfx1100.aot.db
    -> warms the compile cache (runs prefill + a few decode tokens so every hot kernel compiles), records the greedy
       golden tokens, and packs {kernels + fingerprint + golden} into the portable bundle.

ACCEPT+INSTALL (on a fresh box, GPU -- runs the golden probe):
  DEV=AMD JIT=1 python extra/qk/kernel_bundle_build.py install --model <gguf> --bundle qwen3-8b.gfx1100.aot.db
    -> checks the codegen fingerprint, runs the golden probe (re-decode + compare tokens), and ONLY on pass installs
       the kernels into the local cache.db. Fingerprint mismatch or probe failure -> refuse + fall back to cold compile.
"""
import sys, argparse, hashlib, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tinygrad.llm.model import Transformer
from extra.qk import kernel_aot as A

_GOLD_PROMPT = [((i * 7 + 3) % 2000) + 1 for i in range(8)]
_GOLD_NTOK = 8

def _greedy_tokens(model_path:str, max_context:int, prompt, ntok:int) -> list[int]:
  m, _ = Transformer.from_gguf(model_path, max_context)
  g = m.generate(prompt)   # temperature=0 -> greedy, deterministic
  return [int(next(g)) for _ in range(ntok)]

def _golden(tokens:list[int]) -> dict:
  return {"digest": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16],
          "meta": {"prompt": _GOLD_PROMPT, "ntok": len(tokens), "tokens": tokens}}

def make_golden_probe(model_path:str, max_context:int):
  """Return probe_fn(golden_meta)->(ok, detail): re-decode the golden prompt and compare tokens. This is the AUTHORITY
  that a shipped binary computes correctly on THIS box -- catches the codegen input we forgot to fingerprint."""
  def probe(golden:dict):
    want = golden.get("meta", {}).get("tokens", [])
    if not want: return (False, {"reason": "bundle has no golden tokens"})
    got = _greedy_tokens(model_path, max_context, golden["meta"]["prompt"], len(want))
    ok = (got == want)
    return (ok, {"want": want, "got": got, "match": ok})
  return probe

def cmd_build(args):
  mc = int(args.max_context)
  toks = _greedy_tokens(args.model, mc, _GOLD_PROMPT, _GOLD_NTOK)   # this warm-compiles every hot kernel into cache.db
  info = A.pack_bundle(args.out, golden=_golden(toks))
  print(f"built bundle {args.out}: {info['kernels']} kernels, fingerprint {info['fingerprint']['digest']}, "
        f"golden {len(toks)} toks")

def cmd_install(args):
  probe = make_golden_probe(args.model, int(args.max_context))
  res = A.accept_bundle(args.bundle, golden_probe_fn=probe, install=True)
  print(json.dumps({k: res[k] for k in ("fingerprint_ok", "probe_ok", "installed", "used", "reason")
                    if k in res}, indent=2))
  if res.get("probe_ok") is False: print("  probe detail:", res.get("probe_detail"))
  sys.exit(0 if res.get("used") else 1)

if __name__ == "__main__":
  ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
  b = sub.add_parser("build"); b.add_argument("--model", required=True); b.add_argument("--out", required=True)
  b.add_argument("--max_context", default="1024"); b.set_defaults(fn=cmd_build)
  i = sub.add_parser("install"); i.add_argument("--model", required=True); i.add_argument("--bundle", required=True)
  i.add_argument("--max_context", default="1024"); i.set_defaults(fn=cmd_install)
  a = ap.parse_args(); a.fn(a)
