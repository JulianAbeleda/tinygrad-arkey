#!/usr/bin/env python3
"""Capture the exact production NV cubin and launch geometry for one kernel.

Measurement tooling only. It monkeypatches NVProgram to retain the raw cubin
blob, launch geometry, and parameter values the first time a name-prefixed
kernel is constructed and called, then runs a minimal deterministic decode so
the production route compiles and launches it. It writes the cubin plus a JSON
record and makes no production code change.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

CAPTURED: dict[str, dict] = {}


def _sha(b: bytes) -> str:
  return hashlib.sha256(b).hexdigest()


def install_hook(prefixes: list[str]) -> None:
  import tinygrad.runtime.ops_nv as ops_nv

  orig_init = ops_nv.NVProgram.__init__
  orig_call = ops_nv.NVProgram.__call__

  def patched_init(self, dev, name, lib, **kwargs):
    orig_init(self, dev, name, lib, **kwargs)
    if any(name.startswith(p) for p in prefixes) and name not in CAPTURED:
      CAPTURED[name] = {
        "name": name,
        "cubin": bytes(lib),
        "cubin_sha256": _sha(bytes(lib)),
        "regs_usage": getattr(self, "regs_usage", None),
        "shmem_usage": getattr(self, "shmem_usage", None),
        "lcmem_usage": getattr(self, "lcmem_usage", None),
        "max_threads": getattr(self, "max_threads", None),
        "calls": [],
      }

  def patched_call(self, *bufs, global_size=(1, 1, 1), local_size=(1, 1, 1),
                   vals=(), wait=False, timeout=None):
    rec = CAPTURED.get(self.name)
    if rec is not None and len(rec["calls"]) < 8:
      def _buf_meta(b):
        va = getattr(b, "va_addr", None)
        if va is None:
          inner = getattr(b, "_buf", None)
          va = getattr(inner, "va_addr", None)
        sz = getattr(b, "size", None)
        if sz is None:
          inner = getattr(b, "_buf", None)
          sz = getattr(inner, "size", None)
        return {"va_addr": int(va) if va is not None else None,
                "size": int(sz) if sz is not None else None}
      rec["calls"].append({
        "global_size": list(global_size),
        "local_size": list(local_size),
        "vals": list(vals),
        "n_bufs": len(bufs),
        "buf_meta": [_buf_meta(b) for b in bufs],
      })
    return orig_call(self, *bufs, global_size=global_size, local_size=local_size,
                     vals=vals, wait=wait, timeout=timeout)

  ops_nv.NVProgram.__init__ = patched_init
  ops_nv.NVProgram.__call__ = patched_call


def run_capture(prefixes: list[str], out: pathlib.Path, depth: int, model: str, max_context: int = 4608,
                official_loader: bool = False) -> dict:
  install_hook(prefixes)
  from tinygrad import Device
  dev = Device[Device.DEFAULT]
  if official_loader:
    from extra.llm_research.decode.nv_predispatch_full_logits_qualification import _load, _prompt
    mdl = _load(model, max_context); prompt = _prompt(model, depth)
    mdl._decode_direct_greedy_promoted = True; mdl._decode_feedback_pingpong_promoted = True
  else:
    from tinygrad.llm.generate import load_model_and_tokenizer
    mdl, tok = load_model_and_tokenizer(model, max_context, seed=20260617)
    base = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
    prompt = (base * (1 + depth // len(base)))[:depth]

  gen = mdl.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  try:
    next(gen)
    for _ in range(12):
      next(gen)
    dev.synchronize()
  finally:
    gen.close()

  if not CAPTURED:
    result = {"schema": "tinygrad.nv_cubin_capture.v1", "captured": [], "verdict": "NO_MATCH",
              "prefixes": prefixes, "depth": depth, "max_context": max_context,
              "note": "no production NVProgram matched the prefixes"}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result

  captured = []
  out.parent.mkdir(parents=True, exist_ok=True)
  for name, rec in CAPTURED.items():
    cubin_path = out.with_name(f"{name}.cubin")
    cubin_path.write_bytes(rec["cubin"])
    row = {k: rec[k] for k in ("name", "cubin_sha256", "regs_usage", "shmem_usage",
                               "lcmem_usage", "max_threads", "calls")}
    row["cubin_path"] = str(cubin_path)
    row["cubin_bytes"] = len(rec["cubin"])
    captured.append(row)

  result = {"schema": "tinygrad.nv_cubin_capture.v1", "captured": captured,
            "verdict": "CAPTURED", "prefixes": prefixes, "depth": depth, "max_context": max_context,
            "official_loader": official_loader}
  out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return result


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--prefix", default=None, help="single prefix (legacy)")
  ap.add_argument("--prefixes", default=None, help="comma-separated prefixes")
  ap.add_argument("--out", type=pathlib.Path, required=True)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--official-loader", action="store_true")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  args = ap.parse_args()
  prefixes = [p for p in (args.prefixes.split(",") if args.prefixes else (args.prefix or "").split(",")) if p]
  run_capture(prefixes, args.out, args.depth, args.model, args.max_context, args.official_loader)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
