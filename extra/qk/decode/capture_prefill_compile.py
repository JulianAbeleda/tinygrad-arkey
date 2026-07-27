#!/usr/bin/env python3
"""Capture generated AMD source for one production prefill attempt.

This is intentionally a compile probe, not a benchmark. It captures source
before COMGR compilation and records the first compiler failure without PMC,
graph profiling, or kernel edits.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, sys, time


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--max-context", type=int, default=256)
  ap.add_argument("--prompt-tokens", type=int, default=128)
  ap.add_argument("--chunk-size", type=int, default=32)
  ap.add_argument("--out-dir", type=pathlib.Path, required=True)
  args = ap.parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  # Patch both AMD compiler implementations before loading the model. The
  # __main__ guard below is required because model admission may spawn a child.
  from tinygrad.runtime.support.compiler_amd import HIPCompiler, HIPCCCompiler
  records: list[dict] = []

  def patch(cls):
    original = cls.compile
    def capture(self, src: str) -> bytes:
      digest = hashlib.sha256(src.encode()).hexdigest()
      path = args.out_dir / f"{len(records):04d}-{digest[:16]}.hip"
      path.write_text(src)
      row = {"index": len(records), "sha256": digest, "source": str(path),
             "bytes": len(src), "compiler": cls.__name__}
      records.append(row)
      try:
        result = original(self, src)
        row["compiled"] = True
        return result
      except Exception as exc:
        row["compiled"] = False
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        raise
    cls.compile = capture

  patch(HIPCompiler)
  patch(HIPCCCompiler)

  result: dict = {"schema": "tinygrad.prefill.compile-capture.v1",
                  "created_unix_ns": time.time_ns(), "model": str(pathlib.Path(args.model).resolve()),
                  "max_context": args.max_context, "prompt_tokens": args.prompt_tokens,
                  "chunk_size": args.chunk_size, "profile": 0, "pmc": 0, "pmc_graph": 0,
                  "sources": records}
  try:
    from extra.llm.generate import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, args.max_context, seed=20260617)
    base = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + tokenizer.encode(
      "the quick brown fox jumps. " * 800)
    prompt = (base * (1 + args.prompt_tokens // len(base)))[:args.prompt_tokens]
    gen = model.generate(prompt, chunk_size=args.chunk_size, temperature=0.0)
    result["first_token"] = int(next(gen))
    gen.close()
    result["status"] = "success"
    rc = 0
  except Exception as exc:
    result["status"] = "compile_or_execution_failure"
    result["error_type"] = type(exc).__name__
    result["error"] = str(exc)
    rc = 1
  result["sources"] = records
  (args.out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"status": result["status"], "sources": len(records),
                    "failed_source": next((x["source"] for x in records if not x.get("compiled", True)), None),
                    "manifest": str(args.out_dir / "manifest.json")}, indent=2))
  return rc


if __name__ == "__main__":
  raise SystemExit(main())
