#!/usr/bin/env python3
"""Create a CPU-only, content-addressed fixed-depth prompt-token fixture.

Reads GGUF metadata only.  It deliberately does not load model weights, create a
tinygrad device, or dispatch a kernel.  The repeated corpus is the same corpus
used by decode_runtime_overhead.py; the fixture makes its previously implicit
token IDs and BOS policy inspectable.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, tempfile


def _sha_ids(ids:list[int]) -> str:
  return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def _atomic_json(path:pathlib.Path, value:dict) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
  try:
    with open(fd, "w", closefd=True) as f:
      json.dump(value, f, indent=2, sort_keys=True)
      f.write("\n")
      f.flush()
    pathlib.Path(tmp).replace(path)
  except BaseException:
    pathlib.Path(tmp).unlink(missing_ok=True)
    raise


def build_fixture(model:str, lengths:tuple[int, ...]) -> dict:
  from extra.llm.cli import SimpleTokenizer
  from tinygrad.llm.gguf import gguf_load_metadata

  path = pathlib.Path(model).expanduser().resolve()
  stat = path.stat()
  kv, _ = gguf_load_metadata(path)
  tokenizer = SimpleTokenizer.from_gguf_kv(kv)
  base = tokenizer.prefix() + tokenizer.encode("the quick brown fox jumps. " * 800)
  if not base: raise ValueError("tokenizer produced no deterministic base tokens")
  prompts = {}
  for length in lengths:
    if length < 1: raise ValueError("lengths must be positive")
    ids = (base * (1 + length // len(base)))[:length]
    prompts[str(length)] = {"token_ids": ids, "token_count": len(ids), "token_ids_sha256": _sha_ids(ids)}
  return {
    "schema": "tinygrad.decode.token_fixture.v1",
    "generator": "extra/qk/decode/token_fixture.py",
    "model": {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns},
    "tokenizer": {"preset": tokenizer.preset, "bos_token_id": tokenizer.bos_id,
                  "eos_token_id": tokenizer.eos_id, "eot_token_id": tokenizer.eot_id,
                  "prefix_token_ids": tokenizer.prefix(), "add_bos": tokenizer.bos_id is not None},
    "construction": {"base_text": "the quick brown fox jumps. " * 800,
                     "base_token_count": len(base), "base_token_ids_sha256": _sha_ids(base),
                     "algorithm": "(tokenizer.prefix() + tokenizer.encode(base_text)) repeated then truncated"},
    "prompts": prompts,
    "sampling": {"temperature": 0.0, "seed": 20260617,
                 "expectation": "deterministic argmax route; token parity still requires a runtime comparison"},
    "compatibility": {"tinygrad_decode_runtime_overhead_consumes_fixture": False,
                      "llama_bench_consumes_fixture": False,
                      "status": "blocked_pending_bounded_input_adapter",
                      "reason": "the canonical tinygrad authority generates its prompt internally and llama-bench accepts lengths, not token IDs"},
  }


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", required=True)
  ap.add_argument("--lengths", default="128,512,4096")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  lengths = tuple(int(x) for x in args.lengths.split(",") if x)
  _atomic_json(pathlib.Path(args.out), build_fixture(args.model, lengths))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
