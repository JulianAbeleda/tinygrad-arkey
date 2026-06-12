#!/usr/bin/env python3
from __future__ import annotations

import sys

from extra.llm_eval_matrix import load_manifest, main as _llm_main, make_matrix, matrix_markdown

DEFAULT_MANIFEST = "bench/qwen-eval-20260612/manifest.json"

def main() -> int:
  argv = sys.argv[1:]
  if "--manifest" not in argv:
    argv = ["--manifest", DEFAULT_MANIFEST] + argv
  return _llm_main(argv)

if __name__ == "__main__":
  raise SystemExit(main())
