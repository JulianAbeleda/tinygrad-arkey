#!/usr/bin/env python3
from __future__ import annotations

import sys

from extra.llm_eval_harness import (
  DEFAULT_TIMEOUT, _read_jsonl, main as _llm_main, score_prompt, summarize_results, summary_markdown,
)

DEFAULT_OUT = "bench/qwen-eval-20260612/8b-shared"

def main() -> int:
  argv = sys.argv[1:]
  if "--out" not in argv:
    argv = ["--out", DEFAULT_OUT] + argv
  return _llm_main(argv)

if __name__ == "__main__":
  raise SystemExit(main())
