#!/usr/bin/env python3
"""Size the startup tax and split it into Fable's two sub-costs, so we know whether shipping a kernel bundle (kills (b))
is enough or whether we also need to serialize the JIT schedule (kills (a)). Run COLD then WARM:

  # COLD: clear the compile cache first so every kernel is a compile MISS
  rm -f ~/.cache/tinygrad/cache.db*   (or point CACHEDB at a temp file)
  DEV=AMD JIT=1 CACHEDB=/tmp/aot_cold.db  python extra/qk/startup_measure.py --model <gguf>
  # WARM: reuse the now-populated cache -> every kernel is a compile HIT
  DEV=AMD JIT=1 CACHEDB=/tmp/aot_cold.db  python extra/qk/startup_measure.py --model <gguf>

Reported: import, model-load, first-token, next-token; and Compiler.cache_hits/misses. The split:
  (b) kernel-compile subprocess time  ~=  (first-token COLD) - (first-token WARM)   [misses drop to 0 warm]
  (a) graph-capture/render time (irreducible without schedule serialization) ~= (first-token WARM) - weight-copy - fixed
A shipped bundle makes even a fresh box behave WARM for (b). Whatever (a) remains is the residual to attack next.
"""
import time, argparse, pathlib, sys
_t0 = time.perf_counter()
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from tinygrad.device import Compiler
from tinygrad.helpers import CACHEDB
from tinygrad.llm.model import Transformer
_t_import = time.perf_counter() - _t0

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--max_context", default="1024")
  ap.add_argument("--prompt", type=int, default=8)     # short prompt -> exercise prefill + a few decode kernels quickly
  ap.add_argument("--tokens", type=int, default=4)
  args = ap.parse_args()
  h0, m0 = Compiler.cache_hits, Compiler.cache_misses

  t = time.perf_counter()
  mc = args.max_context if args.max_context == "auto" else int(args.max_context)
  model, _ = Transformer.from_gguf(args.model, mc)
  t_load = time.perf_counter() - t
  h1, m1 = Compiler.cache_hits, Compiler.cache_misses

  prompt = [((i * 7 + 3) % 2000) + 1 for i in range(args.prompt)]
  gen = model.generate(prompt)
  t = time.perf_counter(); next(gen); t_first = time.perf_counter() - t
  h2, m2 = Compiler.cache_hits, Compiler.cache_misses
  t = time.perf_counter()
  for _ in range(args.tokens): next(gen)
  t_next = (time.perf_counter() - t) / max(args.tokens, 1)
  h3, m3 = Compiler.cache_hits, Compiler.cache_misses

  print(f"CACHEDB={CACHEDB}")
  print(f"import           : {_t_import*1e3:8.0f} ms")
  print(f"model load       : {t_load*1e3:8.0f} ms   (compile hit/miss: {h1-h0}/{m1-m0})")
  print(f"first token      : {t_first*1e3:8.0f} ms   (compile hit/miss: {h2-h1}/{m2-m1})")
  print(f"next token (avg) : {t_next*1e3:8.0f} ms   (compile hit/miss: {h3-h2}/{m3-m2})")
  print(f"TOTAL compiled(miss)={m3-m0}  cached(hit)={h3-h0}   <- warm run should show miss~=0")

if __name__ == "__main__": main()
