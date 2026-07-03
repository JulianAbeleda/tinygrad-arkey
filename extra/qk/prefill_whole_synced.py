"""THE prefill-throughput authority (synced, TinyJit, min-of-K). This is the ONLY sanctioned way to report a prefill
pp<L> number for tinygrad-vs-llama.

Methodology: warm a TinyJit of the forward at a CONCRETE start_pos, then time a synced burst (dev.synchronize before
and after, min over K bursts) -> the pure prefill-kernel time, with NO generate()/sampling/host-jitter overhead.
whole-prefill@L = sum of per-chunk times over the 512-token chunks covering [0,L).

DO NOT roll your own prefill bench, and NEVER measure prefill via `model.generate` TTFT: TTFT includes generate's Python
overhead + sampling + host jitter and UNDERSTATES prefill by ~3x (proven 2026-07: a generate-ttft harness read 1247
tok/s for 8B @512 while this authority reads ~4400 -> the real number, ~145% of llama). See the memory note
[[prefill-bench-authority-not-ttft]].

  DEV=AMD PREFILL_V2=1 [PREFILL_GRAPH_GEMM=0] PYTHONPATH=. .venv/bin/python extra/qk/prefill_whole_synced.py [--model PATH]

Reference (gfx1100, Qwen3-8B-Q4_K_M, graph-GEMM): ~4408/4215/3822/3230 tok/s @512/1024/2048/4096, ~145% of llama
(~3020-3070). The stale 1983/66% was an older/nosync measurement.
"""
import os, time, argparse
os.environ.setdefault("PREFILL_V2", "1")
from tinygrad import Tensor, Device, TinyJit
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
from tinygrad.llm.model import PREFILL_GRAPH_GEMM

def prefill_authority(model_path:str=DEFAULT_MODEL, chunk_n:int=512, start_positions=(0, 512, 1024, 2048, 3584),
                      whole_lengths=(512, 1024, 2048, 4096), K:int=8, max_context:int=4608, verbose:bool=True) -> dict:
  """Synced whole-prefill throughput for `model_path`. Returns {'chunk_ms': {sp: ms}, 'whole_tok_s': {L: tok/s}, ...}."""
  dev = Device["AMD"]
  m, _ = load_model_and_tokenizer(model_path, max_context, seed=20260617)
  for b in m.blk: b._use_flash, b._prefill_v2 = True, True
  temp = Tensor([0.0])
  chunk = Tensor([[(i * 7) % 1000 for i in range(chunk_n)]], dtype="int32").contiguous()
  def burst(sp_int) -> float:
    j = TinyJit(m.forward)
    for _ in range(4): j(chunk, sp_int, temp).realize()   # warm capture+compile
    dev.synchronize()
    ts = []
    for _ in range(3):
      dev.synchronize(); t0 = time.perf_counter()
      for _ in range(K): j(chunk, sp_int, temp).realize()
      dev.synchronize(); ts.append((time.perf_counter() - t0) / K * 1e3)
    return min(ts)
  chunk_ms = {sp: burst(sp) for sp in start_positions}
  if verbose:
    print(f"PREFILL AUTHORITY (synced, K={K})  model={os.path.basename(model_path)}  GRAPH_GEMM={PREFILL_GRAPH_GEMM}")
    for sp, ms in chunk_ms.items(): print(f"  chunk@start_pos={sp:5}: {ms:6.1f}ms ({chunk_n/ms*1e3:.0f} tok/s)")
  import bisect
  xs = sorted(chunk_ms); ys = [chunk_ms[x] for x in xs]
  def interp(s):
    if s <= xs[0]: return ys[0]
    if s >= xs[-1]: return ys[-1]
    i = bisect.bisect_right(xs, s) - 1; return ys[i] + (ys[i+1]-ys[i]) * (s-xs[i]) / (xs[i+1]-xs[i])
  whole = {L: L / sum(interp(s) for s in range(0, L, chunk_n)) * 1e3 for L in whole_lengths}
  if verbose:
    for L, tps in whole.items(): print(f"  WHOLE-PREFILL@{L}: {tps:.0f} tok/s")
  return {"model": model_path, "chunk_ms": chunk_ms, "whole_tok_s": whole, "graph_gemm": PREFILL_GRAPH_GEMM}

if __name__ == "__main__":
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", default=DEFAULT_MODEL, help="GGUF path (default: harness DEFAULT_MODEL)")
  ap.add_argument("-K", type=int, default=8, help="bursts to min over")
  prefill_authority(**{"model_path": ap.parse_args().model, "K": ap.parse_args().K})
