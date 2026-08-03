"""NV GEMV-class census for the Scope D like-for-like cap settling check.

Captures the DEBUG=2 prime token at a fixed decode depth (d512 default, fused prefill
attention disabled) and classifies the per-kernel rows into GEMV-class vs everything
else, excluding the vocab-head kernels (q6k_gen_coop_151936_4096 with any suffix,
q6k_vocab_scalar_reduce) and the scatter chain (E_1187/r_1187), matching the class
definition in decode-gap-per-target-lever-scope-20260802.md section 8.1. Reports the
GEMV-class sum, a full per-kernel table, and the house pins (fixed-depth token sha256,
first token, decode sha256, census row). No implementation code is touched.
"""
import argparse, contextlib, hashlib, io, json, re, statistics, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
import tinygrad.llm.model as tgm

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")
VOCAB_COOP = "q6k_gen_coop_151936_4096"
VOCAB_REDUCE = "q6k_vocab_scalar_reduce"

def is_gemv_class(name: str) -> bool:
  """GEMV-class = q4k lanemap + q6k coop/partial, minus vocab head and scatter plumbing."""
  if name == VOCAB_REDUCE or name.startswith(VOCAB_COOP + "_") or name == VOCAB_COOP:
    return False
  if "1187" in name:
    return False
  return name.startswith("q4k_g3_lanemap_gemv") or name.startswith("q6k_gen_coop") or \
    name.startswith("q6k_gen_partial")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--nmeas", type=int, default=20)
  ap.add_argument("--reps", type=int, default=3)
  args = ap.parse_args()
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  model, kv = Transformer.from_gguf(MODEL, 4608)
  census = (model.config.admit or {}).get("prefill_overlay_promotion")
  prompt = [1] * args.depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
  buf = io.StringIO()
  marks = []
  with contextlib.redirect_stdout(buf):
    with Context(DEBUG=2):
      next(gen); marks.append(len(buf.getvalue().splitlines()))
      next(gen); marks.append(len(buf.getvalue().splitlines()))
      next(gen); marks.append(len(buf.getvalue().splitlines()))
  gen.close()
  log = buf.getvalue().splitlines()
  prime_end = marks[0]
  per_kernel = {}
  for l in log[:prime_end]:
    m = TM_RE.match(l)
    if not m: continue
    us = float(m.group(2)) * (1e-3 if m.group(3) == "ms" else 1.0)
    per_kernel.setdefault(m.group(1), []).append(us)
  total = sum(len(v) for v in per_kernel.values())
  gemv_rows = {k: v for k, v in per_kernel.items() if is_gemv_class(k)}
  gemv_sum_us = sum(statistics.median(v) * len(v) for v in gemv_rows.values())
  gemv_count = sum(len(v) for v in gemv_rows.values())
  all_sum_us = sum(statistics.median(v) * len(v) for v in per_kernel.values())

  from tinygrad import Device
  dev = Device[Device.DEFAULT]
  tok_s, shas, firsts = [], [], []
  for _ in range(args.reps):
    model.reset_generation_state()
    gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
    next(gen)
    dev.synchronize()
    lat, toks = [], []
    for _ in range(args.nmeas):
      t0 = time.perf_counter()
      toks.append(int(next(gen)))
      lat.append(time.perf_counter() - t0)
    gen.close()
    tok_s.append(args.nmeas / sum(lat))
    shas.append(hashlib.sha256(",".join(map(str, toks)).encode()).hexdigest())
    firsts.append(toks[0])

  out = {
    "depth": args.depth,
    "census_prefill_overlay_promotion": census,
    "kernels_per_token": total,
    "all_kernel_us_total": round(all_sum_us),
    "gemv_class_count": gemv_count,
    "gemv_class_us_sum": round(gemv_sum_us),
    "gemv_class_ms_sum": round(gemv_sum_us / 1e3, 3),
    "per_kernel_us": {k: {"count": len(v), "median_us": round(statistics.median(v), 2),
                           "total_us": round(sum(v), 2), "min_us": round(min(v), 2),
                           "max_us": round(max(v), 2)} for k, v in per_kernel.items()},
    "tok_s_median": round(statistics.median(tok_s), 3),
    "token_sha_reps": shas,
    "first_token_reps": firsts,
  }
  print(json.dumps(out, indent=1))

if __name__ == "__main__":
  main()
