"""NV installed-row timing probe for the flash score/combine kernels (Scope C control).

Captures the DEBUG=2 prime token at a fixed decode depth and reports per-kernel rows for
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` and `flash_fused_gmax_combine_32_128`
(count, median us, total us), plus the house pins: fixed-depth token sha256, first token, decode
sha256 (model_e2e_bench.py convention), and the bench census `prefill_overlay_promotion` row.
No implementation code is touched; this is measurement evidence only.
"""
import argparse, contextlib, hashlib, io, json, re, statistics, sys, time
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
import tinygrad.llm.model as tgm

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
TM_RE = re.compile(r"^\*\*\* NV\s+\d+\s+(\S+)\s+arg\s+\d+.*?tm\s+([\d.]+)(us|ms)/")
SCORE_NAME = "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128"
COMBINE_NAME = "flash_fused_gmax_combine_32_128"

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
  total_us = sum(statistics.median(v) * len(v) for v in per_kernel.values())
  score = per_kernel.get(SCORE_NAME, [])
  combine = per_kernel.get(COMBINE_NAME, [])

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
    "kernel_us_total": round(total_us),
    "score_count": len(score),
    "score_us_median": round(statistics.median(score), 2) if score else None,
    "score_us_min": round(min(score), 2) if score else None,
    "score_us_max": round(max(score), 2) if score else None,
    "combine_count": len(combine),
    "combine_us_median": round(statistics.median(combine), 2) if combine else None,
    "combine_us_min": round(min(combine), 2) if combine else None,
    "combine_us_max": round(max(combine), 2) if combine else None,
    "tok_s_median": round(statistics.median(tok_s), 3),
    "token_sha_reps": shas,
    "first_token_reps": firsts,
  }
  print(json.dumps(out, indent=1))

if __name__ == "__main__":
  main()
