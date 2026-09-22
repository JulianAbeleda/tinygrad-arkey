"""Role-qualified Q4_K FFN-down static oracle.

The down role is deliberately kept out of the gate/up Stream-K provider.  This
module owns the fixed full-grid oracle ABI and the scalar reference used to
check it.  The CUDA tile grammar is shared only as a *textual tile primitive*;
the entry point, shape guards, indexing and launch geometry are independent.
"""
from __future__ import annotations
import numpy as np

M, N, K = 512, 4096, 12288
WORDS_PER_BLOCK = 36

def q4k_scalar(words: np.ndarray, x: np.ndarray, scales: np.ndarray,
               sums: np.ndarray, *, m=M, n=N, k=K) -> np.ndarray:
  """Independent CPU Q4_K x Q8 roundpoint oracle.

  ``x`` is int8[M,K], scales/sums are fp32[M,K/32].  Every arithmetic
  product is accumulated in fp64 and rounded to fp32 only at the output,
  making this useful for catching both packed-address and metadata mistakes.
  """
  w = np.asarray(words, dtype=np.uint32).reshape(n, k // 256, WORDS_PER_BLOCK)
  xi = np.asarray(x, dtype=np.int8).reshape(m, k)
  sc = np.asarray(scales, dtype=np.float32).reshape(m, k // 32)
  sm = np.asarray(sums, dtype=np.float32).reshape(m, k // 32)
  out = np.empty((m, n), dtype=np.float32)
  for col in range(n):
    for row in range(m):
      total = 0.0
      for block in range(k // 256):
        b = w[col, block]
        word0 = int(b[0])
        d = np.frombuffer(np.uint32(word0 & 0xffff).tobytes(), np.float16)[0].astype(np.float32)
        dm = np.frombuffer(np.uint32(word0 >> 16).tobytes(), np.float16)[0].astype(np.float32)
        for g in range(8):
          if g < 4:
            qs, mn = int(b[1 + g] & 63), int(b[1 + 4 + g] & 63)
          else:
            h = int(b[1 + 8 + g - 4])
            qs = (h & 15) | ((int(b[1 + g - 4]) >> 6) << 4)
            mn = (h >> 4) | ((int(b[1 + 4 + g - 4]) >> 6) << 4)
          q = np.empty(32, dtype=np.int8)
          for p in range(32):
            byte = int(b[4 + (g // 2) * 8 + p // 4]) >> ((p % 4) * 8)
            q[p] = (byte >> (4 * (g & 1))) & 15
          j = block * 8 + g
          total += float(d * qs) * float(np.dot(q.astype(np.float64), xi[row, block*256+g*32:block*256+(g+1)*32])) * float(sc[row,j])
          total -= float(dm * mn) * float(sm[row,j])
      out[row, col] = np.float32(total)
  return out

def source() -> str:
  """Return a down-only full-grid kernel source with fixed 2-D indexing."""
  from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC, lexical_src
  s = lexical_src(SRC)
  # lexical_src emits only the static complete entry.  Specialize the ABI to
  # the sole down shape and rename it so cache/program identity cannot alias
  # gate/up assets.
  s = s.replace("q4k_imma_complete", "q4k_down_static_oracle")
  s = s.replace("int mb=blockIdx.y*128,nb=blockIdx.x*128,blocks=K/256;",
                "int mb=blockIdx.y*128,nb=blockIdx.x*128,blocks=48;")
  s = s.replace("row*K+blk*256", "row*12288+blk*256")
  s = s.replace("row*N+col", "row*4096+col")
  return s

def launch_spec() -> dict:
  return {"shape": {"M": M, "N": N, "K": K}, "grid": (N//128, M//128, 1),
          "block": (256, 1, 1), "kernel": "q4k_down_static_oracle",
          "blocks_per_row": K//256}
