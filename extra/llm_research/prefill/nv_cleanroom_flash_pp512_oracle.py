"""Independent deterministic oracle for a clean-room pp512 Flash primitive."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

S, D, HQ, HKV, G = 512, 128, 32, 8, 4
CANARY = np.float32(1234567.0)

@dataclass(frozen=True)
class FlashFixture:
  q: np.ndarray
  k: np.ndarray
  v: np.ndarray
  expected: np.ndarray
  mask: np.ndarray
  canary: np.ndarray

def _head_major(a: np.ndarray, heads: int) -> np.ndarray:
  if a.shape != (heads, S, D) or not a.flags.c_contiguous:
    raise AssertionError(f"expected contiguous ({heads},{S},{D}), got {a.shape} {a.strides}")
  item = a.dtype.itemsize
  if a.strides != (S * D * item, D * item, item): raise AssertionError(f"bad head-major strides: {a.strides}")
  return a

def make_fixture(seed: int = 20260830) -> FlashFixture:
  rng = np.random.default_rng(seed)
  # Production Flash receives fp32 Q and half-rounded fp16 K/V.
  q = np.ascontiguousarray(rng.standard_normal((HQ, S, D), dtype=np.float32) * .125)
  k = np.ascontiguousarray((rng.standard_normal((HKV, S, D), dtype=np.float32) * .125).astype(np.float16))
  v = np.ascontiguousarray((rng.standard_normal((HKV, S, D), dtype=np.float32) * .125).astype(np.float16))
  _head_major(q, HQ); _head_major(k, HKV); _head_major(v, HKV)
  mask = np.tril(np.ones((S, S), dtype=np.bool_))
  out = np.empty((HQ, S, D), dtype=np.float32)
  scale = np.float32(1.0 / np.sqrt(D))
  for h in range(HQ):
    kvh = h // G
    scores = (q[h].astype(np.float32) @ k[kvh].astype(np.float32).T) * scale
    scores[~mask] = -np.inf
    scores -= np.max(scores, axis=1, keepdims=True)
    probs = np.exp(scores, dtype=np.float32)
    probs /= np.sum(probs, axis=1, keepdims=True, dtype=np.float32)
    out[h] = probs @ v[kvh]
  if not np.isfinite(out).all(): raise AssertionError("oracle produced non-finite output")
  canary = np.full((HQ, S, D + 8), CANARY, dtype=np.float32)
  canary[:, :, :D] = out
  return FlashFixture(q, k, v, out, mask, canary)

def check_output(actual: np.ndarray, fixture: FlashFixture, atol: float = 2e-4, rtol: float = 2e-4) -> dict:
  if actual.shape != fixture.expected.shape or not actual.flags.c_contiguous: raise AssertionError("output shape/layout mismatch")
  finite = bool(np.isfinite(actual).all())
  delta = np.abs(actual.astype(np.float64) - fixture.expected.astype(np.float64))
  ok = finite and bool(np.allclose(actual, fixture.expected, atol=atol, rtol=rtol))
  return {"finite": finite, "allclose": ok, "max_abs": float(delta.max()),
          "mean_abs": float(delta.mean()), "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(fixture.expected))}

def fixture_identity(fixture: FlashFixture) -> dict:
  return {"seed_geometry": {"S":S,"D":D,"Hq":HQ,"Hkv":HKV,"gqa":G},
          "q_sha256": hashlib.sha256(fixture.q.tobytes()).hexdigest(),
          "k_sha256": hashlib.sha256(fixture.k.tobytes()).hexdigest(),
          "v_sha256": hashlib.sha256(fixture.v.tobytes()).hexdigest(),
          "expected_sha256": hashlib.sha256(fixture.expected.tobytes()).hexdigest()}

if __name__ == "__main__":
  f = make_fixture(); print(fixture_identity(f)); print(check_output(f.expected, f))
