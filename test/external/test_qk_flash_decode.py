import unittest, numpy as np
from tinygrad.device import Device, Buffer
from tinygrad.dtype import dtypes
from extra.qk_flash_decode import flash_partial_src, flash_reduce_src

class TestFlashDecode(unittest.TestCase):
  def test_exact_vs_reference(self):
    Hd, Hq, Hkv, MAXC = 128, 32, 8, 4096; G = Hq // Hkv
    dev = Device["AMD"]; rng = np.random.default_rng(0)
    for Tc, S in [(3072, 8), (1024, 8), (777, 8), (100, 4)]:
      q = rng.standard_normal((Hq, Hd)).astype(np.float16)
      k = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16); v = rng.standard_normal((Hkv, MAXC, Hd)).astype(np.float16)
      pp = dev.runtime("flash_partial", dev.compiler.compile(flash_partial_src(Hd, Hq, Hkv, S, MAXC)))
      pr = dev.runtime("flash_reduce", dev.compiler.compile(flash_reduce_src(Hd, Hq, Hkv, S, MAXC)))
      def buf(a, dt): b = Buffer("AMD", a.size, dt).ensure_allocated(); b.copyin(memoryview(np.ascontiguousarray(a))); return b
      qb, kb, vb = buf(q, dtypes.half), buf(k, dtypes.half), buf(v, dtypes.half)
      pout = Buffer("AMD", Hq*S*Hd, dtypes.float32).ensure_allocated()
      pm = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated(); pl = Buffer("AMD", Hq*S, dtypes.float32).ensure_allocated()
      out = Buffer("AMD", Hq*Hd, dtypes.float32).ensure_allocated()
      pp(pout._buf, pm._buf, pl._buf, qb._buf, kb._buf, vb._buf, global_size=(Hq*S,1,1), local_size=(Hd,1,1), vals=(Tc,), wait=True)
      pr(out._buf, pout._buf, pm._buf, pl._buf, global_size=(Hq,1,1), local_size=(Hd,1,1), wait=True)
      o = np.empty(Hq*Hd, np.float32); out.copyout(memoryview(o)); got = o.reshape(Hq, Hd)
      qf, kf, vf = q.astype(np.float32), k[:, :Tc].astype(np.float32), v[:, :Tc].astype(np.float32)
      ref = np.zeros((Hq, Hd), np.float32)
      for h in range(Hq):
        kv = h // G; sc = (qf[h] @ kf[kv].T)/np.sqrt(Hd); pw = np.exp(sc-sc.max()); pw /= pw.sum(); ref[h] = pw @ vf[kv]
      self.assertLess(np.abs(got - ref).max(), 2e-2)

if __name__ == "__main__":
  unittest.main()
