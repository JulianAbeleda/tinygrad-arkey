"""One decode attention layer through SharedPrefixKV.attend (the profiled sampler's path), random bf16 K/V.
TC=1 swaps _exact_dot (fp32-widened operands, CUDA cores) for a bf16 x bf16 -> fp32 matmul (tensor cores; the products
are exact either way, only the accumulation order changes). usage: TC=n python3 att_iso.py B P STEP BUCKET"""
import sys, os, time
from tinygrad import Tensor, dtypes, TinyJit, Device, Variable
import tinygrad.llm.nemotron_h_attention as A
if int(os.environ.get("TC", "0")): A._exact_dot = lambda a, b: a.matmul(b, dtype=dtypes.float)
B, P, STEP, BUCKET = (int(x) for x in sys.argv[1:5])
Tensor.manual_seed(0)
layer = A.SharedPrefixKV(B, 8, 128, P, BUCKET)
layer.load_prefix(Tensor.randn(1, 8, P, 128).cast(dtypes.bfloat16), Tensor.randn(1, 8, P, 128).cast(dtypes.bfloat16))
layer.suffix["k"].assign(Tensor.randn(B, 8, BUCKET, 128).cast(dtypes.bfloat16)).realize()
layer.suffix["v"].assign(Tensor.randn(B, 8, 128, BUCKET).cast(dtypes.bfloat16)).realize()
q = Tensor.randn(B, 40, 1, 128).contiguous().realize()
pl, st = Variable("prefix_length", 1, P), Variable("step", 0, BUCKET - 1)
@TinyJit
def att(q, pl, st): return layer.attend(q, pl, st, BUCKET).contiguous().realize()
ts = []
for i in range(20):
  Device.default.synchronize(); t0 = time.perf_counter(); o = att(q, pl.bind(P), st.bind(STEP)); Device.default.synchronize()
  ts.append(time.perf_counter() - t0)
ts = sorted(ts[3:]); import numpy as np
np.save(f"/home/ubuntu/storage/audit-nongemm/att_{B}_{P}_tc{os.environ.get('TC','0')}.npy", o.numpy())
print(f"ATT TC={os.environ.get('TC','0')} B={B} P={P} step={STEP} bucket={BUCKET} median {1e3*ts[len(ts)//2]:.3f} ms")
