"""Decode residual add + RMSNorm + bf16 hi/lo operand for the next GEMM, as the sampler step runs it per block:
hidden = (hidden + mixed).contiguous(); normed = RMSNorm(hidden) -> bf16 hi, lo.  usage: python3 norm_iso.py B"""
import sys, time
from tinygrad import Tensor, dtypes, TinyJit, Device, nn
B, D = int(sys.argv[1]), 3136
Tensor.manual_seed(0)
norm = nn.RMSNorm(D, 1e-5); norm.weight = Tensor.rand(D).contiguous().realize()
h = Tensor.randn(B, 1, D).contiguous().realize(); m = Tensor.randn(B, 1, D).cast(dtypes.bfloat16).contiguous().realize()
@TinyJit
def f(h, m):
  x = (h + m.cast(h.dtype)).contiguous()
  n = norm(x)
  hi = n.cast(dtypes.bfloat16); lo = (n - hi.float()).cast(dtypes.bfloat16)
  return Tensor.realize(x, hi.contiguous(), lo.contiguous())
ts = []
for i in range(30):
  Device.default.synchronize(); t0 = time.perf_counter(); f(h, m); Device.default.synchronize(); ts.append(time.perf_counter() - t0)
ts = sorted(ts[3:]); print(f"NORM B={B} median {1e6*ts[len(ts)//2]:.1f} us host (3 kernels + launch)")
