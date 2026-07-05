import numpy as np, time
from tinygrad import Tensor, dtypes, TinyJit, Device
from tinygrad.helpers import getenv
dev=Device[Device.DEFAULT]

M=getenv("M",512); N=getenv("N",4096); K=getenv("K",4096)
DT=getenv("DT","int8")  # int8 | fp16
flops = 2*M*N*K

if DT=="int8":
  a=Tensor((np.random.randint(-8,8,(M,K))).astype(np.int8)); b=Tensor((np.random.randint(-8,8,(N,K))).astype(np.int8))
  acc=dtypes.int
else:
  a=Tensor(np.random.randn(M,K).astype(np.float16)); b=Tensor(np.random.randn(N,K).astype(np.float16))
  acc=dtypes.float
a.realize(); b.realize()

@TinyJit
def mm(): return a.matmul(b.transpose(), dtype=acc).realize()

for _ in range(5): mm()
Tensor.empty(1).realize()
best=1e9
for _ in range(50):
  st=time.perf_counter(); mm(); dev.synchronize()
  best=min(best, time.perf_counter()-st)
print(f"DT={DT} {M}x{N}x{K} best={best*1e3:.3f}ms  {flops/best/1e12:.2f} TFLOP/s (int:TOP/s)")
