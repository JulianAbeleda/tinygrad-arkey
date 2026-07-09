import os, numpy as np
os.environ["ALLOW_DEVICE_USAGE"]="1"
from tinygrad import Tensor
M=N=K=64
np.random.seed(0)
na=np.random.randn(M,K).astype(np.float16); nb=np.random.randn(K,N).astype(np.float16)
outs=[]
for t in range(2):
  c=Tensor(na).matmul(Tensor(nb)).realize().numpy().astype(np.float32)
  outs.append(c)
  print(f"run{t}: nan%={np.isnan(c).mean():.4f} finite_absmax={np.nanmax(np.abs(c)):.1f} c[0,1]={c[0,1]:.3f} c[2,3]={c[2,3]:.3f}")
a,b=outs
both=np.isfinite(a)&np.isfinite(b)
print("identical where both finite:", np.allclose(a[both],b[both]))
print("same NaN mask:", np.array_equal(np.isnan(a),np.isnan(b)))
