import os, numpy as np
os.environ["ALLOW_DEVICE_USAGE"]="1"
from tinygrad import Tensor
M,N,K=48,48,64
np.random.seed(0)
na=np.random.randn(M,K).astype(np.float16); nb=np.random.randn(K,N).astype(np.float16)
c=Tensor(na).matmul(Tensor(nb)).realize().numpy().astype(np.float32)
ref=na.astype(np.float32)@nb.astype(np.float32)
print("3x3 per-16x16-subtile rmse (nan_count):")
for i in range(3):
  row=[]
  for j in range(3):
    b=c[i*16:i*16+16,j*16:j*16+16]; r=ref[i*16:i*16+16,j*16:j*16+16]
    fin=np.isfinite(b)
    rmse=np.sqrt(((b[fin]-r[fin])**2).mean()) if fin.any() else float('nan')
    row.append(f"r{rmse:8.2f}(n{int((~fin).sum())})")
  print(f" row{i}:", " ".join(row))
