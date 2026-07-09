import os, numpy as np
os.environ["ALLOW_DEVICE_USAGE"]="1"
from tinygrad import Tensor
M=N=K=64
np.random.seed(0)
na=np.random.randn(M,K).astype(np.float16); nb=np.random.randn(K,N).astype(np.float16)
c=Tensor(na).matmul(Tensor(nb)).realize().numpy().astype(np.float32)
ref=na.astype(np.float32)@nb.astype(np.float32)
err=np.abs(c-ref)
# per 16x16 subtile mean err and nan count
print("subtile [row16 x col16] : rmse (nan_count)")
for i in range(4):
  row=[]
  for j in range(4):
    blk=c[i*16:(i+1)*16, j*16:(j+1)*16]
    rblk=ref[i*16:(i+1)*16, j*16:(j+1)*16]
    fin=np.isfinite(blk)
    rmse=np.sqrt(((blk[fin]-rblk[fin])**2).mean()) if fin.any() else float('nan')
    row.append(f"{rmse:8.2f}({int((~fin).sum()):3d})")
  print(f" r{i}:", " ".join(row))
