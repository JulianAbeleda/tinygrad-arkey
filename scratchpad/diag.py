import os, numpy as np
os.environ["ALLOW_DEVICE_USAGE"]="1"
from tinygrad import Tensor
M=N=K=64
np.random.seed(0)
na=np.random.randn(M,K).astype(np.float16); nb=np.random.randn(K,N).astype(np.float16)
c=Tensor(na).matmul(Tensor(nb)).realize().numpy().astype(np.float32)
print("shape",c.shape,"nan%",np.isnan(c).mean(),"zero%",(c==0).mean())
fin=c[np.isfinite(c)]
print("finite count",fin.size,"min",fin.min() if fin.size else None,"max",fin.max() if fin.size else None)
# per-row nan fraction to see structure
nanrow=np.isnan(c).mean(1)
print("row nan fracs (first 8):",np.round(nanrow[:8],2))
print("col nan fracs (first 8):",np.round(np.isnan(c).mean(0)[:8],2))
# raw bytes of a few
print("c[0,:6]",c[0,:6])
print("c[:6,0]",c[:6,0])
