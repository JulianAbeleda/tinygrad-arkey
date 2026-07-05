import numpy as np, os
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import getenv

M = getenv("M", 64); N = getenv("N", 64); K = getenv("K", 64)
FORM = getenv("FORM", 0)  # 0=broadcast-int, 1=plain-dot-int-acc, 2=broadcast-char-mul
np.random.seed(0)
a = np.random.randint(-8, 8, (M, K)).astype(np.int8)
b = np.random.randint(-8, 8, (N, K)).astype(np.int8)
ref = a.astype(np.int32) @ b.astype(np.int32).T

at = Tensor(a); bt = Tensor(b)
if FORM == 0:
  out = (at.reshape(M,1,K).cast(dtypes.int) * bt.reshape(1,N,K).cast(dtypes.int)).sum(2)
elif FORM == 1:
  out = at.matmul(bt.transpose(), dtype=dtypes.int)
elif FORM == 2:
  out = (at.reshape(M,1,K) * bt.reshape(1,N,K)).sum(2, dtype=dtypes.int)
elif FORM == 3:
  out = at.dot(bt.transpose(), dtype=dtypes.int)
res = out.numpy()
diff = np.abs(res.astype(np.int64) - ref.astype(np.int64))
print(f"FORM={FORM} shape={M}x{N}x{K} max_abs_diff={diff.max()} dtype={out.dtype}")
