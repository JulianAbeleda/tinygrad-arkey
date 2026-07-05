import numpy as np
from tinygrad import Tensor, dtypes
np.random.seed(1)
M=N=K=16
a = np.random.randint(-3,4,(M,K)).astype(np.int8)
b = np.random.randint(-3,4,(N,K)).astype(np.int8)
ref = a.astype(np.int32) @ b.astype(np.int32).T
out = Tensor(a).matmul(Tensor(b).transpose(), dtype=dtypes.int).numpy()
print("ref[0,:8]  ", ref[0,:8])
print("out[0,:8]  ", out[0,:8])
print("ref[:8,0]  ", ref[:8,0])
print("out[:8,0]  ", out[:8,0])
# is out a transpose of ref?
print("out==ref.T ?", np.array_equal(out, ref.T))
print("out==ref   ?", np.array_equal(out, ref))
# diagonal check
print("match count", (out==ref).sum(), "/", M*N)
