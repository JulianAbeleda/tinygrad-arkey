import os; os.environ['DEV']='AMD'
from tinygrad import Tensor, dtypes
from tinygrad.helpers import Context
a=Tensor.randn(512,512,dtype=dtypes.half).realize(); b=Tensor.randn(512,512,dtype=dtypes.half).realize()
with Context(TC_OPT=2):
    out=(a@b - (a@b).max(-1,keepdim=True)).numpy()
print("DONE: p2_probe_3")
