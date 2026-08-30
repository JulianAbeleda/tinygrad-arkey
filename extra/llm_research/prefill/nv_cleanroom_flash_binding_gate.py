"""Compile/load gate for the isolated NativeProgram binding."""
from tinygrad import Tensor, dtypes
import os, numpy as np
from extra.llm_research.prefill.nv_cleanroom_flash_pp512_binding import CleanroomFlashPP512Binding

def run() -> dict:
  os.environ['PATH']='/usr/local/cuda-13.2/bin:'+os.environ.get('PATH','')
  b=CleanroomFlashPP512Binding()
  root='/tmp/nv-cleanroom-flash-fixture'
  qn=np.fromfile(root+'/q.bin',dtype=np.float16).reshape(1,32,512,128)
  kn=np.fromfile(root+'/k.bin',dtype=np.float16).reshape(1,8,512,128)
  vn=np.fromfile(root+'/v.bin',dtype=np.float16).reshape(1,8,512,128)
  expected=np.fromfile(root+'/expected.bin',dtype=np.float32).reshape(1,32,512,128)
  q=Tensor(qn,device='NV'); k=Tensor(kn,device='NV'); v=Tensor(vn,device='NV')
  out=b.project(q,k,v).realize()
  vals=out.numpy()
  delta=np.abs(vals-expected)
  close=np.allclose(vals,expected,atol=2e-4,rtol=2e-4)
  return {'status':'PASS' if close else 'FAIL','shape':list(vals.shape),'dtype':str(vals.dtype),'finite':bool(np.isfinite(vals).all()),
    'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'allclose':bool(close)}

if __name__=='__main__': print(run())
