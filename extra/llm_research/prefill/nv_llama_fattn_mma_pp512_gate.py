"""HCQ exact-output gate for the whole-tile llama MMA Flash cubin."""
import numpy as np
from tinygrad import Tensor
from extra.llm_research.prefill.nv_llama_fattn_mma_pp512_binding import project

root="/tmp/nv-cleanroom-flash-fixture/"
q=np.fromfile(root+"q.bin",np.float32).reshape(1,32,512,128)
k=np.fromfile(root+"k.bin",np.float16).reshape(1,8,512,128)
v=np.fromfile(root+"v.bin",np.float16).reshape(1,8,512,128)
ref=np.fromfile(root+"expected.bin",np.float32).reshape(1,32,512,128)
mask=np.where(np.arange(512)[None,:]<=np.arange(512)[:,None],0.0,-65504.0).astype(np.float16).reshape(1,1,512,512)
got=project(Tensor(q,device="NV"),Tensor(k,device="NV"),Tensor(v,device="NV"),Tensor(mask,device="NV")).realize().numpy()
d=np.abs(got-ref);close=np.allclose(got,ref,atol=2e-4,rtol=2e-4)
print({"status":"PASS" if close else "FAIL","finite":bool(np.isfinite(got).all()),"max_abs":float(d.max()),"mean_abs":float(d.mean()),"relative_l2":float(np.linalg.norm(got-ref)/np.linalg.norm(ref)),"allclose":bool(close)})
raise SystemExit(0 if close else 1)
