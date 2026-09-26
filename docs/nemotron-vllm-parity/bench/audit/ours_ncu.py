"""ncu target for BoltBeam's production projection kernels.

tinygrad's NV backend drives the GPU without the CUDA driver, so ncu cannot see it. DEV=CUDA renders the byte-identical
kernel source (checked: only the name hash differs) once the route table is admitted for the CUDA device, which this
script does by reporting the NV target for it. One profiled production call per (role, M) inside cuProfilerStart/Stop;
stdout lists `SHAPE role m kernels` in launch order so ncu rows can be mapped back.
usage: DEV=CUDA ncu --profile-from-start off ... python3 ours_ncu.py ROLES MS [mode]
"""
import sys, ctypes
import tinygrad.llm.dense_candidate_gemm as g
_dt = g.device_target
g.device_target = lambda dev: {**_dt(dev), "backend": "NV"}
from tinygrad import Tensor, dtypes, Device
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.nemotron_h import _Linear
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ours_gemm import ROLES

cuda = ctypes.CDLL("libcuda.so.1")
mode = sys.argv[3] if len(sys.argv) > 3 else "prod"
for role in sys.argv[1].split(","):
  n, k = ROLES[role]
  lin = _Linear(k, n, bias=False)
  lin.weight = (Tensor.randn(n, k) * 0.02).cast(dtypes.bfloat16).contiguous().realize()
  assert g.bind_projection(lin, role) is not None
  for m in (int(v) for v in sys.argv[2].split(",")):
    x = Tensor.randn(m, k).contiguous().realize()
    if mode == "bf16": x = x.cast(dtypes.bfloat16).contiguous().realize()
    def call():
      if mode == "bf16": return g.route_dense_bf16(x, lin._candidate.padded, role, n).contiguous().realize()
      if m > 128:
        with g.route_scratch(): return lin(x).contiguous().realize()
      return lin(x).contiguous().realize()
    call(); call(); Device.default.synchronize()
    k0 = GlobalCounters.kernel_count
    cuda.cuProfilerStart(); call(); Device.default.synchronize(); cuda.cuProfilerStop()
    print(f"SHAPE {role} {m} {GlobalCounters.kernel_count - k0}", flush=True)
