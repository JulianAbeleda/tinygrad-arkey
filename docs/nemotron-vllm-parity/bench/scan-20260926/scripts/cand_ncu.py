"""ncu target: one derived candidate on one exact shape via the DEV=CUDA proxy, labelled for BoltBeam's collector.
usage: DEV=CUDA python3 cand_ncu.py ROLE ROWS 'JSON [geometry, split, pipeline]' ..."""
import sys, json, ctypes
sys.path.insert(0, "/home/ubuntu/tinygrad-self-training")
import extra.llm_research.prefill.dense_bf16_geometry_search as S
from tinygrad import Device
from tinygrad.helpers import GlobalCounters
cuda = ctypes.CDLL("libcuda.so.1")
role, m = sys.argv[1], int(sys.argv[2])
n, k = next((n, k) for r, n, k in S.ROLES if r == role)
ops = S._Operands(m, n, k)
for spec in sys.argv[3:]:
  geom, split, pipe = json.loads(spec)
  row = S.measure(role, ops, tuple(geom), split, tuple(pipe), 2)
  print("warm", row, file=sys.stderr, flush=True)
  Device.default.synchronize()
  k0 = GlobalCounters.kernel_count
  cuda.cuProfilerStart(); S.measure(role, ops, tuple(geom), split, tuple(pipe), 0); Device.default.synchronize(); cuda.cuProfilerStop()
  print(f"SHAPE {role} {m} {GlobalCounters.kernel_count - k0}", flush=True)
