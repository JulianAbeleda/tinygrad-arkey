"""Isolated Nemotron-H SSD mixer (one Mamba layer, 1024-token piece, 4B shapes) through the production ssd_mixer,
with a stub block (random weights). usage: DEBUG=2 DEV=NV python3 ssd_iso.py PREC CHUNK [reps]"""
import sys, time, types, os
from tinygrad import Tensor, dtypes, Device, Variable, TinyJit
sys.path.insert(0, "/home/ubuntu/storage/audit-nongemm"); import ssd_variant
from tinygrad.llm.nemotron_h_ssd import ssd_mixer
PREC, CHUNK = sys.argv[1], int(sys.argv[2]); REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
L, D, H, P, G, N, K = 1024, 3136, 96, 80, 8, 128, 4
inner = H * P; conv_dim = inner + 2 * G * N
Tensor.manual_seed(0)
class Lin:
  def __init__(s, o, i): s.weight = (Tensor.randn(o, i) * i ** -0.5).cast(dtypes.bfloat16).contiguous().realize()
  def __call__(s, x): return x.cast(dtypes.bfloat16) @ s.weight.T
cfg = types.SimpleNamespace(ssm_inner=inner, ssm_groups=G, ssm_state=N, ssm_heads=H, conv_kernel=K, norm_eps=1e-5)
import os
if os.environ.get("NOGEMM"):
  _proj = (Tensor.randn(1, L, inner + conv_dim + H) * 0.5).cast(dtypes.bfloat16).contiguous().realize()
  ssm_in, ssm_out = (lambda x: _proj + x[..., :1] * 0), (lambda x: x)
else: ssm_in, ssm_out = Lin(inner + conv_dim + H, D), Lin(D, inner)
blk = types.SimpleNamespace(config=cfg, ssm_in=ssm_in, ssm_out=ssm_out,
  ssm_conv1d=types.SimpleNamespace(weight=(Tensor.randn(conv_dim, K) * 0.3).contiguous().realize(), bias=Tensor.zeros(conv_dim).contiguous().realize()),
  ssm_dt={"bias": Tensor.randn(H).contiguous().realize()}, ssm_a=(-Tensor.rand(H, 1) * 4 - 0.5).contiguous().realize(),
  ssm_d=Tensor.ones(H, 1).contiguous().realize(), ssm_norm=types.SimpleNamespace(weight=Tensor.ones(inner).contiguous().realize()))
hidden = (Tensor.randn(1, L, D) * 0.5).cast(dtypes.bfloat16).contiguous().realize()
conv = Tensor.zeros(1, K - 1, conv_dim).contiguous().realize()
state = Tensor.zeros(1, H, P, N).contiguous().realize()
v = Variable("valid", 1, L)
@TinyJit
def run(h, c, s, vv):
  out, nc = ssd_mixer(blk, h, {"conv": c, "state": s}, chunk=CHUNK, precision=PREC, valid=vv)
  return out.contiguous().realize(), nc["state"].contiguous().realize()
for r in range(REPS):
  Device.default.synchronize(); t = time.perf_counter()
  o, st = run(hidden, conv, state, v.bind(L)); Device.default.synchronize()
  print(f"rep {r} {PREC} chunk={CHUNK} {1e3*(time.perf_counter()-t):.2f} ms", flush=True)
import numpy as np
tag = f"{PREC}_{CHUNK}_m{os.environ.get('MAT','0')}"
np.save(f"/home/ubuntu/storage/audit-nongemm/y_{tag}.npy", o.float().numpy()); np.save(f"/home/ubuntu/storage/audit-nongemm/s_{tag}.npy", st.numpy())
