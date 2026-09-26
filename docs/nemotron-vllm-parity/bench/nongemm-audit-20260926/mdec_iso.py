"""One Mamba decode step (mamba_replay_buffers_step, the profiled sampler's path) with a stub 4B block, B sequences,
ring slot 8 of 16. Plain bf16 matmuls stand in for ssm_in/ssm_out (their kernels are not part of this audit).
usage: DEBUG=2 python3 mdec_iso.py B"""
import sys, time, types
from tinygrad import Tensor, dtypes, TinyJit, Device, Variable, nn
from tinygrad.llm.nemotron_h_decode import mamba_replay_buffers, mamba_replay_buffers_step, mamba_replay_flush
B = int(sys.argv[1]); D, H, P, G, N, K = 3136, 96, 80, 8, 128, 4
inner = H * P; conv_dim = inner + 2 * G * N
Tensor.manual_seed(0)
class Lin:
  def __init__(s, o, i): s.weight = (Tensor.randn(o, i) * i ** -0.5).cast(dtypes.bfloat16).contiguous().realize()
  def __call__(s, x): return x.cast(dtypes.bfloat16) @ s.weight.T
cfg = types.SimpleNamespace(ssm_inner=inner, ssm_groups=G, ssm_state=N, ssm_heads=H, conv_kernel=K, norm_eps=1e-5)
blk = types.SimpleNamespace(config=cfg, ssm_in=Lin(inner + conv_dim + H, D), ssm_out=Lin(D, inner), attn_norm=nn.RMSNorm(D, 1e-5),
  ssm_conv1d=types.SimpleNamespace(weight=(Tensor.randn(conv_dim, K) * 0.3).contiguous().realize(), bias=Tensor.zeros(conv_dim).contiguous().realize()),
  ssm_dt={"bias": Tensor.randn(H).contiguous().realize()}, ssm_a=(-Tensor.rand(H, 1) * 4 - 0.5).contiguous().realize(),
  ssm_d=Tensor.ones(H, 1).contiguous().realize(), ssm_norm=types.SimpleNamespace(weight=Tensor.ones(inner).contiguous().realize()))
buf = mamba_replay_buffers(blk, Tensor.zeros(B, K - 1, conv_dim).contiguous(), Tensor.randn(B, H, P, N).contiguous(), 16)
h = Tensor.randn(B, 1, D).contiguous().realize()
slot = Variable("slot", 0, 15)
@TinyJit
def step(h, s): return mamba_replay_buffers_step(blk, h, buf, s).contiguous().realize()
@TinyJit
def flush(c): mamba_replay_flush(blk, buf, c)
for fn, args, name in ((step, lambda: (h, slot.bind(8)), "STEP"), (flush, lambda: (Variable("count", 1, 16).bind(16),), "FLUSH")):
  ts = []
  for i in range(12):
    Device.default.synchronize(); t0 = time.perf_counter(); fn(*args()); Device.default.synchronize(); ts.append(time.perf_counter() - t0)
  ts = sorted(ts[3:]); print(f"MDEC {name} B={B} median {1e3*ts[len(ts)//2]:.3f} ms host")
