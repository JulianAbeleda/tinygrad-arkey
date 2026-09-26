"""The sampler tail after the output GEMM, B rows x 131072 vocab.
V=0: NemotronHBatchSampler._logprobs/_sample verbatim.
V=1: same math, the log-sum-exp normalizer materialized ([B,1]) before the per-element subtraction.
V=2: no full log-prob tensor: gumbel argmax on the scaled logits (the normalizer does not move the argmax), chosen
     log prob = scaled logit[token] - logsumexp. usage: V=n python3 tail_iso.py B [reps]"""
import sys, os, time
from tinygrad import Tensor, dtypes, TinyJit, Device
B, VOC = int(sys.argv[1]), 131072
VAR = int(os.environ.get("V", "0"))
Tensor.manual_seed(0)
raw = (Tensor.randn(B, 1, VOC + 64) * 3).contiguous().realize()
bias = Tensor.zeros(VOC).contiguous().realize()
temperature = Tensor([1.0]).realize()
@TinyJit
def tail(raw, temperature):
  logits = raw[:, 0, :VOC]
  logits = (logits.float() + bias).contiguous()
  scaled = logits / temperature
  if VAR == 0:
    logprobs = scaled.log_softmax(-1).contiguous()
  elif VAR >= 1:
    m = scaled.max(-1, keepdim=True).contiguous()
    lse = (m + (scaled - m).exp().sum(-1, keepdim=True).log()).contiguous()
    if VAR == 1: logprobs = (scaled - lse).contiguous()
  if VAR == 2:
    gumbel = -(-(Tensor.rand_like(scaled).maximum(1e-12)).log()).log()
    token = (scaled + gumbel).argmax(-1)
    chosen = scaled.gather(-1, token.unsqueeze(-1))[:, 0] - lse[:, 0]
  else:
    gumbel = -(-(Tensor.rand_like(logprobs).maximum(1e-12)).log()).log()
    token = (logprobs + gumbel).argmax(-1)
    chosen = logprobs.gather(-1, token.unsqueeze(-1))[:, 0]
  return token.cast(dtypes.int32).realize(), chosen.realize()
ts = []
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 20):
  Device.default.synchronize(); t0 = time.perf_counter(); t, c = tail(raw, temperature); Device.default.synchronize()
  ts.append(time.perf_counter() - t0)
ts = sorted(ts[3:]); print(f"TAIL V={VAR} B={B} median {1e3*ts[len(ts)//2]:.3f} ms (host wall, synced)  tok {t.numpy()[:3]} chosen {c.numpy()[:3]}")
