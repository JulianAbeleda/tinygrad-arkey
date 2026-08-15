"""Decode graph position invariance (llama.cpp reference: positions are data, never graph
structure). The decode JIT must capture ONE position-invariant graph and then pure-replay it
at every later start_pos: no per-token function re-trace and no per-token schedule re-build.
Regression: the bound start_pos (BIND node) used to leak into KV store slots, rope reads and
mask extents, so every token at depth>=2048 re-traced and re-scheduled the whole decode graph
(unique function keys + schedule CACHE MISS per token, ~26s/token on NV)."""
import contextlib
import io
import pathlib

import numpy as np
import pytest

import tinygrad.llm.model_route_plan as mrp
import tinygrad.llm.model as tgm
from tinygrad import Tensor
from tinygrad.helpers import Context
from tinygrad.llm.device_facts import DeviceCapabilities, DeviceFacts, ProbeRecord
from tinygrad.uop.ops import UOp


MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
MAXC = 256


@pytest.fixture(scope="module")
def decode_model():
  if not pathlib.Path(MODEL).exists():
    pytest.skip("no local Qwen3 8B GGUF fixture")
  _shared_q8 = mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS
  _custom_prefill = tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS
  _scan_device_facts = tgm.scan_device_facts
  try:
    # closed arm: keep the shared-Q8 decode routes dormant (GPU sm_120 only anyway)
    mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = frozenset()
    tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
    # CPU device facts carry no VRAM scan; inject fake facts so admission can plan.
    cap = DeviceCapabilities(global_allocation_granularity=4096, supports_fp16=False)
    probe = ProbeRecord("probe", "now")
    tgm.scan_device_facts = lambda: DeviceFacts("CPU", "CPU", "cpu", 96 * 2**30, 64 * 2**30, cap, probe, probe)
    from tinygrad.llm.model import Transformer
    model, _ = Transformer.from_gguf(MODEL, MAXC)
    yield model
  finally:
    mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS = _shared_q8
    tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = _custom_prefill
    tgm.scan_device_facts = _scan_device_facts


def _run(model, sp: int, toks: Tensor):
  """One decode/prefill call with DEBUG=2; returns (function keys, schedule keys, cache hits)."""
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  buf = io.StringIO()
  with Context(DEBUG=2), contextlib.redirect_stdout(buf):
    out = model(toks, v_sp.bind(sp), temp, use_flash=False).realize()
  lines = buf.getvalue()
  fkeys = {l.split()[1] for l in lines.splitlines() if l.startswith("function ")}
  skeys = {t for l in lines.splitlines() if l.startswith("scheduled ")
           for t in l.split() if len(t) == 8 and all(c in "0123456789abcdef" for c in t)}
  hits = {t for l in lines.splitlines() if l.startswith("scheduled ") and " cache hit" in l
          for t in l.split() if len(t) == 8 and all(c in "0123456789abcdef" for c in t)}
  return fkeys, skeys, hits, out

def test_symbolic_prefill_and_decode_share_position_invariant_graph(decode_model):
  """Warmup (ignored), then capture at sp=200 and pure exec replay at sp=201/202.
  After capture, later positions must produce zero function and zero schedule lines,
  and the symbolic KV store must land at each exec position's own slot."""
  toks = Tensor([[7]], dtype="int32")

  # warmup: first call installs lazy storage; its graph differs from steady-state (not position)
  _run(decode_model, 0, toks)

  fc, sc, _, _ = _run(decode_model, 200, toks)
  f1, s1, h1, _ = _run(decode_model, 201, toks)
  f2, s2, h2, out = _run(decode_model, 202, toks)

  assert len(fc) == 2, f"expected two function-body keys (block parity), got {fc}"
  assert not f1 and not s1, f"exec at sp=201 re-traced: fkeys={f1} skeys={s1}"
  assert not f2 and not s2, f"exec at sp=202 re-traced: fkeys={f2} skeys={s2}"

  cache = decode_model.blk[0].cache_kv.numpy()
  assert cache[0, 0, 0, 200, :].any() and cache[0, 0, 0, 201, :].any() and cache[0, 0, 0, 202, :].any()
  assert not cache[0, 0, 0, 199, :].any() and not cache[0, 0, 0, 203, :].any()
  assert out.numpy().shape == (1, 1)
  assert all(v == v for v in out.numpy().flat), "decode output contains NaN"


def test_symbolic_prefill_mask_builds_position_invariant_extents(decode_model):
  """Symbolic-T prefill (prompt slice, like generate's chunked prefill) exercises the mask
  path (triu diagonal/extent on the unbound variable). Regression: the mask used the bound
  start_pos while KV reads went unbound, breaking symbolic broadcast at qk+mask."""
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  prompt = Tensor([[1] * 64], dtype="int32")
  for sp in (0, 32):
    toks = prompt[:, v_sp.bind(sp):v_sp.bind(sp) + 32]
    buf = io.StringIO()
    with Context(DEBUG=2), contextlib.redirect_stdout(buf):
      out = decode_model(toks, v_sp.bind(sp), temp, use_flash=False).realize()
    lines = buf.getvalue()
    fkeys = {l.split()[1] for l in lines.splitlines() if l.startswith("function ")}
    skeys = {t for l in lines.splitlines() if l.startswith("scheduled ")
             for t in l.split() if len(t) == 8 and all(c in "0123456789abcdef" for c in t)}
    assert len(fkeys) == 2, f"sp={sp}: expected 2 function keys, got {fkeys}"
    assert len(skeys) >= 1, f"sp={sp}: expected schedule lines"
    assert out.numpy().shape == (1, 1)
    assert all(v == v for v in out.numpy().flat), f"sp={sp}: prefill output contains NaN"


def test_eager_full_logits_realize_honors_bound_position(decode_model):
  """Harness pattern (JIT=0): forward_with_logits under a bound start_pos UOp, then .numpy().
  Regression: once the BIND node stopped leaking into the decode graph (position-invariant
  capture), eager realize lost the position value and raised KeyError('start_pos'); the
  model now carries the bound value on the returned tensors and realize merges it."""
  v_sp = UOp.variable("start_pos", 0, MAXC - 1)
  token = Tensor([[7]], dtype="int32")
  temp = Tensor([0.0])
  outs = []
  with Context(JIT=0):
    for sp in (16, 32):
      _, eager = decode_model.forward_with_logits(token, v_sp.bind(sp), temp)
      outs.append(eager.numpy())
  assert outs[0].shape == outs[1].shape == (1, 151936)
  assert all(np.isfinite(o).all() for o in outs), "eager full logits contain NaN"
  assert not np.allclose(outs[0], outs[1]), "eager logits did not advance with start_pos"
