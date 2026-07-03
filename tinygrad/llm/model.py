from __future__ import annotations
import copy, functools, itertools, os, pathlib
from dataclasses import dataclass, replace
from tinygrad import Tensor, nn, UOp, TinyJit, dtypes, getenv, function, Device
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import prod
from tinygrad.llm.admission import (
  AUTO_MAX_CONTEXT, VRAM_ADMIT_FRACTION, detect_free_vram_bytes, detect_total_vram_bytes,
  resolve_max_context_admission,
)
from tinygrad.llm.gguf import gguf_load, gguf_load_metadata, gguf_load_with_metadata
from tinygrad.llm import route_ops as qk_ops
from tinygrad.llm.decode_routes import clear_vdot_quant_cache, flash_decode_attention_route
from tinygrad.llm.prefill_policy import (
  prefill_concrete_kv_auto_decision, prefill_v2_auto_decision, prefill_v2_realize_bytes,
  prefill_v2_validate_ubatch,
)
from tinygrad.llm.qk_primitives import (
  QK_AMD_GFX1100_ARCH_OK, QKConfig, QKPrimitiveBudget, Q4KPrimitiveRegistry, _demote_q6k_to_q4,
  _install_q4k_fusions, _install_q4k_primitives, _install_q6k_primitives, _qk_storage_summary,
)
from tinygrad.llm.route_policy import (
  _load_qk_generated_policy, _qk_generated_policy_len, _load_qk_route_policy, _set_qk_route_policy,
  _qk_route_policy_selected, _qk_route_policy_selects_q4k_g3, _qk_route_policy_selects_q6k_generated,
  _qk_route_policy_selects_prefill_generated, _validate_qk_route_policy_for_config,
  should_use_flash_decode as _route_should_use_flash_decode,
)
from tinygrad.uop.ops import resolve

# Prefill v2 (opt-in, default off; decode 100% untouched when off). Concrete-ubatch fp16 prefill that lets
# tensor cores apply + the loop-found TC schedule warm-start in. See docs/amd-decode-prefill-v2-gate-20260616.md.
# Costs ~fp16-covered-weight-size extra VRAM (it coexists with the Q4_K decode storage; ~+14GB for 8B) -> OOMs
# small cards. `PREFILL_V2=auto` resolves on/off from detected VRAM in from_gguf (see prefill_v2_auto_decision +
# docs/prefill-default-policy-evaluation-result-20260620.md). Explicit 0/1 always wins.
_PREFILL_V2_ENV = os.environ.get("PREFILL_V2", "")
PREFILL_V2_AUTO = _PREFILL_V2_ENV.strip().lower() == "auto"
PREFILL_V2 = False if PREFILL_V2_AUTO else bool(getenv("PREFILL_V2", 0))
def _set_prefill_v2(val:bool):   # auto policy resolves the module global before Transformer() is constructed
  global PREFILL_V2; PREFILL_V2 = val
PREFILL_UBATCH = getenv("PREFILL_UBATCH", 512)  # concrete token batch; warmstart keys use this N
# Phase-3 routing fix (default ON under PREFILL_V2): route a sub-UBATCH prompt remainder through ONE shifted
# prefill-v2 chunk instead of the slow 32-token symbolic fallback. PREFILL_REMAINDER_FIX=0 reverts. See
# docs/prefill-route-schedule-result-20260620.md.
PREFILL_REMAINDER_FIX = bool(getenv("PREFILL_REMAINDER_FIX", 1))
# Opt-in (within PREFILL_V2): per-layer fp16 overlay. Instead of realizing a resident fp16 copy of every covered
# linear up-front (~fp16-model-size extra VRAM), each block dequants its Q4/Q6 weights to fp16 inside a layer-sized
# TinyJit. Replaying the same captured layer graph with different block tensors reuses the graph's fp16 scratch
# buffers, so peak overlay is one layer signature rather than the whole model. Cost: dequant reruns each prefill.
PREFILL_CHUNKED = bool(getenv("PREFILL_CHUNKED", 0))
# Restricted default-ON (within PREFILL_V2): route eligible fp16 prefill matmuls through the dependency-free
# graph-capturable RDNA3 GEMM. Passed all 4 default-on readiness gates (synced 1.61x, 0/256 greedy mismatches,
# 11/11 fallback, 5/5 OOM; docs/prefill-graph-gemm-default-on-readiness-result-20260620.md). Default-on is
# GUARDED to gfx1100 (the validated arch), decided ONCE HERE at import -- NOT in the route, because Device[...]
# access is disallowed during JIT capture. Non-gfx1100/other-device -> default off. Unsupported shapes (T!=512,
# non-tile-divisible, bias, ineligible role) silently fall back to the normal PREFILL_V2 matmul. Only active when
# PREFILL_V2 is on. Set PREFILL_GRAPH_GEMM=0/1 to override. Absolute-parity drift (max_abs_dNLL 0.0176) accepted
# report-only (greedy byte-identical; harmful dNLL <= 0.0094).
def _prefill_graph_gemm_default() -> int:
  if "PREFILL_GRAPH_GEMM" in os.environ: return getenv("PREFILL_GRAPH_GEMM", 0)   # explicit user override
  try:                                                                            # restricted default-on: gfx1100 only
    if Device.DEFAULT != "AMD": return 0
    return 1 if "gfx1100" in str(getattr(Device["AMD"], "arch", "")) else 0
  except Exception: return 0
PREFILL_GRAPH_GEMM = bool(_prefill_graph_gemm_default())

# Concrete-KV prefill (opt-in, default off): pass a CONCRETE start_pos per prefill chunk so KV=start_pos+T is
# concrete -> the attention's reduce tiles/TC fires (symbolic KV blocks it). ~1.24x e2e, byte-identical. Cost: a
# separate concrete prefill jit per distinct start_pos (0,512,...), precompiled at load -> best WARM/server prefill
# but a load-time precompile tax that loses for cold one-shot short prompts. `PREFILL_CONCRETE_KV=auto` enables it
# only under the server profile (see prefill_concrete_kv_auto_decision). See
# docs/prefill-default-policy-evaluation-result-20260620.md, docs/prefill-concrete-kv-policy-result-20260620.md.
# PREFILL_SERVER_PROFILE=1 is a convenience: serve >1 generation / long prompts -> implies PREFILL_V2=auto (if V2
# unset) + concrete-KV on (when V2 ends up on). One-shot short prompts must NOT set it.
PREFILL_SERVER_PROFILE = bool(getenv("PREFILL_SERVER_PROFILE", 0))
_PREFILL_CKV_ENV = os.environ.get("PREFILL_CONCRETE_KV", "")
PREFILL_CONCRETE_KV_AUTO = _PREFILL_CKV_ENV.strip().lower() == "auto"
PREFILL_CONCRETE_KV = False if PREFILL_CONCRETE_KV_AUTO else bool(getenv("PREFILL_CONCRETE_KV", 0))
def _set_prefill_concrete_kv(val:bool):
  global PREFILL_CONCRETE_KV; PREFILL_CONCRETE_KV = val
# P2: explicit TC attention (Q@Kᵀ TC + fp16 scores + softmax + P@V TC, GQA broadcast) for prefill on CONCRETE KV
# (the only regime where the concrete-shape tensor core fires; symbolic KV blocked it -> 0.79x in-model). Needs
# PREFILL_CONCRETE_KV. Research, dNLL-gated. See docs/prefill-concrete-kv-build-scope-20260619.md.
# DEFAULT-ON, GUARDED to gfx1100 (validated arch), decided ONCE at import (not in the route -- Device[...] is
# disallowed during JIT capture). Only active under PREFILL_V2; the route's isinstance(start_pos,int) guard keeps
# it to CONCRETE chunks (start_pos=0 by default), the only validated regime. Set PREFILL_TC_ATTN=0/1 to override.
# Concrete first chunk: cuts attention ~18%->~5%, reproducible ~1.16x whole-forward, BYTE-IDENTICAL (rel_RMSE 0.0,
# dNLL 0.0, greedy-exact, 3 sessions). A FUSION win, NOT tensor cores (WMMA does not fire; no warmstart TC-opt for
# attention shapes, no BEAM). See docs/prefill-branch-b-tc-attention-result-20260620.md.
def _prefill_tc_attn_default() -> int:
  if "PREFILL_TC_ATTN" in os.environ: return getenv("PREFILL_TC_ATTN", 0)   # explicit user override
  try:                                                                       # restricted default-on: gfx1100 only
    if Device.DEFAULT != "AMD": return 0
    return 1 if "gfx1100" in str(getattr(Device["AMD"], "arch", "")) else 0
  except Exception: return 0
PREFILL_TC_ATTN = bool(_prefill_tc_attn_default())
# HISTORY: the earlier env `PREFILL_TC_ATTENTION` probe reported ~0.8x "REFUTED in-model" -- that was a BROKEN
# harness: it set the typo'd env `PREFILL_TC_ATTENTION` (model reads PREFILL_TC_ATTN) so both arms ran SDPA, AND
# it bound a symbolic start_pos that fails the concrete-int guard so the path never fired. Overturned 2026-06-20
# (correct concrete-int, same-process interleaved synced A/B). See docs/prefill-branch-b-tc-attention-result-20260620.md.
# The loop-found per-shape TC schedule (gate-validated; NO BEAM -- BEAM hangs gfx1100). Forced onto the
# prefill-v2 fp16 matmuls via _WARMSTART_OPTS by shape key. The contraction-heavy shapes (in>out, e.g.
# ffn_down 4096x12288) want UPCAST(0,4); the rest UPCAST(0,2) -- using one schedule for all drops the chain
# to ~9% (verified). See docs/amd-decode-prefill-v2-gate-20260616.md.
def _prefill_v2_opts(out_f:int, in_f:int) -> tuple:
  # UNROLL(reduce,8): unrolling the K loop makes each thread's global->LDS copy loads contiguous, so they fold
  # from per-element global_load_d16 (+ ~8 v_mov register-init/WMMA) to wide global_load_b128 (~2 v_mov/WMMA).
  # +3.7% pp512, no VGPR spill (UNROLL,4 spills 362), dNLL -0.00013. See docs/prefill-cgw3-copy-unroll-result-20260619.md.
  return (Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, 0, 4 if in_f > out_f else 2), Opt(OptOps.UPCAST, 1, 4),
          Opt(OptOps.UNROLL, 0, 8))

def should_use_flash_decode(start_pos, T, use_flash:bool=False) -> bool:
  return _route_should_use_flash_decode(start_pos, T, use_flash, getenv_fn=getenv)

def _pf16(lin, x:Tensor) -> Tensor:
  # prefill v2: a single fp16 matmul (both operands fp16 -> RDNA3 WMMA tensor cores can fire). The primitives'
  # `.weight` is a LAZY Q4_K/Q6_K->fp16 dequant graph (not a realized buffer); using it directly fuses the
  # whole dequant into the matmul -> bandwidth/dequant-bound ~3% peak (no TC win). So we realize a clean fp16
  # weight ONCE per linear (cached as `_pf16_w` by _install_prefill_v2_warmstart) and matmul against that.
  w = getattr(lin, "_pf16_w", None)
  if PREFILL_CHUNKED and w is None:
    # Per-layer overlay: in the chunked path this runs inside a layer-sized TinyJit, whose replay reuses this fp16
    # dequant scratch for every block with the same signature. Do not store it on the Linear; that would pin all blocks.
    w_local = lin.weight.cast(dtypes.float16).contiguous()
    if PREFILL_GRAPH_GEMM:
      routed = qk_ops.route_pf16_graph_gemm(lin, x, w=w_local)
      if routed is not None: return routed
    b = getattr(lin, "bias", None)
    return x.cast(dtypes.float16).linear(w_local.transpose(), b.cast(dtypes.float16) if b is not None else None)
  if PREFILL_GRAPH_GEMM and w is not None:
    routed = qk_ops.route_pf16_graph_gemm(lin, x)
    if routed is not None: return routed
  if w is None: w = lin.weight.cast(dtypes.float16)   # fallback (uncached): lazy, slow -- expect the cache
  b = getattr(lin, "bias", None)
  return x.cast(dtypes.float16).linear(w.transpose(), b.cast(dtypes.float16) if b is not None else None)

@functools.cache
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, device:str|None=None) -> Tensor:
  freqs = 1.0 / (theta ** (Tensor.arange(0, dim, 2)[:(dim // 2)] / dim))
  freqs = Tensor.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
  return freqs.cos().cat(freqs.sin(), dim=-1).clone(device)

class ExpertWeights:
  """Like nn.Linear but with num_experts dimension. Weight shape: (num_experts, out_features, in_features)."""
  def __init__(self, num_experts:int, in_features:int, out_features:int):
    self.weight = Tensor.zeros(num_experts, out_features, in_features)
  def __call__(self, sel:Tensor, x:Tensor) -> Tensor:
    # sel: (B, T, k), x: (B, T, 1, in) or (B, T, k, in) -> output: (B, T, k, out)
    return (x.unsqueeze(-2) @ self.weight[sel].transpose(-1, -2)).contiguous().squeeze(-2)

def apply_rope(x:Tensor, freqs_cis:Tensor) -> Tensor:
  assert x.shape[-1] % 2 == 0
  cos, sin = freqs_cis.reshape(1, 1, x.shape[2], -1).chunk(2, dim=-1)
  x1, x2 = x.chunk(2, dim=-1)
  return (x1 * cos - x2 * sin).cat(x2 * cos + x1 * sin, dim=-1)

def pairwise_topk(x: Tensor, k: int) -> tuple[Tensor, Tensor]:
  n = x.shape[-1]
  vals = Tensor.arange(n).reshape(1,1,n).cast(x.dtype).expand(x.shape)
  cmp = (x.unsqueeze(-1) > x.unsqueeze(-2)) | ((x.unsqueeze(-1) == x.unsqueeze(-2)) & \
    (Tensor.arange(n).reshape(1,1,n,1) < Tensor.arange(n).reshape(1,1,1,n)))
  sel = x.const_like(0).scatter(-1, cmp.sum(axis=-1).cast('int32'), vals)[:,:,n-k:].cast('int32')
  return x.gather(-1, sel), sel

@dataclass(frozen=True)
class SSMConfig:
  conv_kernel: int
  state_size: int
  group_count: int
  time_step_rank: int
  inner_size: int

@dataclass(frozen=True)
class TransformerConfig:
  num_blocks: int
  dim: int
  hidden_dim: int
  n_heads: int
  n_kv_heads: int
  norm_eps: float
  vocab_size: int
  head_dim: int
  rope_theta: float
  rope_dim: int
  v_head_dim: int
  max_context: int = 0
  qk_norm: int = 0
  num_experts: int = 0
  num_experts_per_tok: int = 0
  norm_topk_prob: bool = False
  q_lora_rank: int = 0
  kv_lora_rank: int = 0
  shared_expert_dim: int = 0
  full_attention_interval: int = 0
  attn_output_gate: bool = False
  ssm: SSMConfig|None = None
  shared_expert_gate: bool = True
  leading_dense_blocks: int = 0
  dense_hidden_dim: int = 0
  routed_scaling_factor: float = 1.0
  qkv_bias: bool = False
  expert_bias: bool = False
  admit: dict|None = None   # auto-scan admission report (free/budget/weights/kv/prefill terms); None if not resolved
  kv_quant: bool = False    # KV-quant long-ctx tier: store KV as int8 + fp16 per-(K|V,head,token) scale (halves resident KV)
  ring: bool = False        # StreamingLLM streaming tier (lossy): unbounded logical ctx in the N-token buffer via eviction

class FFNBlock:
  def __init__(self, config:TransformerConfig):
    self.config = config

    # --- RMSNorms --------------------------------------------------------
    self.attn_norm   = nn.RMSNorm(config.dim, config.norm_eps)
    self.ffn_norm    = nn.RMSNorm(config.dim, config.norm_eps)

    # --- feed-forward (MoE or dense) -------------------------------------
    if config.num_experts > 0:
      self.ffn_gate_inp = nn.Linear(config.dim, config.num_experts, bias=False)  # router
      if config.expert_bias: self.exp_probs_b = {"bias": Tensor.zeros(config.num_experts)}
      self.ffn_gate_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_up_exps = ExpertWeights(config.num_experts, config.dim, config.hidden_dim)
      self.ffn_down_exps = ExpertWeights(config.num_experts, config.hidden_dim, config.dim)
      if config.shared_expert_dim > 0:
        self.ffn_gate_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_up_shexp = nn.Linear(config.dim, config.shared_expert_dim, bias=False)
        self.ffn_down_shexp = nn.Linear(config.shared_expert_dim, config.dim, bias=False)
        if config.shared_expert_gate: self.ffn_gate_inp_shexp = {"weight": Tensor.zeros(config.dim)}
    else:
      self.ffn_gate    = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_up      = nn.Linear(config.dim, config.hidden_dim, bias=False)
      self.ffn_down    = nn.Linear(config.hidden_dim, config.dim, bias=False)

  def _feed_forward(self, x:Tensor) -> Tensor:
    if getattr(self, '_prefill_v2', False) and not hasattr(self, 'ffn_gate_exps') and not hasattr(self, 'ffn_gateup'):
      # prefill v2 (dense): fp16 + .contiguous()-isolated matmuls so each is a clean, warmstart-matchable TC
      # kernel (mirrors the gated chained-FFN prefill authority shape). MoE/fused fall through.
      g = _pf16(self.ffn_gate, x).contiguous()
      u = _pf16(self.ffn_up, x).contiguous()
      h = (g.silu() * u).contiguous()
      return _pf16(self.ffn_down, h).contiguous()
    if hasattr(self, 'ffn_gate_exps'):
      h = x.unsqueeze(2)  # (B, T, 1, D) - add expert dim for broadcasting
      logits = self.ffn_gate_inp(x)
      if hasattr(self, 'exp_probs_b'):
        probs = logits.sigmoid()
        _, sel = pairwise_topk(probs + self.exp_probs_b["bias"], self.config.num_experts_per_tok)
        probs = probs.gather(-1, sel)
        if self.config.norm_topk_prob: probs = probs / probs.sum(axis=-1, keepdim=True)
      else:
        vals, sel = pairwise_topk(logits, self.config.num_experts_per_tok)
        probs = vals.softmax(-1) if self.config.norm_topk_prob else logits.softmax(-1).gather(-1, sel)
      probs = probs * self.config.routed_scaling_factor
      x_down = self.ffn_down_exps(sel, (self.ffn_gate_exps(sel, h).silu() * self.ffn_up_exps(sel, h)).contiguous())  # (B, T, k, D)
      out = (x_down * probs.unsqueeze(-1)).sum(axis=2)  # (B, T, D)
      if hasattr(self, 'ffn_gate_shexp'):
        shexp = self.ffn_down_shexp(self.ffn_gate_shexp(x).silu().contiguous() * self.ffn_up_shexp(x))
        if hasattr(self, 'ffn_gate_inp_shexp'): shexp = shexp * (x * self.ffn_gate_inp_shexp["weight"]).sum(axis=-1, keepdim=True).sigmoid()
        out = out + shexp
      return out
    # TODO: remove the need for this contiguous
    if hasattr(self, "ffn_gateup"):  # B1 fused gate/up
      gate, up = self.ffn_gateup(x)
      return self.ffn_down(gate.silu().contiguous() * up)
    if getenv("Q4K_UNFUSE"):  # run FFN matmuls in fp16 so RDNA3 WMMA tensor cores can apply (minimal-overhead)
      xh = x.cast(dtypes.float16)
      return self.ffn_down((self.ffn_gate(xh).silu().contiguous() * self.ffn_up(xh)).cast(dtypes.float16))
    if getenv("DECODE_FUSE_SILU_GATE", 0):
      return self.ffn_down(self.ffn_gate(x).silu() * self.ffn_up(x))
    return self.ffn_down(self.ffn_gate(x).silu().contiguous() * self.ffn_up(x))

  # given the token-prefix match, return how much cached state this block can still reuse
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return prefix_len
  # return writes that reset this block's state after a cache mismatch
  def _state_reset_ops(self) -> list[Tensor]: return []
  def _init_state(self, x:Tensor): raise NotImplementedError
  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor: raise NotImplementedError

  def __call__(self, x: Tensor, start_pos: int|UOp):
    self._init_state(x)
    # StreamingLLM ring: pass the per-step freqs as an ARGUMENT into _run (a nested @function) so it is a true graph
    # input -- reading it off self inside _run would bake it at _run's compile time (attributes don't rebind).
    _rf = getattr(self, "_ring_freqs", None)
    # we pass in the weights implicitly so we unpack the GGUF on the fly
    @function(precompile=True, allow_implicit=True)
    def _run(x:Tensor, start_pos:int|UOp, ring_freqs):
      h =     x + self._attention(self.attn_norm(x), start_pos, ring_freqs)
      return (h + self._feed_forward(self.ffn_norm(h))).contiguous()
    return _run(x, start_pos, _rf)

class TransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    assert config.v_head_dim == config.head_dim, "TransformerBlock requires v_head_dim == head_dim"

    # --- attention projections (all linear, bias-free) ------------------
    q_proj_out       = config.head_dim * config.n_heads * (2 if config.attn_output_gate else 1)
    kv_proj_out      = config.head_dim * config.n_kv_heads
    self.attn_q      = nn.Linear(config.dim, q_proj_out,  bias=config.qkv_bias)
    self.attn_k      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_v      = nn.Linear(config.dim, kv_proj_out, bias=config.qkv_bias)
    self.attn_output = nn.Linear(config.head_dim * config.n_heads, config.dim, bias=False)
    if config.qk_norm: self.attn_q_norm, self.attn_k_norm = nn.RMSNorm(config.qk_norm, config.norm_eps), nn.RMSNorm(config.qk_norm, config.norm_eps)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    if getattr(self, '_prefill_v2', False) and not hasattr(self, "attn_qkv"):  # prefill v2: fp16 isolated q/k/v
      q, k, v = _pf16(self.attn_q, x).contiguous(), _pf16(self.attn_k, x).contiguous(), _pf16(self.attn_v, x).contiguous()
    elif hasattr(self, "attn_qkv"): q, k, v = self.attn_qkv(x)  # B1 fused q/k/v
    else: q, k, v = self.attn_q(x), self.attn_k(x), self.attn_v(x)
    if self.config.qk_norm and self.config.qk_norm != self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    B, T, _ = x.shape
    if self.config.attn_output_gate:
      qg = q.reshape(B, T, self.config.n_heads, 2, self.config.head_dim)
      q, gate = qg[:, :, :, 0, :], qg[:, :, :, 1, :].reshape(B, T, self.config.n_heads * self.config.head_dim)
    q = q.reshape(B, T, self.config.n_heads,    self.config.head_dim).transpose(1, 2)  # (B,H,T,Hd)
    k = k.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    v = v.reshape(B, T, self.config.n_kv_heads, self.config.head_dim).transpose(1, 2)  # (B,KvH,T,Hd)
    if self.config.qk_norm == self.config.head_dim: q, k = self.attn_q_norm(q), self.attn_k_norm(k)

    # rope-at-read (DECODE_ROPE_AT_READ, opt-in; requires full-head rope): store UN-roped K and rotate at read -- the
    # prerequisite for the StreamingLLM ring's position re-basing. Q is never cached, so it is always roped here.
    # The ring supplies a per-step PRE-GATHERED freqs table via the JIT-input attribute _ring_freqs (slot-relative
    # positions); when unset the baked self.freqs_cis is used (absolute positions). _ring_freqs implies rope-at-read.
    _ring_freqs = ring_freqs
    # rope-at-read active if: env flag, OR a ring-decode step (freqs supplied), OR the ring is active this generation
    # (covers PREFILL, which must ALSO store un-roped K so the ring decode reads it consistently).
    _rope_read = (bool(getenv("DECODE_ROPE_AT_READ", 0)) or _ring_freqs is not None or getattr(self, "_ring_active", False)) \
                 and self.config.rope_dim == self.config.head_dim
    _fr = _ring_freqs if _ring_freqs is not None else self.freqs_cis
    # full-ring (ctx>=N): the buffer is full and the write slot wraps, so the live read length is the WHOLE buffer N
    # (all slots valid), not start_pos+T (start_pos is the wrapped write slot, not a length). Selects [0:N] reads + Tc=N.
    _ring_full = getattr(self, "_ring_full", False)
    _rl = self.config.max_context if _ring_full else (start_pos + T)
    # Q is roped via _fr (the gathered ring table when ring, else freqs_cis) indexed by start_pos: in the full ring
    # start_pos is the write slot wp, and _fr[wp] = freqs[pos_of(wp)] = the query's (newest) position -> consistent
    # with the K positions. In fill / non-ring, _fr == freqs_cis and start_pos is the absolute position (unchanged).
    q = apply_rope(q[..., :self.config.rope_dim], _fr[start_pos:start_pos+T]).cat(q[..., self.config.rope_dim:], dim=-1)
    if not _rope_read:
      k = apply_rope(k[..., :self.config.rope_dim], self.freqs_cis[start_pos:start_pos+T]).cat(k[..., self.config.rope_dim:], dim=-1)

    # NOTE: we don't want to change self.cache_kv, the function API doesn't support this well
    if self.config.kv_quant and _rope_read:
      raise NotImplementedError("KV-quant + rope-at-read not yet composed (Q8 stores roped K; the ring's un-roped K "
                                "path is validated fp16 first). Disable one of DECODE_KV_QUANT / DECODE_ROPE_AT_READ.")
    if self.config.kv_quant:
      # KV-quant write: symmetric per-(K|V, head, token) int8 (absmax over head_dim). k is already roped, so we store
      # roped-then-quantized K (Q8 is orthogonal to RoPE). Store int8 KV + fp16 scale; dequant the re-slice to fp16 for
      # the non-flash (SDPA/prefill) consumers (per-layer transient). The flash route reads int8+scale natively.
      _Hkv = self.config.n_kv_heads
      _kv = Tensor.stack(k, v)                                                    # [2,B,Hkv,T,Hd] fp16
      _sc = (_kv.abs().max(axis=-1, keepdim=True) / 127.0).maximum(1e-8)          # [2,B,Hkv,T,1]
      _kvq = (_kv / _sc).round().cast(dtypes.int8)                                # [2,B,Hkv,T,Hd] int8
      _sch = _sc.reshape(2, B, _Hkv, T).cast(dtypes.float16)
      _st_kv = self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(_kvq.uop)
      _st_sc = self.cache_kv_scale[:, :, :, start_pos:start_pos+T].uop.store(_sch.uop)
      assigned_kv = Tensor(self.cache_kv.uop.after(_st_kv))
      assigned_scale = Tensor(self.cache_kv_scale.uop.after(_st_sc))
      _ksc = assigned_scale[0, :, :, 0:start_pos+T].reshape(B, _Hkv, start_pos+T, 1)
      _vsc = assigned_scale[1, :, :, 0:start_pos+T].reshape(B, _Hkv, start_pos+T, 1)
      k = assigned_kv[0, :, :, 0:start_pos+T, :].cast(dtypes.float16) * _ksc
      v = assigned_kv[1, :, :, 0:start_pos+T, :].cast(dtypes.float16) * _vsc
    else:
      assigned_scale = None
      assigned_kv = Tensor(self.cache_kv.uop.after(self.cache_kv[:, :, :, start_pos:start_pos+T, :].uop.store(Tensor.stack(k, v).uop)))
      _kfull = assigned_kv[0]
      if _rope_read:
        # rope-at-read for the NON-flash (SDPA/prefill) consumers: K is stored un-roped. Rotate the FULL concrete-MAXC
        # cache (positions 0..MAXC-1) THEN slice -- roping before the slice avoids indexing freqs by a SYMBOLIC bound
        # (start_pos+T's vmax can exceed MAXC). The full-MAXC rope is DCE'd on the flash decode path (k unused there;
        # the kernel ropes in-register from `freqs`); it materializes only for prefill/SDPA. Unwritten slots (>ctx) are
        # roped garbage but sliced away below. Explicit (not apply_rope) so the concrete last dim needs no -1 inference.
        _rd = self.config.rope_dim; _hh = _rd // 2; _mc = self.config.max_context
        _cos = _fr[:, :_hh].reshape(1, 1, _mc, _hh); _sin = _fr[:, _hh:].reshape(1, 1, _mc, _hh)
        _k1, _k2 = _kfull[..., :_hh], _kfull[..., _hh:_rd]
        _kfull = (_k1 * _cos - _k2 * _sin).cat(_k2 * _cos + _k1 * _sin, _kfull[..., _rd:], dim=-1)
      k = _kfull[:, :, 0:_rl, :]
      v = assigned_kv[1, :, :, 0:_rl, :]

    #self.cache_kv[:, :, :, start_pos:start_pos+T, :].assign(Tensor.stack(k, v))
    #k = self.cache_kv[0, :, :, 0:start_pos+T, :]
    #v = self.cache_kv[1, :, :, 0:start_pos+T, :]

    # NOTE: this mask is causal_lower_right, not the causal_upper_left generated by is_casual = True
    # TODO: this if statement should be removed and it shouldn't generate extra kernels
    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    # ring decode ALWAYS uses flash: start_pos is the wrapped write SLOT (not the ctx), so should_use_flash_decode(slot)
    # could wrongly pick SDPA; the ring context is the whole window and must read via the flash live-split route.
    if _ring_freqs is not None or should_use_flash_decode(start_pos, T, getattr(self, "_use_flash", False)):
      Hq, Hkv, Hd = self.config.n_heads, self.config.n_kv_heads, self.config.head_dim
      out = flash_decode_attention_route(q, assigned_kv, start_pos, T, B, Hq, Hkv, Hd, self.config.max_context,
                                         kv_scale=assigned_scale, freqs=(_fr if _rope_read else None),
                                         ring_full=_ring_full)
      attn = out.reshape(B, Hq, T, Hd).cast(q.dtype)
    elif PREFILL_TC_ATTN and getattr(self, '_prefill_v2', False) and isinstance(start_pos, int) and resolve(T != 1):
      # P2: Option-B explicit TC attention on CONCRETE KV. Q@Kᵀ (TC) -> fp16 scores -> softmax -> P@V (TC),
      # GQA via broadcast (K/V per kv-head expanded over the G group dim). Concrete KV=start_pos+T -> TC fires.
      Hkv, Hd, KV = self.config.n_kv_heads, self.config.head_dim, start_pos + T
      G = self.config.n_heads // Hkv; scale = Hd ** -0.5
      qg = q.reshape(B, Hkv, G, T, Hd).cast(dtypes.float16)
      kg = k.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      vg = v.reshape(B, Hkv, 1, KV, Hd).cast(dtypes.float16)
      s = ((qg @ kg.transpose(-1, -2)).float() * scale + mask.reshape(1, 1, 1, T, KV)).softmax(-1)
      attn = (s.cast(dtypes.float16) @ vg).reshape(B, self.config.n_heads, T, Hd).cast(q.dtype)  # (B,H,T,Hd)
    else:
      attn = q.scaled_dot_product_attention(k, v, attn_mask=mask, enable_gqa=True)   # (B,H,T,Hd)
    attn = attn.transpose(1, 2).reshape(B, T, -1)                                    # back to (B,T,D)
    out_in = attn if not self.config.attn_output_gate else (attn * gate.sigmoid())
    if getattr(self, '_prefill_v2', False): return _pf16(self.attn_output, out_in).contiguous()  # prefill v2
    return self.attn_output(out_in)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_kv"):
      # 8B generated decode attention was validated with fp16 K/V cache storage (TG-P14 KV_BOTH parity and roofline
      # closeout). Keep this shape on fp16 so the generated tile reads the same cache dtype the promotion measured;
      # other shapes keep the default dtype.
      _generated_8b_supported = QK_AMD_GFX1100_ARCH_OK and x.shape[0] == 1 and self.config.n_heads == 32 \
        and self.config.n_kv_heads == 8 and self.config.head_dim == 128
      _kv_dtype = dtypes.float16 if _generated_8b_supported else None
      # Admission guardrail (B5): assert the ACTUAL cache_kv bytes (with the real dtype) fit the admitted VRAM budget
      # before allocating -- converts a silent late OOM into a clear, actionable failure. O(1), no re-probe: the
      # admission report carries free/budget/weights so we re-derive the KV allowance and check this model's total KV.
      _admit = getattr(self.config, "admit", None)
      if _admit and _admit.get("mode", "").split("+")[0] in ("auto", "explicit") and "budget_gb" in _admit:
        _elem = 1 if self.config.kv_quant else (2 if _kv_dtype is dtypes.float16 else dtypes.default_float.itemsize)
        _scale_b = (2 * x.shape[0] * self.config.n_kv_heads * self.config.max_context * 2) if self.config.kv_quant else 0
        _block_kv = 2 * x.shape[0] * self.config.n_kv_heads * self.config.max_context * self.config.head_dim * _elem + _scale_b
        _total_kv = _block_kv * self.config.num_blocks
        _allow = (_admit["budget_gb"] - _admit["weights_gb"] - _admit.get("flash_scratch_gb", 0.0)) * 1e9 \
                 - _admit.get("prefill_gb_per_1k", 0.0) * self.config.max_context / 1000 * 1e9
        if _total_kv > _allow:
          raise RuntimeError(
            f"KV cache admission guard: max_context={self.config.max_context} needs {_total_kv/1e9:.1f}GB of KV "
            f"(dtype {'fp16' if _elem==2 else dtypes.default_float.name}) but only {_allow/1e9:.1f}GB is admissible "
            f"(budget {_admit['budget_gb']:.1f}GB @{VRAM_ADMIT_FRACTION} minus weights {_admit['weights_gb']:.1f}GB + prefill peak). "
            f"Reduce --max_context or use auto. This is the guardrail that prevents a silent OOM.")
      if self.config.kv_quant:
        # KV-quant tier: resident KV is int8 (half the bytes) + a per-(K|V, head, token) fp16 scale buffer. The decode
        # flash route dequants in-register (int8*scale); non-flash consumers dequant their re-sliced K/V to fp16 (per-
        # layer transient). Model-agnostic -- keyed off config.kv_quant, no model-name check.
        self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, dtype=dtypes.int8, device=x.device)
        self.cache_kv_scale = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, dtype=dtypes.float16, device=x.device)
      else:
        self.cache_kv = Tensor.empty(2, x.shape[0], self.config.n_kv_heads, self.config.max_context, self.config.head_dim, dtype=_kv_dtype, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class MLATransformerBlock(FFNBlock):
  def __init__(self, config:TransformerConfig):
    super().__init__(config)
    qk_nope_head_dim = config.head_dim - config.rope_dim
    if config.q_lora_rank > 0:
      self.attn_q_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
      self.attn_q_a_norm = nn.RMSNorm(config.q_lora_rank, config.norm_eps)
      self.attn_q_b = nn.Linear(config.q_lora_rank, config.n_heads * config.head_dim, bias=False)
    else:
      self.attn_q = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
    self.attn_kv_a_mqa = nn.Linear(config.dim, config.kv_lora_rank + config.rope_dim, bias=False)
    self.attn_kv_a_norm = nn.RMSNorm(config.kv_lora_rank, config.norm_eps)
    self.attn_k_b = {"weight": Tensor.zeros(config.n_heads, config.kv_lora_rank, qk_nope_head_dim)}
    self.attn_v_b = {"weight": Tensor.zeros(config.n_heads, config.v_head_dim, config.kv_lora_rank)}
    self.attn_output = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    B, T, _ = x.shape
    q_nope_head_dim = self.config.head_dim - self.config.rope_dim
    q_proj = self.attn_q_b(self.attn_q_a_norm(self.attn_q_a(x))) if self.config.q_lora_rank > 0 else self.attn_q(x)
    q = q_proj.reshape(B, T, self.config.n_heads, self.config.head_dim).transpose(1, 2)
    q_nope, q_rope = q[..., :q_nope_head_dim], q[..., q_nope_head_dim:]
    q = (q_nope @ self.attn_k_b["weight"].transpose(-1, -2)).cat(apply_rope(q_rope, self.freqs_cis[start_pos:start_pos+T]), dim=-1)

    kv_a = self.attn_kv_a_mqa(x)
    c_kv = self.attn_kv_a_norm(kv_a[..., :self.config.kv_lora_rank])
    k_rope = apply_rope(
      kv_a[..., self.config.kv_lora_rank:].reshape(B, T, 1, self.config.rope_dim).transpose(1, 2),
      self.freqs_cis[start_pos:start_pos+T])

    k_store = c_kv.reshape(B, 1, T, self.config.kv_lora_rank).cat(k_rope.reshape(B, 1, T, self.config.rope_dim), dim=-1)
    k = Tensor(self.cache_k.uop.after(self.cache_k[:, :, start_pos:start_pos+T, :].uop.store(k_store.uop)))[:, :, 0:start_pos+T, :]
    v = k[..., :self.config.kv_lora_rank]

    mask = Tensor.full((1, 1, T, start_pos+T), float("-inf"), dtype=x.dtype, buffer=False).triu(start_pos+1) \
      if resolve(T != 1) else None
    attn = q @ k.transpose(-1, -2) * (1.0 / self.config.head_dim ** 0.5)
    if mask is not None: attn = attn + mask
    attn = attn.softmax(-1)
    attn = ((attn @ v) @ self.attn_v_b["weight"].transpose(-1, -2)).transpose(1, 2).reshape(B, T, -1)
    return self.attn_output(attn)

  def _init_state(self, x:Tensor):
    if not hasattr(self, "cache_k"):
      self.cache_k = Tensor.empty(x.shape[0], 1, self.config.max_context, self.config.kv_lora_rank + self.config.rope_dim, device=x.device)
      self.freqs_cis = precompute_freqs_cis(self.config.rope_dim, self.config.max_context, self.config.rope_theta, device=x.device)

class GatedDeltaNetBlock(FFNBlock):
  def __init__(self, config:TransformerConfig, ssm:SSMConfig):
    super().__init__(config)
    self.head_k_dim, self.num_k_heads, self.num_v_heads = ssm.state_size, ssm.group_count, ssm.time_step_rank
    assert self.num_v_heads % self.num_k_heads == 0
    self.head_v_dim, self.ssm_conv_kernel = ssm.inner_size // ssm.time_step_rank, ssm.conv_kernel
    self.conv_channels, self.q_dim = ssm.inner_size + 2*ssm.group_count*ssm.state_size, ssm.state_size*ssm.group_count
    self.attn_qkv, self.attn_gate = nn.Linear(config.dim, self.conv_channels, bias=False), nn.Linear(config.dim, ssm.inner_size, bias=False)
    self.ssm_alpha, self.ssm_beta = nn.Linear(config.dim, self.num_v_heads, bias=False), nn.Linear(config.dim, self.num_v_heads, bias=False)
    self.ssm_conv1d = {"weight": Tensor.zeros(self.conv_channels, self.ssm_conv_kernel)}
    self.ssm_dt = {"bias": Tensor.zeros(self.num_v_heads)}
    self.ssm_a = Tensor.zeros(self.num_v_heads)
    self.ssm_norm, self.ssm_out = nn.RMSNorm(self.head_v_dim, config.norm_eps), nn.Linear(ssm.inner_size, config.dim, bias=False)

  def _attention(self, x:Tensor, start_pos:int|UOp, ring_freqs=None) -> Tensor:
    B, T, _ = x.shape
    assert T == 1, "GatedDeltaNetBlock currently only supports T=1"

    # input processing
    x = x.half()
    out_gate = self.attn_gate(x).reshape(B, 1, self.num_v_heads, self.head_v_dim)
    beta = self.ssm_beta(x).sigmoid().reshape(B, self.num_v_heads, 1, 1)
    alpha = ((self.ssm_alpha(x).float() + self.ssm_dt["bias"]).softplus() * self.ssm_a).reshape(B, self.num_v_heads, 1, 1).exp()

    # qkv conv
    conv_window = self.conv_state.cat(self.attn_qkv(x), dim=1)
    conv_out = (conv_window * self.ssm_conv1d["weight"].T.unsqueeze(0)).sum(1).silu()
    q, k, v = conv_out.split([self.q_dim, self.q_dim, self.conv_channels - 2*self.q_dim], dim=-1)
    q = q.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    k = k.reshape(B, self.num_k_heads, self.head_k_dim).normalize(dim=-1).repeat(1, self.num_v_heads//self.num_k_heads, 1)
    v = v.reshape(B, self.num_v_heads, self.head_v_dim)
    q, k, v = q.mul(self.head_k_dim**-0.5).unsqueeze(-1), k.unsqueeze(-1), v.unsqueeze(-1)

    # recurrent
    recurrent_state = self.recurrent_state * alpha
    recurrent_state = recurrent_state + ((v - recurrent_state@k) * beta)@k.transpose(-1, -2)

    # store the updated state
    conv_state_store = self.conv_state.uop.store(conv_window[:, 1:, :].cast(self.conv_state.dtype).uop)
    recurrent_state_store = self.recurrent_state.uop.store(recurrent_state.cast(self.recurrent_state.dtype).uop)
    recurrent_state = Tensor(self.recurrent_state.uop.after(recurrent_state_store, conv_state_store))

    # output
    core_attn_out = self.ssm_norm((recurrent_state@q).squeeze(-1).reshape(B, 1, self.num_v_heads, self.head_v_dim))
    return self.ssm_out((core_attn_out * out_gate.silu()).reshape(B, 1, -1).cast(x.dtype))

  # recurrent state can't be partially reused after divergence, force a full rebuild
  def _state_reset_ops(self):
    return [self.conv_state.assign(self.conv_state.const_like(0)),
            self.recurrent_state.assign(self.recurrent_state.const_like(0))] if hasattr(self, "conv_state") else []
  def _reusable_prefix_len(self, prefix_len:int, cached_len:int) -> int: return 0 if prefix_len != cached_len else prefix_len

  def _init_state(self, x):
    if not hasattr(self, "conv_state"):
      self.conv_state = Tensor.zeros(x.shape[0], self.ssm_conv_kernel-1, self.conv_channels, device=x.device).clone()
      self.recurrent_state = Tensor.zeros(x.shape[0], self.num_v_heads, self.head_v_dim, self.head_v_dim, device=x.device).clone()

class Transformer:
  def __init__(self, config:TransformerConfig):
    dense_config = replace(config, num_experts=0, num_experts_per_tok=0, shared_expert_dim=0, hidden_dim=config.dense_hidden_dim or config.hidden_dim)
    if config.ssm: config = replace(config, qk_norm=config.head_dim)
    block_cls = MLATransformerBlock if config.kv_lora_rank > 0 else TransformerBlock
    self.blk:list[FFNBlock] = [GatedDeltaNetBlock(config, config.ssm) if config.ssm and (i+1) % config.full_attention_interval != 0 else
                               block_cls(dense_config if i < config.leading_dense_blocks else config) for i in range(config.num_blocks)]
    self.token_embd  = nn.Embedding(config.vocab_size, config.dim)
    self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
    self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
    self.max_context = config.max_context
    self.config = config
    self.has_recurrent_block = any(isinstance(b, GatedDeltaNetBlock) for b in self.blk)
    self._cached_tokens: list[int] = []
    self._q4k_linears = Q4KPrimitiveRegistry()
    # we specialize the JIT for prefill and rollout; rollout_jit_flash is the context-aware flash-decode
    # decode graph (captured lazily when ctx crosses FLASH_DECODE_THRESHOLD), see __call__/generate.
    self.prefill_jit = TinyJit(self.forward)
    self.rollout_jit = TinyJit(self.forward)
    self.rollout_jit_flash = TinyJit(self.forward)
    self.rollout_jit_ring = TinyJit(self.forward_ring)        # ring FILL phase (ctx<N): read [0:start_pos+T], identity freqs
    self.rollout_jit_ring_full = TinyJit(self.forward_ring)   # ring FULL phase (ctx>=N): read [0:N], wrapped write slot + gathered freqs
    # prefill v2 (opt-in): a SEPARATE jit captured with a CONCRETE token batch (T=PREFILL_UBATCH) so tensor
    # cores apply, distinct from the symbolic-batch prefill_jit. Only ever called when PREFILL_V2.
    self.prefill_v2_jit = TinyJit(self.forward)
    self.prefill_v2_jits: dict = {}   # concrete-KV: one prefill jit per concrete start_pos (PREFILL_CONCRETE_KV)
    self.prefill_v2_layer_jits: dict = {}
    # prefill v2 warmstart table is built here but installed into the global codegen knob ONLY for the duration
    # of the prefill-v2 forward (see __call__), to contain that ambient power rather than leave it set process-wide.
    self._pf16_warmstart:dict|None = None
    if PREFILL_V2:
      prefill_v2_validate_ubatch(PREFILL_UBATCH)
      self._pf16_warmstart = self._build_prefill_v2_warmstart()

  # the dense FFN + attn projection linears prefill-v2 accelerates (per block)
  _PREFILL_V2_LINEARS = ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
                         "attn_q", "attn_k", "attn_v", "attn_output")
  def _prefill_v2_dims(self, lin):
    out_f = getattr(lin, "out_features", None) or lin.weight.shape[0]
    in_f  = getattr(lin, "in_features", None)  or lin.weight.shape[1]
    return (out_f, in_f) if isinstance(out_f, int) and isinstance(in_f, int) else (None, None)

  def _prefill_v2_covered(self):
    # (linear, out_f, in_f) for each covered linear with a known concrete shape -- single source for the
    # warmstart table, the VRAM estimate, and the realization, so they can't drift apart.
    for block in self.blk:
      for n in self._PREFILL_V2_LINEARS:
        lin = getattr(block, n, None)
        if lin is None or getattr(lin, "weight", None) is None: continue
        out_f, in_f = self._prefill_v2_dims(lin)
        if out_f is not None: yield lin, out_f, in_f

  def _clone_block_shell(self, block:FFNBlock) -> FFNBlock:
    # Shallow-copy the module tree so rebinding template tensors for JIT inputs never mutates a real decode block.
    tmpl = copy.copy(block)
    for k, v in block.__dict__.items():
      if isinstance(v, dict): setattr(tmpl, k, v.copy())
      elif isinstance(v, list): setattr(tmpl, k, v.copy())
      elif hasattr(v, "__dict__"): setattr(tmpl, k, copy.copy(v))
    return tmpl

  def _set_state_tensor(self, obj, name:str, val:Tensor):
    parts = name.split(".")
    cur = obj
    for p in parts[:-1]:
      cur = cur[int(p)] if isinstance(cur, (list, tuple)) else cur[p] if isinstance(cur, dict) else getattr(cur, p)
    p = parts[-1]
    if isinstance(cur, list): cur[int(p)] = val
    elif isinstance(cur, dict): cur[p] = val
    else: setattr(cur, p, val)

  def _prefill_v2_block_state(self, block:FFNBlock) -> tuple[tuple[str, ...], tuple[int, ...], tuple[Tensor, ...]]:
    sd = nn.state.get_state_dict(block)
    names, vals, val_idx, seen = tuple(sd.keys()), [], [], {}
    for n in names:
      t = sd[n]
      k = t.uop
      if k not in seen:
        seen[k] = len(vals)
        vals.append(t)
      val_idx.append(seen[k])
    return names, tuple(val_idx), tuple(vals)

  def _prefill_v2_layer_key(self, block:FFNBlock, names:tuple[str, ...], val_idx:tuple[int, ...], vals:tuple[Tensor, ...], start_pos:int|UOp):
    sp_key = ("int", start_pos) if isinstance(start_pos, int) else ("sym",)
    return (type(block), tuple((n, vals[i].shape, vals[i].dtype) for n, i in zip(names, val_idx)), val_idx, sp_key)

  def _prefill_v2_layer_jit(self, block:FFNBlock, names:tuple[str, ...], val_idx:tuple[int, ...], vals:tuple[Tensor, ...], start_pos:int|UOp) -> TinyJit:
    key = self._prefill_v2_layer_key(block, names, val_idx, vals, start_pos)
    if key not in self.prefill_v2_layer_jits:
      tmpl, state_names, state_idx = self._clone_block_shell(block), names, val_idx
      tmpl._use_flash, tmpl._prefill_v2, tmpl._ring_freqs, tmpl._ring_full = False, True, None, False
      def bind_state(state_vals:tuple[Tensor, ...]):
        if state_idx and max(state_idx) >= len(state_vals):
          raise RuntimeError(f"prefill layer JIT state mismatch: need {max(state_idx)+1} tensors, got {len(state_vals)}")
        for n, i in zip(state_names, state_idx): self._set_state_tensor(tmpl, n, state_vals[i])
      if isinstance(start_pos, int):
        sp_const = start_pos
        def layer_forward(x:Tensor, *state_vals:Tensor) -> Tensor:
          bind_state(state_vals)
          return tmpl(x, sp_const)
      else:
        def layer_forward(x:Tensor, sp:UOp, *state_vals:Tensor) -> Tensor:
          bind_state(state_vals)
          return tmpl(x, sp)
      self.prefill_v2_layer_jits[key] = TinyJit(layer_forward)
    return self.prefill_v2_layer_jits[key]

  def _build_prefill_v2_warmstart(self) -> dict:
    # The loop-found per-shape TC schedule for the prefill-v2 fp16 FFN/attn matmuls, keyed by the in-model
    # kernel signature (verified): (frozenset({out_features, PREFILL_UBATCH}), in_features). Shapes are
    # config-fixed so this is computable at init from the (pre-primitive) nn.Linears. A kernel whose key is
    # absent (e.g. a silu-fused gate) keeps the heuristic; a key that applies-then-errors falls back too
    # (postrange.py). NOT set into the global here -- installed only around the prefill-v2 forward (__call__).
    return {(frozenset({out_f, PREFILL_UBATCH}), in_f): _prefill_v2_opts(out_f, in_f)
            for _, out_f, in_f in self._prefill_v2_covered()}

  def realize_prefill_v2_weights(self) -> int:
    # Realize a clean fp16 weight per covered linear (cached as `_pf16_w`, read by _pf16). The primitives' lazy
    # Q4_K/Q6_K->fp16 dequant graph, used raw, fuses into the matmul -> ~3% peak (no TC win); a realized fp16
    # buffer makes the prefill-v2 matmul a real TC GEMM (~13x prefill on 8B). COST: ~fp16-model-size extra VRAM
    # (it coexists with the Q4_K decode storage). Gated/opt-in; called at the end of from_gguf when PREFILL_V2.
    covered = list(self._prefill_v2_covered())
    if PREFILL_CHUNKED:
      # VRAM-frugal per-layer path: realize nothing up-front and store no `_pf16_w`; the layer TinyJit dequants to
      # replay-owned fp16 scratch that is reused across block tensor inputs.
      return 0
    # preflight: realizing ~fp16-model-size on top of Q4_K OOMs for 14B/32B -> fail fast with the estimate.
    est_gb = prefill_v2_realize_bytes([(o, i) for _, o, i in covered]) / 1e9
    budget_gb = getenv("PREFILL_V2_MAX_REALIZE_GB", 18)
    if est_gb > budget_gb and not getenv("PREFILL_V2_FORCE_REALIZE", 0):
      raise RuntimeError(f"PREFILL_V2 would realize ~{est_gb:.1f} GB of fp16 weights (on top of Q4_K decode "
                         f"storage), over the ~{budget_gb} GB budget -- likely OOM. This is 8B-sized work; for "
                         f"larger models raise PREFILL_V2_MAX_REALIZE_GB, set PREFILL_V2_FORCE_REALIZE=1 to "
                         f"override, or use PREFILL_CHUNKED=1 (VRAM-frugal per-layer fp16 overlay).")
    n = 0
    for lin, _, _ in covered:
      lin._pf16_w = lin.weight.cast(dtypes.float16).contiguous().realize(); n += 1
    return n

  def precompile_concrete_prefill_jits(self) -> int:
    # Increment 0 ship: with PREFILL_CONCRETE_KV, every prefill chunk runs through a per-start_pos CONCRETE jit
    # (-> the fusion attention path, 1.7-4.4x/chunk faster than the symbolic chunk). Those jits compile on first
    # use (~5s each), so a COLD long prompt pays the tax inline. Precompiling them ONCE at load (here) moves the
    # tax to load time, so every generation -- including the first -- is warm. Bounded: ceil(max_context/UBATCH)
    # jits. Safe to leave the dummy KV behind: a fresh model's first generation starts at start_pos=0 and
    # overwrites the cache in chunk order before any position is read. gfx1100/PREFILL_V2/PREFILL_CONCRETE_KV only.
    if not (PREFILL_V2 and PREFILL_CONCRETE_KV): return 0
    temp = Tensor([0.0])
    dummy = Tensor.zeros(1, PREFILL_UBATCH, dtype="int32").contiguous().realize()
    n = 0
    for sp in range(0, self.max_context - PREFILL_UBATCH + 1, PREFILL_UBATCH):
      self(dummy, sp, temp, use_flash=False).realize(); n += 1   # populates self.prefill_v2_jits[sp]
    return n

  def logits(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens).float()                   # (B, T, D)
    for block in self.blk: x = block(x, start_pos)
    return self.output(self.output_norm(x))

  def logits_prefill_v2_chunked(self, tokens:Tensor, start_pos:int|UOp) -> Tensor:
    x = self.token_embd(tokens).float()
    for block in self.blk:
      block._init_state(x)
      names, val_idx, vals = self._prefill_v2_block_state(block)
      if val_idx and max(val_idx) >= len(vals):
        raise RuntimeError(f"prefill layer state pack mismatch before JIT: need {max(val_idx)+1}, got {len(vals)}")
      jit = self._prefill_v2_layer_jit(block, names, val_idx, vals, start_pos)
      x = jit(x, *vals) if isinstance(start_pos, int) else jit(x, start_pos, *vals)
    return self.output(self.output_norm(x))

  def forward(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    logits = self.logits(tokens, start_pos)[:, -1, :]
    # Gumbel-max trick: argmax(logits/temp - log(-log(uniform))) is equivalent to sampling from softmax(logits/temp)
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_prefill_v2_chunked(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor) -> Tensor:
    logits = self.logits_prefill_v2_chunked(tokens, start_pos)[:, -1, :]
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def forward_ring(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, freqs:Tensor) -> Tensor:
    # StreamingLLM ring decode: `freqs` is a per-step JIT INPUT (the slot-relative pre-gathered cos|sin table). Set it
    # on each block from INSIDE the traced fn so it binds to the graph input (a baked attribute would capture once).
    for block in self.blk: block._ring_freqs = freqs
    return self.forward(tokens, start_pos, temperature)

  def __call__(self, tokens:Tensor, start_pos:int|UOp, temperature:Tensor, use_flash:bool=False,
               ring_freqs:Tensor|None=None, ring_full:bool=False) -> Tensor:
    is_prefill = resolve(tokens.shape[1] != 1)
    # prefill v2: only when opt-in AND this is a CONCRETE-batch prefill chunk. Normal prefill passes a symbolic
    # v_toks (tokens.shape[1] is a UOp -> not int), so the two paths never collide; decode is T==1.
    is_prefill_v2 = PREFILL_V2 and is_prefill and isinstance(tokens.shape[1], int)
    if getenv("Q4K_VDOT_AMORT"): clear_vdot_quant_cache()  # E0: fresh quant cache per forward/trace
    for q4k_linear in self._q4k_linears.linears:
      q4k_linear.decode_enabled = not is_prefill
    # context-aware flash: each block reads _use_flash at trace time; rollout_jit (SDPA) and
    # rollout_jit_flash bake distinct attention -- each is only ever called with its own use_flash, so
    # capture is consistent. The decode-only T==1 guard in _attention ignores it during prefill.
    for block in self.blk:
      block._use_flash, block._prefill_v2, block._ring_freqs, block._ring_full = use_flash, is_prefill_v2, None, ring_full
    # StreamingLLM ring decode: distinct captured graphs with `freqs` as a per-step JIT input (rebound each token). The
    # FULL-phase graph (ring_full, ctx>=N) reads the whole [0:N] cache and writes at the wrapped slot; the FILL-phase
    # graph reads [0:start_pos+T] like normal decode. block._ring_full (baked bool) selects the read mode in _attention.
    if ring_freqs is not None and not is_prefill:
      _rjit = self.rollout_jit_ring_full if ring_full else self.rollout_jit_ring
      return _rjit(tokens.contiguous(), start_pos, temperature, ring_freqs)
    # Per-layer overlay: do not wrap the whole prefill in a TinyJit. The Python loop replays a layer-sized TinyJit
    # across blocks, passing each block's tensors as inputs so fp16 dequant scratch is overwritten, not accumulated.
    if is_prefill_v2 and PREFILL_CHUNKED:
      import tinygrad.codegen.opt.postrange as pr
      saved = pr._WARMSTART_OPTS
      pr._WARMSTART_OPTS = self._pf16_warmstart
      try: return self.forward_prefill_v2_chunked(tokens.contiguous(), start_pos, temperature)
      finally: pr._WARMSTART_OPTS = saved
    # concrete-KV: a CONCRETE int start_pos (KV concrete -> attention TC fires) gets a per-start_pos jit.
    if is_prefill_v2 and isinstance(start_pos, int):
      jit = self.prefill_v2_jits.setdefault(start_pos, TinyJit(self.forward))
    else:
      jit = (self.prefill_v2_jit if is_prefill_v2 else self.prefill_jit) if is_prefill else \
            (self.rollout_jit_flash if use_flash else self.rollout_jit)
    if not is_prefill_v2: return jit(tokens.contiguous(), start_pos, temperature)
    # contain the ambient codegen power: install the warmstart table ONLY around the prefill-v2 forward (it's
    # consulted at kernel-compile time, i.e. this jit's first call), then restore -- decode/other paths never
    # see a populated _WARMSTART_OPTS even within this process.
    import tinygrad.codegen.opt.postrange as pr
    saved = pr._WARMSTART_OPTS
    pr._WARMSTART_OPTS = self._pf16_warmstart
    try: return jit(tokens.contiguous(), start_pos, temperature)
    finally: pr._WARMSTART_OPTS = saved

  @staticmethod
  def from_gguf(gguf:Tensor|str|pathlib.Path, max_context:"int|str|None"=None,
                realize=bool(getenv("REALIZE", 0)), stream:str="auto") -> tuple[Transformer, dict]:
    # Probe free VRAM at ENTRY, before gguf_load makes the weight storage resident -- so `free` is the baseline
    # available for weights+KV (the admission budget then subtracts weights itself; probing after gguf_load would
    # double-count weights already in `used`). Total is stable regardless.
    _total_vram, _free_vram = detect_total_vram_bytes(), detect_free_vram_bytes()
    # TODO: remove the need for copy to default device
    # Q4K_PRIMITIVE defaults ON for a GGUF path ON AMD (the exact ~2.2x decode win, validated on AMD), and
    # OFF for a preloaded Tensor (no GGUF storage to view; primitive paths require a path) or a non-AMD
    # default device (the kernels are AMD-targeted). Set Q4K_PRIMITIVE=0/1 to override.
    q4k_auto = "Q4K_PRIMITIVE" not in os.environ and not isinstance(gguf, Tensor) and Device.DEFAULT == "AMD"
    use_q4k_primitive = bool(getenv("Q4K_PRIMITIVE", 1 if q4k_auto else 0))
    # Q6_K (ffn_down etc. in mixed-quant Q4_K_M) defaults ON with Q4K_PRIMITIVE: it's the decode bottleneck
    # otherwise (the slow fp-dequant fallback ~= 59% of GPU work), Q6_K dequant is exact (identical output),
    # and enabling it is a ~2.2x decode win. Set Q6K_PRIMITIVE=0 to opt out.
    use_q6k_primitive = bool(getenv("Q6K_PRIMITIVE", 1 if use_q4k_primitive else 0))
    qk_generated_policy_path = getenv("QK_GENERATED_POLICY", "")
    use_qk_generated_policy = bool(qk_generated_policy_path)
    qk_route_policy_path = getenv("QK_ROUTE_POLICY", "")
    use_qk_route_policy = bool(qk_route_policy_path)
    if (use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy or use_qk_route_policy) and isinstance(gguf, Tensor):
      raise ValueError("quant primitive paths require a GGUF path, not a preloaded Tensor")
    # QK primitive/generated linears are backed by AMD-targeted custom kernels. Auto-enable is already
    # AMD-only (q4k_auto), so this only catches an *explicit* Q4K_PRIMITIVE/Q6K_PRIMITIVE/QK_GENERATED_POLICY
    # on another backend -- fail fast with a clear message instead of an obscure later kernel failure.
    if (use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy or use_qk_route_policy) and Device.DEFAULT != "AMD":
      raise ValueError(f"QK quant primitive paths (Q4K_PRIMITIVE/Q6K_PRIMITIVE/QK_GENERATED_POLICY/QK_ROUTE_POLICY) require "
                       f"DEV=AMD; the kernels are AMD-targeted. Got Device.DEFAULT={Device.DEFAULT!r}.")
    # Auto-scan replaces the old hardcoded clamp: capture the request now (AUTO_MAX_CONTEXT/None -> auto, or an int).
    # Path GGUFs can be admitted from header metadata before any full tensor realization; this is load-bearing for
    # oversized models because gguf_load_with_metadata otherwise realizes the whole file on the default device first.
    _requested_max_context, _admit_resolved = max_context, False
    _kv_quant = bool(getenv("DECODE_KV_QUANT", 0))   # default off; the tiered admission below may enable it
    _ring_admitted = False                           # set by the admission ring tier (lossy streaming)
    if not isinstance(gguf, Tensor):
      _admit_kv, _admit_meta = gguf_load_metadata(gguf)
      _admit_arch = _admit_kv["general.architecture"]
      _admit_n_heads, _admit_n_kv_heads = _admit_kv[f"{_admit_arch}.attention.head_count"], _admit_kv[f"{_admit_arch}.attention.head_count_kv"]
      _admit_head_dim = _admit_kv.get(f"{_admit_arch}.attention.key_length_mla",
                                      _admit_kv.get(f"{_admit_arch}.attention.key_length",
                                                    _admit_kv[f"{_admit_arch}.embedding_length"] // _admit_n_heads))
      num_blocks = _admit_kv[f"{_admit_arch}.block_count"] - _admit_kv.get(f"{_admit_arch}.nextn_predict_layers", 0)
      trained_ctx = _admit_kv[f"{_admit_arch}.context_length"]
      _q4_bytes = pathlib.Path(gguf).stat().st_size
      _cov = tuple(f"{n}.weight" for n in Transformer._PREFILL_V2_LINEARS)
      _est_fp16 = sum(prod(dims) * 2 for name, dims, _, _ in _admit_meta["tensor_infos"] if any(name.endswith(s) for s in _cov))
      _kv_per_tok = 2 * _admit_n_kv_heads * _admit_head_dim * 2 * num_blocks
      _v2_should_auto = PREFILL_V2_AUTO or (PREFILL_SERVER_PROFILE and "PREFILL_V2" not in os.environ)
      _v2_reason = None
      if _v2_should_auto:
        _v2_on, _v2_reason = prefill_v2_auto_decision(_total_vram, _est_fp16, _q4_bytes, _kv_per_tok * trained_ctx)
        _set_prefill_v2(_v2_on)
      else:
        _v2_on = PREFILL_V2
      if _v2_on and PREFILL_CHUNKED:
        # per-layer overlay: fp16 weights are replay scratch, not a resident copy. Reserve a few layer-sized overlays
        # instead of the whole fp16 model. Tune with PREFILL_CHUNK_RESIDENT_BLOCKS.
        _overlay_resident = (_est_fp16 // max(num_blocks, 1)) * getenv("PREFILL_CHUNK_RESIDENT_BLOCKS", 4)
        _weights = _q4_bytes + _overlay_resident
      else:
        _weights = _q4_bytes + (_est_fp16 if _v2_on else 0)
      _prefill_per_tok = 4 * _admit_n_heads * PREFILL_UBATCH
      _flash_scratch = _admit_n_heads * int(getenv("DECODE_LIVE_SPLIT_S", 48)) * (_admit_head_dim + 2) * 4
      _model_label = f"{_admit_arch} ({_q4_bytes/1e9:.0f}GB Q4)"
      # KV-quant tier available iff the decode live-split structural shape class holds (B=1 decode, Hd=128, Hkv=8,
      # Hq%Hkv==0) -- the only route that dequants int8 KV in-register. scale_per_tok = per-(K|V,head) fp16 scale x blocks.
      _kv_quant_shape = _admit_head_dim == 128 and _admit_n_kv_heads == 8 and _admit_n_heads % _admit_n_kv_heads == 0
      _kv_quant_supported = _kv_quant_shape and not bool(getenv("DECODE_KV_QUANT_DISABLE", 0))
      # ring tier needs the same live-split shape class AND full-head rope (rope_dim==head_dim; ring re-bases positions).
      _admit_rope_dim = _admit_kv.get(f"{_admit_arch}.rope.dimension_count", _admit_head_dim)
      _ring_supported = _kv_quant_shape and _admit_rope_dim == _admit_head_dim
      _stream = str(getenv("STREAM", stream))
      _scale_per_tok = 2 * _admit_n_kv_heads * 2 * num_blocks
      max_context, _kv_quant, _admit = resolve_max_context_admission(
        _requested_max_context, trained_ctx, _free_vram, _weights, _kv_per_tok, _prefill_per_tok, _flash_scratch,
        _model_label, kv_quant_supported=_kv_quant_supported, scale_per_tok=_scale_per_tok,
        stream=_stream, ring_supported=_ring_supported)
      if getenv("DECODE_KV_QUANT", -1) != -1: _kv_quant = bool(getenv("DECODE_KV_QUANT", 0))  # explicit override
      _ring_admitted = _admit.get("ring", False)
      print(f"max_context={_admit['mode']} -> {max_context} "
            f"(free {_admit.get('free_gb', float('nan')):.1f}GB, budget {_admit.get('budget_gb', float('nan')):.1f}GB "
            f"@{VRAM_ADMIT_FRACTION}, weights {_admit.get('weights_gb', _weights/1e9):.1f}GB, "
            f"KV{'(int8)' if _kv_quant else ''} {_admit.get('kv_gb_per_1k', _kv_per_tok*1000/1e9):.2f}GB/1k, "
            f"prefill-peak {_admit.get('prefill_gb_per_1k', _prefill_per_tok*1000/1e9):.2f}GB/1k, "
            f"trained {trained_ctx}, fp16-cap {_admit.get('mc_fp16', '-')}, q8-cap {_admit.get('mc_q8', '-')})")
      if _admit.get("banner"): print(_admit["banner"])
      _admit_resolved = True
    if use_q4k_primitive or use_q6k_primitive or use_qk_generated_policy or use_qk_route_policy:
      kv, state_dict, q4k_meta = gguf_load_with_metadata(gguf)
    else:
      kv, state_dict = gguf_load(gguf.to(None).realize() if isinstance(gguf, Tensor) else gguf)
      q4k_meta = None
    qk_route_policy = _load_qk_route_policy(qk_route_policy_path) if use_qk_route_policy else None

    # all state items should be float16, not float32
    state_dict = {k:v.cast('float16') if getenv("HALF", 1) else v for k,v in state_dict.items()}

    # some models like Llama 3.2 don't have an output.weight, they just tie to the token_embd.weight
    if 'output.weight' not in state_dict: state_dict['output.weight'] = state_dict['token_embd.weight']

    arch = kv['general.architecture']
    n_heads, n_kv_heads = kv[f'{arch}.attention.head_count'], kv[f'{arch}.attention.head_count_kv']

    ssm = None
    if arch in ('qwen35', 'qwen35moe'):
      ssm = SSMConfig(**{k: kv[f'{arch}.ssm.{k}'] for k in ('conv_kernel','state_size','group_count','time_step_rank','inner_size')})
    if arch in ('qwen35', 'qwen35moe', 'glm4moe'):
      state_dict = {k.replace('post_attention_norm', 'ffn_norm'):v for k,v in state_dict.items()}

    kv_lora_rank = kv.get(f'{arch}.attention.kv_lora_rank', 0)
    head_dim = kv.get(f'{arch}.attention.key_length_mla', kv.get(f'{arch}.attention.key_length', kv[f'{arch}.embedding_length'] // n_heads))
    rope_dim = kv.get(f'{arch}.rope.dimension_count', head_dim)

    if not _admit_resolved:
      # Fallback for preloaded Tensor GGUF inputs, where no path header is available before load.
      num_blocks = kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0)
      trained_ctx = kv[f'{arch}.context_length']
      _q4_bytes = pathlib.Path(gguf).stat().st_size if not isinstance(gguf, Tensor) else 0
      _cov = tuple(f"{n}.weight" for n in Transformer._PREFILL_V2_LINEARS)
      _est_fp16 = sum(t.numel() * 2 for k, t in state_dict.items() if any(k.endswith(s) for s in _cov))
      _kv_per_tok = 2 * n_kv_heads * head_dim * 2 * num_blocks
      _v2_should_auto = PREFILL_V2_AUTO or (PREFILL_SERVER_PROFILE and "PREFILL_V2" not in os.environ)
      _v2_reason = None
      if _v2_should_auto:
        _v2_on, _v2_reason = prefill_v2_auto_decision(_total_vram, _est_fp16, _q4_bytes, _kv_per_tok * trained_ctx)
        _set_prefill_v2(_v2_on)
      else:
        _v2_on = PREFILL_V2
      if _v2_on and PREFILL_CHUNKED:
        # per-layer overlay: fp16 weights are replay scratch, not a resident copy. Reserve a few layer-sized overlays
        # instead of the whole fp16 model. Tune with PREFILL_CHUNK_RESIDENT_BLOCKS.
        _overlay_resident = (_est_fp16 // max(num_blocks, 1)) * getenv("PREFILL_CHUNK_RESIDENT_BLOCKS", 4)
        _weights = _q4_bytes + _overlay_resident
      else:
        _weights = _q4_bytes + (_est_fp16 if _v2_on else 0)
      _prefill_per_tok = 4 * n_heads * PREFILL_UBATCH
      _flash_scratch = n_heads * int(getenv("DECODE_LIVE_SPLIT_S", 48)) * (head_dim + 2) * 4
      _model_label = f"{arch} ({_q4_bytes/1e9:.0f}GB Q4)"
      _ring_supported = (head_dim == 128 and n_kv_heads == 8 and n_heads % n_kv_heads == 0 and rope_dim == head_dim)
      max_context, _kv_quant, _admit = resolve_max_context_admission(_requested_max_context, trained_ctx, _free_vram, _weights,
                                                                     _kv_per_tok, _prefill_per_tok, _flash_scratch, _model_label,
                                                                     stream=str(getenv("STREAM", stream)), ring_supported=_ring_supported)
      _ring_admitted = _admit.get("ring", False)
      print(f"max_context={_admit['mode']} -> {max_context} "
            f"(free {_admit.get('free_gb', float('nan')):.1f}GB, budget {_admit.get('budget_gb', float('nan')):.1f}GB "
            f"@{VRAM_ADMIT_FRACTION}, weights {_admit.get('weights_gb', _weights/1e9):.1f}GB, "
            f"KV {_admit.get('kv_gb_per_1k', _kv_per_tok*1000/1e9):.2f}GB/1k, "
            f"prefill-peak {_admit.get('prefill_gb_per_1k', _prefill_per_tok*1000/1e9):.2f}GB/1k, "
            f"trained {trained_ctx}, mem-cap {_admit.get('mc_mem', '-')})")
      if _admit.get("banner"): print(_admit["banner"])

    # Permute RoPE weights from interleaved to half-split layout.
    for name in state_dict:
      if ('attn_q.weight' in name or 'attn_q_b.weight' in name) and (arch == 'llama' or kv_lora_rank):
        w = state_dict[name].reshape(n_heads, state_dict[name].shape[0]//n_heads, -1)
        prefix = head_dim-rope_dim
        state_dict[name] = w[:, :prefix].cat(w[:, prefix:].rearrange("n (h two) d -> n (two h) d", two=2), dim=1).reshape(-1, w.shape[-1])
      elif arch == 'llama' and 'attn_k.weight' in name:
        w = state_dict[name].reshape(n_kv_heads, state_dict[name].shape[0]//n_kv_heads, -1)
        state_dict[name] = w.rearrange("n (h two) d -> n (two h) d", two=2).reshape(-1, w.shape[-1])
      elif kv_lora_rank and 'attn_kv_a_mqa.weight' in name:
        state_dict[name] = state_dict[name][:kv_lora_rank].cat(state_dict[name][kv_lora_rank:].rearrange("(h two) d -> (two h) d", two=2), dim=0)
    config = TransformerConfig(
      num_blocks=kv[f'{arch}.block_count'] - kv.get(f'{arch}.nextn_predict_layers', 0), dim=kv[f'{arch}.embedding_length'],
      hidden_dim=kv.get(f'{arch}.expert_feed_forward_length', kv.get(f'{arch}.feed_forward_length', 0)),
      n_heads=n_heads, n_kv_heads=n_kv_heads, norm_eps=kv[f'{arch}.attention.layer_norm_rms_epsilon'],
      vocab_size=len(kv['tokenizer.ggml.tokens']),
      head_dim=head_dim,
      rope_theta=kv[f'{arch}.rope.freq_base'],
      rope_dim=rope_dim,
      v_head_dim=kv.get(f'{arch}.attention.value_length_mla', kv.get(f'{arch}.attention.value_length', head_dim)),
      max_context=max_context,
      qk_norm=int(state_dict['blk.0.attn_q_norm.weight'].shape[0]) if 'blk.0.attn_q_norm.weight' in state_dict else 0,
      num_experts=kv.get(f'{arch}.expert_count', 0), num_experts_per_tok=kv.get(f'{arch}.expert_used_count', 0),
      norm_topk_prob=kv.get(f'{arch}.expert_weights_norm', arch in ('qwen3moe', 'qwen35moe')),
      kv_lora_rank=kv_lora_rank, q_lora_rank=kv.get(f'{arch}.attention.q_lora_rank', 0),
      leading_dense_blocks=kv.get(f'{arch}.leading_dense_block_count', 0),
      shared_expert_dim=kv.get(
        f'{arch}.expert_shared_feed_forward_length',
        kv.get(f'{arch}.expert_shared_count', 0) * kv.get(f'{arch}.expert_feed_forward_length', 0)),
      shared_expert_gate=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.ffn_gate_inp_shexp.weight" in state_dict,
      dense_hidden_dim=kv.get(f'{arch}.feed_forward_length', 0) if kv.get(f'{arch}.leading_dense_block_count', 0) else 0,
      routed_scaling_factor=kv.get(f'{arch}.expert_weights_scale', 1.0), attn_output_gate=arch in ('qwen35', 'qwen35moe'), ssm=ssm,
      full_attention_interval=kv.get(f'{arch}.full_attention_interval', 0),
      qkv_bias='blk.0.attn_q.bias' in state_dict,
      expert_bias=f"blk.{kv.get(f'{arch}.leading_dense_block_count', 0)}.exp_probs_b.bias" in state_dict,
      admit=_admit, kv_quant=_kv_quant, ring=_ring_admitted)
    _set_qk_route_policy(qk_route_policy, bool(getenv("QK_ROUTE_POLICY_STRICT", 0)),
                         bool(getenv("QK_ROUTE_POLICY_DEBUG", 0)))
    _validate_qk_route_policy_for_config(qk_route_policy, config)
    # TG-P6: PURE_MACHINE_SEARCH_ONLY diagnostic mode. Fail loud (naming route + replacement scope) if any hot
    # default route is not machine-authored/generated. Rollback/oracle routes are tolerated only when explicitly
    # requested (PURE_MACHINE_SEARCH_ALLOW_ROLLBACK=1). No-op unless PURE_MACHINE_SEARCH_ONLY=1.
    if getenv("PURE_MACHINE_SEARCH_ONLY", 0):
      qk_ops.assert_pure_machine_search()
    # Prefill policy auto-resolution, BEFORE Transformer() (its __init__ reads PREFILL_V2 to build the warmstart,
    # and the concrete-KV precompile + generate() read PREFILL_CONCRETE_KV). Explicit 0/1 skip these.
    # PREFILL_SERVER_PROFILE=1 implies PREFILL_V2=auto (when V2 unset) + concrete-KV on (when V2 ends up on).
    # PREFILL_V2 auto was already resolved (conservatively, at trained_ctx) in the admission block above and applied
    # via _set_prefill_v2; report it here against the now-admitted ctx. Reusing that one decision keeps `weights` in
    # the admission budget consistent with what is actually realized (no second, looser re-decide).
    if _v2_should_auto:
      kv_bytes = _kv_per_tok * config.max_context
      print(f"PREFILL_V2=auto -> {'ON' if _v2_on else 'OFF'}: {_v2_reason} "
            f"(fp16 covered {_est_fp16/1e9:.1f}GB, Q4 {_q4_bytes/1e9:.1f}GB, KV {kv_bytes/1e9:.1f}GB @ctx{config.max_context})")
    if PREFILL_CONCRETE_KV_AUTO or PREFILL_SERVER_PROFILE:
      ckv_on, ckv_reason = prefill_concrete_kv_auto_decision(PREFILL_SERVER_PROFILE, PREFILL_V2)
      _set_prefill_concrete_kv(ckv_on)
      print(f"PREFILL_CONCRETE_KV=auto -> {'ON' if ckv_on else 'OFF'}: {ckv_reason}")
    # FAST_EMPTY_INIT: every weight is REPLACED by load_state_dict below, so building the ~254 random init graphs
    # (nn.Linear Tensor.uniform / nn.Embedding glorot_uniform) is wasted work (~2.3s of the load, per profiling).
    # Init EMPTY during construction instead -- correct because nothing reads the random values before they're replaced.
    _saved_init = None
    if getenv("FAST_EMPTY_INIT", 1):
      _saved_init = (Tensor.__dict__.get("uniform"), Tensor.__dict__.get("glorot_uniform"))
      _fe = lambda *shape, **kw: Tensor.empty(*shape)
      Tensor.uniform = Tensor.glorot_uniform = _fe
    try:
      model = Transformer(config)
    finally:
      if _saved_init is not None:
        for _n, _v in zip(("uniform", "glorot_uniform"), _saved_init):
          delattr(Tensor, _n) if _v is None else setattr(Tensor, _n, _v)
    nn.state.load_state_dict(model, state_dict, verbose=False, consume=True, realize=False)  # NOTE: rope_freqs.weight (32,) is unused
    if q4k_meta is not None:
      # auto-enabled primitives default to `shared` storage (view the GGUF in place, storage_bytes=0) so
      # large models (e.g. 32B) stay within VRAM; explicit Q4K_PRIMITIVE keeps `sidecar`; env always wins.
      qk_cfg = QKConfig.from_env(storage_default="shared" if q4k_auto else "sidecar")
      primitive_linears = []
      primitive_budget = QKPrimitiveBudget(qk_cfg.max_storage_bytes, qk_cfg.generated_policy_strict)
      q4_storage_mode, q6_storage_mode = qk_cfg.storage_mode, qk_cfg.q6_storage_mode
      generated_policy = _load_qk_generated_policy(qk_generated_policy_path) if use_qk_generated_policy else None
      if generated_policy is not None:
        if qk_cfg.policy_debug:
          print(f"QK_GENERATED_POLICY_DEBUG loaded={qk_generated_policy_path} entries={_qk_generated_policy_len(generated_policy)}")
        primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q4_storage_mode)
        primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, generated_policy, primitive_budget, q6_storage_mode)
      else:
        if use_q4k_primitive: primitive_linears += _install_q4k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q4_storage_mode)
        if use_q6k_primitive: primitive_linears += _install_q6k_primitives(model, pathlib.Path(gguf), q4k_meta, None, primitive_budget, q6_storage_mode)
      if qk_cfg.storage_debug:
        summary = _qk_storage_summary(primitive_linears)
        cap = -1 if primitive_budget.cap_bytes is None else primitive_budget.cap_bytes
        by_kind_s = ",".join(f"{k}:{v}" for k, v in summary["by_kind"].items()) or "none"
        by_mode_s = ",".join(f"{k}:{v}" for k, v in summary["by_mode"].items()) or "none"
        print(f"QK_PRIMITIVE_STORAGE_DEBUG installed={len(primitive_linears)} source_bytes={summary['source_bytes']} "
              f"storage_bytes={summary['persistent_bytes']} shared_bytes={summary['shared_bytes']} "
              f"nonpersistent_bytes={summary['nonpersistent_bytes']} runtime_cap_bytes={cap} "
              f"runtime_cap_used_bytes={primitive_budget.used_bytes} by_kind={by_kind_s} by_mode={by_mode_s} "
              f"requested_storage_mode={q4_storage_mode} q4_effective_storage_mode={q4_storage_mode} "
              f"q6_effective_storage_mode={q6_storage_mode}")
      if primitive_linears and qk_cfg.demote_targets:  # B3: requant over-provisioned Q6 tensors -> Q4 (searched set)
        primitive_linears = _demote_q6k_to_q4(model, primitive_linears, qk_cfg.demote_targets)
      if primitive_linears: model._q4k_linears = Q4KPrimitiveRegistry(primitive_linears)
      if primitive_linears and qk_cfg.fuse_q4k:  # B1 horizontal-fusion probe -- REFUTED, experimental only
        import sys as _sys
        print("WARNING: Q4K_FUSE is REFUTED (qk-gemv-final-mile-20260617.md): ~-18% decode + prefill fallback "
              "broken on T>32. Enabled only for experiments; do NOT use for shipped benchmarks.", file=_sys.stderr)
        _install_q4k_fusions(model)
    # NOTE: without this contiguous, it unpacks the weights from the model every time. we shouldn't need this, but for now it's faster
    if realize:
      for s in (params:=nn.state.get_parameters(model)): s.replace(s.contiguous())
      Tensor.realize(*params)
    # prefill v2 (opt-in): realize fp16 weights now that primitives are installed (shapes/dequant graphs ready)
    if PREFILL_V2: model.realize_prefill_v2_weights()
    # Increment 0 ship: with PREFILL_CONCRETE_KV, precompile the per-start_pos concrete prefill jits at load so the
    # ~5s/jit compile tax is paid once here, not inline on a cold prompt -> every generation is warm.
    if PREFILL_V2 and PREFILL_CONCRETE_KV: model.precompile_concrete_prefill_jits()
    return model, kv

  def get_start_pos(self, tokens:list[int]) -> int:
    prefix_len = sum(1 for _ in itertools.takewhile(lambda ab: ab[0] == ab[1], zip(tokens[:-1], self._cached_tokens)))
    return min(block._reusable_prefix_len(prefix_len, len(self._cached_tokens)) for block in self.blk)

  def _ring_slot(self, logical_pos:int, N:int, sinks:int) -> int:
    # StreamingLLM ring write slot: FILL in order (the first `sinks` tokens land in slots 0..sinks-1); once FULL,
    # round-robin the WINDOW (slots sinks..N-1) with modulus N-sinks so the sink slots are never overwritten.
    if logical_pos < N: return logical_pos
    return sinks + ((logical_pos - sinks) % (N - sinks))

  def _ring_gather_freqs(self, freqs_full:Tensor, logical_pos:int, N:int, sinks:int) -> Tensor:
    # Pre-gather the rotary table so row==slot -> freqs_full[pos_of(slot)]. FILL: identity (row==slot==abs position, so
    # ctx<N is token-identical). FULL: sinks keep positions 0..sinks-1; window slot s gets recency position
    # N-1 - ((wp - s) mod (N-sinks)) so the newest slot (wp) is N-1 and the oldest window slot is `sinks` -> all rotary
    # phases stay in [0,N) <= trained ctx. Rebuilt each step (shared across layers) and fed as the ring JIT freqs input.
    if logical_pos < N: return freqs_full
    W = N - sinks
    wp = sinks + ((logical_pos - sinks) % W)
    posmap = list(range(sinks)) + [(N - 1) - ((wp - s) % W) for s in range(sinks, N)]
    return freqs_full[Tensor(posmap, dtype=dtypes.int32, device=freqs_full.device)].contiguous()

  def warmup_flash_decode(self):
    # Pre-capture rollout_jit_flash so the in-generation crossover at ctx>=FLASH_DECODE_THRESHOLD does not pay the
    # one-time jit-compile stall inline (the first long generation would otherwise pause ~once at the boundary).
    # Dummy single-token decode at a high symbolic start_pos; cache contents are irrelevant for graph capture.
    if self.has_recurrent_block: return
    ctx = min(max(self.max_context // 2, getenv("FLASH_DECODE_THRESHOLD", 512)), self.max_context - 1)
    if ctx < 1: return
    v_sp = UOp.variable("start_pos", 0, self.max_context - 1)
    dummy, temp = Tensor([[0]], dtype="int32"), Tensor([0.0])
    for _ in range(3):
      try: self(dummy, v_sp.bind(ctx), temp, use_flash=True).realize()
      except Exception: return

  def generate(self, tokens:list[int], chunk_size:int=32, temperature:float=0.0):
    if self.has_recurrent_block: chunk_size = 1
    # ring is enabled by the admission streaming tier (config.ring) or the manual env flag; both require full-head rope.
    _ring = (self.config.ring or bool(getenv("DECODE_RING", 0))) and self.config.rope_dim == self.config.head_dim
    if _ring and len(tokens) > self.max_context:
      # StreamingLLM evicts DURING generation, not prefill: a prompt larger than the physical window N can't be held.
      raise RuntimeError(f"prompt is {len(tokens)} tokens but the streaming window is N={self.max_context}: streaming "
                         f"evicts during generation, not prefill. Shorten the prompt to <={self.max_context} tokens, or "
                         f"use a model/quant that admits a larger window.")
    for _b in self.blk: _b._ring_active = _ring   # make prefill ALSO store un-roped K when the ring is on
    v_start_pos = UOp.variable("start_pos", 0, self.max_context-1)
    v_toks = UOp.variable("toks", 1, chunk_size)
    # TODO: use UOp.variable for temperature once float variables are supported
    temp = Tensor([temperature])
    # assign all input tokens once, then slice from start_pos for the model call
    t = Tensor(tokens + [0] * (self.max_context - len(tokens)), dtype="int32").reshape(1, self.max_context)
    # recompute start_pos from what's currently valid in the caches
    start_pos = self.get_start_pos(tokens)
    if start_pos < len(self._cached_tokens) and (resets := [r for b in self.blk for r in b._state_reset_ops()]): Tensor.realize(*resets)
    # flash-decode selection is centralized in should_use_flash_decode (default FLASH_DECODE=auto, threshold
    # 512): generate passes no use_flash override and lets that single authority decide per captured graph.
    out, prompt_len = None, len(tokens)
    while _ring or len(tokens) < self.max_context:   # ring: unbounded logical context (caller controls when to stop)
      if PREFILL_V2 and (prompt_len - start_pos) >= PREFILL_UBATCH:
        # prefill v2: a CONCRETE-T chunk of all-real prompt tokens (start_pos still symbolic; only the token
        # dim must be concrete for tensor cores). remaining>=UBATCH => start_pos<prompt_len so we slice from t.
        # concrete start_pos -> KV=start_pos+T concrete -> attention TC fires (the validated 1.24x, byte-identical).
        # Default ON for the FIRST chunk (start_pos==0): one cached concrete jit, no multi-chunk compile cost.
        # PREFILL_CONCRETE_KV=1 forces it for ALL chunks (K jits, pays off only when cached / for prompt<=512).
        use_concrete = (start_pos == 0) or PREFILL_CONCRETE_KV
        sp, ntv = (start_pos if use_concrete else v_start_pos.bind(start_pos)), PREFILL_UBATCH
        out = self(t[:, sp:sp+PREFILL_UBATCH], sp, temp, use_flash=False).realize()
      elif PREFILL_REMAINDER_FIX and PREFILL_V2 and start_pos < prompt_len and prompt_len >= PREFILL_UBATCH:
        # Phase-3 fix: a sub-UBATCH PROMPT remainder would otherwise fall to many slow 32-token symbolic calls
        # (the fallback trap). Instead process the LAST PREFILL_UBATCH tokens as ONE prefill-v2 chunk by shifting
        # the window back so it ENDS exactly at prompt_len -> all-real tokens (no padding), last position is
        # prompt_len-1 so out.item() is the next token. Re-processes the small overlap with the prior chunk (same
        # tokens -> same KV) -> correct. Symbolic start_pos reuses the one prefill_v2_jit (no per-remainder compile).
        sp = v_start_pos.bind(prompt_len - PREFILL_UBATCH)   # symbolic offset -> matches the prefill_v2_jit signature
        out = self(t[:, sp:sp+PREFILL_UBATCH], sp, temp, use_flash=False).realize()
        ntv = prompt_len - start_pos                      # advance straight to end of prompt
      elif _ring and start_pos >= prompt_len and out is not None:
        # StreamingLLM ring decode (T=1, past the prompt). Bind the WRAPPED write slot (always in [0,N-1] -> never trips
        # Variable.bind's vmax assert even as the logical position grows unboundedly); feed the per-step pre-gathered
        # freqs (identity while filling -> token-identical; slot-relative once full); ring_full switches to the [0:N]
        # read graph at the wrap. Two graphs total (fill + full), captured once each -- no per-step recompile.
        _N, _sinks = self.max_context, int(getenv("DECODE_RING_SINKS", 4))
        sp = v_start_pos.bind(self._ring_slot(start_pos, _N, _sinks)); ntv = 1
        _rf = self._ring_gather_freqs(next(b.freqs_cis for b in self.blk if hasattr(b, "freqs_cis")),
                                      start_pos, _N, _sinks)
        out = self(out, sp, temp, use_flash=True, ring_freqs=_rf, ring_full=(start_pos >= _N)).realize()
      else:
        sp, nt = v_start_pos.bind(start_pos), v_toks.bind(min(chunk_size, len(tokens) - start_pos))
        ntv = nt.val
        # Select the flash-decode graph (rollout_jit_flash) vs SDPA graph (rollout_jit) per-token by context, so a
        # generation that STARTS short still crosses over to flash once ctx reaches the threshold. Without this the
        # decode graph is baked SDPA at the start ctx and never switches -> short-prompt decode SDPA-degrades the
        # whole way (e.g. 85->54 tok/s by ctx512). should_use_flash_decode returns False for ntv!=1 (prefill chunks).
        _uf = should_use_flash_decode(sp, ntv)
        out = self(t[:, sp:sp+nt] if start_pos < prompt_len or out is None else out, sp, temp,
                   use_flash=_uf).realize()
      start_pos += ntv
      # chunked prefill: keep processing until all prompt tokens are consumed
      if start_pos < len(tokens): continue
      tokens.append(int(out.item()))
      self._cached_tokens = tokens[:-1]
      yield tokens[-1]
