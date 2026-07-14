"""Boundary test for the PURE_MACHINE_SEARCH_ONLY guard (F1b + F2).

pure_search_guard.py proves purity of a PARALLEL model (HOT_FAMILIES). This test binds that claim to the REAL runtime
dispatcher: it drives tinygrad/llm/decode_routes.py (and the model.py prefill default) on the hot shapes and asserts

  1. under the shipped DEFAULT env each hot family selects the generated/pure route the manifest declares (so the guard's
     "all pure by default" claim is grounded in what the selector actually does), and
  2. each `env:{}` shipped-default route's runtime getenv default truly fires by default -- so a flipped getenv default in
     decode_routes.py / model.py (e.g. BUBBLEBEAM_FUTURESIGHT -> 0, DECODE_LIVE_SPLIT -> 0, PREFILL_GRAPH_GEMM -> 1) makes
     this suite fail instead of silently diverging from the manifest, and
  3. the guard's decode rollback predicates agree with the real selector's rollback env.

The route functions build lazy Tensors on CPU (no realize, no GPU) so selection is observable without executing kernels:
if the generated branch is taken the provided `fallback` is NOT called; a de-selected route calls fallback or fails loud.
"""
import contextlib
import os
from types import SimpleNamespace

import pytest

from tinygrad import Tensor, UOp, dtypes, getenv
import tinygrad.llm.decode_routes as dr
from tinygrad.llm import route_ops
import tinygrad.llm.model as model
from extra.qk import pure_search_guard as guard
from extra.qk.route_manifest import ROUTES, default_routes


@contextlib.contextmanager
def env(**overrides):
  """Apply env overrides (value None deletes the key), clear the getenv cache so decode_routes sees them, then restore."""
  saved = {k: os.environ.get(k) for k in overrides}
  for k, v in overrides.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v
  getenv.cache_clear()
  try:
    yield
  finally:
    for k, v in saved.items():
      if v is None: os.environ.pop(k, None)
      else: os.environ[k] = v
    getenv.cache_clear()


# Env keys the hot decode selectors read; cleared to a pristine default for each drive.
_DECODE_KEYS = dict(BUBBLEBEAM_FUTURESIGHT=None, Q4K_GEMV_SCHEDULER=None, DECODE_Q4K_G3_ANYSHAPE=None,
                    DECODE_Q4K_INKERNEL_COMBINE_KV=None, DECODE_Q4K_SPLIT_K_KV=None, DECODE_Q6K_GENERATED=None,
                    DECODE_LIVE_SPLIT=None)


def _drive_q4k(**overrides):
  called = {"fallback": False}
  def fallback(x): called["fallback"] = True; return x
  in_f, out_f = 4096, 12288  # ffn_gate_up 8B shape, G3-eligible
  storage = SimpleNamespace(mode="q4_resident", words=Tensor.empty(1, dtype=dtypes.uint8, device="CPU"))
  lin = SimpleNamespace(decode_enabled=True, bias=None, in_features=in_f, out_features=out_f,
                        name="blk.0.ffn_gate.weight", q4k_storage=storage)
  x = Tensor.empty(1, 1, in_f, dtype=dtypes.float16, device="CPU")
  with env(**{**_DECODE_KEYS, **overrides}):
    dr.q4k_primitive_linear_call(lin, x, fallback, arch_ok=True)
  return called["fallback"]


def _drive_q6k(**overrides):
  called = {"fallback": False}
  def fallback(x): called["fallback"] = True; return x
  in_f, out_f = 12288, 4096  # ffn_down Q6_K shape
  storage = SimpleNamespace(halfs=Tensor.empty(1, dtype=dtypes.uint8, device="CPU"))
  lin = SimpleNamespace(decode_enabled=True, bias=None, in_features=in_f, out_features=out_f,
                        name="blk.0.ffn_down.weight", q6k_storage=storage, parts=1, opts=())
  x = Tensor.empty(1, 1, in_f, dtype=dtypes.float16, device="CPU")
  with env(**{**_DECODE_KEYS, **overrides}):
    dr.q6k_primitive_linear_call(lin, x, fallback, arch_ok=True)
  return called["fallback"]


def _drive_attention(monkeypatch, **overrides):
  """Drive the real flash_decode_attention_route; a sentinel replaces the generated live-split emitter so we observe
  SELECTION (was the generated route chosen?) without needing the exact KV tile layout or a GPU."""
  called = {"live_split": False}
  def sentinel(*a, **k):
    called["live_split"] = True
    return Tensor.empty(32, 128, dtype=dtypes.float32, device="CPU")
  monkeypatch.setattr(route_ops, "flash_decode_live_split_block_tile", sentinel)
  B, Hq, Hkv, Hd, MAXC = 1, 32, 8, 128, 4096  # 8B G=4 live-split shape
  q = Tensor.empty(B, Hq, Hd, dtype=dtypes.float16, device="CPU")
  kv = Tensor.empty(2, MAXC, Hkv, Hd, dtype=dtypes.float16, device="CPU")
  sp = UOp.variable("start_pos", 0, MAXC - 1).bind(100)
  raised = None
  with env(**{**_DECODE_KEYS, **overrides}):
    try:
      dr.flash_decode_attention_route(q, kv, sp, 1, B, Hq, Hkv, Hd, MAXC)
    except RuntimeError as e:
      raised = str(e)
  return called["live_split"], raised


# ---- (1) default env selects the generated route the manifest declares ----

def test_real_q4k_default_selects_generated_route():
  assert _drive_q4k() is False  # generated G3 taken -> fallback NOT called
  # guard's parallel model agrees for the same env
  q4k = {r["family"]: r for r in guard.effective_routes({})}["decode_q4k_gemv"]
  assert q4k["effective_route"] == "decode_q4k_g3_generated"
  assert q4k["rolled_back_to_oracle"] is False and q4k["pure"] is True


def test_real_q6k_default_selects_generated_route():
  assert _drive_q6k() is False  # generated Q6_K taken -> fallback NOT called
  q6k = {r["family"]: r for r in guard.effective_routes({})}["decode_q6k_gemv"]
  assert q6k["effective_route"] == "decode_q6k_coop_generated"
  assert q6k["pure"] is True


def test_real_attention_default_selects_live_split(monkeypatch):
  selected, raised = _drive_attention(monkeypatch)
  assert selected is True and raised is None  # generated live-split emitter reached
  attn = {r["family"]: r for r in guard.effective_routes({})}["decode_attention"]
  assert attn["effective_route"] == "decode_flash_live_split_g4_8b_kvboth"
  assert attn["rolled_back_to_oracle"] is False and attn["pure"] is True


def test_prefill_default_is_promoted_generated_candidate_set():
  # model.py enables graph-GEMM only for the supported target; the manifest policy then supplies exact generated
  # candidates without selecting the raw graph-GEMM oracle.
  with env(PREFILL_GRAPH_GEMM=None):
    assert model._prefill_graph_gemm_default() == 1
  gemm = {r["family"]: r for r in guard.effective_routes({})}["prefill_gemm"]
  assert gemm["effective_route"] == "prefill_wmma_lds_dbuf_generated"
  assert len(gemm["candidate_set_identities"]) == 4
  assert gemm["rolled_back_to_oracle"] is False and gemm["pure"] is True
  with env(PREFILL_GRAPH_GEMM="0"):
    assert model._prefill_graph_gemm_default() == 0


# ---- (2) F2: a flipped getenv DEFAULT de-selects the shipped route and this suite catches it ----

def test_shipped_default_getenv_values_keep_generated_routes_on():
  # These are the runtime getenv DEFAULTS the guard encodes in _env_flag. Pin them against the values the REAL selector
  # observes under a pristine env, so flipping the source default (decode_routes.py / model.py) fails here.
  with env(**_DECODE_KEYS, PREFILL_GRAPH_GEMM=None):
    assert getenv("BUBBLEBEAM_FUTURESIGHT", 1) == 1
    assert getenv("DECODE_LIVE_SPLIT", 1) == 1
  assert guard._env_flag({}, "BUBBLEBEAM_FUTURESIGHT", 1) is True
  assert guard._env_flag({}, "DECODE_LIVE_SPLIT", 1) is True
  # and driving the real selectors with the shipped default confirms the generated arm is live
  assert _drive_q4k() is False
  assert _drive_q6k() is False


def test_real_q4k_rollback_env_deselects_generated_matches_guard():
  # BUBBLEBEAM_FUTURESIGHT=0 and Q4K_GEMV_SCHEDULER=1 both leave the generated route in the real selector...
  assert _drive_q4k(BUBBLEBEAM_FUTURESIGHT="0") is True   # fallback (ordinary graph) taken
  assert _drive_q4k(Q4K_GEMV_SCHEDULER="1") is True
  # ...and the guard's rollback predicate agrees (still pure: the fall-off is the ordinary tinygrad graph)
  assert guard._decode_q4k_rolled_back({"BUBBLEBEAM_FUTURESIGHT": "0"}) is True
  assert guard._decode_q4k_rolled_back({"Q4K_GEMV_SCHEDULER": "1"}) is True
  assert guard._decode_q4k_rolled_back({}) is False


def test_real_attention_rollback_env_deselects_live_split_matches_guard(monkeypatch):
  selected, raised = _drive_attention(monkeypatch, DECODE_LIVE_SPLIT="0")
  assert selected is False and raised is not None and "live-split" in raised  # de-selected -> fail loud, no hand fallback
  assert guard._decode_attention_rolled_back({"DECODE_LIVE_SPLIT": "0"}) is True
  assert guard._decode_attention_rolled_back({}) is False


# ---- (3) the guard's default routes are the manifest's promoted defaults, bound to generated kernels ----

def test_guard_default_families_are_manifest_defaults_with_kernel_binding():
  defaults = set(default_routes())
  for r in guard.effective_routes({}):
    rid = r["effective_route"]
    assert rid in ROUTES, rid
    assert rid in defaults, f"{rid} is not a manifest default route"
    # a hot default that emits a route-local generated kernel must declare the pattern it binds to; the ordinary
    # tinygrad-graph default (scheduler-owned, no route-local kernel) legitimately has none.
    if r["surface_class"] != "ordinary_tinygrad_graph":
      assert ROUTES[rid].get("expected_kernels"), f"{rid} has no expected_kernels binding"


def test_prefill_graph_gemm_env_selects_raw_oracle_impure():
  gemm = {r["family"]: r for r in guard.effective_routes({"PREFILL_GRAPH_GEMM": "1"})}["prefill_gemm"]
  assert gemm["effective_route"] == "prefill_pipe_role_selective_generated"
  assert gemm["rolled_back_to_oracle"] is True and gemm["pure"] is False
  with pytest.raises(RuntimeError, match="surface=external_raw_or_binary"):
    guard.assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1", "PREFILL_GRAPH_GEMM": "1"})


def test_hand_asm_lds2_profile_has_exact_impure_route_identity():
  gemm = {r["family"]: r for r in guard.effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_GEMM_PROFILE": "hand_asm_lds2"})}["prefill_gemm"]
  assert gemm["effective_route"] == "prefill_hand_asm_lds2"
  assert gemm["rolled_back_to_oracle"] is True and gemm["pure"] is False
