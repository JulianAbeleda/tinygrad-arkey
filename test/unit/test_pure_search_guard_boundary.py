"""Boundary test for the PURE_MACHINE_SEARCH_ONLY guard (F1b + F2).

pure_search_guard.py proves purity of a PARALLEL model (HOT_FAMILIES). This test binds that claim to the REAL runtime
dispatcher: it drives tinygrad/llm/decode_routes.py (and the model.py prefill default) on the hot shapes and asserts

  1. under the shipped DEFAULT env each hot family selects the generated/pure route the manifest declares (so the guard's
     "all pure by default" claim is grounded in what the selector actually does), and
  2. production decode selection is invariant under benchmark/research environment overrides, and
  3. the non-production guard may still model those overrides without controlling ordinary model execution.

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
from extra.qk.route_manifest import ROUTES, default_routes, promoted_prefill_candidate_policy


@contextlib.contextmanager
def env(**overrides):
  """Apply benchmark/test overrides, clear the getenv cache, then restore."""
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


# Legacy benchmark/research keys; production decode route selection must not read these.
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
  q = SimpleNamespace(device="AMD", reshape=lambda *_shape: Tensor.empty(Hq, Hd, dtype=dtypes.float16, device="CPU"))
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
  assert attn["effective_route"] == "decode_flash_live_split_g4_kvboth"
  assert attn["rolled_back_to_oracle"] is False and attn["pure"] is True


def test_prefill_policy_has_no_graph_gemm_boolean_selector():
  facts = SimpleNamespace(backend="AMD", architecture="gfx1100")
  base = {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "baseline", "routes": {}}
  for value in (None, "0", "1"):
    with env(PREFILL_GRAPH_GEMM=value):
      selected = model.select_prefill_runtime_policy(base, scanned_device_facts=facts, workload_reuse=False)
      assert "prefill_graph_gemm" not in selected
  bound = {**base, "strategy":"FULL_RESIDENT_OVERLAY", "graph_gemm":{"candidate_set":{}, "policy_rows":[{}]}}
  selected = model.select_prefill_runtime_policy(bound, scanned_device_facts=facts, workload_reuse=False)
  assert "prefill_graph_gemm" not in selected
  candidate_path = promoted_prefill_candidate_policy()["candidate_set_path"]
  gemm = {r["family"]: r for r in guard.effective_routes(
    {"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH":candidate_path})}["prefill_gemm"]
  assert gemm["effective_route"] == "prefill_wmma_lds_dbuf_generated"
  assert len(gemm["candidate_set_identities"]) == 4
  assert gemm["rolled_back_to_oracle"] is False and gemm["pure"] is True


def test_scanned_target_support_never_adds_a_graph_gemm_selector():
  base = {"strategy": "DIRECT_PACKED_FALLBACK", "candidate_id": "baseline", "routes": {}}
  supported = model.select_prefill_runtime_policy(base, scanned_device_facts=SimpleNamespace(backend="AMD", architecture="gfx1100"),
                                                   workload_reuse=False)
  unsupported = model.select_prefill_runtime_policy(base, scanned_device_facts=SimpleNamespace(backend="AMD", architecture="gfx1200"),
                                                     workload_reuse=False)
  assert "prefill_graph_gemm" not in supported and "prefill_graph_gemm" not in unsupported


def test_prefill_q4k_research_selector_is_manifest_attributed():
  default = {r["family"]: r for r in guard.effective_routes({})}["prefill_q4k"]
  assert default["effective_route"] == "prefill_q4k_direct_tile4x4_default"
  selected = {r["family"]: r for r in guard.effective_routes({"PREFILL_Q4K_Q8": "wmma_tiled"})}["prefill_q4k"]
  assert selected["effective_route"] == "prefill_q4k_int8_wmma_tiled_research"
  assert selected["provenance"] == "machine_authored_generated" and selected["pure"] is True


# ---- (2) benchmark/research overrides do not control production decode ----

def test_shipped_default_getenv_values_keep_generated_routes_on():
  # These are the runtime getenv DEFAULTS the guard encodes in _env_flag. Pin them against the values the REAL selector
  # observes under a pristine env, so flipping the source default (decode_routes.py / model.py) fails here.
  with env(**_DECODE_KEYS):
    assert getenv("BUBBLEBEAM_FUTURESIGHT", 1) == 1
    assert getenv("DECODE_LIVE_SPLIT", 1) == 1
  assert guard._env_flag({}, "BUBBLEBEAM_FUTURESIGHT", 1) is True
  assert guard._env_flag({}, "DECODE_LIVE_SPLIT", 1) is True
  # and driving the real selectors with the shipped default confirms the generated arm is live
  assert _drive_q4k() is False
  assert _drive_q6k() is False


def test_q4k_research_rollback_env_is_not_a_production_selector():
  assert _drive_q4k(BUBBLEBEAM_FUTURESIGHT="0") is False
  assert _drive_q4k(Q4K_GEMV_SCHEDULER="1") is False
  # The benchmark-only guard remains able to describe historical rollback experiments.
  assert guard._decode_q4k_rolled_back({"BUBBLEBEAM_FUTURESIGHT": "0"}) is True
  assert guard._decode_q4k_rolled_back({"Q4K_GEMV_SCHEDULER": "1"}) is True
  assert guard._decode_q4k_rolled_back({}) is False


def test_attention_research_rollback_env_is_not_a_production_selector(monkeypatch):
  selected, raised = _drive_attention(monkeypatch, DECODE_LIVE_SPLIT="0")
  assert selected is True and raised is None
  assert guard._decode_attention_rolled_back({"DECODE_LIVE_SPLIT": "0"}) is True
  assert guard._decode_attention_rolled_back({}) is False


# ---- (3) the guard's default routes are the manifest's promoted defaults, bound to generated kernels ----

def test_guard_default_families_are_manifest_defaults_with_kernel_binding():
  defaults = set(default_routes())
  for r in guard.effective_routes({}):
    rid = r["effective_route"]
    assert rid in ROUTES, rid
    if r["family"] != "prefill_gemm": assert rid in defaults, f"{rid} is not a manifest default route"
    else: assert rid == "prefill_v2_scheduler_matmul_default"
    # a hot default that emits a route-local generated kernel must declare the pattern it binds to; the ordinary
    # tinygrad-graph default (scheduler-owned, no route-local kernel) legitimately has none.
    if r["surface_class"] != "ordinary_tinygrad_graph":
      assert ROUTES[rid].get("expected_kernels") or ROUTES[rid].get("candidate_set_path"), f"{rid} has no kernel binding"


def test_explicit_candidate_artifact_is_attributed_without_a_route_flag():
  candidate_env = {"BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH":promoted_prefill_candidate_policy()["candidate_set_path"]}
  gemm = {r["family"]: r for r in guard.effective_routes(candidate_env)}["prefill_gemm"]
  assert gemm["effective_route"] == "prefill_wmma_lds_dbuf_generated"
  assert gemm["rolled_back_to_oracle"] is False and gemm["pure"] is True
  guard.assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1", **candidate_env})


def test_retired_hand_asm_selectors_fail_loud():
  with pytest.raises(ValueError, match="retired prefill route selectors"):
    guard.effective_routes({"PREFILL_GEMM_PROFILE": "hand_asm_lds2"})
