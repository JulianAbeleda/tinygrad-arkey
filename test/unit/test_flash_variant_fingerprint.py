"""LR-071: the decode flash builder's env-gated arms must stay declared and covered.

These are cheap structural checks. The expensive part -- actually building each arm in its own process and
diffing the graphs -- is `extra/audit/flash_variant_fingerprint.py --check`, which is a gate, not a unit test:
it spawns one subprocess per arm and is too slow to run on every pytest invocation.
"""
from __future__ import annotations

import re
import pathlib

from extra.audit import flash_variant_fingerprint as fvf

ROOT = pathlib.Path(fvf.ROOT)


def test_every_env_gate_in_the_builder_is_either_an_arm_or_declared_out_of_scope():
  """The gate is only as good as its arm list. A new getenv branch added to the builder with no corresponding
  arm would be certified by a gate that never compiled it -- which is the exact hole this was built to close."""
  src = (ROOT / "extra/llm_research/flash_kernels.py").read_text()
  found = set(re.findall(r'getenv\(\s*"([A-Z0-9_]+)"', src))
  covered = {k for _, env, _ in fvf.ARMS for k in env}
  # DECODE_STAGE_COALESCE is deliberately out of scope; see the ARMS comment in the gate.
  out_of_scope = {"DECODE_STAGE_COALESCE"}
  assert found - covered - out_of_scope == set(), \
    f"builder reads {sorted(found - covered - out_of_scope)} but no arm sets them and they are not declared out of scope"


def test_the_default_arm_is_first_and_sets_nothing():
  assert fvf.ARMS[0][0] == "default" and fvf.ARMS[0][1] == {} and fvf.ARMS[0][2] == {}


def test_inline_reduce_arm_drives_the_descriptor_field():
  """The inline arm must drive the descriptor-owned reduce_structure field (the only path production
  consults now), while keeping the legacy env alias declared covered for the untouched research builder."""
  arms = {name: (env, kwargs) for name, env, kwargs in fvf.ARMS}
  assert "inline_reduce" in arms
  env, kwargs = arms["inline_reduce"]
  assert env == {"DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE": "1"}
  assert kwargs == {"reduce_structure": "inline"}


def test_dead_split_score_branch_is_not_declared_as_an_arm():
  """The DECODE_ATTN_TILE_SPLIT_SCORE branch no longer exists in the builder at HEAD, so declaring it as an
  arm would certify a graph the builder cannot produce. The stored artifact must not claim that arm either."""
  assert "split_score" not in {name for name, _, _ in fvf.ARMS}


def test_both_score_variants_share_one_merge_tail():
  """LR-070 extracted the duplicated online-softmax merge into `_merge_tail`. If someone re-inlines it into one
  branch, the two arms drift again silently -- the failure this refactor removed."""
  src = (ROOT / "extra/llm_research/flash_kernels.py").read_text()
  assert src.count("def _merge_tail(") == 1
  # Both score variants now feed one shared post-selection merge call.
  assert src.count("mxu = _merge_tail(tt, new_m, corr, p)") == 1
