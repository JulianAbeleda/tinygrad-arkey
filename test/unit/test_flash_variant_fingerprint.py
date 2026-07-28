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
  src = (ROOT / "extra/qk/flash_kernels.py").read_text()
  found = set(re.findall(r'getenv\(\s*"([A-Z0-9_]+)"', src))
  covered = {k for _, env in fvf.ARMS for k in env}
  # DECODE_STAGE_COALESCE is deliberately out of scope; see the ARMS comment in the gate.
  out_of_scope = {"DECODE_STAGE_COALESCE"}
  assert found - covered - out_of_scope == set(), \
    f"builder reads {sorted(found - covered - out_of_scope)} but no arm sets them and they are not declared out of scope"


def test_the_default_arm_is_first_and_sets_nothing():
  assert fvf.ARMS[0][0] == "default" and fvf.ARMS[0][1] == {}


def test_split_score_arm_is_present():
  """This is the arm extra/audit/lowering_baseline.py cannot see, and the reason this gate exists."""
  assert "DECODE_ATTN_TILE_SPLIT_SCORE" in dict(fvf.ARMS)["split_score"]


def test_both_score_variants_share_one_merge_tail():
  """LR-070 extracted the duplicated online-softmax merge into `_merge_tail`. If someone re-inlines it into one
  branch, the two arms drift again silently -- the failure this refactor removed."""
  src = (ROOT / "extra/qk/flash_kernels.py").read_text()
  assert src.count("def _merge_tail(") == 1
  # Both score variants now feed one shared post-selection merge call.
  assert src.count("mxu = _merge_tail(tt, new_m, corr, p)") == 1
