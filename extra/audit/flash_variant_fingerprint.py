#!/usr/bin/env python3
"""LR-071: a two-arm fingerprint for the decode flash tile's env-gated variant matrix.

Why this exists. `extra/audit/lowering_baseline.py` compiles the decode flash tile for three
(model, staging) configurations and proves the ISA is byte-identical. That gate is real, but it only ever
exercises the DEFAULT arm of every env-gated branch inside the builder, because `getenv` is read at
kernel-build time and the gate runs with those variables unset. `flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`
has a `DECODE_ATTN_TILE_SPLIT_SCORE` branch that builds a materially different graph -- a pass-1 LDS score
buffer plus a second reduce loop -- and nothing compiled it. So a refactor touching code shared by both arms
would be certified safe by a gate that had only ever seen one of them.

That is not hypothetical: LR-070 extracted ~18 duplicated lines (the online-softmax merge and PV
accumulation, including a copy-pasted WAR-barrier comment) into a single `_merge_tail` used by both arms.
The default arm was covered by the AMD gate. This gate is what covered the other one.

What it does: builds the kernel's UOp sink under each arm, in a SEPARATE PROCESS per arm, and hashes it.
Separate processes are required, not stylistic -- `getenv` is `@functools.cache`d, so a variable flipped after
the first read inside one process never reaches the builder (see
docs/task_workflow/input/lr-019-gate-mechanism-divergence-scope-20260726.md).

GPU-free and compile-free: it hashes the constructed UOp graph (`sink.key`), never rendering or running a
kernel, so it needs no AMD device and is safe to run anywhere.

Run:
  PYTHONPATH=. python3 extra/audit/flash_variant_fingerprint.py           # write bench/flash-variant-fingerprint/latest.json
  PYTHONPATH=. python3 extra/audit/flash_variant_fingerprint.py --check   # recompute + diff; exit 1 on any change
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "bench" / "flash-variant-fingerprint"
OUT_PATH = OUT_DIR / "latest.json"
SCHEMA = "tinygrad.flash_variant_fingerprint.v1"

# (Hq, Hkv, Hd) per model, matching ATTN_SHAPES in lowering_baseline.py.
MODELS = {"8B": (32, 8, 128), "14B": (40, 8, 128)}
STAGINGS = ("KV_BOTH", "K_ONLY")
MAXC, S, TC = 4096, 4, 1024

# The env-gated arms of the builder's variant matrix. "" is the default arm. Each entry becomes one subprocess.
# DECODE_STAGE_COALESCE is deliberately absent: it changes the staging LaneMap and is covered by its own
# microgate; adding it here would triple the arm count for a branch this gate was not built to certify. Say so
# rather than implying the matrix is complete.
ARMS: tuple[tuple[str, dict[str, str]], ...] = (
  ("default", {}),
  ("split_score", {"DECODE_ATTN_TILE_SPLIT_SCORE": "1"}),
  ("inline_reduce", {"DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE": "1"}),
)

_CHILD = """
import hashlib, json, sys
sys.path.insert(0, {root!r})
from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes
from extra.qk.decode.flash_decode_attention_spec import describe_flash_decode_attention
out = {{}}
for model, (Hq, Hkv, Hd) in {models!r}.items():
  for staging in {stagings!r}:
    spec = describe_flash_decode_attention(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC={maxc}, S={s}, staging=staging)
    fn = spec.emit_tile(UOp.const(dtypes.int32, {tc}))
    W2 = Hd + 2
    pout = UOp.placeholder((Hq * {s} * W2,), dtypes.float32, 0)
    q = UOp.placeholder((Hq * Hd,), dtypes.float16, 1)
    cache = UOp.placeholder((2, 1, Hkv, {maxc}, Hd), dtypes.float16, 2)
    out[f"{{model}}.{{staging}}"] = hashlib.sha256(fn(pout, q, cache).key).hexdigest()
print(json.dumps(out))
"""

# Only these are passed to the child, plus the arm's own variables. An inherited gate variable would silently
# change the graph and be recorded as if it were the default -- the same class of bug LR-019 fixed in the
# lowering fingerprint.
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH", "PYTHONHASHSEED")


def _run_arm(arm_env: dict[str, str]) -> dict[str, str]:
  src = _CHILD.format(root=str(ROOT), models=MODELS, stagings=STAGINGS, maxc=MAXC, s=S, tc=TC)
  env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
  env["PYTHONPATH"] = str(ROOT)
  env.update(arm_env)
  r = subprocess.run([sys.executable, "-c", src], cwd=str(ROOT), capture_output=True, text=True, env=env)
  if r.returncode != 0:
    raise RuntimeError(f"arm {arm_env} failed:\n{r.stderr[-2000:]}")
  return json.loads(r.stdout.strip().splitlines()[-1])


def compute() -> dict[str, dict[str, str]]:
  return {name: _run_arm(env) for name, env in ARMS}


def build_artifact(argv: list[str]) -> dict[str, Any]:
  return {
    "header": {"schema": SCHEMA, "python_version": sys.version,
               "command": "python3 " + " ".join(["extra/audit/flash_variant_fingerprint.py", *argv])},
    "arms": compute(),
  }


def _diff(old: dict, new: dict) -> list[tuple[str, str, str]]:
  rows = []
  for arm in sorted(set(old) | set(new)):
    o, n = old.get(arm, {}), new.get(arm, {})
    for cfg in sorted(set(o) | set(n)):
      a, b = o.get(cfg), n.get(cfg)
      if a is None: rows.append((f"{arm}.{cfg}", "ADDED", str(b)))
      elif b is None: rows.append((f"{arm}.{cfg}", "REMOVED", str(a)))
      elif a != b: rows.append((f"{arm}.{cfg}", "CHANGED", f"{a} -> {b}"))
  return rows


def run_check(argv: list[str]) -> int:
  if not OUT_PATH.is_file():
    print(f"FAIL: no stored artifact at {OUT_PATH}; run without --check first")
    return 1
  stored = json.loads(OUT_PATH.read_text()).get("arms", {})
  fresh = build_artifact(argv)["arms"]
  rows = _diff(stored, fresh)
  for name, status, detail in rows:
    print(f"{name:28s} {status:8s} {detail}")
  n = sum(len(v) for v in fresh.values())
  if not rows:
    print(f"verdict: PASS ({n} kernels across {len(fresh)} variant arms, graphs unchanged)")
    return 0
  print(f"verdict: FAIL ({len(rows)}/{n} differ)")
  return 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Two-arm fingerprint for the decode flash variant matrix.")
  ap.add_argument("--check", action="store_true", help="recompute and diff against the stored artifact")
  args, _ = ap.parse_known_args()
  argv = sys.argv[1:]
  if args.check: return run_check(argv)
  artifact = build_artifact(argv)
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  OUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  n = sum(len(v) for v in artifact["arms"].values())
  print(f"wrote {OUT_PATH} ({n} kernels across {len(artifact['arms'])} arms)")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
