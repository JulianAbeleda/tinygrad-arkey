#!/usr/bin/env python3
"""Decode-route codegen identity check for a SHARED-RENDERER change.

`PREFILL_SOFTMAX_REDUCE_FUSE` (commit 23b8e05fc) does not live in the attention kernel -- it lives in
tinygrad/renderer/cstyle.py, the HIP renderer that EVERY AMD kernel goes through. Its two hunks fire on:

  (a) any `Ops.CUSTOMI` with `dtype is dtypes.float` and `child_count > 1`  -> gets an SSA name instead
      of being inlined;
  (b) `Ops.MAX` whose peer is an already-rendered `__builtin_fmaxf(...)`    -> renders as fmaxf instead
      of decomposing to a select.

The DECODE route builds float CUSTOMI of exactly those kinds:
  * extra/qk/flash_kernels.py -- `__builtin_amdgcn_fdot2(...)` (dtype float)
  * tinygrad/schedule/wmma/softmax.py:amd_gfx1100_broadcast_row_state -- the float "bpermute" CUSTOMI
so "prefill-only flag" is an assumption, not a fact. tinygrad/llm/prefill_policy.py's
_SHARED_ATTENTION_PROOF_FIELDS makes this explicit: promotion requires decode_nonregression_8b AND
decode_nonregression_14b, because "enabling one shared compiler path changes both supported model routes".

This script proves or refutes non-regression the cheap way: compile the REAL decode kernels (the exact
emitters tinygrad/llm/decode_routes.py drives, via extra/qk/flash_decode_attention_executor.py --
NOT flash_kernels' raw callables, which are factory functions with no .key and cannot be handed to
to_program) and sha256 the resulting AMD code objects. Byte-identical code objects mean decode's machine
code is untouched, which is a stronger statement than any throughput measurement.

Method: wrap tinygrad.device.Compiler.compile_cached to record sha256(src) and sha256(lib) for every
kernel compiled while the real decode graph is realized. Compiled libs are captured whether they came
from the disk cache or a fresh compile -- the key is the source, so an ON-arm source change cannot be
masked by a stale cache hit.

FAILS CLOSED: if either arm captured zero kernels (the failure mode where both arms raise the *same*
exception and two identical errors get mistaken for two identical binaries), the verdict is
INCONCLUSIVE, never IDENTICAL.

Run one arm at a time (the flag is read at render time via getenv, and getenv caches). The flag is
DEFAULT-ON since 2026-07-24, so BOTH arms must now be set explicitly; compare() asserts the recorded flag
value per arm and reports INCONCLUSIVE rather than identity if an arm ran in the wrong state:
  PYTHONPATH=. DEV=AMD PREFILL_SOFTMAX_REDUCE_FUSE=0 python3 extra/qk/decode_codegen_identity_check.py --out /tmp/off.json
  PYTHONPATH=. DEV=AMD PREFILL_SOFTMAX_REDUCE_FUSE=1 python3 extra/qk/decode_codegen_identity_check.py --out /tmp/on.json
then
  PYTHONPATH=. python3 extra/qk/decode_codegen_identity_check.py --compare /tmp/off.json /tmp/on.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

# Shapes: the two model routes the manifest admits, at the production decode geometry.
# 8B  -> Hq=32 Hkv=8 Hd=128 ; 14B -> Hq=40 Hkv=8 Hd=128  (tinygrad/llm/decode_routes.py binds both)
SHAPES = {
  "8B":  dict(Hq=32, Hkv=8, Hd=128),
  "14B": dict(Hq=40, Hkv=8, Hd=128),
}
MAXC, TC, SPLIT_COUNT = 512, 400, 4


def _sha(b) -> str:
  if isinstance(b, str): b = b.encode()
  return hashlib.sha256(b).hexdigest()


def collect() -> dict:
  os.environ.setdefault("DEV", "AMD")
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Compiler
  from tinygrad.helpers import getenv
  from extra.qk.flash_decode_attention_executor import flash_decode_live_split_block_tile

  captured: list[dict] = []
  orig = Compiler.compile_cached

  def wrapped(self, src, cache_context=None):
    lib = orig(self, src, cache_context)
    captured.append({"src_sha": _sha(src), "lib_sha": _sha(lib), "src_len": len(src), "lib_len": len(lib),
                     # the two renderer-hunk observables, counted on the emitted source itself
                     "n_bpermute_textual": src.count("__builtin_amdgcn_ds_bpermute"),
                     "n_fmaxf_textual": src.count("__builtin_fmaxf"),
                     "n_fdot2_textual": src.count("__builtin_amdgcn_fdot2")})
    return lib

  Compiler.compile_cached = wrapped
  try:
    result: dict = {"_schema": "decode-codegen-identity.v1",
                    # record the EFFECTIVE value (default is now 1), so an arm run in the wrong state is caught
                    "flag_PREFILL_SOFTMAX_REDUCE_FUSE": int(getenv("PREFILL_SOFTMAX_REDUCE_FUSE", 1)),
                    "shapes": {}}
    for name, g in SHAPES.items():
      Hq, Hkv, Hd = g["Hq"], g["Hkv"], g["Hd"]
      before = len(captured)
      entry: dict = {"geometry": g}
      try:
        rng = np.random.default_rng(20260724)
        qn = rng.normal(0, .04, (1, Hq, 1, Hd)).astype(np.float16)
        cache_np = rng.normal(0, .04, (2, 1, Hkv, MAXC, Hd)).astype(np.float16)
        q_dev = Tensor(qn, device="AMD")
        cache = Tensor(cache_np, device="AMD")
        out = flash_decode_live_split_block_tile(q_dev, cache, TC, Hd, Hq, Hkv, MAXC, SPLIT_COUNT,
                                                staging="KV_BOTH", fused_combine=True)
        arr = out.numpy()   # forces render + compile + execute
        entry["executed"] = True
        entry["out_shape"] = list(arr.shape)
        entry["out_sha"] = _sha(arr.astype("float32").tobytes())
        entry["out_finite"] = bool(np.isfinite(arr).all())
      except Exception as e:
        entry["executed"] = False
        entry["error"] = f"{type(e).__name__}: {e}"[:400]
      entry["kernels"] = captured[before:]
      entry["n_kernels"] = len(entry["kernels"])
      result["shapes"][name] = entry
    return result
  finally:
    Compiler.compile_cached = orig


def compare(off: dict, on: dict) -> dict:
  report: dict = {"gate": "decode_codegen_identity", "shapes": {}}
  problems: list[str] = []
  if off.get("_schema") != "decode-codegen-identity.v1" or on.get("_schema") != "decode-codegen-identity.v1":
    return {"gate": "decode_codegen_identity", "verdict": "INCONCLUSIVE", "reason": "bad or missing arm schema"}
  if off.get("flag_PREFILL_SOFTMAX_REDUCE_FUSE") != 0:
    problems.append(f"OFF arm ran with flag={off.get('flag_PREFILL_SOFTMAX_REDUCE_FUSE')!r}, expected 0")
  if on.get("flag_PREFILL_SOFTMAX_REDUCE_FUSE") != 1:
    problems.append(f"ON arm ran with flag={on.get('flag_PREFILL_SOFTMAX_REDUCE_FUSE')!r}, expected 1")

  for name in SHAPES:
    a, b = off["shapes"].get(name), on["shapes"].get(name)
    s: dict = {}
    if a is None or b is None:
      problems.append(f"{name}: missing from an arm"); report["shapes"][name] = {"status": "MISSING"}; continue
    # FAIL CLOSED: zero kernels captured means nothing was proven, even if both arms "agree".
    if not a.get("n_kernels") or not b.get("n_kernels"):
      problems.append(f"{name}: captured {a.get('n_kernels')} / {b.get('n_kernels')} kernels -- nothing compiled, "
                      f"cannot conclude identity (off_err={a.get('error')!r} on_err={b.get('error')!r})")
      s["status"] = "NOTHING_COMPILED"
    elif not a.get("executed") or not b.get("executed"):
      problems.append(f"{name}: an arm did not execute (off={a.get('error')!r} on={b.get('error')!r})")
      s["status"] = "NOT_EXECUTED"
    s["n_kernels"] = [a.get("n_kernels"), b.get("n_kernels")]
    s["off_lib_shas"] = [k["lib_sha"] for k in a.get("kernels", [])]
    s["on_lib_shas"] = [k["lib_sha"] for k in b.get("kernels", [])]
    s["off_src_shas"] = [k["src_sha"] for k in a.get("kernels", [])]
    s["on_src_shas"] = [k["src_sha"] for k in b.get("kernels", [])]
    s["src_identical"] = s["off_src_shas"] == s["on_src_shas"]
    s["lib_identical"] = s["off_lib_shas"] == s["on_lib_shas"]
    s["out_identical"] = (a.get("out_sha") == b.get("out_sha") and a.get("out_sha") is not None)
    s["textual_counts_off"] = [(k["n_bpermute_textual"], k["n_fmaxf_textual"], k["n_fdot2_textual"])
                               for k in a.get("kernels", [])]
    s["textual_counts_on"] = [(k["n_bpermute_textual"], k["n_fmaxf_textual"], k["n_fdot2_textual"])
                              for k in b.get("kernels", [])]
    if s.get("status") is None:
      s["status"] = "IDENTICAL" if (s["src_identical"] and s["lib_identical"] and s["out_identical"]) else "DIFFERS"
      if s["status"] == "DIFFERS":
        problems.append(f"{name}: decode codegen CHANGED (src_identical={s['src_identical']} "
                        f"lib_identical={s['lib_identical']} out_identical={s['out_identical']}) -- "
                        f"a throughput+correctness decode measurement is now MANDATORY in both arms")
    report["shapes"][name] = s

  report["problems"] = problems
  statuses = {v.get("status") for v in report["shapes"].values()}
  if problems or statuses != {"IDENTICAL"}:
    report["verdict"] = "FAIL" if "DIFFERS" in statuses else "INCONCLUSIVE"
  else:
    report["verdict"] = "PASS"
  report["note"] = {
    "PASS": "every real decode kernel renders to byte-identical source AND a byte-identical code object in both "
            "arms, for every manifest-admitted decode shape; the shared-renderer change cannot affect decode",
    "FAIL": "decode machine code differs -> decode non-regression must be MEASURED, not asserted",
    "INCONCLUSIVE": "the comparison did not actually compile decode in both arms; do NOT read this as identity",
  }[report["verdict"]]
  return report


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out")
  ap.add_argument("--compare", nargs=2, metavar=("OFF", "ON"))
  args = ap.parse_args()
  if args.compare:
    with open(args.compare[0]) as f: off = json.load(f)
    with open(args.compare[1]) as f: on = json.load(f)
    rep = compare(off, on)
    print(json.dumps(rep, indent=2))
    print(f"AUTHORITY_GATE: {rep['verdict']}")
    return 0 if rep["verdict"] == "PASS" else 1
  rep = collect()
  txt = json.dumps(rep, indent=2)
  if args.out:
    with open(args.out, "w") as f: f.write(txt)
  print(txt)
  total = sum(v.get("n_kernels", 0) for v in rep["shapes"].values())
  print(f"CAPTURED_KERNELS: {total}")
  # A collect run that compiled nothing is a failed run, not a passing one.
  return 0 if total > 0 else 1


if __name__ == "__main__":
  sys.exit(main())
