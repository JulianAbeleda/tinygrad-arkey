#!/usr/bin/env python3
"""LR-000: compile-only lowering fingerprint baseline for the lowering-architecture refactor.

Scope: docs/task_workflow/input/lowering-architecture-refactor-scope-20260726.md section 6, Phase 0 (LR-000).

Purpose: freeze a byte-comparable fingerprint of the default-path kernel builders (source hash + resource
facts + shape + launch geometry) *before* any refactor-only lowering change, so a later refactor-only run can
be proven byte-identical:

  - a changed lowering pass shows up as a changed `source_sha256` (and usually `resources`) for the SAME
    kernel_id/shape
  - a changed route selection shows up as a missing/added `kernel_id`
  - a changed model shape shows up as a changed `shape` block for the SAME kernel_id family

This script is compile-only: it never loads a model and never executes a GPU kernel. It builds representative
kernel graphs directly from the existing default-path route/spec builders with fixed, real Qwen3-8B/14B shapes
(drawn from extra/qk/route_manifest.json shape_guards) and calls `tinygrad.codegen.to_program(sink,
Device["AMD"].renderer)` -- the same compile-only entry point `extra/qk/mmq_compile_evidence.py` uses
(`do_to_program` runs under `Context(ALLOW_DEVICE_USAGE=0)`, so this never touches a device even when one is
present). If AMD is unavailable this script fails loudly (RuntimeError), never silently substitutes another
renderer: a fingerprint captured against the wrong target is worse than no fingerprint.

Resource facts (vgpr/sgpr/lds_bytes/scratch_bytes/...) are parsed from the compiled code object with
`extra/qk/mmq_compile_evidence.py:parse_amdgpu_metadata` (the established AMDGPU-metadata reader) and then
typed/validated through `extra/qk/amd_resource_artifact.py:AMDResourceFacts` -- this script does not define a
second resource schema. Source/binary hashing reuses that same module's `artifact_sha256` (sha256 hex of the exact bytes
passed to the compiler) instead of a second hashing scheme.

NOTE on scope narrowing vs `join_amd_resource_artifact`/`AMDResourceArtifact`: that join additionally requires
a per-kernel physical-register role interval map (`role_evidence`, `post_regalloc=True`), which only exists for
kernels that have had their final ISA manually annotated with logical register ownership (e.g. the MMQ atom's
epoch/exec-mask analysis). That annotation does not exist -- and is not generic -- for the seven builder
modules this baseline covers. This baseline therefore reuses `AMDResourceFacts` (the typed, validated resource
record) and the module's hashing convention, but does not construct the full `AMDResourceArtifact` join. This
is a deliberate, documented scope decision, not an oversight.

Covered default-path kernel builders (see COVERAGE below for the exact call sites):
  - extra/qk/prefill/q4k_prefill_route_spec.py   (direct-packed Q4_K prefill)
  - extra/qk/prefill/q6k_prefill_route_spec.py   (direct-packed Q6_K prefill)
  - extra/qk/prefill/flash_prefill_attention_spec.py  (shipped fused prefill attention)
  - extra/qk/decode/flash_decode_attention_spec.py    (shipped live-split decode attention: tile + combine)
  - extra/qk/quant/q4_k_gemv_primitive.py        (Q4_K GEMV/GEMM primitive, direct_out b=1)
  - extra/qk/quant/q6_k_gemv_primitive.py        (Q6_K GEMV/GEMM primitive, direct_out b=1)
  - extra/qk/gemv_g3_codegen_lowering.py         (G3 lanemap-generated Q4_K decode GEMV -- the promoted default)

Run:
  PYTHONPATH=. python3 extra/audit/lowering_baseline.py           # writes bench/lowering-refactor-baseline/latest.json
  PYTHONPATH=. python3 extra/audit/lowering_baseline.py --check   # recompile + diff vs stored latest.json; exit 1 on any change
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_compile_evidence import parse_amdgpu_metadata
from extra.qk.amd_resource_artifact import AMDResourceFacts, artifact_sha256  # noqa: E402  (one hashing rule, not a second)
from extra.qk.layout import (Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, Q6_K_BLOCK_ELEMS)

OUT_DIR = ROOT / "bench" / "lowering-refactor-baseline"
OUT_PATH = OUT_DIR / "latest.json"
SCHEMA = "tinygrad.lowering_refactor_baseline.v1"
TARGET_DEVICE = "AMD"

# Qwen3-8B / Qwen3-14B role shapes, taken verbatim from extra/qk/route_manifest.json shape_guards (decode_q4k_g3_generated,
# decode_q6k_coop_generated, prefill_flash_attention_generated, decode_flash_live_split_g4_kvboth,
# decode_flash_live_split_g5_kvboth, decode_flash_block_tile_g5_konly, packed_wmma_prefill_generated). M=512 is the
# canonical prefill token block used across every prefill shape_guard entry. MAXC/S/Tc for decode match the canonical
# decode authority defaults in extra/qk/decode/decode_harness.py (DEFAULT_MAX_CONTEXT=4608, ckpt 4096) and
# extra/qk/decode/decode_tile_timing.py (SPLITS=48).
M_PREFILL = 512
MAXC_DECODE = 4608
S_DECODE = 48
TC_DECODE = 4096

ROLE_SHAPES = {
  "8B": {"ffn_gate_up": {"K": 4096, "N": 12288}, "ffn_down": {"K": 12288, "N": 4096}, "attn_qo": {"K": 4096, "N": 4096}},
  "14B": {"ffn_gate_up": {"K": 5120, "N": 17408}, "ffn_down": {"K": 17408, "N": 5120}, "attn_qo": {"K": 5120, "N": 5120}},
}
ATTN_SHAPES = {"8B": {"Hq": 32, "Hkv": 8, "Hd": 128}, "14B": {"Hq": 40, "Hkv": 8, "Hd": 128}}


def _amd_renderer():
  """Return the AMD renderer or fail loudly. Never falls back to another device/renderer."""
  try:
    dev = Device[TARGET_DEVICE]
  except Exception as exc:
    raise RuntimeError(f"AMD device is unavailable ({exc!r}); refusing to fingerprint against a different renderer") from exc
  renderer = getattr(dev, "renderer", None)
  if renderer is None or type(renderer).__name__ != "HIPRenderer":
    raise RuntimeError(f"Device[{TARGET_DEVICE!r}].renderer is not the expected HIP renderer (got {type(renderer)!r})")
  return renderer


@dataclass(frozen=True)
class KernelCase:
  kernel_id: str
  route_id: str
  model: str
  shape: dict[str, Any]
  make_sink: Callable[[], UOp]


def _facts_from_program(prog: UOp) -> dict[str, Any]:
  """Compile-only fingerprint of one PROGRAM UOp: name, launch geometry, source/binary hash, resources.

  Per the LR-000 spec: if the program does not carry a compiled BINARY, resources/binary fields are null
  rather than compiling to a binary through some other path.
  """
  if prog.op is not Ops.PROGRAM: raise RuntimeError(f"expected Ops.PROGRAM, got {prog.op}")
  source = next((s.arg for s in prog.src if s.op is Ops.SOURCE), None)
  binary = next((s.arg for s in prog.src if s.op is Ops.BINARY), None)
  if source is None: raise RuntimeError("PROGRAM has no SOURCE; compile-only lowering did not reach rendering")
  global_size = tuple(prog.arg.global_size)
  local_size = tuple(prog.arg.local_size) if prog.arg.local_size is not None else None
  result: dict[str, Any] = {
    "program_name": prog.arg.function_name,
    "source_sha256": artifact_sha256(source, "source"),
    "grid": list(global_size),
    "workgroup": list(local_size) if local_size is not None else None,
    "binary_sha256": None,
    "resources": None,
  }
  if binary is not None:
    result["binary_sha256"] = artifact_sha256(binary, "binary")
    meta = parse_amdgpu_metadata(binary)
    facts = AMDResourceFacts(vgpr=meta["vgpr"], sgpr=meta["sgpr"], lds_bytes=meta["lds_bytes"],
                             scratch_bytes=meta["scratch_bytes"], vgpr_spills=meta["vgpr_spills"],
                             sgpr_spills=meta["sgpr_spills"], workgroup_threads=meta["max_workgroup_threads"],
                             wavefront_size=meta["wavefront_size"])
    result["resources"] = facts.to_json()
  return result


def compile_case(case: KernelCase, renderer) -> dict[str, Any]:
  entry: dict[str, Any] = {"kernel_id": case.kernel_id, "route_id": case.route_id, "model": case.model,
                           "shape": case.shape, "skipped_reason": None}
  try:
    sink = case.make_sink()
  except Exception as exc:
    entry["skipped_reason"] = f"builder could not construct a sink without a loaded model/runtime: {exc!r}"
    return entry
  prog = to_program(sink, renderer)
  entry.update(_facts_from_program(prog))
  return entry


# --------------------------------------------------------------------------------------------------------------
# Kernel case builders
# --------------------------------------------------------------------------------------------------------------

def _q4k_prefill_cases() -> list[KernelCase]:
  from extra.qk.prefill.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
  cases = []
  for model, roles in ROLE_SHAPES.items():
    for role, dims in roles.items():
      rows, k, tokens = dims["N"], dims["K"], M_PREFILL
      shape = {"rows": rows, "k": k, "tokens": tokens, "role": role}
      def make_sink(rows=rows, k=k, tokens=tokens, role=role):
        spec = describe_q4k_packed_prefill(rows=rows, k=k, tokens=tokens, role=role)
        fn = emit_q4k_packed_prefill_kernel(spec)
        k_blocks = spec.k_blocks
        out = UOp.placeholder((tokens, rows), dtypes.float32, 0)
        words = UOp.placeholder((rows * k_blocks * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
        x = UOp.placeholder((tokens * k,), dtypes.float16, 2)
        return fn(out, words, x)
      cases.append(KernelCase(f"prefill_q4k_direct.{role}.{model}", "prefill_q4k_direct_tile4x4_default",
                              model, shape, make_sink))
  return cases


def _q6k_prefill_cases() -> list[KernelCase]:
  from extra.qk.prefill.q6k_prefill_route_spec import describe_q6k_packed_prefill, emit_q6k_packed_prefill_kernel
  cases = []
  for model, roles in ROLE_SHAPES.items():
    role = "ffn_down"
    dims = roles[role]
    rows, k, tokens = dims["N"], dims["K"], M_PREFILL
    shape = {"rows": rows, "k": k, "tokens": tokens, "role": role}
    def make_sink(rows=rows, k=k, tokens=tokens, role=role):
      spec = describe_q6k_packed_prefill(rows=rows, k=k, tokens=tokens, role=role)
      fn = emit_q6k_packed_prefill_kernel(spec)
      k_blocks = spec.k_blocks
      out = UOp.placeholder((tokens, rows), dtypes.float32, 0)
      halfs = UOp.placeholder((rows * k_blocks * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1)
      x = UOp.placeholder((tokens * k,), dtypes.float16, 2)
      return fn(out, halfs, x)
    cases.append(KernelCase(f"prefill_q6k_direct.{role}.{model}", "prefill_q6k_direct_generated", model, shape, make_sink))
  return cases


def _flash_prefill_cases() -> list[KernelCase]:
  from extra.qk.prefill.flash_prefill_attention_spec import describe_flash_prefill_attention
  cases = []
  for model, dims in ATTN_SHAPES.items():
    Hq, Hkv, Hd = dims["Hq"], dims["Hkv"], dims["Hd"]
    q_tokens = kv_tokens = M_PREFILL
    shape = {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "q_tokens": q_tokens, "kv_tokens": kv_tokens, "causal": True}
    def make_sink(Hq=Hq, Hkv=Hkv, Hd=Hd, q_tokens=q_tokens, kv_tokens=kv_tokens):
      spec = describe_flash_prefill_attention(Hq=Hq, Hkv=Hkv, q_tokens=q_tokens, kv_tokens=kv_tokens,
                                              causal=True, scale=1.0 / (Hd ** 0.5))
      fn = spec.emit()
      out = UOp.placeholder((q_tokens * Hq * Hd,), dtypes.float16, 0)
      q = UOp.placeholder((q_tokens * Hq * Hd,), dtypes.float16, 1)
      k = UOp.placeholder((kv_tokens * Hkv * Hd,), dtypes.float16, 2)
      v = UOp.placeholder((kv_tokens * Hkv * Hd,), dtypes.float16, 3)
      return fn(out, q, k, v)
    cases.append(KernelCase(f"prefill_flash_attention.{model}", "prefill_flash_attention_generated", model, shape, make_sink))
  return cases


def _flash_decode_cases() -> list[KernelCase]:
  from extra.qk.decode.flash_decode_attention_spec import describe_flash_decode_attention
  configs = [
    ("8B", "KV_BOTH", "decode_flash_live_split_g4_kvboth"),
    ("14B", "KV_BOTH", "decode_flash_live_split_g5_kvboth"),
    ("14B", "K_ONLY", "decode_flash_block_tile_g5_konly"),
  ]
  cases = []
  for model, staging, route_id in configs:
    dims = ATTN_SHAPES[model]
    Hq, Hkv, Hd = dims["Hq"], dims["Hkv"], dims["Hd"]
    shape = {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC_DECODE, "S": S_DECODE, "Tc": TC_DECODE, "staging": staging}
    W2 = Hd + 2

    def make_tile_sink(Hq=Hq, Hkv=Hkv, Hd=Hd, staging=staging, W2=W2):
      spec = describe_flash_decode_attention(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC_DECODE, S=S_DECODE, staging=staging)
      Tc_u = UOp.const(dtypes.int32, TC_DECODE)
      fn = spec.emit_tile(Tc_u)
      pout = UOp.placeholder((Hq * S_DECODE * W2,), dtypes.float32, 0)
      q = UOp.placeholder((Hq * Hd,), dtypes.float16, 1)
      cache = UOp.placeholder((2, 1, Hkv, MAXC_DECODE, Hd), dtypes.float16, 2)
      return fn(pout, q, cache)
    cases.append(KernelCase(f"decode_flash_tile.{model}.{staging}", route_id, model, shape, make_tile_sink))

    def make_combine_sink(Hq=Hq, Hd=Hd, Hkv=Hkv, staging=staging, W2=W2):
      spec = describe_flash_decode_attention(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC_DECODE, S=S_DECODE, staging=staging)
      fn = spec.emit_combine()
      out = UOp.placeholder((Hq * Hd,), dtypes.float32, 0)
      po = UOp.placeholder((Hq * S_DECODE * W2,), dtypes.float32, 1)
      return fn(out, po)
    cases.append(KernelCase(f"decode_flash_combine.{model}.{staging}", route_id, model, shape, make_combine_sink))
  return cases


def _q4k_gemv_primitive_cases() -> list[KernelCase]:
  from extra.qk.quant.q4_k_gemv_primitive import q4k_gemm_packed_load_direct_out_kernel
  cases = []
  for model, roles in ROLE_SHAPES.items():
    for role, dims in roles.items():
      rows, k, b = dims["N"], dims["K"], 1
      shape = {"rows": rows, "k": k, "b": b, "role": role}
      def make_sink(rows=rows, k=k, b=b):
        kfn = q4k_gemm_packed_load_direct_out_kernel(rows, k, b, "explicit", ())
        k_blocks = k // Q4_K_BLOCK_ELEMS
        out = UOp.placeholder((b, rows), dtypes.float32, 0)
        words = UOp.placeholder((rows * k_blocks * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
        x = UOp.placeholder((b * k,), dtypes.float16, 2)
        return kfn(out, words, x)
      cases.append(KernelCase(f"decode_q4k_gemv_primitive.{role}.{model}", "q4_k_gemv_primitive_direct_out",
                              model, shape, make_sink))
  return cases


def _q6k_gemv_primitive_cases() -> list[KernelCase]:
  from extra.qk.quant.q6_k_gemv_primitive import q6k_gemm_packed_load_direct_out_kernel
  cases = []
  for model, roles in ROLE_SHAPES.items():
    role = "ffn_down"
    dims = roles[role]
    rows, k, b = dims["N"], dims["K"], 1
    shape = {"rows": rows, "k": k, "b": b, "role": role}
    def make_sink(rows=rows, k=k, b=b):
      kfn = q6k_gemm_packed_load_direct_out_kernel(rows, k, b, ())
      k_blocks = k // Q6_K_BLOCK_ELEMS
      out = UOp.placeholder((b, rows), dtypes.float32, 0)
      halfs = UOp.placeholder((rows * k_blocks * Q6K_HALFWORDS_PER_BLOCK,), dtypes.uint16, 1)
      x = UOp.placeholder((b * k,), dtypes.float16, 2)
      return kfn(out, halfs, x)
    cases.append(KernelCase(f"decode_q6k_gemv_primitive.{role}.{model}", "q6_k_gemv_primitive_direct_out",
                            model, shape, make_sink))
  return cases


def _gemv_g3_lanemap_cases() -> list[KernelCase]:
  from extra.qk.gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
  cases = []
  for model, roles in ROLE_SHAPES.items():
    for role, dims in roles.items():
      rows, k = dims["N"], dims["K"]
      shape = {"rows": rows, "k": k, "role": role}
      def make_sink(rows=rows, k=k):
        kfn = q4k_g3_lanemap_gemv_kernel(rows, k)
        k_blocks = k // Q4_K_BLOCK_ELEMS
        out = UOp.placeholder((rows,), dtypes.float32, 0)
        words = UOp.placeholder((rows * k_blocks * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
        x = UOp.placeholder((k,), dtypes.float16, 2)
        return kfn(out, words, x)
      cases.append(KernelCase(f"decode_q4k_g3_lanemap.{role}.{model}", "decode_q4k_g3_generated", model, shape, make_sink))
  return cases


def build_cases() -> list[KernelCase]:
  cases: list[KernelCase] = []
  cases += _q4k_prefill_cases()
  cases += _q6k_prefill_cases()
  cases += _flash_prefill_cases()
  cases += _flash_decode_cases()
  cases += _q4k_gemv_primitive_cases()
  cases += _q6k_gemv_primitive_cases()
  cases += _gemv_g3_lanemap_cases()
  ids = [c.kernel_id for c in cases]
  if len(ids) != len(set(ids)): raise RuntimeError(f"duplicate kernel_id in baseline case list: {ids}")
  return cases


# --------------------------------------------------------------------------------------------------------------
# Header / artifact assembly
# --------------------------------------------------------------------------------------------------------------

def _git(*args: str) -> str:
  return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def build_header(renderer, argv: list[str]) -> dict[str, Any]:
  try:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
  except Exception as exc:
    raise RuntimeError(f"could not determine git revision/dirty state: {exc!r}") from exc
  return {
    "schema": SCHEMA,
    "git_commit": commit,
    "git_dirty": dirty,
    "python_version": sys.version,
    "renderer": type(renderer).__name__,
    "target": str(renderer.target),
    "device": TARGET_DEVICE,
    "command": "python3 " + " ".join(["extra/audit/lowering_baseline.py", *argv]),
  }


def build_artifact(argv: list[str]) -> dict[str, Any]:
  renderer = _amd_renderer()
  header = build_header(renderer, argv)
  entries = [compile_case(case, renderer) for case in build_cases()]
  entries.sort(key=lambda e: e["kernel_id"])
  return {"header": header, "entries": entries}


# --------------------------------------------------------------------------------------------------------------
# --check: recompile and diff against the stored artifact
# --------------------------------------------------------------------------------------------------------------

_ENTRY_DIFF_FIELDS = ("route_id", "model", "shape", "source_sha256", "binary_sha256", "resources",
                     "workgroup", "grid", "skipped_reason")


def _classify_entry_diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
  changes = []
  if old.get("shape") != new.get("shape"): changes.append("shape")
  if old.get("route_id") != new.get("route_id"): changes.append("route")
  if old.get("source_sha256") != new.get("source_sha256"): changes.append("source_hash")
  if old.get("binary_sha256") != new.get("binary_sha256"): changes.append("binary_hash")
  if old.get("resources") != new.get("resources"): changes.append("resources")
  if (old.get("workgroup"), old.get("grid")) != (new.get("workgroup"), new.get("grid")): changes.append("launch_geometry")
  if old.get("skipped_reason") != new.get("skipped_reason"): changes.append("skipped_reason")
  return changes


def run_check(argv: list[str]) -> int:
  if not OUT_PATH.is_file():
    print(f"FAIL: no stored baseline at {OUT_PATH}; run without --check first")
    return 1
  stored = json.loads(OUT_PATH.read_text())
  fresh = build_artifact(argv)
  stored_by_id = {e["kernel_id"]: e for e in stored["entries"]}
  fresh_by_id = {e["kernel_id"]: e for e in fresh["entries"]}
  all_ids = sorted(set(stored_by_id) | set(fresh_by_id))
  rows = []
  any_diff = False
  for kid in all_ids:
    old, new = stored_by_id.get(kid), fresh_by_id.get(kid)
    if old is None:
      rows.append((kid, "ADDED", "kernel present in fresh compile, absent from stored baseline")); any_diff = True; continue
    if new is None:
      rows.append((kid, "REMOVED", "kernel present in stored baseline, absent from fresh compile")); any_diff = True; continue
    changes = _classify_entry_diff(old, new)
    if changes:
      rows.append((kid, "CHANGED", ",".join(changes))); any_diff = True
  print(f"{'kernel_id':60s} {'status':8s} changed")
  for kid, status, detail in rows:
    print(f"{kid:60s} {status:8s} {detail}")
  if not any_diff:
    print(f"verdict: PASS ({len(all_ids)} kernels, byte-identical)")
    return 0
  print(f"verdict: FAIL ({len(rows)}/{len(all_ids)} kernels differ)")
  return 1


def main() -> int:
  ap = argparse.ArgumentParser(description="Compile-only lowering fingerprint baseline (LR-000).")
  ap.add_argument("--check", action="store_true", help="recompile and diff against the stored latest.json; write nothing")
  args, unknown = ap.parse_known_args()
  argv = sys.argv[1:]
  if args.check:
    return run_check(argv)
  artifact = build_artifact(argv)
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  OUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
  n_ok = sum(1 for e in artifact["entries"] if e["skipped_reason"] is None)
  n_skip = len(artifact["entries"]) - n_ok
  print(f"wrote {OUT_PATH} ({n_ok} compiled, {n_skip} skipped)")
  print(f"verdict: {'PASS' if n_skip == 0 else 'PASS_WITH_SKIPS'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
