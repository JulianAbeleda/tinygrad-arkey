#!/usr/bin/env python3
"""Static probe: what fraction of the Metal fused-dequant prefill GEMM loop body is which kind of
work, at the MSL-source and (when the Metal Toolchain is installed) AIR-IR levels?

Metal analogue of scratchpad/kv_tile_amortization_probe.py. Same shape: compile the production
emitter, locate the loop body, classify every statement against an INST_CLASS regex table, emit a
group/count/share table. This probe runs NO GPU workload -- rendering and (when available)
`xcrun metal -c` frontend compilation are compile-only operations; nothing is dispatched to a
device queue.

Docs: docs/task_workflow/input/metal-prefill-loop-body-decomposition-scope-20260730.md (MP0).

## Production-kernel reconstruction, not a weaker fallback proxy

The three target kernels (`r_16_256_8_16_4_3_16_4_2_8_4`, `r_16_64_8_16_4_4_48_2_2_2_16_2`,
`r_16_64_8_16_4_4_16_4_2_16_2` -- depth-512 prefill profile, scope section "The target kernels")
are FFN/attention-projection GEMMs. Tracing `tinygrad/llm/prefill_routes.py::route_prefill_linear`
and `tinygrad/llm/model.py` shows the exact production compute graph for Metal's dense fp16
fallback (the only reachable route here -- `FULL_RESIDENT_OVERLAY` is memory-infeasible on Metal
per `metal-prefill-schedule-search-scope-20260730.md` 2.3, so `lin._pf16_w` is always None, so
`_build_prefill_v2_warmstart` produces an EMPTY dict, so `_WARMSTART_OPTS`/`_prefill_v2_opts` never
fires on Metal -- confirmed by reading `tinygrad/codegen/opt/postrange.py::apply_opts` and
`tinygrad/llm/model.py:926-936`) is exactly:

    w = lin.weight.cast(dtypes.float16)             # lazy GGUF dequant -> fp16, still lazy
    out = x.cast(dtypes.float16).linear(w.transpose(), bias)

and the weight is a lazy `ggml_data_to_tensor(raw_bytes, n, ggml_type)` graph (Q4_K=12 for
gate/up/qkv, Q6_K=14 for down -- llama.cpp's Q4_K_M mix keeps `ffn_down`/`attn_v` at Q6_K). This
probe reconstructs that graph directly with empty/uninitialized backing tensors (no checkpoint file
needed, no weight VALUES needed -- only shapes/dtypes affect generated code) and renders it through
the exact same `to_program` + default-heuristic (`hand_coded_optimizations`, since no warmstart
opts apply) pipeline production uses. **The rendered kernel name's shape/opt-encoding prefix matches
the profiled production kernel name exactly, digit for digit, for all three kernels** (verified
below at import/run time, not asserted) -- this is strong evidence of AST identity (name encodes
shape + full applied-opts sequence), not mere prefix resemblance. What was NOT done: an end-to-end
call through `model.py`'s `Transformer.__call__`/`route_prefill_linear` with a real loaded
checkpoint and a real `PrefillRouteAttachment`. That is why this is reported as "reconstructed
production kernel, name-prefix-verified" rather than "traced production call".
"""
from __future__ import annotations
import json, re, shutil, subprocess, sys, tempfile
from collections import Counter
from pathlib import Path
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.cstyle import MetalRenderer
from tinygrad.uop.ops import Ops
from tinygrad.llm.gguf import ggml_data_to_tensor, _GGML_QUANT

PREFILL_UBATCH = 512  # tinygrad/llm/model.py PREFILL_UBATCH

# The three target kernels from the depth-512 unbatched prefill profile (scope "The target
# kernels"), and the (role, out_features, in_features, ggml_type) that reproduces each one.
# ggml_type 12 = Q4_K, 14 = Q6_K (llama.cpp's Q4_K_M mix keeps ffn_down/attn_v at Q6_K).
KERNELS = (
  # label,   out_f,  in_f,   ggml_type, profiled_name_prefix,                    share,  gflops
  ("gate_up", 12288,  4096,  12,        "r_16_256_8_16_4_3_16_4_2_8_4",          0.366,  2070),
  ("down",     4096, 12288,  14,        "r_16_64_8_16_4_4_48_2_2_2_16_2",        0.305,   676),
  ("qkv_q",    4096,  4096,  12,        "r_16_64_8_16_4_4_16_4_2_16_2",          0.117,  2183),
)

# ---------------------------------------------------------------------------------------------
# MSL-level classification (statement/line granularity -- MSL has no instruction stream, so the
# unit here is one rendered C statement, mirroring one rendered line from MetalRenderer's cstyle
# output). Order matters: earlier patterns take priority (e.g. a ternary-masked load is "select/
# compare (mask)", not "global load" -- same precedence choice AMD's v_cndmask-gated loads got).
MSL_INST_CLASS = (
  ("barrier",          re.compile(r"threadgroup_barrier")),
  ("simdgroup_matrix",  re.compile(r"simdgroup_(matrix|multiply_accumulate|load|store)\b|__WMMA_")),
  ("threadgroup_ldst",  re.compile(r"\bthreadgroup\b")),
  ("shuffle",           re.compile(r"\bsimd_(shuffle|broadcast|ballot|and|or|xor)\b|\bquad_(shuffle|broadcast)\b")),
  ("transcendental",    re.compile(r"\b(exp2|exp|log2|log|sqrt|rsqrt|sin|cos|tan|tanh|pow)\s*\(")),
  ("select_mask",       re.compile(r"\?[^:;]*:|^\s*bool\b")),
  ("global_load",       re.compile(r"data\d+_\d+")),
  ("index_math",        re.compile(r"^\s*for\s*\(|^\s*(int|uint)\d*\s+\w+\s*=[^=]")),
  ("other_arith",       re.compile(r"=")),
)
USEFUL_MSL = {"simdgroup_matrix"}


def classify_msl(line: str) -> str:
  for name, pat in MSL_INST_CLASS:
    if pat.search(line):
      return name
  return "unclassified"


def find_msl_loop_body(src: str) -> tuple[int, int, list[str]]:
  """The reduce (K) loop is the only real `for` construct MetalRenderer emits for these GEMMs --
  everything else is straight-line prologue (thread-index setup, accumulator zero-init) or
  epilogue (final store). Brace-match from the first `for (int Ridx` to its closing brace,
  mirroring the AMD probe's backward-branch loop detection but at C-statement granularity."""
  lines = src.splitlines()
  start = next(i for i, l in enumerate(lines) if re.match(r"\s*for \(int Ridx", l))
  depth, end = 0, None
  for i in range(start, len(lines)):
    depth += lines[i].count("{") - lines[i].count("}")
    if i > start and depth == 0:
      end = i
      break
  if end is None:
    raise RuntimeError("unbalanced braces locating the MSL reduce loop")
  return start, end, lines[start:end + 1]


def classify_msl_body(body: list[str]) -> Counter:
  counts = Counter()
  for l in body:
    s = l.strip()
    if not s or s in ("{", "}"):
      continue  # structural brace-only lines carry no operation
    counts[classify_msl(l)] += 1
  return counts


# ---------------------------------------------------------------------------------------------
# AIR-level classification (LLVM-IR instruction granularity, via `xcrun metal -c` + `xcrun
# metal-objdump --disassemble`). Scoped to the WHOLE kernel function, not loop-body-only: LLVM
# preserves all four nested loop levels of this kernel as real backward branches (verified by
# inspecting `br i1 ..., label %N, label %M, !llvm.loop` back-edges), so an ISA-style single-span
# "loop body" is not a clean cut without deeper per-loop-level CFG analysis; that is out of this
# probe's compile-only budget. This is stated explicitly wherever the AIR table is reported, and
# is why the AIR and MSL tables are NOT expected to reconcile 1:1 in instruction count (only in
# which groups are populated at all).
AIR_INST_CLASS = (
  ("barrier",          re.compile(r"air\.wg\.barrier|air\.barrier|threadgroup_barrier")),
  ("simdgroup_matrix",  re.compile(r"air\.simdgroup_matrix|simdgroup_multiply_accumulate")),
  ("threadgroup_ldst",  re.compile(r"addrspace\(3\)")),
  ("shuffle",           re.compile(r"air\.simd_shuffle|air\.simd_broadcast|air\.simd_ballot|air\.quad_shuffle")),
  ("transcendental",    re.compile(r"air\.(fast_)?(exp|log|sqrt|rsqrt|sin|cos|tan|pow)\b")),
  ("select_mask",       re.compile(r"^\s*%\d+\s*=\s*(select|icmp)\b|br i1\b")),
  ("global_load",       re.compile(r"^\s*%\d+\s*=\s*load\b|^\s*store\b")),
  ("index_math",        re.compile(r"^\s*%\d+\s*=\s*(phi|getelementptr|extractelement|sext|zext|trunc)\b|br label\b|^\s*ret\b")),
  ("other_arith",       re.compile(r"^\s*%\d+\s*=\s*(fadd|fsub|fmul|fdiv|add|sub|mul|udiv|sdiv|shl|lshr|ashr|and|or|xor"
                                    r"|fptrunc|fpext|fptoui|fptosi|uitofp|sitofp|bitcast|insertelement|shufflevector"
                                    r"|(tail\s+)?call)\b")),
)
USEFUL_AIR = {"simdgroup_matrix"}


def classify_air(line: str) -> str:
  for name, pat in AIR_INST_CLASS:
    if pat.search(line):
      return name
  return "unclassified"


def classify_air_body(dis_text: str) -> Counter:
  lines = dis_text.splitlines()
  start = next(i for i, l in enumerate(lines) if l.startswith("define void @"))
  end = next(i for i in range(start, len(lines)) if lines[i] == "}")
  counts = Counter()
  for l in lines[start + 1:end]:
    s = l.strip()
    if not s or s.startswith(";") or re.match(r"^\d+:", s):
      continue  # comments and basic-block labels carry no operation
    counts[classify_air(l)] += 1
  return counts


def metal_toolchain_available() -> bool:
  return shutil.which("xcrun") is not None and subprocess.run(
    ["xcrun", "--find", "metal-objdump"], capture_output=True).returncode == 0


def render_kernel(label: str, out_f: int, in_f: int, ggml_type: int, m: int = PREFILL_UBATCH) -> tuple[str, str]:
  """Reconstruct the production Q4_K/Q6_K lazy-dequant -> fp16 -> matmul graph and render its MSL,
  exactly as `route_prefill_linear`'s dense-fallback tail does on Metal. No checkpoint, no realize,
  no execution -- shapes/dtypes alone determine the generated code."""
  n = out_f * in_f
  nblk, nbytes_per_blk = _GGML_QUANT[ggml_type]
  raw = Tensor.empty((n // nblk) * nbytes_per_blk, dtype=dtypes.uint8, device="METAL")
  w = ggml_data_to_tensor(raw, n, ggml_type).reshape(out_f, in_f).cast(dtypes.float16)
  x = Tensor.empty(1, m, in_f, dtype=dtypes.float16, device="METAL")
  out = x.cast(dtypes.float16).linear(w.transpose())
  calls = [c for c in out.schedule_linear().src if c.op is Ops.CALL]
  if len(calls) != 1:
    raise RuntimeError(f"{label}: expected exactly one CALL in the schedule, got {len(calls)}")
  ren = MetalRenderer(Target.parse("METAL:METAL:Apple9"))
  prog = to_program(calls[0].src[0], ren)
  src = next(u.arg for u in prog.src if u.op is Ops.SOURCE)
  name = re.search(r"kernel void (\w+)\(", src).group(1)
  return name, src


def compile_to_air_disasm(src: str, workdir: Path) -> str | None:
  """Compile-only: `xcrun metal -c` runs the Metal *frontend* compiler to an .air object; this
  never touches a GPU command queue. Returns None (no exception) if the toolchain is absent, per
  scope 3/MP0b -- MP0 must not block on it."""
  if not metal_toolchain_available():
    return None
  metal_path, air_path = workdir / "kernel.metal", workdir / "kernel.air"
  metal_path.write_text(src)
  subprocess.run(["xcrun", "metal", "-c", str(metal_path), "-o", str(air_path)], check=True, capture_output=True)
  dis = subprocess.run(["xcrun", "metal-objdump", "--disassemble", str(air_path)],
                        check=True, capture_output=True, text=True)
  return dis.stdout


def probe(label: str, out_f: int, in_f: int, ggml_type: int, profiled_prefix: str, share: float, gflops: float) -> dict:
  name, src = render_kernel(label, out_f, in_f, ggml_type)
  prefix_matches = name.startswith(profiled_prefix + "_") or name == profiled_prefix
  start, end, body = find_msl_loop_body(src)
  msl_counts = classify_msl_body(body)
  msl_total = sum(msl_counts.values())
  msl_useful = sum(v for k, v in msl_counts.items() if k in USEFUL_MSL)

  result = {
    "kernel_role": label, "out_features": out_f, "in_features": in_f, "ggml_type": ggml_type,
    "rendered_kernel_name": name, "profiled_name_prefix": profiled_prefix,
    "name_prefix_match": prefix_matches, "profiled_share": share, "profiled_gflops": gflops,
    "msl_loop_body_lines": f"{start}-{end}", "msl_total_statements": msl_total,
    "msl_useful_simdgroup_matrix": msl_useful,
    "msl_by_class": dict(msl_counts.most_common()),
    "msl_unclassified": msl_counts.get("unclassified", 0),
    "msl_coverage_fraction": round(1 - msl_counts.get("unclassified", 0) / msl_total, 4) if msl_total else None,
    "air": None,
  }

  with tempfile.TemporaryDirectory() as td:
    dis = compile_to_air_disasm(src, Path(td))
  if dis is not None:
    air_counts = classify_air_body(dis)
    air_total = sum(air_counts.values())
    air_useful = sum(v for k, v in air_counts.items() if k in USEFUL_AIR)
    result["air"] = {
      "scope": "whole-kernel (not loop-body-only; see module docstring)",
      "total_instructions": air_total, "useful_simdgroup_matrix": air_useful,
      "by_class": dict(air_counts.most_common()),
      "unclassified": air_counts.get("unclassified", 0),
      "coverage_fraction": round(1 - air_counts.get("unclassified", 0) / air_total, 4) if air_total else None,
    }
  return result


if __name__ == "__main__":
  results = [probe(*k) for k in KERNELS]
  out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
  print(json.dumps(results, indent=2))
  if out:
    out.write_text(json.dumps(results, indent=2) + "\n")
