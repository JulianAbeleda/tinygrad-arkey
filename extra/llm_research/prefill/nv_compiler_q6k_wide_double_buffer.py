"""Bounded two-stage software pipeline for compiler-emitted wide Q6_K CUDA.

The source kernel publishes one K64 phase into a 20,480-byte shared window,
then consumes it.  Its first barrier protects that single window from the
previous phase and its second barrier publishes the current phase.  This
transform gives adjacent phases disjoint ping/pong windows.  The steady state
can therefore publish phase N+1 while phase N is consumed, followed by one
barrier that both publishes N+1 and releases N-1.
"""
from __future__ import annotations

import re

_DECL = "  __shared__ __align__(16) signed char buf1[20480];\n"
_LOOP_RE = re.compile(r"(?P<indent>^[ ]*)for \(int Ridx0 = (?P<begin>[^;]+); Ridx0 < (?P<end>[^;]+); Ridx0\+\+\) \{", re.MULTILINE)
_BARRIER = "__syncthreads();"


def _block_end(source: str, open_brace: int) -> int:
  depth = 0
  for pos in range(open_brace, len(source)):
    if source[pos] == "{": depth += 1
    elif source[pos] == "}":
      depth -= 1
      if depth == 0: return pos
  raise ValueError("unterminated wide Q6 phase loop")


def bound_compiler_q6k_wide_k256(source: str) -> str:
  """Restrict the compiler kernel to one K256 epoch with the promoted unroll."""
  match = _LOOP_RE.search(source)
  if match is None: raise ValueError("compiler wide Q6 phase loop not found")
  if match.group("begin").strip() != "0" or match.group("end").strip() != "192":
    raise ValueError("one-K256 gate requires the direct 0..192 compiler loop")
  loop = f'{match.group("indent")}#pragma unroll 2\n{match.group("indent")}for (int Ridx0 = 0; Ridx0 < 4; Ridx0++) {{'
  return source[:match.start()] + loop + source[match.end():]


def transform_compiler_q6k_wide_double_buffer(source: str) -> str:
  """Pipeline the existing K64 producer/consumer through two shared buffers."""
  if source.count(_DECL) != 1: raise ValueError("compiler wide Q6 shared declaration not found")
  match = _LOOP_RE.search(source)
  if match is None: raise ValueError("compiler wide Q6 phase loop not found")
  open_brace = source.find("{", match.start(), match.end())
  close_brace = _block_end(source, open_brace)
  body = source[open_brace + 1:close_brace]
  barriers = [m for m in re.finditer(re.escape(_BARRIER), body)]
  if len(barriers) != 2: raise ValueError(f"wide Q6 phase body must have two barriers, found {len(barriers)}")

  # Loads before barrier zero are already register prefetches.  With disjoint
  # windows they join decode/publication between the two old barriers.
  producer = body[:barriers[0].start()] + body[barriers[0].end():barriers[1].start()]
  consumer = body[barriers[1].end():]
  mma_call = "__WMMA_8_16_32_signed_char_int("
  if mma_call in producer or mma_call not in consumer:
    raise ValueError("wide Q6 producer/consumer split lost the IMMA boundary")
  if producer.count("*(buf1+") != 80:
    raise ValueError(f"wide Q6 producer must publish 80 values, found {producer.count('*(buf1+')}")

  producer = re.sub(r"\bbuf1\b", "q6_stage", producer)
  consumer = re.sub(r"\bbuf1\b", "q6_stage", consumer)
  indent = match.group("indent")
  begin, end = match.group("begin").strip(), match.group("end").strip()
  pipeline = f"""{indent}for (int RidxP = {begin}; RidxP <= {end}; RidxP++) {{
{indent}  if (RidxP < {end}) {{
{indent}    int Ridx0 = RidxP;
{indent}    signed char *q6_stage = buf1 + ((Ridx0&1)*20480);{producer}
{indent}  }}
{indent}  if (RidxP > {begin}) {{
{indent}    int Ridx0 = RidxP-1;
{indent}    signed char *q6_stage = buf1 + ((Ridx0&1)*20480);{consumer}
{indent}  }}
{indent}  __syncthreads();
{indent}}}"""
  transformed = source[:match.start()] + pipeline + source[close_brace + 1:]
  transformed = transformed.replace(_DECL, "  __shared__ __align__(16) signed char buf1[40960];\n", 1)
  if transformed.count("signed char *q6_stage") != 2 or transformed.count(_BARRIER) != source.count(_BARRIER)-1:
    raise ValueError("wide Q6 double-buffer topology did not close")
  return transformed


__all__ = ["bound_compiler_q6k_wide_k256", "transform_compiler_q6k_wide_double_buffer"]
