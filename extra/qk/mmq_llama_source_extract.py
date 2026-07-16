"""Fail-closed extraction of the llama.cpp gfx1100 Q4_K x Q8_1 MMQ contract.

This module deliberately knows C++ spellings, not tinygrad descriptor values.  A
result is returned only when all mutually-dependent declarations and kernel
statements can be found exactly once in authoritative llama.cpp source text.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
import subprocess
from typing import Mapping


PINNED_LLAMA_REVISION = "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
DEFAULT_LLAMA_ROOT = Path("/home/ubuntu/env/llama.cpp")
SOURCE_FILES = (
  "ggml/src/ggml-common.h",
  "ggml/src/ggml-cuda/mmq.cuh",
  "ggml/src/ggml-cuda/mma.cuh",
  "ggml/src/ggml-cuda/quantize.cu",
)


class SourceContractError(ValueError):
  """The source does not uniquely prove the requested MMQ contract."""


@dataclass(frozen=True)
class SourceProof:
  file: str
  anchor: str
  excerpt: str
  line: int
  digest: str


@dataclass(frozen=True)
class GridFormulas:
  tile_rows: str
  tile_columns: str
  batch: str


@dataclass(frozen=True)
class LlamaQ4KQ8MMQProof:
  revision: str
  architecture: str
  tile_rows: int
  tile_columns: str
  k_epoch: int
  waves: int
  wave_size: int
  threads: int
  q8_record_values: int
  q8_record_bytes: int
  q8_ds4_order: tuple[str, ...]
  q4_block_values: int
  q4_row_bytes: int
  lds_order: tuple[str, ...]
  lds_total_formula: str
  lifecycle: tuple[str, ...]
  barriers_per_epoch: int
  wmma_builtin: str
  wmma_calls: int
  wmma_signed_controls: tuple[tuple[bool, bool, bool], ...]
  correction_expression: str
  grid: GridFormulas
  proofs: tuple[SourceProof, ...]

  def proof_for(self, anchor: str) -> SourceProof:
    matches = [proof for proof in self.proofs if proof.anchor == anchor]
    if len(matches) != 1:
      raise SourceContractError(f"proof anchor {anchor!r} is not unique")
    return matches[0]


def _one(pattern: str, text: str, what: str, flags: int = re.MULTILINE | re.DOTALL) -> re.Match[str]:
  found = list(re.finditer(pattern, text, flags))
  if len(found) != 1:
    raise SourceContractError(f"expected exactly one {what}, found {len(found)}")
  return found[0]


def _body(text: str, signature: str, what: str) -> tuple[str, int]:
  match = _one(signature, text, what, re.MULTILINE)
  start = text.find("{", match.end())
  if start < 0:
    raise SourceContractError(f"missing body for {what}")
  depth = 0
  for pos in range(start, len(text)):
    if text[pos] == "{": depth += 1
    elif text[pos] == "}":
      depth -= 1
      if depth == 0: return text[start + 1:pos], start + 1
  raise SourceContractError(f"unterminated body for {what}")


def _proof(file: str, anchor: str, text: str, start: int, end: int) -> SourceProof:
  excerpt = text[start:end].strip()
  return SourceProof(file, anchor, excerpt, text.count("\n", 0, start) + 1, sha256(excerpt.encode()).hexdigest())


def _require(snippet: str, text: str, file: str, anchor: str, proofs: list[SourceProof]) -> None:
  matches = [m for m in re.finditer(re.escape(snippet), text)]
  if len(matches) != 1:
    raise SourceContractError(f"{file}: expected exactly one {anchor} anchor, found {len(matches)}")
  proofs.append(_proof(file, anchor, text, matches[0].start(), matches[0].end()))


def extract_from_sources(sources: Mapping[str, str], *, revision: str = PINNED_LLAMA_REVISION) -> LlamaQ4KQ8MMQProof:
  missing = set(SOURCE_FILES) - set(sources)
  if missing: raise SourceContractError(f"missing authoritative sources: {sorted(missing)}")
  common, mmq, mma, quant = (sources[name] for name in SOURCE_FILES)
  proofs: list[SourceProof] = []

  qkk = int(_one(r"^#define\s+QK_K\s+(\d+)\s*$", common, "QK_K").group(1))
  qk8 = int(_one(r"^#define\s+QK8_1\s+(\d+)\s*$", common, "QK8_1").group(1))
  kscale = int(_one(r"^#define\s+K_SCALE_SIZE\s+(\d+)\s*$", common, "K_SCALE_SIZE").group(1))
  q4_match = _one(r"typedef\s+struct\s*\{(?:(?!typedef\s+struct).)*?ggml_half2\s+dm;.*?uint8_t\s+scales\[K_SCALE_SIZE\];.*?uint8_t\s+qs\[QK_K/2\];.*?\}\s*block_q4_K;", common, "block_q4_K")
  q4_assert = _one(r"static_assert\(sizeof\(block_q4_K\)\s*==\s*2\*sizeof\(ggml_half\)\s*\+\s*K_SCALE_SIZE\s*\+\s*QK_K/2", common, "block_q4_K extent")
  proofs.append(_proof(SOURCE_FILES[0], "q4_row_extent", common, q4_match.start(), q4_match.end()))
  proofs.append(_proof(SOURCE_FILES[0], "q4_extent_assert", common, q4_assert.start(), q4_assert.end()))

  iter_k = int(_one(r"^#define\s+MMQ_ITER_K\s+(\d+)\s*$", mmq, "MMQ_ITER_K").group(1))
  nwaves = int(_one(r"^#define\s+MMQ_NWARPS\s+(\d+)\s*$", mmq, "MMQ_NWARPS").group(1))
  _require("return 128;\n#endif // defined RDNA1", mmq, SOURCE_FILES[1], "gfx1100_tile_rows", proofs)
  _require("case GGML_TYPE_Q4_K:\n        case GGML_TYPE_Q5_K:\n            return MMQ_Q8_1_DS_LAYOUT_DS4;", mmq, SOURCE_FILES[1], "q4_ds4_selection", proofs)
  _require("half2 ds4[4];   // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored as d0,s0,d1,s1,d2,s2,d3,s3", mmq, SOURCE_FILES[1], "ds4_order", proofs)
  record_assert = "static_assert(sizeof(block_q8_1_mmq) == 4*QK8_1 + 4*sizeof(half2), \"Unexpected block_q8_1_mmq size\");"
  _require(record_assert, mmq, SOURCE_FILES[1], "q8_record_extent", proofs)

  process, process_pos = _body(mmq, r"static\s+__device__\s+__forceinline__\s+void\s+mul_mat_q_process_tile\s*\(", "mul_mat_q_process_tile")
  lifecycle_pattern = re.compile(r"load_tiles\(x, tile_x,.*?tile_y\[l\]\s*=\s*by0\[l\];.*?(__syncthreads\(\);).*?vec_dot\(tile_x, tile_y, sum, 0\);.*?(__syncthreads\(\);).*?tile_y\[l\]\s*=\s*by0\[l\];.*?(__syncthreads\(\);).*?vec_dot\(tile_x, tile_y, sum, MMQ_TILE_NE_K\);.*?(__syncthreads\(\);)", re.DOTALL)
  life = _one(lifecycle_pattern.pattern, process, "two-phase lifecycle")
  if process.count("__syncthreads();") != 4: raise SourceContractError("tile lifecycle must contain exactly four barriers")
  proofs.append(_proof(SOURCE_FILES[1], "two_phase_lifecycle", mmq, process_pos + life.start(), process_pos + life.end()))

  _require("int * tile_y = data_mul_mat_q + mmq_x;\n    int * tile_x = tile_y + GGML_PAD(mmq_x*MMQ_TILE_Y_K, nwarps*warp_size);", mmq, SOURCE_FILES[1], "lds_order", proofs)
  lds_formula = "nbs_ids + nbs_x + GGML_PAD(nbs_y, nwarps*warp_size*sizeof(int))"
  _require("return " + lds_formula + ";", mmq, SOURCE_FILES[1], "lds_total", proofs)
  _require("const dim3 block_dims(warp_size, nwarps, 1);", mmq, SOURCE_FILES[1], "launch_shape", proofs)
  grid_rows = "(args.nrows_x   + mmq_y - 1) / mmq_y"
  grid_cols = "(args.ncols_max + mmq_x - 1) / mmq_x"
  grid_batch = "args.nchannels_y * args.nsamples_y"
  for anchor, declaration in (("grid_rows", "const int nty  = " + grid_rows + ";"),
                              ("grid_columns", "const int ntx  = " + grid_cols + ";"),
                              ("grid_batch", "const int ntzw = " + grid_batch + ";")):
    _require(declaration, mmq, SOURCE_FILES[1], anchor, proofs)

  builtin = "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32"
  rdna_match = _one(r"#elif defined\(RDNA3\)\n(?P<body>\s*using int32x4_t.*?" + re.escape(builtin) + r"\(true, a_vec\[0\], true, b_vec\[0\], acc\[0\], true\);.*?" + re.escape(builtin) + r"\(true, a_vec\[1\], true, b_vec\[1\], acc\[0\], true\);)\n#endif // RDNA4", mma, "RDNA3 signed WMMA branch")
  rdna3, rdna3_pos = rdna_match.group("body"), rdna_match.start("body")
  calls = list(re.finditer(re.escape(builtin) + r"\(true,\s*a_vec\[(\d)\],\s*true,\s*b_vec\[\1\],\s*acc\[0\],\s*true\)", rdna3))
  if [m.group(1) for m in calls] != ["0", "1"]: raise SourceContractError("RDNA3 must issue exactly indexed WMMA calls 0 and 1 with explicit signed controls")
  if rdna3.count(builtin) != 2: raise SourceContractError("RDNA3 branch must contain exactly two integer WMMA calls")
  proofs.append(_proof(SOURCE_FILES[2], "rdna3_signed_wmma", mma, rdna3_pos, rdna3_pos + len(rdna3)))

  correction = "dmA.x*dsB.x*C.x[l] + dmA.y*dsB.y"
  _require("sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.x*dsB.x*C.x[l];\n                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.y*dsB.y;", mmq, SOURCE_FILES[1], "q4_correction", proofs)
  _require("y[ib].ds4[iqs/32] = make_half2(d, sum);", quant, SOURCE_FILES[3], "ds4_producer", proofs)

  if (qkk, qk8, iter_k, nwaves) != (256, 32, 256, 8):
    raise SourceContractError(f"source constants do not describe the pinned gfx1100 contract: {(qkk, qk8, iter_k, nwaves)}")
  return LlamaQ4KQ8MMQProof(
    revision, "gfx1100/RDNA3", 128, "mmq_x (runtime selected, granularity 16 or 32)", iter_k,
    nwaves, 32, nwaves * 32, 4 * qk8, 4 * qk8 + 4 * 4,
    ("d0", "s0", "d1", "s1", "d2", "s2", "d3", "s3"), qkk, 2 * 2 + kscale + qkk // 2,
    ("ids", "q8 tile_y", "q4 tile_x"), lds_formula,
    ("load_q4_and_q8_half0", "barrier", "wmma_half0", "barrier", "load_q8_half1", "barrier", "wmma_half1", "barrier"),
    4, builtin, 2, ((True, True, True), (True, True, True)), correction,
    GridFormulas(grid_rows, grid_cols, grid_batch), tuple(proofs))


def extract_llama_checkout(root: Path | str = DEFAULT_LLAMA_ROOT) -> LlamaQ4KQ8MMQProof:
  root = Path(root).resolve()
  try:
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    raise SourceContractError(f"cannot establish llama.cpp revision at {root}") from exc
  if revision != PINNED_LLAMA_REVISION:
    raise SourceContractError(f"wrong llama.cpp revision: expected {PINNED_LLAMA_REVISION}, got {revision}")
  try: sources = {name: (root / name).read_text() for name in SOURCE_FILES}
  except (OSError, UnicodeError) as exc: raise SourceContractError(f"cannot read authoritative llama.cpp sources at {root}") from exc
  return extract_from_sources(sources, revision=revision)
