#!/usr/bin/env python3
"""Compiler-path gate for the logical Q4_K -> signed-int8 fragment provider.

This gate intentionally qualifies the raw integer contraction only.  It proves
that ordinary Tensor.matmul, PackedWeightTransform ownership, postrange LDS
production, and NVIDIA IMMA preserve every packed nibble.  A complete Q4_K
projection additionally needs the per-K32 scale/min accumulator contract; the
gate records that boundary instead of mislabelling a raw dot as a projection.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, subprocess
from dataclasses import dataclass
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import PackedWeightTransform, Q4KInt8FragmentProvider
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry


M, N, K = 32, 16, 256


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q4KInt8FragmentProvider
  pipeline: None = None


@dataclass(frozen=True)
class _DenseContext:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  pipeline: None = None


def _carrier(words:Tensor, transform:PackedWeightTransform) -> Tensor:
  """Movement-only logical char view; postrange replaces every dummy byte."""
  blocks, halfwords = transform.rows * transform.blocks_per_row, int(transform.block_bytes)//2
  # Reuse the proven packed-half carrier's movement graph so the logical K
  # owner remains one range through pm_split_ranges.  The terminal cast is
  # dummy data only; the typed provider replaces it before LDS production.
  return words.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128-halfwords))) \
    .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(20260828)
  raw = rng.integers(0, 256, (N, 144), dtype=np.uint8)
  # Every row has alternating low/high nibbles across all four payload regions.
  for row in range(N): raw[row, 16:] = np.uint8(0xD2 ^ ((row & 3) * 0x11))
  q8 = rng.integers(-127, 128, (M, K), dtype=np.int16).astype(np.int8)
  q4 = np.empty((N, K), dtype=np.int8)
  for group in range(8):
    payload = raw[:, 16+(group//2)*32:16+(group//2+1)*32]
    q4[:, group*32:(group+1)*32] = ((payload >> (4*(group&1))) & 15).astype(np.int8)
  return raw.view(np.uint32).reshape(-1).copy(), q8, q4


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  transform = PackedWeightTransform("Q4_K", N, K)
  provider = Q4KInt8FragmentProvider(transform)
  geometry = KernelTileGeometry((M, N, 64), (1, 1), 32, 32,
    (KernelLDSWindow("A", 0, M*64, 64), KernelLDSWindow("B", M*64, M*64+N*64, 64)))
  identity = hashlib.sha256(repr((geometry, provider.identity)).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, transform, provider)
  words_np, q8_np, q4_np = _fixture()
  words = Tensor(words_np, device="NV").contiguous().realize()
  q8 = Tensor(q8_np, device="NV").contiguous().realize()
  key = warmstart_key({M, N}, K, transform.storage_dtype)
  from tinygrad.codegen import to_program_cache
  to_program_cache.clear()
  with warmstart_candidate_state({key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}):
    out = q8.matmul(_carrier(words, transform).transpose(), dtype=dtypes.int).contiguous()
    out.realize()
    programs = list(to_program_cache.values())
  got, reference = out.numpy(), q8_np.astype(np.int32) @ q4_np.astype(np.int32).T
  sources = [u.arg for p in programs for u in p.src if u.op is Ops.SOURCE and isinstance(u.arg, str)]
  bound_identities = [getattr(getattr(p.src[0].arg, "candidate_context", None), "canonical_identity", None) for p in programs]
  # Diagnostic control: the same precontract geometry with an ordinary dense
  # int8 B tensor.  If this also fails, the remaining defect is below the typed
  # packed provider in NVIDIA precontract fragment loading/mapping.
  dense_context = _DenseContext("boltbeam.full_kernel_candidate.v1", "d"*64, geometry)
  dense_key = warmstart_key({M, N}, K)
  dense_b = Tensor(q4_np, device="NV").contiguous().realize()
  to_program_cache.clear()
  with warmstart_candidate_state({dense_key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {dense_key:dense_context}):
    dense_out = q8.matmul(dense_b.transpose(), dtype=dtypes.int).contiguous()
    dense_out.realize()
    dense_programs = list(to_program_cache.values())
  dense_got = dense_out.numpy()
  source_path = pathlib.Path(args.out).with_suffix(".cu")
  if sources: source_path.write_text("\n\n".join(sources))
  binaries = [u.arg for p in programs for u in p.src if u.op is Ops.BINARY and isinstance(u.arg, bytes)]
  cubin_path, sass_path, sass = pathlib.Path(args.out).with_suffix(".cubin"), pathlib.Path(args.out).with_suffix(".sass"), ""
  if binaries:
    cubin_path.write_bytes(binaries[0])
    nvdisasm = pathlib.Path(__file__).resolve().parents[3]/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
    env = dict(os.environ, NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
    cp = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-sass", str(cubin_path)], capture_output=True, text=True, env=env)
    sass = cp.stdout+cp.stderr; sass_path.write_text(sass)
  rec = {
    "schema":"tinygrad.nv_compiler_packed_fragment_gate.v1",
    "shape":{"M":M, "N":N, "K":K},
    "identity":identity,
    "ordinary_matmul":True,
    "packed_global_bytes":int(words_np.nbytes),
    "expanded_global_weight_allocation":False,
    "raw_dot":{"exact":bool(np.array_equal(got, reference)), "max_abs":int(np.abs(got-reference).max()),
               "nonzero":int(np.count_nonzero(got))},
    "dense_precontract_control":{"exact":bool(np.array_equal(dense_got, reference)),
      "max_abs":int(np.abs(dense_got-reference).max()), "nonzero":int(np.count_nonzero(dense_got)),
      "programs":len(dense_programs), "diagnostic_expanded_weight":True},
    "source":{"programs":len(programs), "path":str(source_path) if sources else None,
              "signed_imma":any("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in s for s in sources),
              "candidate_identity_exact":bound_identities == [identity]},
    "sass":{"path":str(sass_path) if sass else None, "imma_16832_s8":sass.count("IMMA.16832.S8.S8"),
            "hmma":sass.count("HMMA"), "idp4a":sass.count("IDP.4A")},
    "complete_q4_projection":False,
    "complete_projection_wall":"ordinary two-input int matmul has no Q8 scale/raw-sum inputs and no per-K32 Q4 scale/min accumulator boundary",
  }
  rec["passed"] = (rec["raw_dot"]["exact"] and rec["raw_dot"]["nonzero"] > 0 and
                   rec["source"]["candidate_identity_exact"] and rec["sass"]["imma_16832_s8"] > 0)
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(rec, indent=2)+"\n")
  print(json.dumps(rec, sort_keys=True))
  Device["NV"].synchronize()
  if not rec["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
