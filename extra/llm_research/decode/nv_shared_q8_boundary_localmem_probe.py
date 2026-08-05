#!/usr/bin/env python3
"""One-kernel native-NV local-memory differential for the shared-Q8 Q4 consumer.

This is deliberately *not* a model route.  It takes an actual Qwen3-8B Q4_K
attention-Q payload and an actual token-embedding activation, packs that
activation to Q8_1 on the CPU, then executes exactly one of the experimental
Q4 shared-Q8 consumer kernels.  The two arms differ only at the typed program
boundary:

* ``research`` supplies an explicit ``Tensor.empty`` to
  ``execute_research_program``;
* ``promoted`` lets ``OutputSpec`` allocate the same output through
  ``execute_promoted_program``.

Before either arm can submit, the program is lowered and its NV resource/QMD
state is recorded.  This separates a generic QMD/local-memory defect from a
model-graph lifetime or scheduling defect without launching the known-faulting
multi-consumer graph.
"""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path

import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.gguf import gguf_load, gguf_load_metadata
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
                                         execute_promoted_program, execute_research_program)
from tinygrad.uop.ops import Ops

from extra.llm_research.decode.q4q4q6_shared_q8_microgate import K, emit_q4, emit_q6

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
ROWS = 4096


def _sha(obj: object) -> str:
  data = obj if isinstance(obj, bytes) else repr(obj).encode()
  return hashlib.sha256(data).hexdigest()


def _actual_payload(model: str) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
  """Return real blk.0 Q payload plus the dequantized real token-1 embedding.

  The GGUF backing is memory mapped by tinygrad.  Only the 9 MiB packed Q
  tensor and one 4096-wide embedding row are materialized for this probe.
  """
  _kv, metadata = gguf_load_metadata(model)
  infos = {x[0]: x for x in metadata["tensor_infos"]}
  expected = {"blk.0.attn_q.weight": (ROWS, 12, 144, np.uint32), "blk.0.attn_k.weight": (1024, 12, 144, np.uint32),
              "blk.0.attn_v.weight": (1024, 14, 210, np.uint16)}
  packed_weights = {}
  for name, (rows, ggml_type, block_bytes, dtype) in expected.items():
    _name, shape, got_type, offset = infos[name]
    if tuple(shape) != (K, rows) or got_type != ggml_type: raise RuntimeError(f"unexpected authority payload: {infos[name]!r}")
    raw = np.memmap(model, mode="r", dtype=np.uint8, offset=metadata["data_start"] + offset,
                    shape=(rows * K // 256 * block_bytes,))
    packed_weights[name] = np.asarray(raw).copy().view(dtype)
  # Use a genuine model activation. gguf_load is lazy; indexing one row keeps
  # the numerical work local rather than instantiating a model/route.
  _kv, state = gguf_load(model)
  activation = state["token_embd.weight"][1].numpy().astype(np.float16, copy=False)
  group = activation.astype(np.float32).reshape(K // 32, 32)
  scales = np.maximum(np.max(np.abs(group), axis=1) / 127.0, 1e-12).astype(np.float32)
  q = np.clip(np.rint(group / scales[:, None]), -127, 127).astype(np.int8).reshape(K // 4, 4).view(np.uint8)
  packed = (q[:, 0].astype(np.uint32) | (q[:, 1].astype(np.uint32) << 8) |
            (q[:, 2].astype(np.uint32) << 16) | (q[:, 3].astype(np.uint32) << 24))
  return packed_weights, packed, scales


def _lower_nv(t: Tensor):
  linear, var_vals = Tensor.linear_with_vars(t)
  if var_vals: raise RuntimeError(f"expected static graph, got vars={var_vals}")
  lowered = []
  for call in linear.src:
    ast = call.src[0]
    if ast.op is Ops.SINK: ast = to_program(ast, Device["NV"].renderer)
    if ast.op is not Ops.PROGRAM: continue
    lowered.append((get_runtime("NV", ast), ast))
  if not lowered: raise RuntimeError("no NV programs lowered")
  return lowered


def _qmd_fields(prg) -> dict[str, int]:
  # V3 and V5 spell the same fields differently (V5 carries ``_SHIFTED4``),
  # so enumerate the actual QMD schema instead of accidentally omitting the
  # local-memory field on Blackwell.
  fields = prg.qmd.fields[prg.qmd.pref]
  names = sorted(name.lower() for name in fields if any(tag in name for tag in ("LOCAL_MEMORY", "SHARED_MEMORY", "REGISTER_COUNT")))
  names += [name for name in ("cta_raster_width", "grid_width") if name.upper() in fields]
  return {name: prg.qmd.read(name) for name in names}


def run(boundary: str, model: str, group: bool=False) -> dict:
  if Device.DEFAULT != "NV": raise RuntimeError("set DEV=NV; this probe is native-NV only")
  if boundary not in ("research", "promoted"): raise ValueError(boundary)
  weights, packed, scales = _actual_payload(model)
  dev = Device.DEFAULT
  w = Tensor(weights["blk.0.attn_q.weight"], dtype=dtypes.uint32, device=dev).contiguous().realize()
  xp = Tensor(packed, dtype=dtypes.uint32, device=dev).contiguous().realize()
  xs = Tensor(scales, dtype=dtypes.float32, device=dev).contiguous().realize()
  emitter = emit_q4(ROWS)
  program = KernelProgram("research.nv_shared_q8_boundary", "q4_consumer",
    KernelProgramProvenance.RESEARCH_ONLY if boundary == "research" else KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    emitter, output_spec=OutputSpec((ROWS,), dtypes.float32))
  execute = (lambda p, *inputs: execute_research_program(Tensor.empty(p.output_spec.shape if p.output_spec else ROWS,
             dtype=dtypes.float32, device=dev), *inputs, program=p)) if boundary == "research" else \
            (lambda p, *inputs: execute_promoted_program(None, *inputs, program=p))
  out = execute(program, w, xp, xs)
  if group:
    # The only permitted escalation: three independent consumers, all driven
    # by the CPU-packed genuine activation. There is no model hook, cache
    # mutation, or generated-token loop in this graph.
    wk = Tensor(weights["blk.0.attn_k.weight"], dtype=dtypes.uint32, device=dev).contiguous().realize()
    wv = Tensor(weights["blk.0.attn_v.weight"], dtype=dtypes.uint16, device=dev).contiguous().realize()
    pk = KernelProgram("research.nv_shared_q8_boundary", "k_consumer", program.provenance, emit_q4(1024),
                       output_spec=OutputSpec((1024,), dtypes.float32))
    pv = KernelProgram("research.nv_shared_q8_boundary", "v_consumer", program.provenance, emit_q6(1024),
                       output_spec=OutputSpec((1024,), dtypes.float32))
    # Keep all three calls live in a single scheduler graph; scalar reductions
    # are merely a safe completion sink, not a numerical model operation.
    out = out.sum() + execute(pk, wk, xp, xs).sum() + execute(pv, wv, xp, xs).sum()
  lowered = _lower_nv(out)
  before = [{"regs_usage": prg.regs_usage, "shmem_usage": prg.shmem_usage, "lcmem_usage": prg.lcmem_usage,
             "slm_per_thread": prg.dev.slm_per_thread, "qmd": _qmd_fields(prg),
             "source_sha256": _sha(ast.src[4].arg), "lib_sha256": _sha(prg.lib),
             "program_name": prg.name} for prg, ast in lowered]
  out.realize(); Device[dev].synchronize()
  values = np.asarray(out.numpy()).reshape(-1)
  return {"boundary": boundary, "group": group, "metadata_before_exec": before,
          "output_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
          "output_prefix": [float(x) for x in values[:4]]}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--boundary", choices=("research", "promoted"), required=True)
  ap.add_argument("--model", default=MODEL)
  ap.add_argument("--group", action="store_true", help="run the isolated real Q/K/V three-consumer graph")
  ap.add_argument("--out", type=Path)
  args = ap.parse_args()
  result = run(args.boundary, args.model, args.group)
  encoded = json.dumps(result, indent=2, sort_keys=True)
  print(encoded)
  if args.out: args.out.write_text(encoded + "\n")


if __name__ == "__main__": main()
