#!/usr/bin/env python3
"""Graph-safe ordered scratch gate for the finalized pp512 Q4 IMMA chain.

This keeps one physical partial/id workspace and threads its returned AFTER
epochs through producer -> main(write) -> fixup(read) -> next main(write).
It exercises normal lazy scheduling and TinyJit capture; no direct launch or
raw alias is used.
"""
from __future__ import annotations
import argparse, json, pathlib, time
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from tinygrad.uop.ops import Ops
from extra.llm_research.layout import packed_u32_slice, read_metadata
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_q4_imma_provider import M, N, K, PARTIAL_SLOTS, compile_provider, provider_programs
from extra.llm_research.prefill.nv_q8_compact_producer_gate import SRC as Q8_SOURCE

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def _program_calls(linear, name:str):
  return [u for u in linear.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM and u.src[0].arg.name == name]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=MODEL)
  ap.add_argument("--chains", type=int, choices=(2, 72), default=2)
  ap.add_argument("--out", default="")
  args = ap.parse_args()

  dev = Device["NV"]
  provider = compile_provider(dev)
  main_program, fixup_program = provider_programs(provider)
  qlib = NVRTCCompiler(dev.arch, ptx=False, cache_key="q8_ordered_scratch_gate_v1").compile(Q8_SOURCE)
  producer = native_nv_program("q8_compact", qlib, global_size=(M, 8, 1), local_size=(128, 1, 1),
    globals=(0, 1, 2, 3), outs=(1, 2, 3), ins=(0,))

  model_path = pathlib.Path(args.model)
  metadata = read_metadata(model_path)
  info = next(i for i in metadata.infos if i.name == "blk.0.ffn_gate.weight")
  words = packed_u32_slice(model_path, metadata, info, device="NV")
  slotmap = Tensor(provider.slotmap, device="NV").contiguous().realize()
  partials = Tensor.empty(PARTIAL_SLOTS * 128 * 128, dtype=dtypes.float32, device="NV").realize()
  ids = Tensor.empty(PARTIAL_SLOTS, dtype=dtypes.int32, device="NV").realize()

  @TinyJit
  def run(x0:Tensor, x1:Tensor):
    partial_epoch, id_epoch = partials, ids
    outputs = []
    for chain in range(args.chains):
      x = x0 if chain % 2 == 0 else x1
      q8 = Tensor.empty(M*K, dtype=dtypes.int8, device="NV")
      scales = Tensor.empty(M*(K//32), dtype=dtypes.float32, device="NV")
      sums = Tensor.empty(M*(K//32), dtype=dtypes.float32, device="NV")
      _, q8, scales, sums = x.uop_program(q8, scales, sums, fxn=lambda *_: producer)
      out = Tensor.empty(M*N, dtype=dtypes.float32, device="NV")
      out, partial_epoch, id_epoch, _, _, _, _ = out.uop_program(
        partial_epoch, id_epoch, words, q8, scales, sums, fxn=lambda *_: main_program)
      out, partial_epoch, _ = out.uop_program(partial_epoch, slotmap, fxn=lambda *_: fixup_program)
      outputs.append(out)
    return tuple(outputs)

  flat = np.arange(M*K, dtype=np.int32)
  x0 = Tensor((((flat % 257) - 128).astype(np.float32) / 128), device="NV").contiguous().realize()
  x1 = Tensor(((((flat * 7) % 251) - 125).astype(np.float32) / 64), device="NV").contiguous().realize()
  walls = []
  for _ in range(3):
    st = time.perf_counter()
    outputs = run(x0, x1)
    dev.synchronize()
    walls.append((time.perf_counter() - st) * 1e3)

  indices = sorted(set((0, 1, args.chains-1)))
  reference = {i:outputs[i].numpy() for i in indices}
  st = time.perf_counter()
  swapped = run(x1, x0)
  dev.synchronize()
  swap_wall = (time.perf_counter() - st) * 1e3
  swap_exact = {}
  finite = True
  for i in indices:
    got = swapped[i].numpy()
    want = reference[i ^ 1] if i < 2 else reference[1 if i % 2 == 0 else 0]
    swap_exact[str(i)] = bool(np.array_equal(got, want))
    finite &= bool(np.isfinite(got).all() and np.isfinite(reference[i]).all())

  linear = run.captured.linear
  producers = _program_calls(linear, "q8_compact")
  mains = _program_calls(linear, main_program.arg.name)
  fixups = _program_calls(linear, fixup_program.arg.name)
  partial_bases = {u.src[2].buf_uop for u in mains + fixups}
  id_bases = {u.src[3].buf_uop for u in mains}
  workspace_bytes = PARTIAL_SLOTS * 128 * 128 * dtypes.float32.itemsize + PARTIAL_SLOTS * dtypes.int32.itemsize
  distinct_workspace_bytes = args.chains * workspace_bytes
  result = {
    "schema":"tinygrad.nv_q4_imma_ordered_scratch_gate.v1",
    "status":"PASS" if finite and all(swap_exact.values()) and len(producers) == len(mains) == len(fixups) == args.chains
      and len(partial_bases) == len(id_bases) == 1 else "FAIL",
    "chains":args.chains,
    "call_census":{"producer":len(producers), "main":len(mains), "fixup":len(fixups)},
    "ordered_workspace":{"partial_physical_bases":len(partial_bases), "id_physical_bases":len(id_bases),
      "bounded_bytes":workspace_bytes, "distinct_bytes":distinct_workspace_bytes,
      "saved_bytes":distinct_workspace_bytes-workspace_bytes},
    "correctness":{"finite":finite, "swap_rebind_exact":swap_exact,
      "reference_minmax":{str(i):[float(reference[i].min()), float(reference[i].max())] for i in indices}},
    "timing":{"capture_sequence_ms":walls, "hot_swapped_ms":swap_wall},
  }
  payload = json.dumps(result, sort_keys=True, indent=2)
  print(payload)
  if args.out:
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n")
  if result["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__": main()
