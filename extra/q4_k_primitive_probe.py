#!/usr/bin/env python3
import argparse, pathlib, struct, time

from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import UOp, KernelInfo

from extra.qk_layout import GGML_Q4_K, pick_tensor, q4_k_weight_bytes, read_metadata, tensor_shape

def u32_copy_kernel(nwords:int):
  def copy_words(out:UOp, words:UOp) -> UOp:
    gid = UOp.special(nwords, "gidx0")
    return out[gid].store(words[gid]).sink(arg=KernelInfo(name="q4k_u32_copy_probe", opts_to_apply=()))
  return copy_words

def bench(label:str, iters:int, nbytes:int, fn):
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  dt = (time.perf_counter() - st) / iters
  print(f"{label}: {dt*1000:.3f} ms, {nbytes/dt/1e9:.2f} GB/s, kernels={GlobalCounters.kernel_count/iters:.1f}")

def read_words(path:pathlib.Path, byte_off:int, nwords:int) -> list[int]:
  with path.open("rb") as f:
    f.seek(byte_off)
    return list(struct.unpack("<" + "I"*nwords, f.read(4*nwords)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Probe the primitive path for wide Q4_K uint32 loads")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--device", default=None)
  parser.add_argument("--iters", type=int, default=10)
  parser.add_argument("--source", choices=("bitcast", "disk-u32"), default="bitcast",
                      help="bitcast starts from uint8 bytes; disk-u32 opens the GGUF file as uint32 storage")
  args = parser.parse_args()

  meta = read_metadata(args.gguf)
  info = pick_tensor(meta.infos, args.tensor)
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")

  byte_start = meta.data_start + info.off
  q4_bytes = q4_k_weight_bytes(info)
  nwords = q4_bytes // 4
  if byte_start % 4 != 0 or q4_bytes % 4 != 0:
    raise ValueError(f"Q4_K byte range is not uint32 aligned: start={byte_start} q4_bytes={q4_bytes}")
  print(f"tensor={info.name} shape={tensor_shape(info)} q4_bytes={q4_bytes} nwords={nwords} "
        f"source={args.source} device={args.device or 'default'}")

  if args.source == "bitcast":
    raw = Tensor(args.gguf, device=args.device)
    raw_u8 = raw[byte_start:byte_start + q4_bytes]

    # This preparation step is intentionally explicit: today, converting the GGUF uint8 bytes
    # to uint32 through Tensor.bitcast materializes a scalar byte-packing kernel before the
    # custom primitive sees the data.
    raw_u32 = raw_u8.bitcast(dtypes.uint32).reshape(nwords).contiguous().realize()
  else:
    raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
    raw_u32 = raw_words[byte_start//4:byte_start//4+nwords].to(args.device).contiguous().realize()

  sample = min(16, nwords)
  got_words = raw_u32[:sample].numpy().tolist()
  if got_words != read_words(args.gguf, byte_start, sample):
    raise AssertionError(f"{args.source} source failed raw uint32 word correctness")
  print(f"source_correctness: PASS {sample} raw uint32 words match file bytes")
  out = Tensor.empty(nwords, dtype=dtypes.uint32, device=args.device)

  def custom_copy():
    return out.custom_kernel(raw_u32, fxn=u32_copy_kernel(nwords))[0]

  copied = custom_copy().realize()
  if (copied == raw_u32).all().numpy().item() is not True:
    raise AssertionError("uint32 copy probe failed correctness")
  print("correctness: PASS uint32 copy equals prepared uint32 view")
  bench("q4k_u32_copy_probe", args.iters, q4_bytes, custom_copy)
