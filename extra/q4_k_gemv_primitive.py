#!/usr/bin/env python3
import argparse, pathlib, time
from math import prod

from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.q4_k_bench import GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, pick_tensor, q4_k_reference, read_metadata, tensor_shape

Q4K_WORDS_PER_BLOCK = Q4_K_BLOCK_BYTES // 4

def _f16_word(word:UOp, high:bool) -> UOp:
  bits = (word.rshift(16) if high else word).bitwise_and(0xffff)
  return bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def q4k_gemv_kernel(rows:int, k:int):
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    blk = UOp.range(k_blocks, 1, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 2, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    scale_base = base + 1

    w0 = words[base]
    d, dmin = _f16_word(w0, False), _f16_word(w0, True)

    def scale_byte(idx:int) -> UOp:
      return words[scale_base + idx//4].rshift((idx%4)*8).bitwise_and(0xff)

    contrib = UOp.const(dtypes.float32, 0.0)
    for grp in range(8):
      if grp < 4:
        sc = scale_byte(grp).bitwise_and(63)
        mn = scale_byte(4+grp).bitwise_and(63)
      else:
        high = scale_byte(8+grp-4)
        sc = high.bitwise_and(0xf).bitwise_or(scale_byte(grp-4).rshift(6).lshift(4))
        mn = high.rshift(4).bitwise_or(scale_byte(4+grp-4).rshift(6).lshift(4))

      qword = words[base + 4 + (grp//2)*8 + pos//4]
      q = qword.rshift((pos%4)*8 + (grp%2)*4).bitwise_and(0xf)
      w = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
      contrib = contrib + w * x[blk*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)

    acc = out[row].set(0.0)
    acc = out[row].set(acc.after(blk, pos)[row] + contrib, end=pos)
    return acc.end(row, blk).sink(arg=KernelInfo(name=f"q4k_gemv_ref_{rows}_{k}", opts_to_apply=()))

  return kernel

def bench(label:str, iters:int, q4_bytes:int, fn) -> None:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  dt = (time.perf_counter() - st) / iters
  print(f"{label}: {dt*1000:.3f} ms, {q4_bytes/dt/1e9:.2f} Q4-GB/s, kernels={GlobalCounters.kernel_count/iters:.1f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Correctness-first custom Q4_K GEMV primitive probe")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--device", default=None)
  parser.add_argument("--rows", type=int, default=2)
  parser.add_argument("--iters", type=int, default=3)
  args = parser.parse_args()

  meta = read_metadata(args.gguf)
  info = pick_tensor(meta.infos, args.tensor)
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  n, shape = prod(info.dims), tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = min(args.rows, shape[0]), shape[1]
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q4_K block aligned")
  byte_start = meta.data_start + info.off
  if byte_start % 4 != 0: raise ValueError(f"Q4_K tensor byte offset is not uint32 aligned: {byte_start}")

  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  q4_bytes = rows * row_bytes
  nwords = q4_bytes // 4
  print(f"tensor={info.name} full_shape={shape} primitive_shape=({rows},{k}) q4_bytes={q4_bytes} nwords={nwords} device={args.device or 'default'}")

  raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+nwords].to(args.device).contiguous().realize()
  x = Tensor.ones(k, dtype=dtypes.float16, device=args.device).realize()
  out = Tensor.empty(rows, dtype=dtypes.float32, device=args.device)

  raw_u8 = Tensor(args.gguf)[byte_start:byte_start+q4_bytes].to(args.device)
  decoded = q4_k_reference(raw_u8, rows*k).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  def primitive():
    return out.custom_kernel(words, x, fxn=q4k_gemv_kernel(rows, k))[0]

  got = primitive().realize()
  max_abs = (got - ref).abs().max().item()
  print(f"correctness: max_abs={max_abs:.6g}")
  if max_abs > 1e-2:
    print("got", got.numpy())
    print("ref", ref.numpy())
    raise AssertionError("Q4_K GEMV primitive correctness failed")
  bench("q4k_gemv_primitive", args.iters, q4_bytes, primitive)
