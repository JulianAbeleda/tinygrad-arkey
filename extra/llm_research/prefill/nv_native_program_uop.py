"""Capturable wrapper for an already-compiled, single-entry NV cubin."""
from __future__ import annotations

from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

def native_arg_offsets(layout) -> tuple[tuple[int, int], ...]:
  """Return exact (offset,size) for ordered native ABI slots.

  Five-field entries are ``kind, source, size, alignment, explicit_offset`` and
  are used for CUDA parameter-bank ABIs whose packing is not host-ABI-derived.
  Four-field entries retain the inferred legacy layout.
  """
  off, out = 0, []
  for entry in layout:
    kind, _source, size, alignment = entry[:4]
    if kind not in ("ptr", "u32", "u64", "blob") or size <= 0 or alignment <= 0 or alignment & (alignment-1):
      raise ValueError("invalid native argument layout entry")
    if len(entry) == 5:
      explicit = entry[4]
      if explicit < off or explicit % alignment: raise ValueError("invalid native argument offset")
      off = explicit
    else: off = (off + alignment - 1) & ~(alignment - 1)
    out.append((off, size)); off += size
  return tuple(out)


def native_nv_program(name:str, cubin:bytes, *, global_size:tuple[int, int, int],
                      local_size:tuple[int, int, int], globals:tuple[int, ...],
                      outs:tuple[int, ...]=(), ins:tuple[int, ...]=(), vals:tuple[int, ...]=(), shared_mem:int=0,
                      arg_blobs:tuple[tuple[int, bytes, int], ...]=(),
                      arg_layout:tuple[tuple[str, int, int, int], ...]=()) -> UOp:
  if not cubin or cubin[:4] != b"\x7fELF": raise ValueError("native NV program requires an ELF cubin")
  if shared_mem < 0: raise ValueError("shared_mem must be non-negative")
  for index, blob, alignment in arg_blobs:
    if not isinstance(index, int) or index < 0 or not isinstance(blob, bytes) or not blob:
      raise ValueError("native argument blobs require positive index and non-empty bytes")
    if alignment <= 0 or alignment & (alignment - 1): raise ValueError("native argument blob alignment must be a power of two")
  if arg_layout: native_arg_offsets(arg_layout)
  # The empty SINK/LINEAR and retained source are identity/evidence nodes. The
  # binary and launch ABI are consumed directly by get_runtime/exec_kernel.
  fixed_vars=tuple(UOp.variable(f"{name}_arg{i}", value, value) for i,value in enumerate(vals))
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, arg=KernelInfo(name=name)), UOp(Ops.DEVICE, arg="NV"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=f"// precompiled native cubin: {name}"), UOp(Ops.BINARY, arg=cubin)),
    arg=ProgramInfo(name=name, global_size=global_size, local_size=local_size, vars=fixed_vars, globals=globals,
                    outs=outs, ins=ins, aux=((shared_mem,) if shared_mem else ()), arg_blobs=arg_blobs, arg_layout=arg_layout))


def call_native(program:UOp, *tensors, wait:bool=False) -> None:
  """Launch now, or contribute the native call to an active TinyJit capture."""
  from tinygrad.engine.realize import capturing, run_linear
  args=tuple(t.uop.buf_uop for t in tensors)
  call=program.call(*args)
  linear=UOp(Ops.LINEAR, src=(call,))
  var_vals={v.expr:v.arg[1] for v in program.arg.vars}
  if capturing and CAPTURING:
    capturing[0].add_linear(linear, var_vals)
  else:
    run_linear(linear, var_vals=var_vals, wait=wait, jit=False)
