"""Capturable wrapper for an already-compiled, single-entry NV cubin."""
from __future__ import annotations

from tinygrad.helpers import CAPTURING
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp


def native_nv_program(name:str, cubin:bytes, *, global_size:tuple[int, int, int],
                      local_size:tuple[int, int, int], globals:tuple[int, ...],
                      outs:tuple[int, ...]=(), ins:tuple[int, ...]=(), vals:tuple[int, ...]=(), shared_mem:int=0) -> UOp:
  if not cubin or cubin[:4] != b"\x7fELF": raise ValueError("native NV program requires an ELF cubin")
  if shared_mem < 0: raise ValueError("shared_mem must be non-negative")
  # The empty SINK/LINEAR and retained source are identity/evidence nodes. The
  # binary and launch ABI are consumed directly by get_runtime/exec_kernel.
  fixed_vars=tuple(UOp.variable(f"{name}_arg{i}", value, value) for i,value in enumerate(vals))
  return UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, arg=KernelInfo(name=name)), UOp(Ops.DEVICE, arg="NV"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=f"// precompiled native cubin: {name}"), UOp(Ops.BINARY, arg=cubin)),
    arg=ProgramInfo(name=name, global_size=global_size, local_size=local_size, vars=fixed_vars, globals=globals,
                    outs=outs, ins=ins, aux=((shared_mem,) if shared_mem else ())))


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
