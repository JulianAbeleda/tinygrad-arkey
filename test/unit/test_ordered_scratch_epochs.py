"""Generic scheduler contract for bounded scratch reused by opaque programs.

The production-shaped dependency is:

  producer -> main(write scratch) -> fixup(read scratch) -> next main(write scratch)

Every ``uop_program`` result carries the completing CALL in an AFTER.  Threading
the scratch result through that chain is the explicit epoch contract; passing
the raw allocation to two writers remains rejected.
"""
import numpy as np
import pytest

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.schedule.rangeify import _after_writes_buffer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo


N = 4


def _producer(q:UOp, x:UOp) -> UOp:
  i = UOp.range(N, 0)
  return q[i].store(x[i] + 1).end(i).sink(arg=KernelInfo(name="epoch_producer"))


def _main(out:UOp, scratch:UOp, q:UOp) -> UOp:
  i = UOp.range(N, 0)
  return out[i].store(q[i]).end(i).sink(scratch[i].store(q[i] * 2).end(i), arg=KernelInfo(name="epoch_main"))


def _fixup(out:UOp, scratch:UOp) -> UOp:
  i = UOp.range(N, 0)
  return out[i].store(out[i] + scratch[i]).end(i).sink(arg=KernelInfo(name="epoch_fixup"))


def _input(offset:int=0) -> Tensor:
  return Tensor(np.arange(N, dtype=np.float32) + offset, device="PYTHON").realize()


def _chains(inputs:list[Tensor], scratch:Tensor, *, ordered:bool=True) -> tuple[list[Tensor], UOp]:
  outputs:list[Tensor] = []
  epoch = scratch
  for x in inputs:
    q, _ = Tensor.empty(N, device="PYTHON").uop_program(x, fxn=_producer)
    out = Tensor.empty(N, device="PYTHON")
    out, written, _ = out.uop_program(epoch if ordered else scratch, q, fxn=_main)
    out, read = out.uop_program(written, fxn=_fixup)
    if ordered: epoch = read
    outputs.append(out)
  return outputs, epoch.uop


def test_two_producer_main_fixup_chains_share_one_ordered_workspace():
  scratch = Tensor.empty(N, device="PYTHON").realize()
  outputs, final_epoch = _chains([_input(0), _input(100)], scratch)
  Tensor.realize(*outputs)
  np.testing.assert_array_equal(outputs[0].numpy(), np.array([3, 6, 9, 12], dtype=np.float32))
  np.testing.assert_array_equal(outputs[1].numpy(), np.array([303, 306, 309, 312], dtype=np.float32))
  assert final_epoch.buf_uop is scratch.uop.buf_uop


def test_raw_shared_workspace_without_epoch_order_fails_closed():
  outputs, _ = _chains([_input(0), _input(100)], Tensor.empty(N, device="PYTHON").realize(), ordered=False)
  with pytest.raises(RuntimeError, match="unordered repeated write epochs"):
    Tensor.realize(*outputs)


def test_72_ordered_chains_need_one_workspace_and_remain_exact():
  scratch = Tensor.empty(N, device="PYTHON").realize()
  outputs, final_epoch = _chains([_input(i) for i in range(72)], scratch)
  Tensor.realize(*outputs)
  for i in (0, 1, 35, 71):
    np.testing.assert_array_equal(outputs[i].numpy(), 3 * (np.arange(N, dtype=np.float32) + i + 1))
  assert final_epoch.buf_uop is scratch.uop.buf_uop
  assert scratch.uop.buf_uop.arg * scratch.dtype.itemsize == N * dtypes.float32.itemsize


def test_ordered_workspace_capture_replay_rebinds_inputs():
  scratch = Tensor.empty(N, device="PYTHON").realize()

  @TinyJit
  def run(a:Tensor, b:Tensor):
    outputs, _ = _chains([a, b], scratch)
    return tuple(outputs)

  for replay in range(5):
    out0, out1 = run(_input(replay), _input(100 + replay))
    np.testing.assert_array_equal(out0.numpy(), 3 * (np.arange(N, dtype=np.float32) + replay + 1))
    np.testing.assert_array_equal(out1.numpy(), 3 * (np.arange(N, dtype=np.float32) + 100 + replay + 1))
  assert run.captured is not None


def test_finalized_program_output_abi_types_write_and_read_epochs():
  sink = UOp(Ops.SINK, arg=KernelInfo(name="synthetic_native"))
  program = UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="PYTHON"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=b"synthetic")),
    arg=ProgramInfo(name="synthetic_native", globals=(0, 1), outs=(0,), ins=(1,)))
  out, scratch = UOp.new_buffer("PYTHON", N, dtypes.float32), UOp.new_buffer("PYTHON", N, dtypes.float32)
  call = program.call(out, scratch)
  assert _after_writes_buffer(out.after(call))
  assert not _after_writes_buffer(scratch.after(call))


def _access_program(name:str, *, write:bool) -> UOp:
  sink = UOp(Ops.SINK, arg=KernelInfo(name=name))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="PYTHON"), UOp(Ops.LINEAR),
    UOp(Ops.SOURCE, arg=""), UOp(Ops.BINARY, arg=name.encode())),
    arg=ProgramInfo(name=name, globals=(0,), outs=((0,) if write else ()), ins=(() if write else (0,))))


def test_multi_call_after_read_then_write_is_writable():
  buf = UOp.new_buffer("PYTHON", N, dtypes.float32)
  read, write = _access_program("read_then_write_read", write=False).call(buf), \
                _access_program("read_then_write_write", write=True).call(buf)
  assert _after_writes_buffer(buf.after(read, write))


def test_multi_call_after_write_then_read_is_writable():
  buf = UOp.new_buffer("PYTHON", N, dtypes.float32)
  write, read = _access_program("write_then_read_write", write=True).call(buf), \
                _access_program("write_then_read_read", write=False).call(buf)
  assert _after_writes_buffer(buf.after(write, read))


def test_multi_call_after_all_read_is_not_writable():
  buf = UOp.new_buffer("PYTHON", N, dtypes.float32)
  read0, read1 = _access_program("all_read_0", write=False).call(buf), _access_program("all_read_1", write=False).call(buf)
  assert not _after_writes_buffer(buf.after(read0, read1))
