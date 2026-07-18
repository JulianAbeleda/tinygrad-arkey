import itertools
from types import SimpleNamespace

from tinygrad.device import BufferSpec
from tinygrad.runtime.ops_amd import AMD_KERNARGS_BUFFER_SPEC
from tinygrad.runtime.support.hcq import BumpAllocator, HCQBuffer, HCQProgram


class _ArgsState:
  def __init__(self, buf, _program, _bufs, vals=()):
    self.buf, self.vals = buf, vals


def test_hcq_program_kernargs_honor_program_alignment():
  dev = SimpleNamespace(
    kernargs_buf=HCQBuffer(0x1000, 0x1000),
    kernargs_offset_allocator=BumpAllocator(0x1000, wrap=False),
    prof_prg_counter=itertools.count(),
  )
  program = HCQProgram(_ArgsState, dev, "test", kernargs_alloc_size=40, kernargs_alignment=64)
  first = program.fill_kernargs(())
  second = program.fill_kernargs(())
  assert first.buf.va_addr == 0x1000
  assert second.buf.va_addr == 0x1040
  assert first.buf.va_addr % 64 == second.buf.va_addr % 64 == 0


def test_amd_kernargs_pool_is_cpu_visible_and_uncached():
  assert AMD_KERNARGS_BUFFER_SPEC == BufferSpec(cpu_access=True, uncached=True)
