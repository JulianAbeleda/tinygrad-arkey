from tinygrad.uop.ops import ProgramInfo

from extra.qk.mmq_llama_five_buffer_abi_bisect import CASES, LOCAL, build_bisect_sink


def test_abi_bisect_probes_keep_exact_five_pointer_abi_and_launch_shape():
  for case in CASES:
    info = ProgramInfo.from_sink(build_bisect_sink(case))
    assert info.globals == (0, 1, 2, 3, 4)
    assert info.global_size == (1, 1, 1)
    assert info.local_size == (LOCAL, 1, 1)
