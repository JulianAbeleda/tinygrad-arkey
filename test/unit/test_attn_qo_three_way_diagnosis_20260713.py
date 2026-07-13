from tinygrad.uop.ops import Ops

from extra.qk.prefill.attn_qo_three_way_diagnosis_20260713 import compile_pipe_program


def test_pipe_compile_uses_builder_owned_one_wave_geometry():
  program, evidence = compile_pipe_program()
  assert evidence["passed"] is True
  assert evidence["schedule"] == {
    "tile_m": 32, "tile_n": 32, "tile_k": 16, "threads": 32,
    "waves_m": 1, "waves_n": 1, "buffer_count": 2, "lds_bytes": 1,
  }
  assert program.arg.global_size == (128, 16, 1)
  assert program.arg.local_size == (32, 1, 1)
  assert evidence["argument_order"] == ["a", "b", "output"]


def test_pipe_compile_is_structurally_lds_free():
  program, _ = compile_pipe_program()
  source = next(u.arg for u in program.src if u.op is Ops.SOURCE).lower()
  assert "global_load" in source and "v_wmma" in source
  assert all(marker not in source for marker in ("ds_load", "ds_store", "s_barrier"))
