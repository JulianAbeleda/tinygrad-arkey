import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_kernel
from tinygrad.llm.kernel_program import (KernelProgram, KernelProgramProvenance, OutputSpec,
  DeclaredTypedOutput, TypedLayout, TypedViewRequest, execute_promoted_program,
  _validated_typed_view)
from tinygrad.uop.ops import Ops
from extra.llm_research.decode.route_class_numerics import _make_q4k_words

ROWS, K = 32, 1024
words, _ = _make_q4k_words(ROWS, K, seed=1)
words_t = Tensor(words.copy(), dtype=dtypes.uint32, device="CPU").contiguous().realize()
hidden = Tensor.empty((K,), dtype=dtypes.float16, device="CPU")

def producer(store_fp16, declared):
  typed_output = (DeclaredTypedOutput(TypedLayout(dtypes.float16, (ROWS,), (1, 1, ROWS)),
                                      combine_fusion_admitted=False, epilogue_absorption_admitted=True)
                  if declared else None)
  prog = KernelProgram("cpu_topology", "w1w3", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    q4k_g3_lanemap_gemv_w1w3_kernel(ROWS, K, load_style="scalar", store_fp16=store_fp16),
    output_spec=OutputSpec((ROWS,), dtypes.float16 if store_fp16 else dtypes.float32, typed_output=typed_output))
  return execute_promoted_program(None, words_t, words_t, hidden, program=prog)

req = TypedViewRequest(slot=1, dtype=dtypes.float16, flat_shape=(ROWS,), route_role="ffn_down",
                       requires_combine_fusion=False, requires_epilogue_absorption=True)
consumer = KernelProgram("decode_q4k_x", "down.gemv", KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
  q4k_g3_lanemap_gemv_w1w3_kernel(1, K, load_style="scalar"),
  output_spec=OutputSpec((1,), dtypes.float32), typed_input_views=(req,))

for label, sf, dec in [("typed", True, True), ("negative", True, False)]:
  z = producer(sf, dec)
  v = z.cast(dtypes.float16).contiguous()
  u = v.uop
  print(f"{label}: u.op={u.op.value} base={u.base.op.value if u.base is not None else '-'}")
  view, reason = _validated_typed_view(u, req, consumer)
  print(f"  view={'OK' if view is not None else None} reason={reason}")
