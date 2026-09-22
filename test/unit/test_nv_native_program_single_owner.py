import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_llama_packed_q4k_o_pp512_binding import _single_owner_main


class TestNVNativeProgramSingleOwner(unittest.TestCase):
  def test_workspace_is_raw_and_main_has_one_lazy_owner(self):
    main=native_nv_program("single_owner_test",b"\x7fELFtest",global_size=(1,1,1),local_size=(1,1,1),
                           globals=(0,1,2,3),outs=(2,3),ins=(0,1))
    words=Tensor.empty(4,dtype=dtypes.uint32,device="NV")
    record=Tensor.empty(4,dtype=dtypes.uint32,device="NV")
    out=Tensor.empty(4,dtype=dtypes.float32,device="NV")
    workspace=Tensor.empty(4,dtype=dtypes.float32,device="NV")
    owned_out,raw_workspace=_single_owner_main(words,record,out,workspace,main)
    self.assertIs(raw_workspace,workspace)
    calls=[u for u in owned_out.uop.toposort() if u.op is Ops.CALL and u.src[0].op is Ops.PROGRAM]
    self.assertEqual(len(calls),1)
    self.assertEqual(calls[0].src[0].arg.name,"single_owner_test")
    self.assertFalse(any(u.op is Ops.CALL for u in raw_workspace.uop.toposort()))


if __name__ == "__main__": unittest.main()
