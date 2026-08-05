from extra.llm_research.decode.nv_overlap_resource_join import resource_complete
from extra.llm_research.decode.route_b3_dag_attribution import CallRecord, CallAccess, attach_compiled_descriptors, B3AttributionError
import pytest
def test_resource_join_refuses_partial_resource_rows():
  assert not resource_complete({"grid":[1,1,1],"block":[32,1,1],"registers_per_thread":32,"static_smem_bytes":0,"dynamic_smem_bytes":0,"local_mem_bytes":None})

def test_descriptor_collector_is_occurrence_exact_and_observational():
  calls=[CallRecord(0,"a",[CallAccess("x",0,4,True)]),CallRecord(1,"b",[CallAccess("x",0,4,False)])]
  d={i:{"binary_sha256":"0"*64,"grid":[1,1,1],"block":[32,1,1],"registers_per_thread":1,"static_smem_bytes":0,"dynamic_smem_bytes":0,"local_mem_bytes":0} for i in range(2)}
  assert [x["index"] for x in attach_compiled_descriptors(calls,d)] == [0,1]
  with pytest.raises(B3AttributionError): attach_compiled_descriptors(calls,{0:d[0]})
