"""GPU-free Q4/Q6-compatible Stream-K schedule contract."""
from dataclasses import dataclass
from extra.llm_research.prefill.nv_streamk_schedule_spec import StreamKScheduleSpec

@dataclass(frozen=True)
class Q4StreamKScheduleSpec(StreamKScheduleSpec):
  M: int = 512; N: int = 12288; K: int = 4096
  tile_m: int = 128; tile_n: int = 128; tile_k: int = 64; owners: int = 170
  def validate(self):
    return super().validate()

DEFAULT_Q4_STREAMK = Q4StreamKScheduleSpec()
