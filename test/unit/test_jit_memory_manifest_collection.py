from tinygrad import Tensor, TinyJit
from tinygrad.schedule.memory import ScheduleMemoryManifest, collect_memory_plan_manifests


def test_ordinary_cpu_schedule_collects_only_when_requested():
  with collect_memory_plan_manifests() as manifests:
    linear = (Tensor(list(range(16)), device="CPU") + 1).contiguous().schedule_linear()
  assert linear.src and len(manifests) == 1
  assert isinstance(manifests[0], ScheduleMemoryManifest)

  inactive = []
  (Tensor(list(range(16)), device="CPU") + 1).contiguous().schedule_linear()
  assert inactive == []


def test_tinyjit_cpu_capture_lowering_collects_without_changing_execution():
  @TinyJit
  def add_one(x:Tensor) -> Tensor: return (x + 1).contiguous()

  assert add_one(Tensor(list(range(8)), device="CPU").realize()).tolist() == list(range(1, 9))
  capture_input = Tensor(list(range(8)), device="CPU").realize()
  with collect_memory_plan_manifests() as manifests:
    result = add_one(capture_input)
  assert result.tolist() == list(range(1, 9))
  assert len(manifests) == 1
  assert isinstance(manifests[0], ScheduleMemoryManifest)

  replay_input = Tensor(list(range(8)), device="CPU").realize()
  with collect_memory_plan_manifests() as replay_manifests:
    assert add_one(replay_input).tolist() == list(range(1, 9))
  assert replay_manifests == []
