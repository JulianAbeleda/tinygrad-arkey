import numpy as np

from extra.qk.prefill.guarded_execution import (GuardedBuffer, GuardedExecutionHooks, make_tinygrad_executable_hooks,
                                                make_tinygrad_guarded_hooks, run_guarded_execution)


class Fake:
  def __init__(self, mutate=False, unhealthy=False): self.data = {}; self.mutate = mutate; self.unhealthy = unhealthy
  def allocate(self, name, value, policy):
    self.data[name] = np.zeros_like(value); return GuardedBuffer(name, name, policy.prefix_bytes, policy.suffix_bytes)
  def upload(self, buffer, value): self.data[buffer.name][...] = value
  def readback(self, buffer): return self.data[buffer.name].copy()
  def guards_intact(self, buffer): return not (self.mutate and buffer.name == "output")
  def dispatch(self, executable, buffers):
    self.data["output"][...] = self.data["a"] + self.data["b"]
    return 0.001
  def health(self): return not self.unhealthy
  def release(self, buffer): pass


def hooks(fake):
  return GuardedExecutionHooks(fake.allocate, fake.upload, fake.readback, fake.guards_intact,
                               fake.dispatch, fake.health, fake.release)


def test_guarded_execution_is_transport_neutral_and_full_output_gated():
  fake = Fake()
  result = run_guarded_execution(executable="either", inputs={"a": np.array([1, 2], dtype=np.float16),
    "b": np.array([3, 4], dtype=np.float16)}, reference=np.array([4, 6], dtype=np.float32), hooks=hooks(fake))
  assert result["passed"] is True
  assert result["guards_intact"] is True and result["inputs_unchanged"] is True


def test_guarded_execution_refuses_health_failure_before_dispatch():
  fake = Fake(unhealthy=True)
  result = run_guarded_execution(executable="either", inputs={"a": np.array([1, 2]), "b": np.array([3, 4])},
    reference=np.array([4, 6], dtype=np.float32), hooks=hooks(fake))
  assert result["passed"] is False and result["dispatch_performed"] is False


def test_guarded_execution_fails_closed_on_guard_corruption():
  fake = Fake(mutate=True)
  result = run_guarded_execution(executable="either", inputs={"a": np.array([1, 2]), "b": np.array([3, 4])},
    reference=np.array([4, 6], dtype=np.float32), hooks=hooks(fake))
  assert result["passed"] is False and result["guards_intact"] is False


def test_tinygrad_buffer_hooks_place_and_check_real_payload_guards_on_npy():
  def dispatch(_, buffers):
    a = buffers["a"].resource["payload"].numpy()
    b = buffers["b"].resource["payload"].numpy()
    buffers["output"].resource["payload"].copyin(memoryview(np.ascontiguousarray(a + b)))
    return 0.001
  hooks = make_tinygrad_guarded_hooks("CPU", dispatch, lambda: True)
  result = run_guarded_execution(executable="either", inputs={"a": np.array([1, 2], dtype=np.float16),
    "b": np.array([3, 4], dtype=np.float16)}, reference=np.array([4, 6], dtype=np.float32), hooks=hooks)
  assert result["passed"] is True and result["guards_intact"] is True


def test_tinygrad_executable_adapter_binds_logical_abi_order_without_transport_branching():
  class Payload:
    def __init__(self, value): self.value = value
    def get_buf(self, device): return (device, self.value)
  class Executable:
    def dispatch(self, *args): self.args = args; return 0.001
  executable = Executable()
  hooks = make_tinygrad_executable_hooks("CPU", lambda: True, ("b", "a", "output"))
  buffers = {name: GuardedBuffer(name, {"payload": Payload(name)}, 1, 1) for name in ("a", "b", "output")}
  assert hooks.dispatch(executable, buffers) == 0.001
  assert executable.args == (("CPU", "b"), ("CPU", "a"), ("CPU", "output"))


def test_dispatch_exception_is_conservatively_recorded_as_an_attempt():
  fake = Fake()
  fake.dispatch = lambda executable, buffers: (_ for _ in ()).throw(RuntimeError("submit failed"))
  result = run_guarded_execution(executable="either", inputs={"a": np.array([1, 2]), "b": np.array([3, 4])},
    reference=np.array([4, 6], dtype=np.float32), hooks=hooks(fake))
  assert result["passed"] is False and result["dispatch_performed"] is True
