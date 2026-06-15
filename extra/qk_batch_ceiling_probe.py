import sys, time
from tinygrad import Tensor
from tinygrad.uop.ops import UOp
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.model import Transformer
model, kv = Transformer.from_gguf(sys.argv[1], 4096)
mc = model.max_context
v_sp = UOp.variable("start_pos", 0, mc-1)
v_tk = UOp.variable("toks", 1, 32)
temp = Tensor([0.0])
t = Tensor([[1]*mc], dtype="int32")
print("T   fwd_ms   ms/tok   mem_MB")
for T in [1,2,4,8,16,32]:
  sp = v_sp.bind(0); nt = v_tk.bind(T)
  inp = t[:, sp:sp+nt]
  model(inp, sp, temp).realize(); model(inp, sp, temp).realize()  # double warmup
  GlobalCounters.reset(); R=5; st=time.perf_counter()
  for _ in range(R): out = model(inp, sp, temp).realize()
  dt=(time.perf_counter()-st)/R
  print(f"{T:<3} {dt*1000:7.2f} {dt/T*1000:8.2f} {GlobalCounters.global_mem/1e6:8.0f}")
