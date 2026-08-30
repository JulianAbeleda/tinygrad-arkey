"""Exactness gate for the scheduler-generated two-stage DS4 producer."""
import argparse, json, os, time, statistics
import numpy as np
from tinygrad import Device, Tensor, dtypes
from .nv_q8_ds4_native_producer import (DS4ProducerSpec, cpu_pack_ds4,
  emit_ds4_metadata_stage, emit_ds4_pack_stage, flat_ds4_inputs,
  realize_ds4_metadata)

def run(rows: int, k: int) -> dict:
  spec = DS4ProducerSpec(M=rows, K=k)
  rng = np.random.default_rng(20260830)
  host = (rng.standard_normal((rows, k), dtype=np.float32) * 3.0).astype(np.float16)
  x = Tensor(host, device="NV").realize()
  meta = realize_ds4_metadata(x, spec).realize()
  if rows == 512 and k == 4096:
    np.save('/tmp/ds4-workspace.npy', meta.numpy())
  xf, mf = flat_ds4_inputs(x, meta)
  out = Tensor.empty(spec.records*72, dtype=dtypes.uint16, device="NV").realize()
  out, _, _ = out.uop_program(xf, mf, fxn=emit_ds4_pack_stage(spec))
  got = out.realize().numpy().view(np.uint8).reshape(spec.records, 144)
  want = cpu_pack_ds4(host.astype(np.float32))
  llama = None
  if rows == 512 and k == 4096:
    from .nv_llama_packed_q4k_pp512_binding import LlamaPackedQ4KPP512Binding
    binding = LlamaPackedQ4KPP512Binding.compile(Device["NV"])
    lr = Tensor.empty((spec.records*144 + 128*144)//4, dtype=dtypes.uint32, device="NV").realize()
    _, lr = x.uop_program(lr, fxn=lambda *_: binding.producer)
    llama = lr.realize().numpy().view(np.uint8)[:spec.records*144].reshape(spec.records, 144)
  neq = np.flatnonzero(got != want)
  result = {"rows":rows,"k":k,"bytes":int(got.size),"mismatch_bytes":int(neq.size),
    "mismatch_records":int(np.unique(neq//144).size),"exact":bool(np.array_equal(got,want)),
    "first_mismatch":int(neq[0]) if neq.size else None}
  if llama is not None:
    np.savez('/tmp/ds4-capture.npz', input=host, native=got, llama=llama, cpu=want)
    nl = np.flatnonzero(got != llama); cl = np.flatnonzero(want != llama)
    result.update(llama_mismatch_bytes=int(nl.size), cpu_llama_mismatch_bytes=int(cl.size),
      llama_exact=bool(np.array_equal(got,llama)), metadata_native_cpu=int(np.count_nonzero((got[:,:16]!=want[:,:16]))),
      q_native_cpu=int(np.count_nonzero((got[:,16:]!=want[:,16:]))))
  return result

def benchmark(rows: int, k: int, rounds: int) -> dict:
  spec = DS4ProducerSpec(M=rows, K=k); rng = np.random.default_rng(20260830)
  host = (rng.standard_normal((rows, k), dtype=np.float32) * 3.0).astype(np.float16)
  x = Tensor(host, device="NV").realize(); meta = realize_ds4_metadata(x, spec).realize()
  xf, mf = flat_ds4_inputs(x, meta); out = Tensor.empty(spec.records*72, dtype=dtypes.uint16, device="NV").realize()
  pack_fn = lambda: out.uop_program(xf, mf, fxn=emit_ds4_pack_stage(spec))[0].realize()
  # Compile/warm each path before collecting synchronized wall samples.
  meta.realize(); pack_fn(); Device["NV"].synchronize()
  def timed(fn):
    vals=[]
    for _ in range(rounds):
      Device["NV"].synchronize(); t=time.perf_counter_ns(); fn(); Device["NV"].synchronize(); vals.append((time.perf_counter_ns()-t)/1e6)
    return {"samples_ms":vals,"median_ms":statistics.median(vals),"min_ms":min(vals),"max_ms":max(vals)}
  metadata_t=timed(lambda: realize_ds4_metadata(x, spec).realize())
  pack_t=timed(pack_fn)
  combined_t=timed(lambda: (realize_ds4_metadata(x, spec).realize(), pack_fn()))
  from .nv_llama_packed_q4k_pp512_binding import LlamaPackedQ4KPP512Binding
  binding=LlamaPackedQ4KPP512Binding.compile(Device["NV"]); lr=Tensor.empty((spec.records*144+128*144)//4,dtype=dtypes.uint32,device="NV").realize()
  def llama_fn():
    nonlocal lr
    _,lr=x.uop_program(lr,fxn=lambda *_:binding.producer); lr.realize()
  llama_t=timed(llama_fn)
  return {"rows":rows,"k":k,"rounds":rounds,"native":{"metadata":metadata_t,"pack":pack_t,"combined":combined_t},"llama":{"producer":llama_t},
    "launch":{"native_pack_global":(spec.records*72,1,1),"llama_producer_global":(rows,8,1),"llama_producer_local":(128,1,1)},"resources":{"native":None,"llama":None}}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--benchmark",action="store_true"); ap.add_argument("--rounds",type=int,default=9); a=ap.parse_args()
  with open(a.out,"w") as f:
    result={"schema":"nv_q8_ds4_native_producer_gate/v1","status":"PASS"}
    try:
      result["cases"]=[run(1,4096),run(512,4096)]
      result["status"]="PASS" if all(c["exact"] for c in result["cases"]) else "FAIL"
      if a.benchmark: result["benchmark"]=benchmark(512,4096,a.rounds)
    except Exception as e:
      result.update(status="ERROR",error=f"{type(e).__name__}: {e}")
    json.dump(result,f,indent=2,sort_keys=True); f.write("\n")
  print(json.dumps(result,sort_keys=True))
if __name__ == "__main__": main()
