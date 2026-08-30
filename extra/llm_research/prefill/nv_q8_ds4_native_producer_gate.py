"""Exactness gate for the scheduler-generated two-stage DS4 producer."""
import argparse, json, os, time
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
  xf, mf = flat_ds4_inputs(x, meta)
  out = Tensor.empty(spec.records*72, dtype=dtypes.uint16, device="NV").realize()
  out, _, _ = out.uop_program(xf, mf, fxn=emit_ds4_pack_stage(spec))
  got = out.realize().numpy().view(np.uint8).reshape(spec.records, 144)
  want = cpu_pack_ds4(host.astype(np.float32))
  neq = np.flatnonzero(got != want)
  return {"rows":rows,"k":k,"bytes":int(got.size),"mismatch_bytes":int(neq.size),
    "mismatch_records":int(np.unique(neq//144).size),"exact":bool(np.array_equal(got,want)),
    "first_mismatch":int(neq[0]) if neq.size else None}

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args()
  with open(a.out,"w") as f:
    result={"schema":"nv_q8_ds4_native_producer_gate/v1","status":"PASS"}
    try:
      result["cases"]=[run(1,4096),run(512,4096)]
      result["status"]="PASS" if all(c["exact"] for c in result["cases"]) else "FAIL"
    except Exception as e:
      result.update(status="ERROR",error=f"{type(e).__name__}: {e}")
    json.dump(result,f,indent=2,sort_keys=True); f.write("\n")
  print(json.dumps(result,sort_keys=True))
if __name__ == "__main__": main()
