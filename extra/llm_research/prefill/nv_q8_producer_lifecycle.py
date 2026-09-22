#!/usr/bin/env python3
"""Research-only pp512 Q8_1 producer lifecycle ledger.

This does not wire a model route.  It records the exact llama CUDA MMQ packet
ABI and computes which conversions can be shared without changing consumers.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

M, K = 512, 4096
QK8_1, VALUES_PER_RECORD, RECORD_BYTES = 32, 128, 144

def packet_bytes(m=M, k=K):
  assert k % VALUES_PER_RECORD == 0
  return m*(k//VALUES_PER_RECORD)*RECORD_BYTES

def parse_trace_summary(path: Path):
  # nsys stats output may have a notice preamble; find its real CSV header.
  lines=path.read_text().splitlines()
  start=next(i for i,x in enumerate(lines) if x.startswith("Time (%),Total Time (ns),"))
  rows=list(csv.DictReader(lines[start:]))
  out={}
  for r in rows:
    name=r["Name"]
    if "quantize_mmq_q8_1" not in name: continue
    layout="DS4" if "layout)1" in name else "D4" if "layout)0" in name else "D2S6"
    out[layout]={"calls":int(r["Instances"]), "total_ns":int(r["Total Time (ns)"]), "avg_ns":float(r["Avg (ns)"])}
  return out

def lifecycle(trace=None):
  # Qwen3-8B Q4_K_M pp512: final FFN is pruned by llama, hence 35 FFNs.
  consumers={"attention_qkv":108, "attention_o":36, "ffn_gate_up":70, "ffn_down":35}
  # Existing ABI: DS4 may serve Q4 siblings; D4 is a distinct packet.
  producers={"attention_qkv":54, # 18 all-Q4 groups: 1 each; 18 Q4/Q4/Q6: DS4+D4
             "attention_o":36, "ffn_gate_up":35, "ffn_down":35}
  calls=sum(consumers.values()); shared=sum(producers.values())
  measured=parse_trace_summary(Path(trace)) if trace else {
    "DS4":{"calls":214,"total_ns":680651,"avg_ns":3180.6},
    "D4":{"calls":35,"total_ns":177955,"avg_ns":5084.4}}
  total_ns=sum(x["total_ns"] for x in measured.values())
  # Every ABI-preserving reuse here avoids a K=4096 DS4 conversion.
  saved_calls=calls-shared; saved_ms=saved_calls*measured["DS4"]["avg_ns"]/1e6
  base_ms=36.608; new_ms=base_ms-saved_ms
  return {
    "schema":"tinygrad.nv_q8_prefill_producer_lifecycle.v1",
    "workload":{"m":M,"k":K,"layers":36,"final_ffn_pruned":True},
    "abi":{"record_values":128,"record_bytes":144,"qs_bytes":128,"metadata_bytes":16,
      "DS4":"four half2(d,sum), one per 32 values; Q4_K/Q5_K",
      "D4":"four float32 d, one per 32 values; Q6_K",
      "quant":"d=amax/127; q=round(x/d); DS4 sum is raw pre-quant fp32 subgroup sum",
      "layout":"records are transposed by activation column; each record is one column's 128-K slice"},
    "packet":{"bytes_per_512x4096":packet_bytes(),"MiB_per_512x4096":packet_bytes()/2**20,
      "input_fp32_bytes":M*K*4,"compression_vs_fp32":M*K*4/packet_bytes()},
    "observed":measured,
    "ordinary":{"producer_calls":calls,"consumer_calls":calls},
    "abi_preserving_shared":{"producer_calls":shared,"consumer_calls":calls,"avoided_calls":saved_calls,
      "groups":producers,"saved_ms_estimate":saved_ms,"llama_wall_ms_estimate":new_ms,
      "prompt_tok_s_estimate":512/(new_ms/1000),"avoided_ctas":saved_calls*4096,
      "avoided_input_read_bytes":saved_calls*M*K*4,"avoided_packet_write_bytes":saved_calls*packet_bytes(),
      "compressed_dense_lifecycle_ms_estimate":27.263-saved_ms},
    "constraints":["DS4 and D4 records have equal size but incompatible metadata interpretation",
      "Q4 gate and up may share one DS4 packet", "Q/K/V may share one DS4 packet only when all three weights use DS4",
      "mixed Q4/Q4/Q6 QKV needs one DS4 plus one D4 producer unless Q6 consumer ABI is generalized"],
  }

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--nsys-kernel-summary"); ap.add_argument("--output")
  a=ap.parse_args(); row=lifecycle(a.nsys_kernel_summary)
  text=json.dumps(row,indent=2)+"\n"
  if a.output: Path(a.output).write_text(text)
  else: print(text,end="")
if __name__ == "__main__": main()
