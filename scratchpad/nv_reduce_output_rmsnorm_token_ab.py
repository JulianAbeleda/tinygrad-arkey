#!/usr/bin/env python3
"""Default-off full-token A/B for cooperative REDUCE_OUTPUT RMSNorm."""
import argparse

def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("control","candidate"),required=True)
  ap.add_argument("--out",required=True); ap.add_argument("--nmeas",type=int,default=12); ap.add_argument("--reps",type=int,default=3)
  ap.add_argument("--census")
  args=ap.parse_args()
  # Setup-only authority pin used by the historical native d512 campaign:
  # prefill populates KV through the stable non-fused route; decode remains
  # production-selected and is the only interval measured here.
  import tinygrad.llm.generate as generate
  original_load=generate.load_model_and_tokenizer
  def stable_prefill(*a,**kw):
    from dataclasses import replace
    model,tokenizer=original_load(*a,**kw)
    model.config=replace(model.config, prefill_tc_attn=False, prefill_custom_kernel_attn=False)
    for block in model.blk:
      block.config=replace(block.config, prefill_tc_attn=False, prefill_custom_kernel_attn=False)
    if args.mode == "candidate":
      # Lease only the explicit decode call sites.  Ordinary prefill invokes
      # nn.RMSNorm directly and remains byte-identical to the control arm.
      model._decode_reduce_output_rmsnorm_promoted=True
      for block in model.blk: block._decode_reduce_output_rmsnorm_promoted=True
    return model,tokenizer
  generate.load_model_and_tokenizer=stable_prefill
  from extra.llm_research.decode.decode_runtime_overhead import main as authority
  argv=["--ckpts","512","--nmeas",str(args.nmeas),"--reps",str(args.reps),
        "--warmup-decode","3","--chunk-size","32","--skip-dispatch-diagnostic","--out",args.out]
  if args.census: argv += ["--graph-admission-out",args.census]
  return authority(argv)

if __name__ == "__main__": raise SystemExit(main())
