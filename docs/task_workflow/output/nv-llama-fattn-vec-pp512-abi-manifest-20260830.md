# llama fattn-vec pp512 ABI audit

The authoritative Phase 3 trace uses the MMA template
`<DKQ128,DV128,ncols1=16,ncols2=4,false,false>`. Its main launch is grid
`(340,parallel_blocks,1)`, block `(32,4,1)`, with `37,120` bytes of shared
memory and `34.944 us` active time. The general fixup launch is grid
`(340,16,4)`, block `(128,1,1)`, with `11.264 us` active time. The previous
one-query CTA transcription and its `16,384`-CTA result are not this contract.

The fixture boundary is fp32 Q and fp16 K/V, with production tensors shaped
`[B,Hq,T,D]` and `[B,Hkv,KV,D]`, plus cache-after storage `[2,B,Hkv,max_context,D]`; any clean-room kernel must
cache-after storage `[2,B,Hkv,max_context,D]`; any clean-room kernel must
explicitly bridge cache offsets and lower-right causal semantics.

The exact vector route is numerically correct but rejected at `322 us/layer`.
Verdict: no mechanical transcription is admitted. The ABI manifest is
diagnostic only; the next valid work is a clean-room query-tiled design gated
by the independent raw fixture and full-output canaries.
