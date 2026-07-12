# Q6_K LM-head Prefill Authority

The real 8B output projection during prefill is `M=512, N=151936, K=4096`.
It is selected after the transformer stack and before `[:, -1, :]` slicing, so
the existing `M=1` Q6_K GEMV artifact is not an authority for prefill.

Use the existing whole-prefill authority with the direct packed Q6_K route:

```sh
PREFILL_ROUTE=direct_packed \
PREFILL_Q6K_PACKED_LOAD=1 PREFILL_DIRECT_OUT=1 \
PREFILL_DIRECT_TENSORS=output.weight \
PREFILL_GRAPH_GEMM=0 PREFILL_V2=0 \
python3 extra/qk/prefill_whole_synced.py --mode authority -K 8 \
  --warmups 4 --rounds 3 --whole-lengths 512 --pin-clock
```

The report must join: route census (`prefill_q6k_direct_generated` and
`lm_head`), generated kernel identity
`q6k_gen_prefill_direct_out_151936_4096_512`, compiler resource metadata,
full-output correctness against the non-packed Q6_K reference, and pinned
median timing. The current host route test pins the descriptor identity; no
AMD result is claimed until this exact command completes with those joins.

The existing Q6_K generated-coop artifact is decode/GEMV (`M=1`) and must not
be used as the LM-head prefill timing authority.
