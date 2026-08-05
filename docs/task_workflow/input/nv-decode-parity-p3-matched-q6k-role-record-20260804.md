# P3 matched Q6_K K/V role record — 2026-08-04

## Claim tested

Measure the *complete currently emitted tinygrad Q6_K K/V role* at the
highest-value 1024x4096 shape against the same packed operand contract as the
llama Q8->Q6 diagnostic.  This is an isolated primitive diagnostic, not token
wall time and not a default-route qualification.

## Construction

`scratchpad/q6k_matched_tinygrad_role_benchmark.py` uses the pinned llama CPU
reference packer only to make a Q6_K weight payload and Q8_1 activation
payload.  Python independently decodes both.  The exact same Q6_K bytes feed
tinygrad's observed `q6k_gen_partial_1024_4096_4`; the current role's required
`Tensor.sum(axis=1).contiguous()` tail is included in the same CUDA graph.

tinygrad's installed input ABI is fp16, unlike llama MMQ's Q8_1 ABI.  The
activation is consequently decoded from the same Q8_1 payload then rounded to
fp16 before entering tinygrad.  This bridge changes any activation element by
at most 0.000244140625.  The result must not be described as an exact
end-to-end llama Q8-producer replacement.

## Measured row

One flocked `DEV=CUDA` RTX 5090 run used five 500-replay CUDA-event arms.
Raw artifact:
`/tmp/tinygrad_q6k_matched_1024x4096.json`, SHA-256
`737bc0d28e0750daff5e35b361df480b27b2943f5b8fb51e99a3d92d446b89f5`.

| implementation | graph nodes | median event time | numeric guard |
|---|---:|---:|---|
| tinygrad partial Q6_K + required external sum | 2 | **42.995 us** | PASS; 7.63e-6 max abs vs fp16-ABI reference |

The two nodes are the Q6 partial kernel and its reduction tail.  The output's
max delta from the pure-Q8 reference is 0.0016363, bounded separately so
compute error is not confused with the ABI bridge.

## Completed exact-llama control

The gate above is now closed by the enhanced exact-cubin oracle.  Its captured
graph has exactly two kernel nodes with one dependency edge, ordered
`q8 -> q6`; this is not a graph that omitted the Q8 producer.  Five
1000-replay CUDA-event arms measured **4.099 us median per graph replay**.
The separately measured direct Q8-only steady-state number is 3.363 us, but
it cannot be arithmetically added to the graph throughput number: direct
kernel launches and repeated graph replays have different launch-amortization
and scheduling behavior.  The oracle's single-replay event samples are noisy
(5.472 us median) and are retained as an observation, not promoted to a
latency claim.

The GPU Q8 payload differs from the CPU packer in 131 bytes, all in the
`s` auxiliary half field.  The MMQ-consumed fields are exact: `d` has zero
mismatches, `qs` has zero mismatches, and independently dequantized Q8 values
have zero maximum absolute difference.  Thus this is not an operand-correctness
failure.

| implementation | complete isolated role / graph | median event throughput | correctness |
|---|---|---:|---|
| exact llama cubins | Q8_1 quantizer -> Q6_K MMQ | **4.099 us** | PASS; 3.58e-6 max abs; graph edge present |
| installed tinygrad route | Q6_K partial x4 -> external sum | **42.995 us** | PASS; 7.63e-6 max abs vs fp16 ABI reference |

The measured isolated diagnostic ratio is **10.49x** (42.995 / 4.099).  It is
large enough to reject the prior working assumption that the Q6 K/V role is a
sub-microsecond residual.  It still is not token wall time, and it does not by
itself identify how much of the token gap belongs to this role after graph
scheduling, cache effects, or the non-identical fp16-vs-Q8 activation ABI.

The compact immutable measurement fields and the raw-artifact identity are in
`docs/task_workflow/output/nv-decode-parity-p3-matched-q6k-role-20260804.json`
(SHA-256 `27474ee96c47351829a98afc61bbd1135f7cf35532c14c1aebd32998e7374e96`).
