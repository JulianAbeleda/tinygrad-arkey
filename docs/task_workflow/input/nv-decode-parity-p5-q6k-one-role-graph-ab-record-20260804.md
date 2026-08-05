# P5 Q6_K one-role real-token CUDA-graph A/B — 2026-08-04

## Verdict

`PASS_CORRECTNESS_WALL_NEUTRAL_WITHIN_SHORT_SAMPLE_NOISE`.  This closes the
previous ABI uncertainty: an extracted llama Q6_K MMQ can be wired into the
real tinygrad CUDA decode graph while retaining the existing fused downstream
reduction/KV-store consumer.  It does **not** establish a route or parity win.

At d512, the `DEV=CUDA` control using tinygrad's native role and the rewired
A/B generated the identical five-token sequence
`[474, 330, 3817, 3302, 474]`.  After explicitly discarding the first
graph-build/reinstantiate sample, the four-sample medians were 5.622302 ms
(CUDA control) and 5.609985 ms (A/B), a -12.317 us delta.  That is wall-neutral at
this sample size, not evidence of a meaningful speedup.

## Exact scope

The single tinygrad `q6k_gen_partial_1024_4096_4` graph node (call 9, 32 graph
nodes before mutation) was destroyed.  Four nodes replaced it:

- a visible fp16-to-fp32 source adapter;
- llama's exact GPU q8_1 quantizer;
- llama's extracted Q6_K MMQ cubin, launched with the verified `(32,4,1)` block;
- a visible contiguous-to-four-partial scatter.

The original immediate fused reduction/KV-store consumer remained in place.
Its one dependency was rewired from the destroyed native partial node to the
scatter.  The graph therefore increased from 32 to 35 nodes exactly.  The
adapter is source-equivalent to tinygrad's live fp16 activation boundary; it
is intentionally not represented as llama's activation-production path.

## Interpretation

This is the first correctness-passing full-token control for the role that had
the strongest isolated-kernel evidence.  It says that a single Q6 attention
K/V-style role cannot explain the missing decode wall by itself once the
required q8 and layout boundaries are priced in.  The next A/B must select a
role family or repeated population with a predicted token-wall contribution
large enough to clear timing noise; do not promote or alter defaults from this
diagnostic.

The machine-readable result is
`docs/task_workflow/output/nv-decode-q6k-llama-one-role-graph-ab-20260804.json`.
