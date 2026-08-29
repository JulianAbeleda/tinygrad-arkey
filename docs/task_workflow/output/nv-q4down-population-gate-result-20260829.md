# Q4-down population gate — 2026-08-29

The saved-z FP16→Q8 producer is exact for the real blk.4/type12 K=12288
fixture (q, scales, and raw sums all bit-exact). The next gate attempted to
compose that packet with the existing compiler Q4_K primitive at M=512,
N=4096, K=12288.

## Result

**STOP: failed before population timing.** The ordinary compiler-generated
matmul cannot construct the requested role geometry. During postrange
precontract construction it raises:

`current atomic staging requires at least two tensor-core K steps`

The requested K=12288 and current tile/precontract selection produced fewer
than two tensor-core K steps for this path. Therefore no full-output oracle,
readonly/sentinel gate, FP16 comparator, population timing discriminator, or
model integration was run. There is no performance claim for Q4-down.

The attempted command was the existing production gate with
`--role blk.4.ffn_down.weight --n 4096 --k 12288`; it failed while compiling
the first ordinary generated matmul, before a result buffer was launched.

## Next substrate required

Add a role-matched Q4-down compiler geometry/provider that supplies at least
two tensor-core K steps (or a separate static full-output oracle), then rerun
the gates in order: independent oracle → readonly/sentinel/finite → matched
FP16 comparator → population timing → only then model integration.
