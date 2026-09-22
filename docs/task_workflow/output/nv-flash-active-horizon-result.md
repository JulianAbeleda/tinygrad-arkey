# Flash active-horizon result

## Verdict

`ACTIVE_HORIZON_PRIMITIVE_PROMOTED__GRAPH_BUCKETING_INVEST_PASS__S6_TO_768_S8_AFTER`

The remaining wide-Flash cold sensitivity has a concrete contributor. The
installed kernel derives eight 128-token partitions from the 1024-token cache
capacity, even when the active context is near 512. Its `token < Tc` predicate
masks the arithmetic result after the wide K/V loads have been formed, so
logically empty upper partitions still issue memory traffic and execute their
kernel body.

Llama does not use a hidden cache policy. Its d512 graph uses a smaller
physical service horizon and six partitions. The generic principle is to
compile/dispatch the next safe 128-token active-horizon bucket, while retaining
the real cache stride. It is not to specialize on a model or prompt depth.

## Test matrix

Every microtest retained MAXC 1024 unless explicitly labeled otherwise. The
96-MiB conditioner is the same read-only float stream used by the matched
llama/tinygrad experiment.

| arm at Tc 512 | hot | conditioned | semantic result | decision |
|---|---:|---:|---|---|
| installed-shape S8 | 4.544 us | 5.296-5.408 us | reference | control |
| S8 with explicitly gated K/V loads | 4.064 us | 6.048-6.064 us | raw bit-exact | no-go cold service |
| separately allocated K/V, S8 | 4.640 us | 5.536-5.584 us | raw bit-exact | no-go |
| separate K/V plus gated loads | 4.128 us | 6.368-6.384 us | raw bit-exact | no-go |
| fixed-stride S4, bound 512 | 3.320-3.328 us | 4.832-4.880 us | exact at Tc <= 512 | primitive pass |

Cache base-address offsets from 0 through 1.5 MiB moved conditioned S8 time
within roughly 5.22-5.50 us, but did not approach the bounded candidate and did
not produce a portable color rule. Address coloring is not promoted.

## Semantic horizon sweep

The sweep establishes the required boundary behavior rather than assuming it:

| logical Tc | smallest exact bucket | hot | conditioned |
|---:|---:|---:|---:|
| 513 | S5 / 640 | 3.592 us | 4.864 us |
| 576 | S5 / 640 | 3.600 us | 4.864 us |
| 640 | S5 / 640 | 3.504 us | 4.784 us |
| 641 | S6 / 768 | 4.224 us | 5.280 us |
| 704 | S6 / 768 | 4.192 us | 5.280 us |
| 768 | S6 / 768 | 4.088 us | 5.120 us |

S4 becomes numerically wrong at Tc 513, S5 becomes wrong at Tc 641, and S6
remains exact through Tc 768. Thus the admissible rule is
`ceil(Tc / 128)` partitions, or a graph horizon known to be at least that
large. A smaller fixed partition count is not generally safe.

## Counter proof

The exact generated S8 and S6 kernels use the same 56 registers/thread and
2080 bytes of shared memory. NCU reports:

| cold metric | S8 / 1024 | S6 / 768 | change |
|---|---:|---:|---:|
| DRAM bytes | 4.232 MB | 3.183 MB | -24.8% |
| L2 bytes | 17.460 MB | 13.176 MB | -24.5% |
| L1 bytes | 22.086 MB | 16.564 MB | -25.0% |
| executed instructions | 1.069 M | 0.802 M | -25.0% |
| standalone hot time | 4.248 us | 3.825 us | -10.0% |

S4/512 halves the control traffic and instruction count: 2.134 MB DRAM, 8.882
MB L2, 11.043 MB L1, and 0.500 M instructions. The cause is therefore both
extra bytes and extra instructions from excess physical partitions, not a
cache-set-only effect.

## Production conversion

The bounded S5 graph was installed under a closed research lease and confirmed
to emit `flash_vec_llama_score_pv_32_128_5_widekv16` plus the S5 combine. It
preserved the token stream but recovered only 1.95 us/token in its short
reverse bracket, so it did not clear both controls.

The broader S6/768 lease ran 144 timed tokens per arm, remained below its
semantic bound, and passed both controls:

| arm | wall | tok/s |
|---|---:|---:|
| control midpoint | 4.097748 ms/token | 244.037 |
| bounded S6 | 4.088264 ms/token | 244.603 |
| delta | **-9.484 us/token** | **+0.566** |

All three token-stream hashes were identical. This is a real wall conversion.
The bounded primitive, closed research lease, and measurement substrate are
promoted as reusable infrastructure. The fixed S6 route is not made the
unconditional production default because it becomes incorrect at Tc 769. The
installed endpoint therefore remains 4.094502 ms/token / 244.230 tok/s until
the generic graph-bucket selector is separately tested and admitted.

## Implementation implication

A production implementation must select or cache graphs by an active-context
horizon bucket and advance to the next bucket before Tc crosses its bound. It
must not install S6 unconditionally for a 1024-token cache, because that becomes
incorrect at Tc 769. The measured S6 wall result supports investing in that
generic graph-bucketing substrate, but the expected endpoint value is presently
about half a tok/s, not the earlier 1.4-tok/s synthetic ceiling.

## Graph-bucketing investment test

The remaining horizons were tested with full 117-token reverse brackets rather
than extrapolated from isolated kernel timing:

| candidate and tested interval | wall delta | tok/s delta | decision |
|---|---:|---:|---|
| S5/640, Tc 512-636 | -5.408 us/token | +0.324 | wall pass |
| S6/768, early interval through Tc 663 | -9.484 us/token | +0.566 | wall pass |
| S6/768, Tc 640-764 | -20.299 us/token | +1.206 | wall pass |
| S7/896, Tc 768-892 | +3.797 us/token | -0.225 | no-go |

All token-stream hashes matched their reverse controls. The S7 result agrees
with its weak conditioned microgate: removing one physical partition is not
automatically faster in the production lifecycle. Conversely, S6 remains a
strong pass across both halves of its safe range. S5 is a real primitive pass,
but S6 is both safe and faster in the same early interval, so the measured
service policy is **S6 through Tc 768, then S8**. It is not the mechanically
smallest `ceil(Tc/128)` graph at every token.

Using the current 4.094502-ms endpoint as a common translation baseline:

- while S6 is active near the d512 endpoint, the measured early bracket is
  worth about +0.57 tok/s;
- averaged over Tc 513-768, the two S6 brackets imply about 14.892 us/token,
  or roughly +0.89 tok/s for that 256-token region;
- amortized over a complete 512-token continuation from 513 through 1024,
  including an unchanged S8 second half, the estimate is 7.446 us/token,
  moving 244.230 to about **244.675 tok/s (+0.445)**.

This is a pre-installation return estimate, not a booked endpoint. It does not
charge one-time graph capture/compile cost or an unmeasured transition cost.
The implementation investment is bounded because the model already owns and
selects distinct ordinary and alternate-split Flash TinyJits per token. The
required extension is a typed S6 wide-Flash graph variant with a 768 semantic
bound and an S8 fallback, followed by the standard hash, graph-census, and
reverse wall qualification. No S5 or S7 production graph is justified by the
current service results.

## Evidence

- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/matrix-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/matrix-r2.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/matrix-tc*.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/bounded-counters-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/bounded-s6-counters-r1.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/profile-s5.jsonl`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/wall-s5-r9.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/wall-s6-r9.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-matrix-tc769.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-matrix-tc832.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-matrix-tc896.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-wall-s5-full-r9.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-wall-s6-late-r9.json`
- `docs/task_workflow/evidence/nv-flash-kv-layout-matrix-20260826/invest-wall-s7-full-r9.json`
- `extra/llm_research/decode/nv_flash_kv_layout_matrix.py`
- `extra/llm_research/decode/nv_flash_bounded_counter_probe.py`
- `extra/llm_research/decode/nv_flash_bounded_wall.py`
