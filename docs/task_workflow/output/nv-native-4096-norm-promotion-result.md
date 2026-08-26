# Native 4096 RMSNorm promotion and 240+ result

## Outcome

The selective native 4096-wide RMSNorm route is promoted on NV sm_120 and the
bounded dense d512 token path reaches **240.612 tok/s** in the decisive
candidate/control/candidate bracket.

The win came in two accounting steps:

1. native lowering removed the ordinary reduction/epilogue pairs at the FFN,
   output, and already-direct attention sites;
2. attention-route completion allowed the remaining 19 non-shared attention
   norms to use the same native primitive instead of being intercepted by the
   older reduce-output route.

Q/K head norms remain on their fused norm+RoPE/cache path. The 17 shared-Q8
attention blocks retain their fused norm/provider ownership.

## Selective primitive qualification

The initial native route is intentionally site-filtered:

| site | native route |
|---|---|
| attention 4096 norm | enabled, except shared provider owns its fused site |
| FFN 4096 norm | enabled |
| output 4096 norm | enabled |
| Q head norm | disabled |
| K head norm | disabled |

The independent Q/K bracket was exact but lost 1.792 us/token. The 4096 route
removed 35 nodes and reduced device union by 12.250 us in its first qualified
form. A candidate/control/candidate confirmation booked 12.317 us/token.

The native warp-geometry sweep tested 1, 2, 4, 8, and 16 warps. Every output
was bit-exact; the installed 16-warp geometry remained the fastest captured
graph replay. No geometry change was promoted.

## Attention completion mechanism

The post-primitive lifecycle census exposed a missed population:

```text
38 native 4096 norm bodies
17 fused norm + shared-Q8 provider bodies
19 legacy reduce_output_rmsnorm_1_4096 bodies
```

The global reduce-output policy intercepted those 19 attention sites after
the native flag had been selected. The corrected decision lets native
attention sites bypass the global reduce-output marker while preserving a
shared-Q8 lease's stronger fused ownership.

After the change:

```text
56 native 4096 norm bodies
17 fused norm + shared-Q8 provider bodies
 0 legacy reduce_output_rmsnorm_1_4096 bodies
```

The profile changed from 419 to 418 scheduled nodes and device union moved
from 4,091.000 to 4,025.250 us/token: **65.750 us** recovered in the profile
domain. The exact Q/K norm+RoPE/cache populations remained unchanged.

## Decisive wall bracket

The control sets `TINYGRAD_NATIVE_ATTN_NORM_COMPLETION_DISABLE=1`, restoring
only the 19 legacy attention bodies. All other promoted routes remain active.

| arm | ms/token | tok/s |
|---|---:|---:|
| candidate A | 4.154789 | 240.686 |
| rollback control | 4.219059 | 237.020 |
| candidate C | 4.157339 | 240.539 |
| candidate midpoint | **4.156064** | **240.612** |

Recovery versus rollback control is **62.995 us/token**. All token hashes
match across 432 timed tokens. Both candidate arms independently exceed 240.
The 62.995 us wall recovery agrees with the 65.750 us profile-union reduction.

## PDL follow-up

The new native producer legitimately reopened one topology test. A narrow
native-norm-to-gate/up PDL arm produced exactly 36 armed pairs, all real data
edges and no incidental pairs. The consumer was prelaunched with an entry
wait, but the timestamp census measured no resident execution overlap.

The r7 bracket provisionally recovered 23.726 us/token. The reps=9
candidate/control/candidate confirmation collapsed to only 2.888 us/token.
PDL is therefore not installed. The construction exists, but its launch-shadow
recovery is wall-neutral under the qualification gate.

## Current position

| authority | latency | throughput | relative to 240 | relative to fresh llama official |
|---|---:|---:|---:|---:|
| tinygrad candidate midpoint | 4.156064 ms | 240.612 tok/s | 10.603 us faster | 134.343 us / 8.10 tok/s behind |
| llama official | 4.021721 ms | 248.711 tok/s | 144.946 us faster | reference |

The campaign has crossed 240 but has not reached llama. The new frame remains
wall-relative: future work must shorten the 4.156 ms token, not chase a moving
competitor number.

## Evidence

- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/wall-r9-reverse/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/profile/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/geometry/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/native-to-gemv-pdl-census/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/native-to-gemv-pdl-r9/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/attention-native-completion/`
- `docs/task_workflow/evidence/nv-native-4096-norm-promotion-20260826/attention-native-completion-wall-r9/`
