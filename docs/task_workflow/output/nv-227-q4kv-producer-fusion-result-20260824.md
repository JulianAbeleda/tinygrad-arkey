# NV 227 push: ordinary Q4/Q4 K/V producer fusion result

Date: 2026-08-24
Base checkpoint: `a8f153e1d`
GPU: RTX 5090 (`NV sm_120`), model: Qwen3-8B-Q4_K_M

## Outcome

The first Q4/Q4 K/V dual-output producer is promoted for the nine ordinary
Q4/Q4 attention blocks on NV `sm_120`. It replaces two vector Q4_K projection
launches with one bit-exact producer while retaining the promoted terminal
K/V cache sink.

The stabilized reps=9 composed bracket recovers `13.919 us/token`, or about
`+0.686 tok/s` in that same session. The conservative no-rollback reps=15
endpoint is `4.518148 ms/token = 221.330 tok/s`, leaving `112.862 us/token`
to 227 (`4.405286 ms/token`). The cross-session change from the preceding
`4.556418 ms/token` endpoint is `-38.269 us/token`, but only the same-session
bracket is attributed causally to this route.

## Native gate

The native control is two installed
`q4k_g3_lanemap_gemv_vec_1024_4096` launches. The candidate is one
`q4k_g3_lanemap_gemv_pair_vec_1024_4096` launch with separate K and V output
buffers.

| row | control pair | candidate pair | change |
| --- | ---: | ---: | ---: |
| hot CUDA-event median | 5.460832 us | 3.731072 us | -1.729760 us (-31.7%) |
| projected over all 18 Q4/Q4 pairs | - | - | -31.136 us/token ceiling |

All 2048 fp32 output words match bit-for-bit. The candidate uses 67 registers,
has no spill loads/stores, and keeps the installed vector per-lane dot-product
association.

## Boundary failure and correction

The first model attempt returned K and V as slices of one `[2,1024]` opaque
output. That shape was numerically correct but failed the semantic K
RMSNorm+RoPE identity proof. Each affected K norm fell from one
`reduce_output_rmsnorm_rope_8_128` call to three scheduler kernels
(`E_2_8_16_4`, `r_8_16_8`, `E_8_2_16_4`). The graph regressed from 516 to
525 nodes and device union was flat (`-1.0 us/token`).

The corrected boundary exposes K and V as two caller-owned writable buffers
from the same promoted program call. Both tensors are direct AFTER outputs, so
the K semantic route and V consumer bind without transports. The isolated
legacy-store graph then becomes 516 to 507 nodes, with node sum
`-22.304 us/token` and union `-22.250 us/token` in the matched artifacts.

## Wall and composition qualification

With the cache sink disabled in both arms, the ordinary pair passes reps=7:

```text
control A  4.562282 ms/token
candidate  4.538827 ms/token
control C  4.558456 ms/token
recovery      21.543 us/token
```

With the shipped producer cache sink enabled in both arms, the graph is:

| row | control | candidate | delta |
| --- | ---: | ---: | ---: |
| nodes | 480 | 471 | -9 |
| node sum | 4303.456 us | 4283.728 us | -19.728 us |
| device union | 4299.750 us | 4281.250 us | -18.500 us |

The first composed reps=7 bracket was correctly retained as a no-go: the
candidate lost to a closing control that was `20.326 us` faster than Control A.
A stabilized reps=9 confirmation then passed both controls:

```text
control A  4.519055 ms/token
candidate  4.498261 ms/token
control C  4.505304 ms/token
midpoint   4.512179 ms/token
recovery      13.919 us/token
```

Every profile and wall arm has an identical token-stream hash.

## Promotion boundary

`decode-q4k-kv-pair-route-policy.json` promotes only `("NV","sm_120")`.
The loader installs admissions only after primitive replacement and after the
shared-Q8 leases are known. It accepts exactly nine ordinary blocks whose K
and V linears are both Q4_K `1024x4096` `attn_kv` roles.

The route deliberately excludes:

- the nine Q4/Q4 pairs behind shared-Q8;
- all 18 mixed Q4/Q6 K/V pairs;
- non-NV targets and non-decode shapes.

`TINYGRAD_Q4K_KV_PAIR_DISABLE=1` restores the two installed single-projection
launches at model load.

## Installed ledger and next lever

The no-rollback installed profile reports:

```text
nodes           471
node sum        4284.784 us/token
device union    4281.750 us/token
overlap            3.755 us/token
pair producers      9
terminal sinks     36
generic stores      0
```

The 227 target still needs `112.862 us/token`; this landing alone cannot cover
it. The next producer-side test is the separate shared-Q8 Q4/Q4 pair emitter:
reuse the existing Q8 provider and fuse its K and V consumers without changing
their exact four-warp partial merge. After that, the 18 mixed Q4/Q6 pairs must
be tested as their own grammar. Timestamp-only independent cache writers are
not ranked ahead of these body-reduction constructions because installed
overlap remains only `3.755 us/token`.

Evidence: `docs/task_workflow/evidence/nv-227-q4kv-producer-fusion-20260824/`.

Verdict: `PROMOTED_ORDINARY_Q4Q4_KV_PAIR_221_330_TOK_S_227_NOT_REACHED`.
