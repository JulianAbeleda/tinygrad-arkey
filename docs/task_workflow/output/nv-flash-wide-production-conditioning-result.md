# Wide-flash production-conditioning result

## Verdict

Priority 1 is now mechanism-resolved enough to change the optimization target:
the installed wide score kernel is at llama service parity when hot, and the
majority of its production loss is working-set conditioning at the L2-capacity
boundary. The immediate Q/K/V producer chain is not the cause.

This is not a booked endpoint win. The follow-on eviction-policy tests found a
real primitive and a small directionally positive token conversion, but no
candidate cleared the campaign's 50-us/token booking bar.

Decision:
`CACHE_POLICY_TURNABLE_IN_ISOLATION__TOKEN_SCOPE_WALL__NO_BOOKING`

## Exact staged replay

The probe captured the installed MAXC1024/S8 score program and its live layer-0
buffers, then replayed that exact program on native NV HCQ. Every sample first
ran one untimed score to establish the same hot state; only the final score was
timestamped. The fresh production reference was collected in the same run.

| arm | score time | delta from hot midpoint | reading |
|---|---:|---:|---|
| hot A/C midpoint | **4.536 us** | reference | body parity with llama |
| Q + KV completions | 4.512 us | -0.024 us | ruled out |
| full local Q/K/V prefix | 4.544 us | +0.008 us | ruled out |
| 64-MiB read working set | 4.544 us | +0.008 us | below capacity knee |
| 96-MiB read working set | 5.856 us | **+1.320 us** | capacity knee reproduced |
| 128-MiB read working set | 5.776 us | +1.240 us | same cold regime |
| 96 MiB then exact Q/K/V prefix | 5.680 us | **+1.144 us** | production-shaped conditioning |
| fresh production graph | **6.180 us** | **+1.644 us** | authority |

The retained llama score row is about 4.526 us/layer. Thus tinygrad's hot
4.536-us body is already effectively tied. Its fresh production graph instead
pays 1.644 us/layer, or 59.184 us/token across 36 layers.

The read-only capacity construction reproduces 1.320 us/layer in isolation.
When followed by the exact local Q/K/V producer chain, it reproduces 1.144
us/layer, or 41.184 us/token. The Q/K/V chain by itself accounts for only
0.008 us/layer. The remaining production-shaped residual is about 0.500
us/layer, or 18 us/token, and is not assigned to a mechanism by this test.

## Matched llama conditioning discriminator

The same read-only float-stream conditioner was added to the standalone build
of llama's own `flash_attn_ext_vec`. Every observation used the sequence
`flash reheat -> conditioner -> flash`, and CUPTI timed the final flash launch
separately. Two reverse S6 brackets and an S8 geometry check gave:

| implementation and geometry | hot | after 96 MiB | conditioning penalty |
|---|---:|---:|---:|
| llama S6, physical extent 768 | 4.032 us | 4.672 us | **+0.640 us** |
| llama S8, physical extent 768 | 3.968 us | 4.576 us | **+0.608 us** |
| tinygrad installed S8, MAXC 1024 | 4.512 us | 5.808 us | **+1.296 us** |

The llama S6 96-MiB result repeated at 4.672 us, and both hot reverse arms
were 4.032 us. A 64-MiB stream left both implementations at their hot floor;
the capacity knee therefore agrees. Changing llama from six to eight splits
did not recreate tinygrad's larger penalty, so split count is not the cause.

This falsifies both extreme explanations. Llama is not cold-insensitive and
does not preserve all K/V through a hidden cache policy, but tinygrad is about
2.0x more sensitive to the same capacity event. Matching llama's cold
sensitivity would expose about 0.656 us/layer, or 23.616 us/token, worth a
ceiling of about 245.65 tok/s at the current endpoint. This remains exposure,
not booked recovery.

Raw hot times are not used as a cross-runtime verdict here: CUPTI observes
llama's CUDA-runtime kernels, while tinygrad's native NV submission is timed by
HCQ signals and is not surfaced as CUDA kernels in the Nsight trace. The
within-path hot-to-conditioned deltas are the comparable result.

The unresolved mechanism is now narrower: cache allocation/address layout or
generated load/address service, not launch partition count and not an explicit
llama cache policy. The next admissible discriminator is a tinygrad matrix over
MAXC stride, combined versus separately allocated K/V, and base-address color,
with hot/cold L2 and DRAM counters. It must show whether the extra 0.656
us/layer is extra refetch traffic or slower service for the same bytes before a
production implementation is justified.

That discriminator has now run. The installed S8 kernel executes logically
empty upper partitions and issues their wide K/V loads because the validity
predicate masks arithmetic after load formation. A safe active-horizon S6
candidate reduces cold DRAM/L2/L1 traffic and executed instructions by about
25%, preserves the token stream through its 768-token bound, and converts
9.484 us/token (+0.566 tok/s) in a reverse production bracket. Explicit gated
loads, separate K/V allocations, and address coloring did not beat the
active-horizon construction. The result is recorded in
`nv-flash-active-horizon-result.md`; it is below the 50-us booking bar and is
not promoted.

## Translation

At the current 4.094502-ms endpoint:

- the production-shaped cache-conditioning exposure is 41.184 us/token, a
  ceiling of about 246.71 tok/s (+2.48 tok/s);
- the read-only cold-state exposure is 47.520 us/token, a ceiling of about
  247.10 tok/s (+2.87 tok/s);
- the complete hot-versus-production residual is 59.184 us/token, a ceiling
  of about 247.81 tok/s (+3.58 tok/s).

None is claimable until a production cache-policy candidate shortens token
wall. Serial prefetch merely moves the same reads earlier and is not a valid
recovery. The admissible next construction inside this priority is an eviction
or residency policy that protects useful KV lines from the model-wide weight
stream without reducing projection service rate.

## Eviction-policy turnability

The aggregate primitive modeled all 36 layers' depth-512 fp16 K/V state as one
72-MiB read-only footprint and placed a 256-MiB weight-like stream between its
prime and timed reload. Every arm used the native NV HCQ timestamp path.

| policy | reload | recovery of ordinary-stream penalty | result |
|---|---:|---:|---|
| hot control | 11.488 us | reference | L2-resident floor |
| ordinary prime + ordinary stream | 45.024 us | 0% | cold control |
| evict-last prime + ordinary stream | 35.408 us | 28.7% | partial protection |
| ordinary prime + evict-first stream | 11.456 us | 100.1% | primitive pass |
| evict-last prime + evict-first stream | 11.648 us | 99.5% | no added value |

Thus the hardware/compiler primitive is real. The clean theory was to mark
one-use quantized weight loads evict-first and leave K/V on normal priority.

The token-path tests did not preserve the primitive result:

| production scope | control midpoint | candidate | wall delta | tok/s delta | disposition |
|---|---:|---:|---:|---:|---|
| Q/K/V projection weights only | 4.099436 ms | 4.093107 ms | **-6.329 us** | **+0.377** | mechanism-only; below booking bar |
| every installed dense quantized weight consumer | 4.105838 ms | 4.129426 ms | **+23.588 us** | **-1.391** | no-go wall |

All three arms in both brackets produced the same canonical token-stream hash.
The whole-dense attribution capture found essentially no flash conversion: its
fresh production score was 6.158 us/layer versus the prior fresh 6.180-us
reference, only about 0.8 us/token across 36 layers. At the same time,
evict-first loads increased service time in the large gate/up and FFN-down
consumers. A matrix is one-use at token scale, but packed headers and words can
still have useful intra-kernel cache reuse; blanket streaming destroys part of
that reuse.

Priority 1 is therefore closed at the current policy granularity. Reopen only
with a policy that distinguishes truly one-touch lines from reused packed
metadata/data inside a kernel, or with native-NV persisting-window support that
protects K/V without changing weight-load service. The current renderer switch
is research-only, closed by default, and not promoted.

## Evidence

- `docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r1.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r3.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/priority1-conditioning-r3.json.profile.jsonl`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/l2-priority-r1.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/l2-streaming-qkv-wall-r1.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/l2-streaming-whole-dense-wall-r1.json`
- `docs/task_workflow/evidence/nv-flash-wide-conditioning/l2-streaming-whole-dense-profile-r1.json`
- `docs/task_workflow/evidence/nv-llama-flash-matched-conditioning-20260826/exact-read/`
- `docs/task_workflow/evidence/nv-llama-flash-matched-conditioning-20260826/tinygrad-exact-r1.json`
- `extra/llm_research/decode/nv_flash_wide_conditioning_probe.py`
- `extra/llm_research/decode/nv_flash_l2_priority_turnability.py`
- `extra/llm_research/decode/nv_flash_l2_streaming_weight_wall.py`
- `extra/llm_research/microbench/llama_fattn_vec_iso.cu`
- `docs/task_workflow/output/nv-flash-active-horizon-result.md`
