# Llama-style flash wide-load result

## Verdict

The wide/coalesced KV-load substrate passes and is installed for the qualified
NV sm_120 G4 fp16-KV shape. It reverses the scalar prototype's isolated no-go,
remains token-exact in the model, and books a positive unprofiled token wall.

The important result is smaller than the isolated headline: matching llama's
16-byte copy grammar makes the score body fast in isolation, but the full
production graph converts only part of that recovery. The S6 proof recovered
about 41 us/token; the official MAXC1024/S8 bracket recovered 81.468 us/token.

## What changed

The original transcription loaded fp16 K/V scalars. The renderer consequently
generated many narrow transactions and about fifteen times llama's L1 traffic.
The repaired spelling gives each eight-lane group llama's split dimension
ownership and reads each contiguous eight-fp16 slice through an aligned
`uint4` load. Two 16-byte loads cover each lane's 16 dimensions.

The model boundary also required a real substrate fix. Production passes the
KV cache as `AFTER(cache, store)`. A size-changing bitcast of that ordered view
was being materialized as a new uint32 buffer, corrupting the reinterpretation.
The custom-kernel boundary now preserves the zero-copy bitcast and its store
ordering edge. This is generic boundary behavior; it is not tied to the model's
weights or layer count.

## Isolated gates

| gate | scalar S6 prototype | wide S6 prototype | reading |
|---|---:|---:|---|
| native score, per layer | 9.25 us | **4.25 us** | wide grammar reverses the score no-go |
| native score + combine | about 10.4 us | **about 5.5 us** | below the installed pair's roughly 6.5 us |
| L1 traffic | 205.31 MB | **13.62 MB** | matches llama's 13.47-MB regime |
| thread instructions | 0.983 M | **0.799 M** | below llama's retained 0.895 M |
| wide-load rendering | narrow fp16 loads | aligned `uint4` | exact intended copy grammar |

The normal numerical probe differed in only a few fp16 output words at the
accepted rounding scale; zero and dynamic-range probes were bit-exact. Empty
partitions retain PV=0, denominator=0, maximum=-inf.

## Full-model profiled bracket

The reps-7 control/candidate/control bracket used depth 512, physical extent
768, S6, and 32 generated tokens per window. All token-stream hashes matched.

| region, 36 layers per token | installed control | S6 wide candidate | delta |
|---|---:|---:|---:|
| score | about 208.9 us | 215.1 us | about +6.2 us |
| combine | about 101.3 us | 48.5 us | about -52.7 us |
| score + combine | about 310.2 us | **263.6 us** | **about -46.5 us** |
| complete GPU node sum | 4,014.6 us | **3,953.3 us** | **-61.3 us** |

This is the key correction to the isolated story. The score body is not faster
than the installed score under production profiling; the booked flash-island
recovery is primarily the six-part, 128-thread combine. The aligned-load work
is nevertheless enabling: without it, the S6 score body was too slow to retain
the combine win.

The profiled wall was 18.2 us/token better than the reverse-control midpoint,
but its two controls drifted by roughly 392 us/token. That wall field is not
used as the endpoint authority.

## Unprofiled token wall and tok/s

The clean reps-9 bracket used 24-token windows so prelude, warmup, and all nine
windows fit the fixed 768-token graph. Every arm produced the same token hash.

| arm | median latency | throughput |
|---|---:|---:|
| control A | 4.232821 ms/token | 236.25 tok/s |
| candidate | **4.207259 ms/token** | **237.68 tok/s** |
| control C | 4.263686 ms/token | 234.54 tok/s |
| reverse-control midpoint | 4.248253 ms/token | 235.39 tok/s |

The candidate is 40.995 us/token faster than the control midpoint and faster
than both controls. On this bracket that is +2.294 tok/s. Applying only the
causal latency delta to the established 4.166708-ms installed endpoint gives a
projection of 4.125713 ms/token, or about 242.38 tok/s. That is a projection,
not a newly observed installed endpoint.

## Installed S8/MAXC1024 result

Production derives the partition count from physical cache extent: one
128-token partition per extent block. The official MAXC1024 graph therefore
uses S8; the earlier MAXC768 research graph uses S6. Explicit graph geometry,
including the qualified request-horizon S64 route, retains precedence.

The installed-shape reps-9 reverse bracket measured:

| arm | latency | throughput |
|---|---:|---:|
| control A | 4.146328 ms/token | 241.18 tok/s |
| installed candidate | **4.077968 ms/token** | **245.22 tok/s** |
| control C | 4.172544 ms/token | 239.66 tok/s |
| control midpoint | 4.159436 ms/token | 240.42 tok/s |

All token hashes matched. The candidate recovered 81.468 us/token and 4.803
tok/s against the reverse-control midpoint, beating both controls.

The standalone default-path endpoint then measured **4.094502 ms/token =
244.230 tok/s** over nine accepted windows and 144 timed tokens. Relative to
the previous installed 4.166708-ms endpoint, this is an observed +4.232 tok/s.
The refreshed installed profile records 3,954.656 us/token of GPU node sum;
wide score is 222.656 us/token and S8 combine is 50.272 us/token.

## Disposition

The construction has passed the test-before-invest gates:

1. aligned-load code generation and cache-traffic closure;
2. isolated timing and numerical qualification;
3. full-model token-stream equality;
4. profiled device-ledger recovery;
5. reps-9 unprofiled reverse-bracket recovery.

Do not describe the old scalar no-go as a topology wall. It was a turnable load
grammar and boundary-ownership wall, and both missing facts have now been
supplied. Production admission remains fail-closed to NV sm_120, Hq32/Hkv8/
Hd128, fp16 KV, and the qualified MAXC768/S6 or MAXC1024/S8 extents. Other
dense head geometries and physical cache extents remain unqualified.

## Evidence

- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-wide-model-bracket-r2.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-wide-wall-r2.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-wide-substrate-summary.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/s8-wide-wall-installed-r1.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-endpoint-r9.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/s6-substrate-summary.json`
- `extra/llm_research/decode/nv_flash_body_device_timing.py`
- `extra/llm_research/decode/nv_flash_llama_vec_wide_qualification.py`
- `extra/llm_research/decode/nv_flash_llama_vec_wide_wall.py`
