# NV vector-load reopen result

Date: 2026-08-24  
Repo: `/home/ubuntu/tinygrad-arkey`  
Branch: `nvidia-bringup-20260731`, base HEAD `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090 (`sm_120`), locked SM 2790 / memory 14001 MHz  
Model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, single-token decode

## Finding

The loads pay. The earlier no-go was a false negative caused by a scheduler
boundary-copy regression that the structural census did not count. After the
copy is removed, single-projection Q4_K vector loads are bit-exact and recover
`66.56-88.36 us/token` in two independent depth-512 production brackets; a
depth-128 bracket recovers `74.60 us/token`. The vector spelling is now the
NV sm_120 default, with `TINYGRAD_Q4K_SCALAR_LOAD=1` as the rollback/control.

This does not reach 240 tok/s. The conservative accepted endpoint is
`4.658910 ms/token = 214.642 tok/s`, leaving `492.244 us/token` to the
`4.166667 ms` target.

## What invalidated the roofline conclusion

The old experiment did change all 83 intended Q4_K projection bodies, but the
vector residual-add kernel had a different name:

```text
scalar: q4k_g3_lanemap_gemv_epi_resadd_4096_4096
vector: q4k_g3_lanemap_gemv_vec_epi_resadd_4096_4096
```

`_elide_residual_transports` recognized only the exact scalar prefix. The
vector path therefore scheduled one extra fp32[4096] identity transport at
each block boundary. The old structural census observed admitted opaque
program families, not final scheduled invocation cardinality, so its claim of
"zero new copy kernels; same node count" was wrong.

Installed profile evidence makes the cancellation visible:

| path | scheduled calls | `E_32_32_4` calls | copy row us | node sum us |
| --- | ---: | ---: | ---: | ---: |
| scalar control midpoint | 560 | 38 | 55.024 | 4508.128 |
| vector, broken boundary | 597 | 75 | 96.384 | 4495.968 |
| vector, fixed boundary | 560 | 38 | 55.200 | 4460.512 |

The broken vector bodies themselves recovered `51.24 us`, but the 37 hidden
copies cost `41.36 us`, leaving a small net result vulnerable to wall noise.
With the semantic residual-add match fixed, device node sum recovers
`47.616 us` and device union recovers `48.375 us` versus the scalar-control
midpoint.

The regression is covered for both scalar and vector spellings at two points:
declared output must survive precompiled function substitution, and the final
compiled consumer topology must not contain the fp32 identity transport.

## Cold counters: same bytes, higher rate

The original document asked whether wider loads could raise DRAM rate without
changing bytes. They do. These are isolated production-CUDA renderings built
with nvcc sm_120a and measured by Nsight Compute with explicit
`--cache-control all`; they are causal counter probes, not installed NAK wall
timings.

| shape | arm | DRAM bytes | duration us | DRAM % | instructions |
| --- | --- | ---: | ---: | ---: | ---: |
| 4096x4096 | scalar | 9,453,312 | 11.168 | 48.22 | 5,238,784 |
| 4096x4096 | vector | 9,452,800 | 9.184 | 58.66 | 4,784,128 |
| 1024x4096 | scalar | 2,375,424 | 5.632 | 24.02 | 1,309,696 |
| 1024x4096 | vector | 2,375,168 | 4.640 | 29.35 | 1,196,032 |

The byte deltas are one or two 256-byte sectors, effectively zero. Both shapes
execute `8.68%` fewer instructions, complete `17.6-17.8%` sooner in the cold
counter replay, and sustain a higher fraction of peak DRAM throughput. Equal
bytes do not imply equal time: the scalar instruction/load-issue path was
limiting the achieved streaming rate before the physical DRAM ceiling.

This also corrects Phase 5's attribution error. Its `1.35-1.50 TB/s` evidence
was for gate/up and down kernels, not for the affected Q/K/O bodies, and it
could not establish that the vector bodies were already at their roofline.

## Production wall qualification

Every arm is a fresh process, 32 timed tokens per repetition, candidate
sandwiched between scalar controls. Acceptance here requires the candidate to
beat both controls, not only their midpoint.

| bracket | reps | control A ms | vector ms | control C ms | recovery us | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| depth 512 A | 9 | 4.728365 | 4.634587 | 4.717525 | 88.358 | pass |
| depth 512 B | 9 | 4.735032 | 4.658910 | 4.715910 | 66.561 | pass |
| depth 128 | 9 | 6.663164 | 6.594072 | 6.674174 | 74.597 | pass |

All arms produce token-stream hash
`28b0923439dde9076100800bfaed6a9a8a7e00e396691776a16514a609e0543a`.
The depth-128 samples advance through context and therefore trend upward, but
the vector arm is faster at every corresponding sample position.

The independent gate/up question is also closed: at reps=9 its vector arm is
`4.714229 ms` against scalar controls `4.729900/4.753132 ms`, recovering
`27.287 us/token` with the same token-stream hash. The installed gate/up
vector default remains justified.

## Installed result and the 240 ledger

The no-environment census now reports 83 vector and zero scalar single
projections, plus 36 vector gate/up kernels. The fresh installed device profile
has 560 scheduled calls, 38 `E_32_32_4` calls, `node_sum=4463.360 us`,
`union=4460.875 us`, and only `2.485 us` of overlap.

Against the retained llama PDL-off device ledger, the residual is:

| pool | delta us/device-token |
| --- | ---: |
| GEMVs: K/V + down + Q + gate/up + O | 359.967 |
| flash score + combine | 140.187 |
| vocab main + tail | 65.372 |
| total node-sum delta | 585.170 |

The individual GEMV residuals are K/V `104.127`, down `82.999`, Q `71.452`,
gate/up `52.238`, and O `49.151 us`. The historical `~440 us` was an aggregate
llama delta across affected and unaffected bodies; it was never the expected
recovery from changing these 83 Q4_K calls.

The conservative current endpoint is the slower of the two accepted
depth-512 vector medians: `4.658910 ms/token` (`214.642 tok/s`). The remaining
wall gap to 240 is `492.244 us/token`. Cross-session subtraction from the old
`4.697289 ms` authority is not booked as a candidate gain; only same-session
reverse brackets are.

## Which lever is next

Of topology, overlap, and byte reduction, the actionable next lever is
**quantized-kernel streaming rate/topology**, now aimed at the untouched down
and shared-Q8/Q6 families and at Q/K completion overhead:

- Existing overlap is only `2.485 us`, and prior multi-queue/PDL attempts did
  not produce an accepted wall gain. Overlap remains a large theoretical
  opportunity, but there is no validated construction to promote.
- Weight matrices are already streamed once per token. Material byte reduction
  would be powerful, but no bit-exact representation or reuse candidate is
  currently identified; activation/intermediate byte savings are small beside
  the weights.
- This reopen proves that codegen can raise achieved DRAM rate at fixed bytes.
  The remaining GEMV pool is `~360 us`, led by K/V and down, so rate/topology
  work has the largest measured and technically reachable surface.

That pool alone is not enough to guarantee 240 on the conservative wall
endpoint. Reaching 240 likely also requires closing much of flash
score/combine (`140 us`) and vocab (`65 us`); a single gate/up topology change
cannot cover the `492 us` wall gap.

## Changes and verification

- `tinygrad/schedule/__init__.py`: residual transport elision matches the
  semantic Q4_K residual-add family, including vector names.
- `tinygrad/llm/decode_routes.py`: vector single-projection loads are the
  sm_120 default; `TINYGRAD_Q4K_SCALAR_LOAD=1` restores scalar.
- `test/unit/test_projection_epilogue_boundary_census.py`: scalar/vector
  boundary regressions.
- `test/unit/test_llm_decode_routes.py`: target-scoped default and rollback.
- Counter/bracket/census/reconciliation tools were updated to preserve scalar
  controls and understand the promoted vector names.

Verification: `37 passed` across the complete two affected unit-test modules;
focused route/boundary run `5 passed`; all edited Python files compile.

## Evidence

The structured summary and hashes are under
`docs/task_workflow/evidence/nv-vector-load-reopen-20260824/`. Raw source
artifacts remain in the referenced profile, wall, cold-counter, census, and
gate/up evidence directories listed by that manifest.

Verdict: `PROMOTED_VECTOR_LOADS_240_NOT_REACHED`.
