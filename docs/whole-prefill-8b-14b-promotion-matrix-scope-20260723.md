# Final whole-prefill 8B/14B promotion matrix scope

Status: scoped, not runnable as a candidate promotion matrix  
Audit revision: `6c3673360`  
Canonical timing owner: `extra/qk/prefill_whole_synced.py`

## Decision

Do not create another benchmark harness. Extend the existing whole-prefill
authority only after production policy selects a proven native attention
lowering.

The current harness already owns the correct production closure:

- concrete `start_pos` enters the model-owned per-position TinyJit;
- warmstart projection schedules remain installed;
- compilation and capture happen before timed replay;
- the device is synchronized around every timed burst;
- 8B and 14B share one model/profile policy surface;
- raw per-chunk samples and device profile events are retained.

The current report is not a final shared-attention promotion authority. Its
`shared_attention_attribution()` deliberately reports
`selected_lowering: ordinary_sdpa`, `fusion_proven: false`, and
`promotion_eligible: false`. The dense projection route census does not prove
which attention program executed.

## Current gaps

| Requirement | Current state | Required delta |
|---|---|---|
| Full-model production closure | Present | Reuse unchanged |
| Synced TinyJit replay | Present | Reuse unchanged |
| Contexts 512/1024/2048/4096 | Whole-length output exists | Measure every 512-token chunk; do not interpolate unmeasured continuation chunks |
| Repeated samples | Three burst samples by default | Promotion profile requires at least 10 synchronized samples per chunk and reports median/p10/p90 |
| Native attention fired | Hardcoded ordinary SDPA | Join content-addressed compiler captures to every timed attention program |
| Projection route fired | Candidate-set census exists | Preserve it and require complete dense-role ownership |
| Candidate/fallback A/B | No attention selector in harness | Add diagnostic-disable baseline and proof-required candidate modes |
| Full numeric gate | External optional quality JSON only | Add same-input, fresh-state full-output candidate/comparator artifact |
| Attention roofline | Not separated | Attribute fp16 QK/PV time and compute/byte facts by program identity |
| Whole-model roofline | Wall time only | Account for the executed 8B fp16-overlay or 14B packed-Q4 substrate separately |
| Whole wall reconciliation | Profile events are summarized by name | Join typed program roles and reconcile attributed device time to synchronized wall |

## Smallest reusable harness change

Make one versioned extension to `prefill_whole_synced.py`; do not add a parallel
timing loop.

1. Add a `promotion` run profile in `prefill_harness.py`.
2. Set `K=1`, `warmups>=3`, `rounds>=10`, and measure all start positions
   `0,512,1024,1536,2048,2560,3072,3584`.
3. Keep each round as one synchronized TinyJit replay. Do not hide variation by
   averaging eight replays inside one sample or selecting the minimum.
4. Retain the existing authority profile unchanged for compatibility.
5. Add `--attention-mode fallback|candidate`.
6. `fallback` may only disable an otherwise admitted attention route.
7. `candidate` must never force admission. It must require the immutable model
   policy to select native attention and fail if any expected layer falls back.
8. Add `--attention-proof`, `--numeric-gate`, and `--paired-artifact` joins.
9. Add a typed program-role manifest to the report and partition the already
   collected profile events by program identity, not substring guesses.
10. Emit schema `prefill-whole-promotion-matrix.v1`; do not overload the current
    `prefill-whole-synced-authority.v1` meaning.

No renderer or scheduler change belongs in this harness slice.

## Exact matrix

The final report has eight paired rows.

| Model profile | Weight substrate | Hq/Hkv/G/Hd | Whole prompt |
|---|---|---|---:|
| `qwen3_8b_q4k_m_gfx1100` | resident fp16 projection overlay | 32/8/4/128 | 512 |
| `qwen3_8b_q4k_m_gfx1100` | resident fp16 projection overlay | 32/8/4/128 | 1024 |
| `qwen3_8b_q4k_m_gfx1100` | resident fp16 projection overlay | 32/8/4/128 | 2048 |
| `qwen3_8b_q4k_m_gfx1100` | resident fp16 projection overlay | 32/8/4/128 | 4096 |
| `qwen3_14b_q4k_m_gfx1100` | bounded packed Q4 projection route | 40/8/5/128 | 512 |
| `qwen3_14b_q4k_m_gfx1100` | bounded packed Q4 projection route | 40/8/5/128 | 1024 |
| `qwen3_14b_q4k_m_gfx1100` | bounded packed Q4 projection route | 40/8/5/128 | 2048 |
| `qwen3_14b_q4k_m_gfx1100` | bounded packed Q4 projection route | 40/8/5/128 | 4096 |

Every row compares native-attention candidate and ordinary-SDPA fallback while
holding the model, weights, projection route, token input, clocks, process
settings, and commit fixed.

## Timing protocol

Use separate fresh processes for candidate and fallback, then repeat the pair
in alternating order. One process must not mutate a captured model from one
attention mode into the other.

For each side:

1. Verify the model file content identity and model profile.
2. Load once with the same deterministic seed.
3. Materialize the same deterministic token sequence.
4. Warm/capture every required concrete start position outside timing.
5. Confirm no compilation occurs during sampled rounds.
6. Synchronize before and after each single replay.
7. Collect at least 10 positive samples per chunk.
8. Report raw samples, median, p10, p90, CV, and spread.
9. Sum measured chunk medians for each whole length. Do not interpolate.
10. Run pre/post GPU health checks and retain clock-pin provenance.

Promotion uses medians. Minimum-of-burst remains a diagnostic field only.

## Native attention route proof

A request flag, semantic boundary, kernel name, WMMA macro, or profile-event
substring is not route proof.

The candidate artifact must join compiler-owned captures by canonical program
identity. For every model profile and measured start position it must include:

- model profile and exact `B,Hq,Hkv,G,T,KV,Hd,dtype,causal` geometry;
- selected lowering identifier and semantic attention owner identity;
- canonical graph hash, source hash, binary hash, and compiler commit;
- one fused attention call with role-attributed QK and PV WMMA source and ISA;
- complete allocation census with zero full score/probability buffers;
- final VGPR, SGPR, LDS, private/scratch, and spill counts;
- launch grid/local geometry and layer call count;
- proof that every expected layer used that native program family;
- proof that no ordinary-SDPA attention program executed in the candidate run.

The fallback artifact must prove the inverse: ordinary SDPA executed and no
native candidate call did. Projection route census must be identical across the
pair.

## Full numeric gate

Numeric qualification is outside timed replay and uses fresh, equivalent model
state per side.

For each model and whole length:

1. Start from an empty KV cache in a fresh process.
2. Feed identical token IDs sequentially through every 512-token chunk.
3. Compare the complete returned output for every chunk, not one sampled lane.
4. Record max absolute error, max relative error, relative L2, finite count,
   output shape/dtype, and candidate/comparator hashes.
5. Require identical greedy token output for a fixed continuation.
6. Join the existing whole-model dNLL/quality artifact.
7. Reject missing prefix chunks, stale-cache reuse, different token inputs, or
   different projection routes.

Tolerance must be fixed before measurement and justified for fp16 activation
rounding. A candidate cannot set its tolerance from observed errors.

## Separate roofline accounting

### Common fp16 attention roofline

Attention is fp16 activation work for both model routes. For each measured
attention invocation record:

```text
attention_flops = 4 * B * Hq * T * KV * Hd
attention_bytes = bytes(Q) + bytes(K) + bytes(V) + bytes(O) + measured auxiliary traffic
attention_oi = attention_flops / attention_bytes
attention_roof = min(empirical_fp16_wmma_flops, empirical_hbm_bytes * attention_oi)
attention_achieved = attention_flops / attributed_attention_device_time
```

Do not include a full score/probability tensor in candidate bytes; its presence
is already a residency failure. Use empirical device ceilings captured in the
same clock regime, not marketing peaks.

### Route-specific whole-model roofline

Keep this separate from attention.

- 8B accounts for the actually executed resident fp16 projection overlay and
  its activation traffic.
- 14B accounts for packed Q4 weight bytes, metadata/scales, dequant work, fp16
  activations, and its exact generated projection programs.
- Both include common fp16 attention compute exactly once.
- Neither may apply Q4 byte intensity to the fp16 attention kernel.

The report must provide total model FLOPs, executed weight bytes, activation
bytes, attributed device time by role, synchronized wall time, achieved rate,
applicable roof, and percent of roof.

Attributed device time must reconcile to wall time within a predeclared bound.
Any unattributed or overlapping graph time is reported explicitly rather than
assigned to attention or Q4.

## Artifact schema minimum

Each paired matrix row requires:

```json
{
  "profile": "qwen3_8b_q4k_m_gfx1100",
  "whole_length": 4096,
  "candidate": {
    "raw_chunk_samples_ms": {},
    "whole_median_ms": 0.0,
    "whole_tok_s": 0.0,
    "attention_program_census": {},
    "projection_route_census": {},
    "resources": {},
    "rooflines": {}
  },
  "fallback": {},
  "numeric_gate": {"status": "PASS"},
  "paired_delta": {"latency_percent": 0.0},
  "promotion": "PASS"
}
```

Top-level provenance must include git commit/dirty state, model SHA256, proof
SHA256, compiler/source/binary hashes, environment, clock state, device facts,
sample policy, thresholds, and raw artifact paths.

## Command shape after the candidate is admitted

The current command remains the timing owner. The promotion profile supplies
all concrete chunk positions explicitly:

```bash
DEV=AMD JIT=1 PROFILE=1 PYTHONPATH=. .venv/bin/python \
  extra/qk/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --model-profile 8b --mode promotion --attention-mode candidate \
  -K 1 --warmups 3 --rounds 10 \
  --start-positions 0,512,1024,1536,2048,2560,3072,3584 \
  --whole-lengths 512,1024,2048,4096 --max-context 4096 \
  --pin-clock --logits-only --attention-proof <proof.json> \
  --numeric-gate <numeric.json> --artifact <candidate.json>
```

Run the same command with `--attention-mode fallback`, the same projection
`--require-route`, and a distinct artifact. Repeat for `--model-profile 14b`.
Then join the four candidate/fallback artifact pairs into one matrix.

These future flags are intentionally not implemented while the selected
lowering is ordinary SDPA. Candidate mode must fail closed until immutable
policy admission and compiler captures exist.

## Promotion gate

All eight rows must pass:

- full numeric and quality gates;
- complete native candidate and fallback route census;
- no candidate score/probability materialization;
- QK and PV WMMA attributed in source and ISA;
- no spills/private memory regression and acceptable VGPR/LDS occupancy;
- at least 10 synchronized samples per measured chunk;
- candidate whole-prefill median faster beyond the predeclared noise threshold;
- candidate absolute attention device time lower beyond noise;
- attention and whole-model roofline efficiency do not regress;
- no decode correctness or promoted decode performance regression.

One failed model/context row blocks global promotion. A bounded geometry may be
retained as research only if its exact domain is explicit and production still
fails closed elsewhere.

## Current blockers

1. Production attribution selects `ordinary_sdpa`; no distinct candidate matrix
   can be measured honestly.
2. Current authority start positions interpolate several continuation chunks.
3. Default authority records three burst samples and selects the minimum.
4. Projection route census has no attention program identity.
5. Compiler capture and profile events are not joined by a typed program-role
   manifest.
6. No same-input full-output candidate/fallback model artifact exists.
7. No reconciled attention-versus-route-specific whole-model roofline artifact
   exists.

Until blocker 1 is removed by a proven scheduler-native lowering, the correct
status is `NO-GO`, not an empty or duplicated benchmark run.
