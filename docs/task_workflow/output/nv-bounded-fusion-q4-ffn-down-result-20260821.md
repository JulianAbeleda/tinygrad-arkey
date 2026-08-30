# NV bounded-fusion and Q4 FFN-down result

Date: 2026-08-21
Branch: `nvidia-bringup-20260731`, HEAD `6570abc02`
Target: RTX 5090, `DEV=NV`, sm_120, Qwen3-8B-Q4_K_M, d512
Scope: `nv-split-phase-pdl-causal-design-review-scope-20260820.md`
sections 8.2 and 8.3, plus the H8 recomposition.

Status: measurement record. No production runtime, renderer, scheduler, or
model file was changed. Every GPU arm ran as a fresh process under
`/tmp/gpu-bench.lock`. Notation: `O` observed, `I` inferred, `U` unmeasured.

## 1. Locked endpoint

| quantity | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4723.214 | 4005.709 | +717.505 |
| S1, Q to O | 1152.250 | 517.916 | +634.334 |
| device union | 4732.500 | 3892.777 | +839.723 |
| node sum | 4738.496 | 5020.797 | -282.301 |
| overlap mass | 5.996 | 1128.020 | -1122.024 |

These numbers are the locked inputs; nothing in this packet changes them.

## 2. Correction adopted first

The scope and the original H8 row quoted llama Q4 FFN-down at 11.776
us/node. That is attention-O's value. The 08-13 correction pins FFN-down at
19.232 us/node. The current four-warp body median is 20.900 us/node, so the
remaining body ceiling is:

```text
18 x (20.900 - 19.232) = 30.016 us/token
```

not the previously stated 164.224 us. The dependent current-head docs were
updated with this correction; historical 08-12 records were left untouched as
their own evidence.

## 3. Cell 8.2: Q4 FFN-down

The four-warp fp16 geometry route is already promoted at HEAD. The remaining
legal candidates were already measured and close against named blockers.

| candidate | endpoint/topology result | blocker | verdict |
| --- | --- | --- | --- |
| quad-u128-smem load pattern | in-loop 34.48 vs control 26.24-26.29 us/node, wall +38.3 us/token | standalone winner does not survive the real loop | NO-GO |
| DP4A resadd, all 18 blocks | wall -25.6 us/token (+0.50%), topology clean | full-logit relative L2 2.98e-3 exceeds the 1e-3 gate | PRECISION_BLOCKED |
| producer-folded DP4A (`scalar_q8_packet`) | wall +45 us/token | folded Q8 producer cost cancels the datapath win | NO_GO_WALL |
| owned-fp32-boundary Q8 | layer-8 singleton wall -4.28 us/token | full-route precision still above gate; pre-wired full gate vacuous after M2a/M2b landed | PRECISION_BLOCKED |

Evidence: `nv-q4-down-quad-re-census-20260813.md`,
`nv-q4-down-dp4a-resadd-18block-gate-20260814.md`,
`nv-q4-down-fp16-geometry-promotion-20260815.md`,
`nv-q4-down-producer-owned-q8-gate-20260813.md`,
`nv-q4k-ffn-down-owned-boundary-rebracket-record-20260812.md`.

Verdict: `CLOSED_BLOCKED`. The body is 1.668 us/node above the corrected
llama floor. Every legal path that would recover it is blocked by precision
or by an added producer that costs more than it saves. No promotion is
warranted and none was made.

## 4. Cell 8.3: bounded FFN reduce-output fusion

Harness:
`extra/llm_research/decode/nv_ffn_reduce_output_ab.py`. Control = production
default (q/k promoted, FFN site closed). Candidate = only
`_decode_reduce_output_ffn_rmsnorm_promoted` opened on the model and all 36
blocks.

### 4.1 CPU census gate

Evidence:
`docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/ffn-cpu-census.json`.

| family | before | after | delta |
| --- | ---: | ---: | ---: |
| fused `reduce_output_rmsnorm_1_4096` bodies | 0 | 36 | +36 |
| FFN ordinary reduce (`r_1024_4`) | 73 | 37 | -36 |
| FFN ordinary epilogue (`E_1024_4`) | 181 | 145 | -36 |
| weight materializations | 0 | 0 | 0 |

The site swap itself is net -36 programs with zero weight materializations
(O). The capture also shows the documented CPU-only GEMV merge, an additional
net -36 from the attention-o reduce folding into ffn-down, pre-explained by
the census harness `_gemv_shift_explanation` (O). It does not appear on the
GPU graph and is not a q/k change.

### 4.2 Exact-logits and topology gate

Evidence:
`docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/gpu-logits-gate.json`.

| gate | value |
| --- | --- |
| stacked full-logit fp32 SHA-256 | identical `c36d7268...` |
| token stream SHA-256 | identical `52aa34c5...` |
| bitwise equality | true |
| `r_16_256` delta | 37 -> 1 (-36) |
| `E_32_32_4_f14a5cc0` delta | 37 -> 1 (-36) |
| `reduce_output_rmsnorm_1_4096` delta | 19 -> 55 (+36) |
| q/k fused bodies | 36/36 unchanged |
| weight materializations | 0 unchanged |
| program count | 595 -> 559 (-36) |

The one remaining `r_16_256`/`E_32_32_4` pair is the final output norm, which
the per-block FFN knob correctly does not touch. Only the three expected
program families changed. Verdict: `PASS`.

### 4.3 Fresh reverse wall bracket

Evidence:
`docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/gpu-wall-bracket.json`.
Control / candidate / control, settled continuous windows, 4 reps x 24 tokens,
identical token stream hash `1e73e557...` in all three arms.

| arm | median ms/token |
| --- | ---: |
| control A | 4.7340135 |
| candidate | 4.8014143 |
| control B | 4.7223320 |

| quantity | value |
| --- | ---: |
| candidate minus control A | +67.401 us/token |
| candidate minus control B | +79.082 us/token |
| candidate minus control midpoint | +73.242 us/token |
| promotion bar | -50 us/token vs both controls |
| verdict | `NO_GO_WALL` |

The candidate deletes 36 programs and is bitwise exact, but each new 1_4096
body plus its output materialization costs more at this head than the cheap
ordinary reduce/epilogue pair it replaces. This repeats, at the current
4.723 ms baseline, the 08-13 net-negative outcome. The FFN site stays closed.

## 5. H8 recomposition, corrected

Evidence:
`docs/task_workflow/output/nv-h8-fusion-composition-20260821.json`.
The script reproduces the locked critical path 4249.216 us exactly.

| zero-cost scenario | CP us | ceiling us |
| --- | ---: | ---: |
| base | 4249.216 | 0 |
| residual only | 4100.480 | 148.736 |
| reduce only | 4045.536 | 203.680 |
| vocab tail only | 3935.584 | 313.632 |
| Q4-down at corrected floor | 4219.200 | 30.016 |
| all four | 3553.152 | 696.064 |

The composition is exactly additive: alternate-path takeover/interference is
0.000 us in this DAG. Even if every currently legal fold converted at zero
cost, the corrected envelope leaves 21.441 us of the 717.505 us wall gap and
exceeds the 634.334 us S1 gap by 61.730 us. Measured conversion is far below
the ceiling everywhere:

- FFN reduce-output fusion: +67.4/+79.1 us/token slower (this packet, O).
- q/k reduce-output fusion: +38.7/+43.2 us/token at its old baseline (O).
- vocab top1 fusion: NO_GO_WALL (O).
- Q4 DP4A: -25.6 us/token but precision-blocked (O).
- copy-free native RMSNorm: +12.5 to +17.1 us/token (O).

H8 is therefore refuted as a closed-wall mechanism. The binding limit is
wall conversion, not path interference and not raw ceiling size (O/I).

## 6. What this closes and what remains open

Closed now:

- 8.2 Q4 FFN-down: no legal body change promotes; precision and producer
  cost are the named blockers.
- 8.3 bounded FFN fusion: exact and clean, but wall-negative at HEAD.
- H8 fusion-only sufficiency: refuted by measured endpoints; the corrected
  zero-cost envelope is 696.064 us.

Still open (`U` unless noted):

- The above-floor DRAM mass has no hardware DRAM-counter attribution yet.
- The 1122 us overlap delta is schedule structure, not candidate kernel
  speed; the PDL packet already showed the exposed S1 support serialization
  cannot be expressed by the current name-pinned arm.
- The edge-aware runtime-hook construction is drafted in
  `nv-edge-aware-pdl-runtime-hook-scope-20260821.md`; its staged gates are
  the next construction, not another endpoint re-run.
- One continuous graph versus the current five replay groups remains
  unmeasured as a construction (`U`).

No performance claim is made. The 717.505 us wall gap and 634.334 us S1 gap
carry forward unchanged.

## 7. Evidence index

- CPU census:
  `docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/ffn-cpu-census.json`
- Exact logits/topology:
  `docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/gpu-logits-gate.json`
- Wall bracket:
  `docs/task_workflow/evidence/nv-bounded-fusion-ffn-20260821/gpu-wall-bracket.json`
- Corrected H8 recomposition:
  `docs/task_workflow/output/nv-h8-fusion-composition-20260821.json`
- Harness:
  `extra/llm_research/decode/nv_ffn_reduce_output_ab.py`
