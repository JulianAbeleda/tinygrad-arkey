# Pure vs S9 GEMM-route gap — session handoff (2026-07-12)

Handoff for continuing the "why is pure prefill slower than the 4.4k S9 line" investigation.
Read this before re-deriving anything — several intuitive theories were **measured and
falsified** this session; don't repeat them.

## The question

8B Qwen3 prefill, ctx512. The historical **S9 / hybrid** route (hand-ASM backend atoms,
`prefill_pipe_role_selective_generated`) hits ~4.4k tok/s. The **pure / generated** route
(`prefill_wmma_lds_dbuf_generated`, tinygrad-scheduler-generated GEMMs
selected via the BoltBeam candidate set) is stuck ~3.3–3.5k. Goal: explain and close the gap.

## THE reliable measurement (trust this)

Same-session, pinned, single-variable A/B. Only differ by whether the BoltBeam candidate set
is loaded (candidate set OFF = S9 hand route; ON = pure/generated route). Everything else
(attention, softmax, norm, LM head) is identical.

```
env: PYTHONPATH=. DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 [+BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_PATH=...]
cmd: python3 extra/qk/prefill_whole_synced.py --model .../Qwen3-8B-Q4_K_M.gguf \
       --model-profile qwen3_8b_q4k_m_gfx1100 --mode authority -K 8 --warmups 4 --rounds 3 \
       --start-positions 0 --whole-lengths 512 --max-context 4608 --pin-clock
```

| route | family | pinned ctx512 | tok/s |
|---|---|---:|---:|
| HAND (S9)  | `prefill_pipe_role_selective_generated` | **124.9 ms** | 4099 |
| PURE (gen) | `prefill_wmma_lds_dbuf_generated` | **155.9 ms** | 3285 |

**Gap = ~31 ms, and it is 100% the GEMM route** (single variable). The `route_attribution.
prefill_route_family` confirms the flip actually happened — always verify this, don't trust flags.

## Per-role attribution (pinned, single-variable, trust this)

Build single-role candidate sets (one entry each from the full `candidate-set.json`) so exactly
one role runs generated and the other three fall back to S9. Baseline = all-S9 (125.4 ms).

| role generated | pinned ms | delta vs all-S9 |
|---|---:|---:|
| attn_qo | 140.3 | **+14.8 ms** |
| ffn_down | 139.0 | **+13.6 ms** |
| attn_kv | 134.9 | +9.5 ms |
| ffn_gate_up | 130.0 | +4.6 ms |
| all four | 155.9 | +30.5 ms |

Per-role deltas sum to 42.5 ms vs the 30.5 ms full gap → **sub-additive** (cross-kernel overlap
in the graph absorbs ~12 ms; each role's marginal cost in the full route is less than isolated).

## THE mechanism (real in-model kernels, clock-independent — trust this)

Hooked `AMDProgram.__init__` during a real model run to read each GEMM kernel's descriptor
(`group_segment_size` = LDS, `private_segment_size` = scratch) + VGPR via
`extra/qk/mmq_compile_evidence.parse_amdgpu_metadata`. Script: see "Repro" below.

| role | S9 (hand) kernel | PURE (candidate) kernel |
|---|---|---|
| attn_qo | LDS=**1** (no LDS staging) | LDS=**40960**, vgpr=188, occ ≤8 waves |
| ffn_down | LDS=**1** | LDS=40960, vgpr=188, occ ≤8 |
| attn_kv | LDS=**1** | LDS=40960, vgpr=188, occ ≤8 |
| ffn_gate_up | LDS=**40960** | LDS=40960, vgpr=188, occ ≤8 |

**The pure/candidate route forces one 40 KB-LDS cooperative "buffer2" kernel on ALL four roles.
The S9 route uses that LDS kernel only for ffn_gate_up; for attn_qo/attn_kv/ffn_down it uses a
leaner NON-LDS kernel (LDS≈0).** The per-role gap tracks this exactly: the three non-LDS-hand
roles are where pure loses (+14.8/+13.6/+9.5), and the one both-LDS role (ffn_gate_up) is nearly
matched (+4.6). The 40 KB LDS + 188 VGPR pins the candidate kernel to ≤8 waves/SIMD occupancy;
a non-LDS kernel is not LDS-constrained → more occupancy → better latency hiding for these shapes.

Caveat: `parse_amdgpu_metadata` returned VGPR=None for the S9 hand kernels (note format differs;
parse failed). We have hand **LDS** (from descriptor, reliable) but not hand VGPR/occupancy.

## The actionable fix (proposed, partially evidenced)

Route policy, not codegen: **do not apply the 40 KB-LDS buffer2 candidate kernel to
attn_qo/attn_kv/ffn_down; use the leaner non-LDS kernel there (as S9 does), keep the LDS kernel
only for ffn_gate_up.** Evidence already in hand: `candidate set = {ffn_gate_up only}` runs at
130.0 ms (+4.6) while all-four is 155.9 (+30.5) — i.e. keeping only ffn_gate_up on the LDS kernel
already recovers ~26 ms. Next confirmation step: verify `{ffn_gate_up only}` pinned lands near the
S9 125 ms band, then make it the route policy and re-run the full pinned sweep (512/1024/2048/4096).

## Theories tested and FALSIFIED this session — do NOT reopen

1. **"Pure GEMM kernels are slower per-dispatch."** DEBUG=2 per-kernel times suggested pure GEMMs
   were *faster* (58 vs 41k GFLOPS). This was a **clock-ramp artifact** — DEBUG runs are NOT
   pinned (GFLOPS ramps 35k→41k/58k within a run). Per-kernel DEBUG timing across two runs is
   invalid. Ignore it.
2. **"Wait-amortization is the cause."** ISA trace (`extra/qk/prefill/kernel_lifecycle_trace.py`)
   shows generated has 3.19 waits/WMMA vs hand 0.39, P8 FAIL vs PASS. But the causal test refuted
   it: sweeping `loc` (LOCAL), throughput went 26.3→32.0 TFLOPS while waits/WMMA stayed **fixed at
   3.02**. Throughput moves via cooperative-LDS/occupancy, NOT wait count. Also the ISA trace uses
   a generic DBUF schedule, not the actual in-model kernels — it's a proxy. Wait-amortization is
   not the demonstrated lever.
3. **"Isolated GEMM gap = 1.95x."** The schedule-search microbench (cold/cache-resident) said
   hand/gen GEMM ratio ~1.95x. This does NOT reconcile with the whole-model 1.27x at 79% GEMM
   share (would predict hand=5656 tok/s vs actual 4413). The real in-model GEMM ratio is ~1.34x.
   Isolated microbench overstates the gap — do not use it for magnitude.
4. **"LM head is the priority lane."** Refuted earlier (see
   `docs/lane-a-lm-head-packed-refutation-20260712.md`): LM head is 7% of device time, near its
   resident-fp16 floor; packing it is a +87 ms regression.
5. **Semantic-metadata attribution.** `ProfileGraphEntry.metadata` is always `None` in real
   captures (graph batching drops per-op metadata). Role tagging via `role_metadata` is inert for
   the graph export. Attribution must be name/shape-based (see
   `extra/qk/graph_profile_attribution.py`; `PROVEN_NAMES` corrected this session).

## Measurement pitfalls (RDNA3 gfx1100 / this harness)

- **Pin clocks for any timing.** Unpinned runs vary ~10%; DEBUG runs are effectively unpinned
  (they ramp). Use the harness `--pin-clock`; the pin only covers the timed rounds, NOT DEBUG dumps.
- **Per-kernel in-model timing is broken.** The whole-prefill runs as an HCQGraph → emits
  `ProfileGraphEvent` (dispatch list laid end-to-end, cannot show real gaps), NOT per-kernel
  `ProfileRangeEvent`. `PROFILE=1` `device_profile.device_ms` = 0 for graph runs. There is no
  clean per-kernel wall-time tool for the graph path — use whole-model single-variable A/B instead.
- **No single tool gives ISA waits AND GPU time for the same kernel.** The trace tool
  (`AMD:ISA`, static) and the schedule search (GPU timing, plain-Opt path) build different
  kernels. Don't correlate them.
- **The schedule-search table is clobbered for candidate roles.** `_install_candidate_matmul`
  (`extra/qk/prefill_graph_gemm_route.py:153`) overwrites `_WARMSTART_OPTS` with the
  candidate_context right before codegen. So tuning `prefill_v2_schedule_table.json` has zero
  effect on candidate-covered roles. The candidate schedule (mechanism B, geometry via
  `candidate_context`) is what runs, not the Opt table (mechanism A).
- **Selecting routes:** `PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` + candidate set = pure; same without
  candidate set = S9. `PREFILL_ROUTE=direct_packed` needs `DEVICE_IN_FUNCTION_BUG=1
  ALLOW_DEVICE_USAGE=1` or it asserts inside a Function.

## Repro pointers

- Route A/B + per-role: candidate sets built by filtering `bench/prefill-pure-full-kernel/
  multirole-buffer2-candidate-set-v1/candidate-set.json` to one `entries[]` element each.
- Real-kernel resources: monkeypatch `tinygrad.runtime.ops_amd.AMDProgram.__init__` to record
  `name, group_segment_size, private_segment_size` + `parse_amdgpu_metadata(lib)`, then
  `runpy` the authority harness (K=1, warmups 2, rounds 1). Filter names `E_4_*` (pure GEMMs) and
  `prefill_gen_sched_gemm_*` (S9 GEMMs).
- ISA trace: `extra/qk/prefill/kernel_lifecycle_trace.py --kind hand-lds2 | --active-generated
  --kind generated` on `--target AMD:ISA:gfx1100 --json` (instruction counts / P8, no timing).

## Current head state

Committed this session (pushed to master): LM-head refutation + docs, corrected `PROVEN_NAMES`,
device busy-time union, `PREFILL_LM_HEAD_DIRECT` knob, semantic-metadata carrier (inert in export).
No route-policy change made — the LDS-strategy fix above is proposed, not yet implemented.
