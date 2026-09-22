# NV d512 flash single-stage reopen scope

Date: 2026-08-05
Status: CPU/static construction result; implementation and GPU promotion are closed.
Question: can the four d512 split partials for each head be produced and combined inside one
ordinary-UOp cooperative workgroup, removing global partial traffic and the combine launch?

## Finding

**Yes, it is expressible and resource-feasible as a bounded d512 candidate.** It is not established
as faster. The mapping is one 32-lane warp per `(split, grouped_query_head)` pair. Qwen3-8B has
`G=Hq/Hkv=4`; with `S=4`, one KV-head workgroup has 16 warps / 512 threads. The old route launches
`Hkv*S = 32` 128-thread workgroups and then a combine kernel; the candidate launches `Hkv = 8`
512-thread workgroups and writes the final attention output directly. Total score/PV thread count
stays 4096, but block count and scheduling flexibility fall by 4x.

The executable construction oracle is
`extra/llm_research/decode/flash_single_stage_plan.py`; hermetic gates are in
`test/unit/test_flash_single_stage_plan.py`. For the admitted shape it proves a bijection over all
16 `(split,head-within-GQA)` owners, a legal 512-thread block, a 32,768-byte upper bound for four
simultaneous fp16 K/V tiles, an 8,320-byte fp32 split-state exchange, and a combined conservative
41,088-byte shared-memory bound. This is below the scope's 96 KiB admission cap.

## Exact semantic contract

Do not replace the current math with a different online reduction tree. Each owner warp must run
the existing tile body for its split and materialize exactly `(acc[128], den, max)` into LOCAL
memory. After one workgroup barrier, the `split=0` warp for each grouped head reads split states in
the existing combine order `s=0,1,2,3`: find global max in that order, form each split weight,
accumulate numerator and denominator in that order, divide, and cast to fp16 only when the current
M5 output contract requests it. This preserves the current split association; full-logit equality
remains a required empirical gate because renderer scheduling and contraction can still change bits.

## Why this is not another closed geometry sweep

The prior 36-value and 189-row tile sweeps kept the two-program/global-partial decomposition. This
candidate changes ownership and the communication boundary: the old `pout[Hq,S,Hd+2]` buffer and
`flash_fused_gmax_combine*` program disappear. It therefore attacks launch plus global handoff, not
`LANES`, `TK`, stage width, or split size values. The maximum directly owned saving is the measured
combine row (~3.39 us x 36 = 122 us/token), plus partial stores/loads and any cache-locality gain.
The earlier ~157 us score-kernel delta is not automatically recoverable; no accounting may debit it
until native wall evidence does.

## Relationship to llama.cpp

The inspected CUDA source (`ggml-cuda/fattn-vec.cuh`, `flash_attn_ext_vec`, 128 threads) performs
online attention per parallel KV block. `launch_fattn` in `fattn-common.cuh` deliberately launches a
separate `flash_attn_ext_vec_fixup` when `parallel_blocks > 1`. Thus llama's measured score+combine
pair validates the split/fixup semantics, but does **not** demonstrate this single-workgroup design.
This is a tinygrad-specific structural hypothesis, not a transcription of llama assembly.

## Phase gates

1. **Ordinary-UOp construction (CPU only).** Add a new emitter; do not modify the legacy emitter.
   It must render on NV and the established HIP/Metal render arms, contain one GLOBAL KV-head axis,
   LOCAL lane/warp axes `(32,16)`, LOCAL partial state, at least two explicit barriers (tile reuse and
   pre-combine), and exactly one direct output store family. No global `pout` argument is allowed.
2. **Hermetic semantics.** Compare the candidate's split-state merge with the current tile+combine
   reference on normal, all-zero, large-dynamic-range, ragged `Tc=513`, and split-boundary cases.
   Require finite output and the existing dtype/layout contract. AST topology must prove one emitted
   program versus two and no global partial role.
3. **Native isolated gate (GPU authorization required).** Same inputs, reverse launch order, >=1000
   iterations after warmup. Require exact/approved numerical contract and candidate time below
   `score + combine`; separately report score-equivalent time and launch removal. Any resource launch
   rejection, spill explosion, or occupancy below one resident block/SM is NO-GO, not a hardware verdict.
4. **One-block full-logit lease.** Closed-default route only for the exact d512 shape. Require finite
   full logits, token/argmax equality, numeric threshold fixed before timing, and topology `-1`
   program/layer with no adapter. Then reverse A/B wall. Scale only after >=50 us/token same-session
   wall recovery.

## Risks and cheapest falsifiers

- 512 threads and the per-warp accumulator may produce high register pressure or spills. Render and
  inspect resource metadata before a model run.
- Four-way simultaneous tile staging uses 32 KiB; if the renderer allocates multiple live generations or
  adds padding beyond 96 KiB, stop. A sequential split loop is semantically possible but destroys
  split parallelism and is a separate candidate.
- Eight large blocks may underfill the 170-SM RTX 5090 more than 32 smaller blocks. The isolated
  native gate is decisive; program-count savings alone do not qualify it.
- UOp barriers must be uniform. No barrier may sit behind `warp_active`, split-owner, token-valid, or
  ragged-tail predicates.
- At longer contexts `S` is much larger; this mapping is intentionally d512-bounded. Do not generalize
  it or alter the production split policy without a new resource and occupancy proof.

## Hard stops

No route flip, no default change, no tolerance relaxation, no GPU run without root authorization,
and no booking of the 122 us launch ceiling or the 157 us score delta as wall recovery. Failure to
render is a substrate result; failure to launch is a candidate resource result; neither is proof that
single-stage attention is impossible.

## CPU construction addendum

Phase 1 construction now exists as the closed-default
`flash_single_stage_d512_kernel` in `tinygrad/llm/flash_decode_attention.py`. It has no descriptor,
route-policy row, admission hook, or model call site. Its public ABI is exactly `(out, q, cache)`;
there is no global partial argument. Both NV CUDA C and gfx1100 HIP render arms complete without a
GPU or compiler invocation. The NV render proves:

- `__launch_bounds__(512)`, decomposed as local dimensions `(32,4,4)`;
- grid extent 8 (one workgroup per KV head);
- two fp16 LOCAL arrays of 8192 elements each and one fp32 LOCAL array of 2080 elements, exactly
  41,088 bytes total;
- uniform barriers after each staged tile and before the ordered combine;
- one direct fp16 output store guarded to the four split-0 owner warps.

The emitter retains the legacy eight `TK=16` tile iterations per 128-token split, fdot2 + staged
warp-sum score path, and online `(acc,den,max)` updates. Each of the 16 owner warps writes that exact
state to LOCAL memory. The split-0 warp then traverses split states in increasing `0..3` order with
the legacy global-max pass followed by the legacy weighted numerator/denominator pass. The CPU
association oracle independently gathers this same order from the warp-owner layout.

Hermetic result: `7 passed` across `test_flash_single_stage_plan.py` and
`test_flash_single_stage_emitter.py`; `git diff --check` is clean for this lane. This settles
construction/render feasibility. It does not settle native compilation resource metadata, register
spills, numerical equality, or wall performance; those remain Phase 3 GPU gates.

## Native isolated qualification result

Authority: `docs/task_workflow/output/nv-flash-single-stage-microgate-20260805.json`. The probe ran
under `timeout` and `/tmp/gpu-bench.lock` on native `DEV=NV`; it did not touch the model route.

Correctness **PASS**: fp32 and fp16 outputs are bitwise identical to the included legacy tile plus
combine pair on normal seeded data, all-zero data, and a large deterministic dynamic-range case.
All outputs are finite and every `max_abs` is `0.0`. The harness realizes and copies the legacy
authority before constructing/realizing the candidate; this ordering is load-bearing because an
earlier pending-graph resource-inspection draft allowed allocator reuse to invalidate the held
comparison. Resource inspection is now a separate fresh graph.

Native compilation succeeds, but the resource increase is substantial:

| program | registers | shared bytes | local bytes |
| --- | ---: | ---: | ---: |
| legacy score/PV | 40 | 10,240 | 576 |
| legacy combine | 36 | 2,176 | 576 |
| single-stage candidate | 64 | 43,136 | 576 |

Synchronized reverse included-cost timing, 200 graph replays per sample and five samples per arm:

| arm | median us/graph |
| --- | ---: |
| legacy control A | 63.615 |
| single-stage B | 146.517 |
| legacy control C | 64.279 |
| legacy midpoint | 64.012 |
| candidate delta | **+82.505** |

Candidate samples span 123.643-159.984 us; neither the fastest candidate sample nor the median is
competitive with either control. Verdict: **NO-GO** for the 512-thread single-stage mapping. The
candidate is semantically exact, so the result specifically falsifies the assumption that retaining
the same total score/PV thread count while collapsing 32 small blocks into eight large blocks would
preserve effective parallelism. The 64-register, 43-KiB blocks and eight-block grid expose too little
device parallelism; eliminating one launch/global partial handoff cannot repay it.

Per the phase gates, there is no model integration, full-logit run, route policy, default change, or
wall booking. A 256-thread two-splits-per-warp design is not an automatic follow-up: it serializes
two legacy split chains inside each warp and therefore requires a separate association-preserving
emitter and cheap isolated gate. It should rank below score instruction mapping or a design that
retains at least the legacy 32-block population.
