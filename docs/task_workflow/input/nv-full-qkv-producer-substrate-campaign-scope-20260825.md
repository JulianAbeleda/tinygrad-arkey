# NV full-population QKV producer substrate campaign scope

Date: 2026-08-25

Repository: `/home/ubuntu/tinygrad-arkey`

Branch: `nvidia-bringup-20260731`

Recorded starting commit: `5e7f36945215ebc4ed2efbaf887ef241dafce7fd`

Target: Qwen3-8B-Q4_K_M dense single-token decode at depth 512 on RTX 5090
through the tinygrad `NV sm_120` backend.

## 1. Campaign objective

Build and qualify one reusable full-grid Q/K/V producer substrate across the
complete 36-layer attention population. Q4/Q4/Q4 is the proof grammar, not
the campaign endpoint. The campaign ends only when every population below has
either:

1. a bit-exact, wall-positive route included in a composed installed bracket;
   or
2. a complete, repaired, evidence-backed wall that identifies why it cannot
   contribute.

The final performance objective is the physical token wall. `240 tok/s` is a
useful checkpoint, not an optimization rule and not the stopping condition.
The longer-term objective is proximity to the measured hardware/model wall.

## 2. Thesis being tested

The full Q plus K/V projection pool accounts for approximately 529 MB and
529 us per token in the current device ledger. The pool is split into small
projection bodies whose aggregate service rate is below the large streaming
bodies. A full-grid producer may improve the pool by:

- retaining 4,096 Q CTAs rather than collapsing Q parallelism;
- attaching one K-or-V row to the first 2,048 Q CTAs;
- sharing one launch and, where applicable, one activation provider;
- preserving each projection's exact arithmetic association;
- exposing outputs directly to their final owners without slice/copy kernels;
- and, where necessary, composing terminal cache ownership so body recovery
  survives producer-completion boundaries.

The original rate model gave gross whole-pool recoveries of about 89, 152, or
199 us/token at 1.2, 1.4, or 1.6 TB/s. Those are roofline ceilings, not booked
gains. From the demonstrated clean 4.2506 ms/token endpoint, they correspond
to approximately 240.3, 244.0, or 246.8 tok/s. From a variable 234 tok/s
endpoint, the same recoveries correspond to approximately 239.0, 242.6, or
245.4 tok/s. Therefore “the substrate opens 240+” is conditional on broad
population coverage and realized token-wall recovery; it is not a claim that
one nine-block grammar alone reaches 240.

## 3. Population ledger

The campaign owns exactly these four populations:

| ID | activation / arithmetic grammar | blocks | current state |
| --- | --- | ---: | --- |
| S44 | shared-Q8 Q4/Q4/Q4 | 9 | full-grid wall pass, +16.119 us/token; research admission only |
| O44 | ordinary-fp16 Q4/Q4/Q4 | 9 | body pass, production wall negative because recovery does not survive current completion contract |
| S46 | shared-Q8 Q4/Q4/Q6 | 8 | K/V pair exists; full-grid triple not implemented or qualified |
| O46 | ordinary-fp16 Q4/Q4/Q6 | 10 | full-grid triple not implemented or qualified |

The counts sum to all 36 attention blocks. A scope that tests fewer blocks is
a phase result, not campaign completion.

### Current measured accounting

| surface | measured result | booking status |
| --- | ---: | --- |
| S44 full-grid wall | +16.119 us/token | technically passed; not yet composed/default-installed |
| O44 isolated cold span | +2.380 us/group, about +21.420 us/token ceiling | diagnostic only |
| O44 production GPU union | +31.500 us/token | trapped recovery, not booked |
| O44 production wall | -27.746 us/token | no-go for current output/completion contract |
| S46 full grid | unmeasured | open |
| O46 full grid | unmeasured | open |

At the 235.259 tok/s clean endpoint, 240 requires about 83.98 us/token. S44
plus the trapped O44 device recovery accounts for 47.62 us/token. If O44 is
unlocked, the two remaining mixed populations need roughly 36.36 us/token in
aggregate, about 2.02 us/block, to cross 240 in that regime. This is a target
for measurement, not an additive promise.

## 4. What is generic and what remains grammar-specific

### Generic substrate, implemented once

The reusable layer must own:

- a 4,096-CTA Q grid;
- a workgroup-uniform conditional region keyed only by global CTA id;
- legal inner barriers for grammars whose K/V reduction spans multiple warps;
- packed K-then-V row addressing without changing compulsory bytes;
- separate caller-owned Q/K/V outputs, or a typed direct-sink alternative;
- explicit output identities usable by norm, RoPE, cache, and flash consumers;
- fail-closed admission by target, shape, dtype, format, and population;
- fair control allocation so packed persistent views exist in both wall arms;
- exact graph-cycle parsing that follows the final steady topology;
- and rollback that restores the complete installed route.

### Thin grammar adapters

The arithmetic layer may differ only where the format requires it:

- ordinary Q4: fp16 activation, one-warp lane map, vector-load dot;
- shared Q4: Q8_1 activation packets, four-warp direct-output merge;
- ordinary Q6 V: fp16 activation and the installed Q6 exact reduction;
- shared Q6 V: shared-Q8 packet consumer and its exact four-warp reduction.

Do not fork four unrelated producer frameworks. Factor the grid, conditional
region, row mapping, output ownership, and admission logic. Each adapter must
preserve the currently installed left-to-right accumulation and reduction
association for that format.

Performance qualification remains population-specific. A generic compiler
proof does not imply that every grammar pays at token wall.

## 5. Current substrate and reproducibility prerequisite

The working full-grid emitters call:

```python
anchor.post_barrier_region(global_cta_gate, workgroup_uniform=True)
```

The corresponding compiler contract currently exists as uncommitted worktree
changes in:

- `tinygrad/uop/ops.py`;
- `tinygrad/codegen/__init__.py`;
- `test/unit/test_post_barrier_region.py`.

Those changes add the `workgroup_uniform` type bit, prove gates from constants
and global `gidx` ALU only, admit inner barriers only under that proof, and
reject local-id/load/range-dependent predicates. Before claiming clone-level
reproducibility, audit and land only these relevant hunks with their tests.
Do not stage the unrelated dirty split-phase changes in
`tinygrad/renderer/cuda.py` or other concurrent worktree files.

Required Phase 0 tests:

```bash
.venv/bin/python -m pytest -q \
  test/unit/test_post_barrier_region.py \
  test/unit/test_shared_q8_pair.py \
  test/unit/test_q4k_kv_pair.py
```

Required properties:

- global CTA predicates admit an inner barrier;
- local-thread predicates fail closed;
- ordinary predicated regions still reject inner barriers;
- CUDA and PTX renderers retain explicit region ordering;
- the S44 and O44 emitters render from a clean checkout plus committed scope
  commits, not only from the current dirty worktree.

## 6. Campaign phase plan

### Phase 0 — land and verify the generic compiler substrate

Goal: make the already-measured S44/O44 evidence reproducible from committed
state.

Actions:

1. Inspect the three relevant dirty files listed above.
2. Run their focused tests before staging.
3. Stage only the uniform-region hunks and associated test.
4. Re-run the S44 and O44 emitter render/unit gates.
5. Commit separately from candidate route changes.

Advance only if all tests pass and `git diff --cached` contains no unrelated
PDL, renderer-policy, JIT, HCQ, or runtime changes.

### Phase 1 — identify the O44 recovery-loss boundary

Known facts:

- output words and token hashes are exact;
- the complete isolated span is faster;
- the final steady profile is 490 to 480 nodes;
- device union improves by 31.5 us/token;
- the reps-7 A/B/A wall loses 27.746 us/token;
- packed K/V allocation is equal in both arms;
- Q, K, and V already use separate caller-owned outputs;
- the candidate covers all nine O44 blocks.

Therefore do not tune CTA geometry again. Test completion/ownership mechanisms.

#### Phase 1A — boundary-matched discriminator

Build a control that retains installed Q plus K/V-pair arithmetic but presents
the same output/program ownership and graph-call boundary as the full-grid
candidate. Its purpose is to partition:

```text
full-grid arithmetic/geometry
versus
multi-output program/completion ownership
versus
downstream readiness/cache composition
```

Required observations:

- exact Q/K/V buffer identities and AFTER relationships;
- final steady graph pattern and node count;
- producer end to Q-norm start;
- producer end to K-norm/RoPE start;
- V availability to terminal cache producer;
- flash-score readiness after the later cache dependency;
- device node sum/union and unprofiled wall.

If the boundary-matched control loses the same amount, promote the common
boundary as causal. If it does not, inspect producer readiness and lost Q/KV
concurrency before changing runtime infrastructure.

#### Phase 1B — compose direct V-cache ownership

Test the most bounded output-boundary repair:

```text
full-grid QKV producer
  -> direct Q output
  -> direct K output for exact K norm/RoPE
  -> V writes its final current-token cache slot directly
terminal K norm/RoPE producer
  -> writes only K cache
flash score
  -> joins the two cache AFTER identities
```

This composes the full-grid producer with the already-promoted producer-owned
cache-sink contract. It must not reintroduce a generic cache store, output
slice, stack/cat copy, or a host-visible synchronization.

Exactness must cover fp16 and fp32 cache dtypes, first/interior/last cache
slots, untouched-slot sentinels, and full token-stream hashes. The first
production lease remains O44-only.

Advance if the candidate beats both controls in reps>=7 A/B/A. If it is
positive but controls are separated by more than the candidate recovery, run
reps>=9 confirmation. If it is negative after exact boundary accounting,
close O44 and carry only the generic lessons forward.

### Phase 2 — factor the reusable producer specification

Do this after Phase 1 identifies the viable output contract, so the abstraction
does not encode a losing boundary.

The reusable specification should describe:

- Q rows, K/V rows, input width, CTA size, and active CTA interval;
- Q, K, and V dot adapters;
- per-adapter warp count and merge requirement;
- packed K/V storage layout;
- output modes: separate tensor, direct V cache, and any later proven sink;
- exact KernelProgram output specifications;
- target and route admission facts;
- program names that distinguish every grammar and output mode;
- and a complete ordinary fallback.

Keep S44 and O44 legacy emitter entry points temporarily as wrappers so their
existing evidence and rollback paths remain usable during refactoring.

Required refactor gate:

- pre/post-refactor generated source hashes match for the already-qualified
  S44 body, unless a deliberate source change is separately requalified;
- O44 microgate outputs remain bit-exact;
- unit topology counts and barrier contracts remain unchanged;
- no wall gain is booked from refactoring.

### Phase 3 — implement and qualify S46

Population: eight shared-Q8 Q4/Q4/Q6 blocks.

Control must be the currently installed shared-Q8 Q projection plus the
promoted shared-Q8 Q4/Q6 K/V pair. Candidate uses the same Q8 provider and one
full-grid Q4-Q/Q4-K/Q6-V producer.

Special requirements:

- Q4 and Q6 weight storage remain separate or use a typed composite view;
- no physical concatenation is allowed unless both arms allocate it and the
  byte/address effect is explicitly measured;
- Q4 and Q6 reductions preserve their distinct exact associations;
- inactive CTAs issue no K/V loads or arithmetic;
- no packed-output slices or generic copy kernels;
- all eight installed blocks are covered.

Gate order:

1. deterministic bit-exact Q/K/V microgate;
2. hot and pointer-rotated cold complete-span timing;
3. registers, spills, DRAM bytes, and duration counters when available;
4. one-block output-ownership/profile lease;
5. eight-block final steady census;
6. depth-512 reps>=7 A/B/A wall;
7. reps>=9 confirmation only when direction/noise requires it.

### Phase 4 — implement and qualify O46

Population: ten ordinary-fp16 Q4/Q4/Q6 blocks.

Control is the exact installed ordinary Q4 Q projection plus installed Q4-K
and Q6-V paths. If a promoted mixed K/V pair exists at execution time, it is
the control; otherwise the two installed single projections are the control.
Record the actual runtime census rather than assuming admission objects imply
execution.

Candidate uses the generic full-grid/output contract selected by Phase 1 and
the ordinary Q4/Q6 arithmetic adapters. Direct V-cache output should be tested
as the primary output mode if Phase 1B passes.

Use the same gate order as S46 and cover all ten blocks. A partial Q4-only or
Q6-only subset is a diagnostic, not a population result.

### Phase 5 — install each winner behind independent policy

Each passing population gets:

- its own generated route-policy record;
- NV `sm_120` target scoping;
- a specific disable environment variable;
- load-time census validation;
- fail-closed shape/format/dtype checks;
- no effect on prefill or other models/targets;
- and a focused rollback bracket.

Do not use one global flag that prevents isolating regressions by population.
Do not enable a population merely because the generic substrate passed
elsewhere.

### Phase 6 — composed qualification and fresh ledger

Run these compositions in fresh processes:

1. installed control;
2. S44 only;
3. all shared winners;
4. all ordinary winners;
5. all producer winners together;
6. all winners with individual rollback arms when composition is non-additive.

Required final wall:

```text
depth 512
count 32 tokens per repetition
reps >= 9
control / candidate / control
identical token stream hashes
locked or recorded memory and SM clocks
```

Run a fresh unprofiled installed endpoint after policy landing. Report both the
fast demonstrated regime and any slower stable/variable regime; do not replace
one with the other. Then rebuild the full device ledger and translate only the
same-session booked wall into tok/s.

## 7. Common microgate and production rules

### Exactness

- Compare every fp32 Q/K/V output word bit-for-bit.
- For direct sinks, compare the entire cache allocation byte-for-byte.
- Preserve signed zero and exact fp16 conversion points.
- Preserve left-to-right accumulation and warp/shared-memory merge order.
- Token-stream SHA equality is mandatory in every production arm.

### Cold service rate

- Hot timing is diagnostic only.
- Use pointer-rotated complete spans larger than L2.
- Keep control and candidate payload and allocations symmetric.
- Report DRAM bytes and rate separately; unchanged bytes with higher rate is a
  valid mechanism.
- Header/vector load reductions that hit cache are not booked as DRAM savings.

### Output ownership

- Outputs must be caller-owned writable buffers or typed direct sinks.
- Reject slice-based packed outputs when they materialize transport kernels.
- Reject hidden `cat`, `stack`, `contiguous`, cast, or copy work at token time.
- Persistent packed weight views must be built at load time and equalized in
  both wall arms.

### Profile parsing

- Discover the final repeated graph cycle, not the most frequent compile
  pattern.
- Prefer the latest repeated stable pattern and a bounded final replay window.
- Report graph pattern, node count, node sum, union, overlap, and span.
- Inspect raw program names to prove the candidate actually ran.
- An admission-object census without the emitted candidate kernel is a miss.

### Wall authority

- An isolated or profiled gain never overrides an unprofiled wall loss.
- Candidate must beat both controls for direct promotion.
- If facts predict a pass but the first wall fails, audit population, outputs,
  allocations, graph parsing, and runtime route execution before closing it.
- After those are complete, a negative wall is a real promoted wall.

## 8. Tok/s accounting

Always convert from the relevant latency authority:

```text
tok/s = 1e6 / us_per_token
delta_tok/s = 1e6 / (control_us - recovery_us) - 1e6 / control_us
```

Maintain three distinct columns:

- isolated ceiling;
- device-ledger recovery;
- booked unprofiled wall recovery.

Only the third moves the installed tok/s ledger. Do not translate a slower
qualification harness's absolute control rate into the installed endpoint.

Reference checkpoints:

| endpoint authority | latency | throughput | gap to 240 |
| --- | ---: | ---: | ---: |
| clean demonstrated regime | about 4.2506 ms | about 235.26 tok/s | about 84 us |
| newer variable regime | about 4.282-4.292 ms | about 233-234 tok/s | about 115-125 us |

The campaign must state which authority it uses whenever it says “to 240.”

## 9. Evidence and documentation layout

Campaign evidence root:

```text
docs/task_workflow/evidence/nv-full-qkv-producer-substrate-20260825/
  phase0-compiler/
  phase1-o44-boundary/
  phase2-generic-spec/
  phase3-s46/
  phase4-o46/
  phase5-policy/
  phase6-composed/
  campaign-ledger.json
```

Existing controlling evidence must be referenced, not copied:

- `nv-shared-q8-qkv-producer-20260825/full-split-output-smoke.json`;
- `nv-shared-q8-qkv-producer-20260825/production-wall-triple-r7-split.json`;
- `nv-ordinary-q4-qkv-full-20260825/microgate-current-r7.json`;
- `nv-ordinary-q4-qkv-full-20260825/production-profile-accounted.json`;
- `nv-ordinary-q4-qkv-full-20260825/production-wall-r7.json`.

Maintain one campaign result document with a population matrix whose status is
one of:

- `NOT_STARTED`;
- `CONSTRUCTION`;
- `MICROGATE_PASS`;
- `ACCOUNTING_REPAIR`;
- `WALL_PASS`;
- `WALL_CLOSED`;
- `INSTALLED`.

Every phase update records commit, exact command, output artifact, token hash,
population count, and decision. Interim false starts stay in evidence but are
not presented as controlling results.

## 10. Commit and worktree discipline

The repository contains unrelated dirty and untracked research work. Preserve
it. Never reset, checkout, clean, or stage broadly.

For every commit:

1. stage explicit owned paths or hunks;
2. inspect `git diff --cached --stat` and `git diff --cached`;
3. run focused tests;
4. commit one causal unit;
5. push `nvidia-bringup-20260731` after verification;
6. record the commit in the campaign result.

Compiler substrate, generic producer refactor, each grammar, policy landing,
and final ledger should be separate commits.

## 11. Stop and promotion conditions

### Promote a population when

- full population and exact route execute;
- outputs/cache and token hashes are exact;
- no hidden transport or allocation asymmetry exists;
- final steady graph accounting is complete;
- unprofiled candidate beats both controls;
- rollback restores the preceding installed graph;
- and composition does not erase the win.

### Close a population when

- the same full checks are complete;
- a repaired reps>=7 bracket is negative or flat;
- and no named missing observable remains.

### Revisit instead of closing when

- expected blocks are missing;
- candidate kernel names are absent;
- output slices add transports;
- packed allocation differs between arms;
- graph parser selects compile/warmup topology;
- profiling cannot see the active runtime construction;
- or a short/noisy wall contradicts strong body evidence without a boundary
  discriminator.

### Campaign completion

The campaign is complete only after:

1. all four population rows have `INSTALLED` or `WALL_CLOSED` status;
2. all winners pass one composed reps>=9 wall;
3. a clean installed endpoint is measured;
4. the full device/token ledger is rebuilt;
5. tok/s movement is attributed only to booked wall deltas;
6. remaining distance is expressed against the hardware/model wall, with 240
   and retained llama shown only as reference checkpoints.

## 12. Immediate next action

Start with Phase 0, then Phase 1A. Do not implement S46/O46 before the O44
boundary discriminator identifies the reusable completion/output contract;
otherwise the mixed emitters may reproduce a known losing boundary.

The first new evidence question is:

> Does an output/program-boundary-matched control reproduce the O44 wall loss,
> or does the loss arise specifically because the full-grid producer changes
> Q/K/V readiness relative to the installed Q plus K/V-pair schedule?

That answer selects the generic substrate repair and prevents another series
of one-off kernels whose body wins cannot reach token wall.
