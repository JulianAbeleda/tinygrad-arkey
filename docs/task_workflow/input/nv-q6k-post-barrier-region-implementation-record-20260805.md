# NV Q6_K post-barrier region implementation record — 2026-08-05

## Outcome

The structured-control blocker is resolved generically. Tinygrad can now
express this safe producer/consumer workgroup sequence without route-local
CUDA source:

```text
producer lanes publish shared state
all workitems reach one workgroup barrier
enter a typed predicated region
consumer lanes load shared state, reduce, and store
```

For the Q6_K four-warp candidate, NVRTC lowers the generic region to the exact
desired machine lifetime: after the barrier, threads 32–127 execute a
predicated `EXIT`; only warp 0 reaches the three shared loads and five-shuffle
reduction.

The completed timing gate is a no-go for this topology as the next wall lever.
In an interleaved 15-pair A/B/A, the llama-shaped 384-byte stage is
**+0.18535 us** slower than the flat control. It still beats the installed Q6
route by **1.35711 us**. The conclusion is narrow and useful: missing
producer-warp retirement was a real emitter gap, but it does not explain the
remaining Q6 wall gap once implemented.

## Generic primitive

`PostBarrierRegion` is a typed UOp argument with two construction methods:

```python
region = ready_barrier.post_barrier_region(bool_gate)
value = shared_buffer.after(region)[index]
done = region.end_region(final_store)
```

It deliberately models a predicated region rather than a literal kernel
return. This is safer across structured renderers, leaves compiler freedom to
choose a branch or an early exit, and has the semantics required by the
algorithm: non-consumer workitems skip the entire body after all workitems have
crossed the barrier.

The validator requires all of the following:

1. The renderer explicitly advertises support.
2. The opening IF has exactly a boolean gate and an `Ops.BARRIER` anchor.
3. The anchor appears before the region.
4. ENDIF references the opening IF and at least one body root.
5. Every body root depends on the IF, preventing the scheduler from hoisting
   region work before the predicate.
6. No workgroup barrier occurs inside the predicated region.
7. Untyped graph-authored IF/ENDIF remains rejected. The existing gated-store
   cleanup is unchanged.

Only the two proved NV render paths opt in: CUDA C-style and PTX. HIP, Metal,
LLVM, NIR, Python, CPU, and native ISA renderers remain closed by default even
where they already possess ordinary IF syntax. They may opt in only after a
backend-specific barrier/region proof. This deliberately minimizes renderer
surface while keeping the UOp contract vendor-neutral.

The only optimizer change is adding `Ops.IF` to the side-effect boundary set
used when canonicalizing `AFTER`. This matters only when a graph contains the
new typed region. Existing graphs cannot contain IF before the late gated-store
cleanup, so unused behavior remains byte-identical.

## Bank decision

**Bank the substrate, closed-default; do not promote a route.** It closes a
generic expressibility hole with an exact machine proof, fails closed on every
unproved renderer, and is inert unless a caller explicitly constructs the
typed region. The Q6 experiment does not justify it as a performance recovery:
the faithful region still loses to the flat control. Its value is reusable
compiler vocabulary and a falsified hypothesis, not an immediate token win.

## Hermetic proof

`test/unit/test_post_barrier_region.py` covers:

- graph order: barrier < typed IF < local load < typed ENDIF;
- PTX branch placement around the shared-memory body;
- bool-gate and barrier-anchor construction failures;
- unsupported-renderer failure;
- rejection of a barrier inside the region;
- rejection of an independent body root;
- continued rejection of untyped graph IF;
- identity of the unused cleanup path.

The existing Q6 contracts also continue to pass.

## NV SASS proof

The production-shape Q6 candidate emitted this sequence:

```text
/*2c80*/ BAR.SYNC.DEFER_BLOCKING 0x0
/*2c90*/ @P1 EXIT                    # P1 = threadIdx.x >= 32
...
/*2d00*/ LDS ...
/*2d10*/ LDS ...
/*2d20*/ LDS ...
...
/*2d60*/ SHFL.BFLY ... 0x10 ...
/*2d80*/ SHFL.BFLY ... 0x8  ...
/*2da0*/ SHFL.BFLY ... 0x4  ...
/*2dc0*/ SHFL.BFLY ... 0x2  ...
/*2de0*/ SHFL.BFLY ... 0x1  ...
```

This proves all producer warps exit before any LDS or shuffle. It is stronger
than merely observing a branch around a predicated store. The machine artifact
also retains 38 registers, 384 explicit shared bytes, three LDS, and five
shuffles.

The research SASS census now records this ordering proof automatically when it
finds `barrier < predicated EXIT < 3 LDS < 5 SHFL`.

## Correctness

The independent Q8 oracle passed:

| Metric | Value |
|---|---:|
| candidate max abs vs Q8 oracle | 0.01093578 |
| allowed absolute tolerance | 0.02 |
| candidate max abs vs flat control | 0.000003815 |

The tiny candidate/control difference is the expected floating-point reduction
association change. This is primitive correctness only; it is not a full-logit
promotion qualification.

## Wall gate

The first blocked A/B/A run developed severe one-sided clock/outlier drift and
was rejected. The authoritative rerun used interleaved A/B/A triplets so each
candidate arm is bracketed by its controls.

| Metric | Result |
|---|---:|
| method | 15 interleaved A/B/A pairs, 100 replays/arm |
| flat control median | 64.66083 us |
| lane-stage median | 65.06629 us |
| median paired candidate delta | **+0.18535 us** |
| pair signs | 8 slower / 7 faster |
| median after removing one abs(delta) > 2 us outlier | **+0.33155 us** |
| candidate minus installed | **-1.35711 us** |
| gate | **FAIL** |

The sign margin is narrow, but both the robust median and outlier-trimmed median
reject a win. There is no basis to promote or further tune the 384-byte stage.

## Updated ledger

- **Emitter/control-flow blocker:** resolved.
- **Llama-shaped shared reduction:** implemented faithfully; no-go as the next
  wall lever.
- **Flat four-warp Q8+DP4A route:** remains the measured primitive winner versus
  installed.
- **Next decisive work:** integrate that flat Q6 consumer, closed-default, into
  the mixed Q4/Q4/Q6 shared-Q8 model group and enforce the full-logit semantic
  gate at g1/g4/g8/g12 before settled model wall timing.

No production route was enabled. No commit or push was made.

## Artifacts

- Machine record:
  `docs/task_workflow/output/nv-q6k-post-barrier-region-gate-20260805.json`
- Gate implementation:
  `extra/llm_research/decode/q6k_q8_warp_direct_microgate.py`
- Hermetic tests: `test/unit/test_post_barrier_region.py`
- Raw authoritative timing:
  `/tmp/q6k_q8_warp_lane_stage_region_interleaved.json`
- Raw SHA256:
  `30bb1ce43f55ef4f0dc13424c735e2fcf3aa354fc9ee95fb1feed84cf6513664`
