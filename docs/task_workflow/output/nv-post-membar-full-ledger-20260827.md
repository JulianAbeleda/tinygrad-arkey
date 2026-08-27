# Full installed ledger after internal-QMD membar promotion

## Verdict

The fresh stable unprofiled endpoint is **4094.098 us/token = 244.254
tok/s**.  The 15 measurement windows span only 4088.170--4100.848 us/token,
none were rejected, and every window has the same token hash.  Against the
retained llama authority of **4021.721 us/token = 248.650 tok/s**, the fresh
wall debt is **72.377 us/token = 4.396 tok/s**.

The earlier **4060.523 us/token = 246.274 tok/s** endpoint remains a real
historical measurement, but it is not the current-session authority.  The
**3992.993 us/token = 250.439 tok/s** number was a continuity estimate formed
by subtracting the membar bracket recovery from that older endpoint; it was
never a freshly measured installed endpoint.

## Device ledger

The fresh profiled ledger closes exactly:

```text
node sum  3846.848 us/token
overlap      0.098 us/token
union     3846.750 us/token
```

The capture reported 2872 MHz graphics and 14001 MHz memory clocks, 502 W,
and 54 C.  This is a warm, high-power GPU state; low clocks are not an
adequate explanation for the fresh wall result.

The retained llama profiled device union is 3888.240 us/token.  Tinygrad's
fresh profiled union is therefore 41.490 us lower.  This comparison does not
prove a 41.490 us wall advantage: tinygrad HCQ timestamps and llama CUPTI are
different timing domains, and the profiled tinygrad wall contains 1113.599 us
of instrumentation/host delay.  It does prove that the remaining unprofiled
wall gap cannot honestly be assigned wholesale to slower tinygrad kernel
bodies.

## Current reading

| authority | tinygrad | llama | tinygrad - llama |
|---|---:|---:|---:|
| unprofiled token wall | 4094.098 us | 4021.721 us | +72.377 us |
| throughput | 244.254 tok/s | 248.650 tok/s | -4.396 tok/s |
| profiled device union | 3846.750 us | 3888.240 us | -41.490 us |

The next audit target is the translation boundary between device completion
and the unprofiled token wall, measured in one common protocol.  Kernel-row
optimization can still improve absolute speed, but the current ledger no
longer supports calling slower aggregate tinygrad device work the reason for
the llama wall gap.

## Same-session and sustained-state reconciliation

The 246.274-tok/s authority was promoted.  Its wide-Flash commit
`79b740fd9` is an ancestor of the current `e411a292a` endpoint; the current
route census also observes the promoted Flash spelling.  The lower fresh run
is therefore not an uninstalled-promotion failure.

A same-session rerun of the pinned llama binary measured a median repetition
of **4008.703 us/token = 249.457 tok/s**.  Its first repetition was slower;
the following six clustered at 248.979--249.974 tok/s.  This same-session
median is the primary comparator, while 4021.721 us/token remains the retained
continuity reference.

After sustained GPU work, the current tinygrad submit-ahead reverse bracket
measured its control midpoint at **4025.790 us/token = 248.399 tok/s**.  The
candidate was **4025.054 us/token = 248.444 tok/s**, only 0.736 us/token
faster.  Every token hash matched.  Submit-ahead is therefore still neutral
and is not promoted.

The sustained same-session wall gap is **17.086 us/token**, or about **1.058
tok/s**.  The earlier 4094.098-us fresh endpoint remains valid as a cold/fresh
operating-point observation, not the sustained ceiling.

| endpoint class | tinygrad | llama | gap |
|---|---:|---:|---:|
| fresh stable run | 244.254 tok/s | retained 248.650 tok/s | -4.396 tok/s |
| sustained same-session | 248.399 tok/s | 249.457 tok/s | -1.058 tok/s |
| historical tinygrad | 246.274 tok/s | retained 248.650 tok/s | -2.376 tok/s |

## Translation-boundary adjudication

The current marker-light run measures a 3858.608-us native device window and
215.705 us before the first graph call.  A same-session llama Nsight trace
measures a 4030.962-us median graph span, although Nsight inflates that span
past llama's unprofiled wall by about 22 us.  The direction is clear: tinygrad
has the faster device program and more host-side staging around it.  But the
submit-ahead causal test recovers only 0.736 us, proving that the apparent
pre-first interval is mostly overlapped or shifted accounting, not a
215.705-us exposed recovery pool.

The honest remaining target is now the approximately 17-us sustained
same-session wall residual.  It must be attacked with causal full-wall tests;
neither the profiled host-gap subtraction nor the raw pre-first interval is
bookable.

## Token-readback cause and promotion

The lifecycle comparison was not semantically symmetric.  The pinned
`llama-bench` generation loop calls `llama_decode`, synchronizes, and then
supplies another random CPU token.  It does not sample logits or feed the
model's selected token back.  Tinygrad's measured `generate()` includes GPU
argmax, scalar token delivery to the host, and device-to-device feedback into
the next decode.

A ceiling arm retained tinygrad's GPU greedy feedback but suppressed host
`Tensor.item()` delivery.  It measured 3824.676 us/token versus 4117.085
us/token control, exposing a 292.409-us synchronization/readback pool.  This
arm is not promotable because it does not deliver tokens to the caller.

Inspection found two waits in generic HCQ copyout: a CPU wait for the compute
timeline followed by a copy-queue command that waits on that same timeline,
then the required copy-completion wait.  Suppressing only the redundant first
wait retained ordinary token delivery and reproduced in two reverse brackets:

| qualification | control midpoint | candidate | recovery | token hashes |
|---|---:|---:|---:|---|
| monkeypatch discriminator | 4115.681 us | 4069.214 us | 46.467 us | identical |
| actual NV opt-in | 4111.519 us | 4079.271 us | 32.248 us | identical |

The NV default now declares that its ordered copy queue owns source ordering.
The copy queue still waits on the producer timeline and the CPU still waits
for copy completion.  `NV_COPYOUT_SKIP_PRESYNC=0` restores the conservative
generic HCQ sequence.

The promoted 15-window endpoint is **4036.879 us/token = 247.716 tok/s**.
Every window is accepted and every token hash matches.  Relative to the fresh
pre-promotion 4094.098-us authority this is **57.219 us/token** or about
**3.462 tok/s** faster.  The remaining comparison to llama must henceforth
label llama-bench as decode-only/random-feedback; a full greedy-generation
parity claim requires a llama sampling+feedback harness.

## Ordered campaign to the delivered-token ceiling

Q projection work is deliberately deferred.  The immediate objective is to
translate the already measured no-host-delivery ceiling into a correct,
host-delivered token path.  That ceiling arm measured **3824.676 us/token =
261.460 tok/s**; it is a recovery bound, not a production endpoint.

The tests must run in this order:

1. qualify direct GPU writes to each available CPU-visible NV allocation
   class, including cached/uncached coherency, stale-read stress, exact scalar
   values, and completion-signal ownership;
2. make argmax emit a host-visible scalar while preserving the ordinary GPU
   token used for greedy device-to-device feedback;
3. test signal-only completion and remove only waits proven redundant by the
   producer timeline;
4. exhaust the remaining four-byte copyout setup, DMA, and CPU wakeup costs;
5. run matched real-greedy tinygrad and llama lifecycle accounting, then lock a
   fresh delivered-token ledger.

Every candidate follows test-then-invest: exact output/guard qualification,
reverse A/B/A brackets with at least seven repetitions, a reproducible full
token-wall win, and an explicit rollback.  Each proven promotion is committed
and pushed independently.  An expected-pass failure is first treated as a
possible information or measurement wall and revisited with a stronger
discriminator before the mechanism is closed.

Only after this non-Q path either reaches approximately **260 delivered
tok/s** or is causally exhausted does the campaign reopen Q.  The deferred Q
sequence remains service-rate microgate, cold full-token bracket, and generic
shape sweep; no Q estimate is counted in the current endpoint or the
261.460-tok/s delivery ceiling.

### Host-visible scalar discriminator

Direct GPU writes were coherent for 10,000 alternating-value iterations in
each of CPU-mapped VRAM, pinned cached host memory, and uncached system memory.
There were no stale values or guard corruptions.  The exact native argmax then
wrote both its ordinary GPU feedback token and a host mirror; 17 selected
winners matched the ordinary path, GPU output, and mirror.  Its isolated
synchronized gate recovered about 44 us against `Tensor.item()`.

That isolated saving did not translate to the production token wall:

| mirror placement | control midpoint | candidate | delta | exact tokens |
|---|---:|---:|---:|---:|
| pinned cached host | 4067.144 us | 4065.480 us | -1.664 us | yes |
| CPU-mapped VRAM | 4064.221 us | 4064.747 us | +0.526 us | yes |

Neither candidate beat both controls, so neither is promoted.  The existing
ordered four-byte copy is not the large remaining recovery pool.  The
261.460-tok/s no-delivery arm must now be interpreted primarily as a cadence
ceiling: suppressing every host rendezvous permits queued decode work, whereas
a correct streaming generator must expose each selected token.  The next
discriminator is bounded multi-token run-ahead with exact GPU feedback and a
host-visible result ring, followed by batched synchronization and delivery.
This tests cadence without charging Q or changing the greedy token sequence.

## Evidence

- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/installed-wall-r15.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/device-ledger.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/current-marker-host-partition-r24.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/llama-same-session-summary.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/submit-ahead-r7.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/token-readback-r7.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/token-readback-skip-presync-r7.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/token-readback-optin-r7.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/installed-copyout-promoted-r15.json`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/scalar-matrix.json`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/argmax-host-mirror-r9.json`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/argmax-host-mirror-gpu-only-r7.json`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/production-host-mirror-r7.json`
- `docs/task_workflow/evidence/nv-host-visible-token-delivery/production-mapped-vram-mirror-r7.json`
