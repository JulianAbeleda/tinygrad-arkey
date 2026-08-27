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

## Evidence

- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/installed-wall-r15.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/device-ledger.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/current-marker-host-partition-r24.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/llama-same-session-summary.json`
- `docs/task_workflow/evidence/nv-post-membar-full-ledger-20260827/submit-ahead-r7.json`
