# B3 prefill host overhead - independent runtime lever scope (AMD control required)

Date: 2026-08-03

Status: scoped, not implemented. Authorized by
`nv-campaign-forward-review-amendment-20260803.md` sections 3 Q4 and 4.2: B3 is an
independent prefill runtime lever, scoped on its own with a required AMD control, and it
does not gate on decode progress. Branch boundary: tinygrad `nvidia-bringup-20260731` at
`fcf3774f9`. This document does not authorize implementation, promotion to
`dev`/`exp`/`master`, or any code change.

Bans for this scope: no code changes; no GPU use in this doc-only pass; never touch
`extra/llm_research/microbench/dp4a_peak_cuda*` or `scratchpad/t6_metal_admission_probe.py`;
never commit to `master`/`dev`/`exp`.

---

## 1. The measured evidence - why B3 is the last prefill lever

All rows below are same-session warm measurements on the production tuned schedule
(`04e500079`), from `nv-performance-campaign-scope-20260801.md` section 11.1 (sources
collected verbatim):

| quantity | measured | source |
| --- | ---: | --- |
| warm pp512 wall (steady) | 44-46 ms | `/tmp/measure_warm_prefill.py` passes_s `[5.76, 1.91, 0.046, 0.045]`; first replay 1.9s is one-time graph instantiation, steady state is the last two |
| GPU busy (warm replay) | 24.1 ms / 8 graph groups | `DEBUG=2` + `GlobalCounters.time_sum_s`, `/tmp/measure_busy_debug2.py`: 2 copies + batched 32/64/128/256/512/27 |
| NV `wait()` CPU time | 23.7-23.8 ms across 10 waits | `/tmp/probe_warm2.py`, HCQSignal.wait monkeypatch, pass2/pass3 |
| wall/busy | ~1.9x | 46 / 24.1 |

The llama envelope (campaign doc section 8.4): warm wall tracks GPU busy within
**1.15-1.35x** (llama warm avg: 37.76 ms wall vs ~32.95 ms busy = 1.15x). Ours at 1.9x is
above that envelope, so P3's criterion is not met.

The to_mv tax (`/tmp/nv_exec_profile.log`, cProfile, warm run): **1,352,164 `to_mv` calls,
all reached from `hcq.py:285 wait()`** via `HCQSignal.value` -> `cpu_view().view()` ->
`HcqView.__init__` -> `to_mv(addr, nbytes).cast(fmt)` (current code:
`hcq.py:262` `value` property, `hcq.py:18` `HcqView.__init__`). 71 waits, ~19k polls each:
**one fresh memoryview + cast per poll iteration of the `wait()` spin loop**, not per-buffer
copies in dispatch. NV never sleeps (`NVSignal._sleep` at `ops_nv.py:27-31` only sleeps after
200 ms elapsed, and `NVKIface.sleep` at `ops_nv.py:567` is `pass`), so `wait()` busy-spins at
Python speed, ~2.5 us/poll -> ~23.7 ms of the 44-46 ms wall. The `_sleep` +
`time.perf_counter` pair (1.35M each) is the same loop.

Structure of the wall: `wait()` CPU time (23.7 ms) tracks GPU busy (24.1 ms), so the waits
are mostly overlapped with execution; the **non-overlapped host submit cost is wall - busy =
~20-22 ms across 10 submits (~2 ms each)** (campaign doc section 11.2).

Why B3 is the last prefill lever: the fp16-path busy ceiling is **512/24.1 ms = 21.2k tok/s,
above llama's 14,250** (same-session P5: 14,468.4). The promoted prefill path already reaches
0.77x-1.05x of llama at pp512-4096 (pp2048 measured above llama). The entire remaining
pp512 gap vs llama is this host factor. B3 is a host-side runtime build, not a kernel.

## 2. Fix shapes, ranked (campaign doc sections 8.5 and 11.3)

In order of size:

1. **(a) Cache the `HcqView` per signal.** Build the `cpu_view().view(0, 8, 'Q')` once per
   signal and reuse it across polls, so each poll is one plain memory read (~0.1 us instead
   of ~2.5 us). Smallest blast radius: touches only the signal value/timestamp read path in
   shared HCQ code.
2. **(b) Real blocking wait on the NV uvm semaphore.** Replace the sleep stub with the actual
   wait primitive the uvm semaphore exposes, so `wait()` blocks instead of spinning at Python
   speed. Larger blast radius than (a) (driver-facing, timeout/fault semantics) and requires
   an NV-specific path.
3. **(c) Graph replay of the whole schedule.** Already active at the group level (the warm
   pass is 8 replayed groups). Extending replay to the whole schedule removes submits, which
   is where the ~20-22 ms non-overlapped cost lives. This is the endgame shape.

Two facts decide the shape (campaign doc section 11.3): wait time (23.7 ms) approximately
equals GPU busy (24.1 ms), so **(a)/(b) alone may not cut single-pass wall** - they remove
the constant factor, but the ~20-22 ms non-overlapped submit is the real target; and the
submit path is shared HCQ code that also serves AMD, so a blind change without an AMD control
risks the shared target.

Recommended ordering: (a) is the smallest first probe (closed-default, measurable on both
targets, and it buys the AMD runtime row cheaply); (c) is the endgame that can actually land
the envelope. (b) is the riskiest shape and should only be opened if (a)'s measured wait-time
collapse proves insufficient because the remaining wall is the non-overlapped submit.

## 3. AMD control requirement

The submit path is shared HCQ code that also serves AMD (`HCQSignal.wait`/`value` ->
`cpu_view().view()` -> `to_mv().cast()` is target-independent). The campaign guardrail's pg2
AMD control is compile-only rendered-source equality, which is not a runtime perf control
(campaign doc sections 12.3, 13.2). A blind change to the shared polling path therefore risks
the shared target unless AMD is runtime-measured.

This scope requires **same-session NV+AMD measurement of the polling change before landing**:
the before/after rows from section 4.1 must be produced for both targets in the same session
family, and the AMD rows must not regress.

The P5 pp512/pp1024 warm rows (pp512 11,158 tok/s, pp1024 14,003 tok/s, campaign doc section
13.1) are NV rows from the RTX 5090 session; they are the NV baseline, not the AMD control.
The AMD control is a **runtime re-run of the campaign's AMD e2e bench** with the change in
place, compared against the AMD baseline row measured in the same session before the change.
No AMD GPU exists on this machine (5090 only, no ssh target; campaign doc section 7 finding
6), so the AMD leg must run on AMD-capable hardware and the change must not land from this
NV-only session. The compile-only AMD render equality remains required per the campaign
guardrail.

## 4. Measurement protocol

### 4.1 Same-session warm pp512 wall + busy + wait-time, before and after

Each of the three instruments from section 1, run as a before/after pair in the same session
family on the production tuned schedule, on each target:

| instrument | pattern | reports |
| --- | --- | --- |
| wall | `/tmp/measure_warm_prefill.py` | steady-state passes_s entries (last two, after the one-time 1.9s graph instantiation) |
| busy | `/tmp/measure_busy_debug2.py` (`DEBUG=2` + `GlobalCounters.time_sum_s`) | 8 graph groups, ms |
| wait() CPU time | `/tmp/probe_warm2.py` (HCQSignal.wait monkeypatch) | total ms across the 10 waits, pass2/pass3 |

### 4.2 Llama envelope as the criterion

Acceptance is the same-session warm wall/busy ratio landing **within llama's measured
1.15-1.35x envelope**, not a single absolute number (the withdrawn 92% / cold-cross-run
failure mode from campaign doc section 8.4 is explicitly not revived). NV must move from
~1.9x into the envelope; AMD must show the pp512/pp1024 warm rows unchanged within spread.

### 4.3 Correctness pins

Pins from the campaign P5 controls (section 13.2), unchanged by this scope:

- First-token digits unchanged:
  `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080, 25, 576, 35264, 5624]`.
- Decode sha256 unchanged:
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`.
- Bench census row unchanged:
  `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f`.

A host polling change must not move any of them: the same kernels produce the tokens, only
the wait path changes.

### 4.4 Closed-default gate

The runtime change is an **env/record opt-in, not a default-on change**: the cached-view or
blocking-wait path is selected by an env key (e.g. a `getenv` flag in the HCQ signal path),
and the default remains the memoryview-per-poll path until a landing record with both
targets' same-session measurements (4.1), the envelope result (4.2), and the pins (4.3)
exists. Landing flips the default in a separate reviewed change, never in the same commit as
the mechanism.

## 5. Independence from decode (amendment Q4)

**B3 does NOT wait on decode progress, and decode work does NOT wait on B3.** The amendment
Q4 answer states B3 is an independent prefill runtime lever with its own AMD control
requirement and that nothing in the decode boundary decision licenses serializing it behind
decode; section 4.2 item 3 adds "measure the polling change against both NV and AMD before
landing; do not use decode progress as its gate." B3's landing gate is its own NV+AMD
measurement and the pins above, never a decode milestone, and decode sequencing (amendment
section 4.1) proceeds without waiting on B3.

## 6. HARD STOP + settling command

HARD STOP after this section. No implementation until this scope is reviewed. When
implemented, each change lands closed-default with its own commit and record, and the
settling command is the wait-time probe, run **before and after on both NV and AMD**:

```text
# before: capture the baseline on each target, same session family
python /tmp/probe_warm2.py                 # NV and AMD

# after: same probe, same target, same session family, same env except the opt-in key
python /tmp/probe_warm2.py                 # NV and AMD
```

The probe pattern is the one the campaign already used: monkeypatch `HCQSignal.wait` to wrap
the original with `time.perf_counter` timing, accumulate the time spent inside `wait()` per
call, run a warm pp512 prefill pass (pass2/pass3 steady state), and report the total
`wait()` CPU time across the 10 waits (measured NV baseline: 23.7-23.8 ms). Together with
the wall (`/tmp/measure_warm_prefill.py`) and busy (`/tmp/measure_busy_debug2.py`)
instruments, the after-run must show wait() time collapsing and wall/busy inside the llama
1.15-1.35x envelope on NV, with AMD warm rows unchanged. If the AMD leg cannot run on AMD
hardware from this machine, the change does not land from this NV-only session.

**Report when the scope is executed:** the before/after table for NV and AMD (wall, busy,
wait-time), the envelope ratio, the three correctness pins, the env key and its default, and
which fix shape (a), (b), or (c) each landed change used.
