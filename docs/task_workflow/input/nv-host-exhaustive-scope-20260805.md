# NV decode host / outside-window workstream — exhaustive scope

Date: 2026-08-05
Status: **scope decided; CPU-only pieces landed this turn; every GPU arm
parked**
Authority: `nv-decode-final-composed-same-session-record-20260805.md`
Frame: Qwen3-8B-Q4_K_M, depth 512, RTX 5090 / driver 595.84, native `DEV=NV`
Constraint in force: no GPU use. Host is the **last** campaign workstream; it
runs only after overlap and fusion have been exhausted.

## Question

What exactly is the host (outside-window) workstream, item by item, and what
can be developed on CPU today without touching a device or a tinygrad
default? Each item below states (a) mechanism, (b) arithmetic of the
recoverable amount and tok/s, (c) exact A/B or gate design, (d) correctness
contract, and (e) CPU-doable-now vs GPU-parked.

## Frame and authority

The composed same-session baseline is native `5.3242440 ms/token`
(187.82 tok/s). The outside-window delta is **239.805 us/token** — native
`321.784` versus llama `81.979` (`nv-decode-native-d512-host-partition-record-20260804.md`;
audit exact `239.804933` in `nv-decode-final-accounting-audit-20260805.md`).
The marker-free native partition closes within tolerance:

| outside component | us/token |
| --- | ---: |
| preparation before first graph call | 247.557 |
| already-drained scalar copyout + `Tensor.item()` | 83.247 |
| Python append/yield tail | 3.096 |
| redundant synchronize after `next()` | 4.358 |
| disjoint sum | 338.458 |

`338.458 - 321.784 = 16.674`, inside `max(50 us, 2%)`. The drain arm proved
the `item()` interval is 4914.801 us apparent but only 83.247 us real after
device drain; the rest is already-submitted device work and is **not
additive** to the outside term (`nv-decode-native-d512-host-partition-record-20260804.md`).

The pre-first 247.557 us sub-partition (`nv-decode-native-d512-predispatch-breakdown-record-20260805.md`,
KEY FACTS): defensive copy/rebind 121.380 + JIT input/signature reconstruction
77.927 (structural graph rewrite 49.284 + input rediscovery/validation 15.479
+ variable/expected metadata 6.312) + pre-TinyJit 35.637 + graph-cache lookup
1.112 = 236.056; the remaining 11.501 us lives in the minor rows (name/expected
checks 2.966, captured-call/run-linear handoff 0.742, `run_linear` entry 3.246,
lookup-return 2.605) plus cross-session drift. The two largest terms
199.307 us are 77.6% of the observed boundary.

Already banked **inside** the composed baseline: P1 descriptor cache + reusable
input shadow `66.662094` (`nv-decode-p0-p1-prefill-and-predispatch-record-20260805.md`,
default-on, full-logit oracle SHA-256 `71c0a2...ae0f0`) and P5 two-capture
feedback ping-pong `91.6365625` (`nv-decode-feedback-pingpong-record-20260805.md`,
redirect-on midpoint). Their sum `158.2986565` gives the honest remaining host
delta at the composed baseline:

```text
239.804933 - 66.6620940 - 91.6365625 = 81.506277 us/token
```

The parent scope's "composed host residual ~210.5 us" is a different rough
frame: `1318.5672 (composed gap) - 1108.082 (support exposure) = 210.485 us`;
it folds quant cores, host, and reconciliation into one number and is not a
ledger subtraction. The two frames coexist; this scope uses the strict delta
frame above and the marginal tok/s table below.

## Item scope

### 0. Accounting: close the P6 unbooked label (CPU, this turn)

(a) Mechanism: the master P6 ledger labels the 69.166-us predispatch result
`unbooked_for_logits` while the composed baseline already contains P1, which
recovered that cost with a full-logit oracle (`nv-decode-final-accounting-audit-20260805.md`
Finding 1; `nv-decode-parity-campaign-reconciled-ledger-20260805.md`).
(b) Arithmetic: zero remaining credit; the label fix does not move tok/s.
(c) A/B: none; a ledger edit.
(d) Contract: the P1 full-logit SHA-256 `71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0`
stays the admission evidence; the 65.536 and 28.372 component rows are never
added (combined arm only).
(e) CPU now: the ledger update is CPU-only and is done by this scope document
and the ledger tool.

### 1. Full-logit predispatch oracle A/B — 69.166 us

(a) Mechanism: the two reversible settled-replay switches —
`JIT_INPUT_DESCRIPTOR_CACHE` (structural descriptor memoization) and
`JIT_REUSE_WRITTEN_INPUT_SHADOWS` (reusable private feedback shadow) — remove
repeated host reconstruction and fresh shadow allocation per token
(`scratchpad/nv_decode_predispatch_ab.py`,
`extra/llm_research/decode/nv_predispatch_full_logits_qualification.py`).
The diagnostic combined arm measured `-69.1655 us/token` with a sampled-token
contract only; P1 then booked `66.662094` with the full-logit oracle.
(b) Arithmetic: at the composed baseline the marginal row is
`1000 / (5.3242440 - 0.069166) = 190.29 tok/s` (+2.47), but the recovery is
**already inside the baseline** via P1; additional credit is 0.
(c) A/B design: fresh-process OFF-A / ON / OFF-B reverse bracket, 16 steady
tokens, same token stream, full-logit equality across all arms
(`nv-decode-p0-p1-prefill-and-predispatch-record-20260805.md`).
(d) Correctness: full-logit SHA-256 equality, `sample.item() ==
logits.argmax()`, snapshots change with position (stale-return rejection),
unchanged program census.
(e) CPU now: hermetic cache/shadow semantics tests exist; the remaining work
is the accounting label (item 0). Any **future** predispatch change must rerun
the full-logit oracle before booking: GPU-parked.

### 2. JIT input/signature reconstruction — 77.927 us (49.284 structural)

(a) Mechanism: `_prepare_jit_inputs` reruns per settled token. The 77.927 us
is structural signature rebuild via UOp substitution + graph rewrite + unbind
(49.284), input rediscovery/validation (15.479), and variable/expected-info
metadata rebuild (6.312). P1's identity-strict descriptor cache already
eliminates the structural rewrite on hit (prepare median 78.095 -> 28.092 us).
(b) Arithmetic: P1-banked ≈ 50.003; remaining ≈ `77.927 - 50.003 = 27.924 us`.
Marginal (full bucket, parent frame): `1000 / (5.3242440 - 0.077927) =
190.61 tok/s` (+2.79). Composed-additional (remaining only):
`1000 / (5.3242440 - 0.027924) = 188.81 tok/s` (+0.99).
(c) A/B: extend the identity cache to the metadata/discovery fast path
(closed default, hermetic unit tests, JIT off = identical behavior), then a
reverse-bracket marker-off A/B/A with token + full-logit equality; reject
unless wall moves ≥ 25 us (`nv-decode-native-d512-predispatch-breakdown-record-20260805.md`).
(d) Contract: identity-strict misses take the full oracle path; changed rank,
dtype, device, or a new allocation with the same contract must reach the
existing fail-closed `JitError` checks; no concrete buffer is ever cached.
(e) CPU now: the cache is shipped (P1, default-on). The remaining 27.9 us is
discovery/metadata; a CPU prototype is possible, wall credit needs the GPU
A/B: GPU-parked.

### 3. Defensive copy/rebind feedback — 121.380 us

(a) Mechanism: the capture writes one input, so `CapturedJit` inserts an
alias-firewall copy (`_copy_input`) every token: fresh buffer/copy-UOp build
16.892 + generic eager-copy execution 99.088. P1-B reuses one private shadow
(28.372 us, copy kept); P5 removes the copy itself by alternating two captures
with distinct fixed return buffers (91.6365625 us redirect-on).
(b) Arithmetic: banked `28.372 + 91.6365625 = 120.0085625`; remaining
`121.380 - 120.0085625 = 1.371 us` (noise). Marginal (full bucket):
`1000 / (5.3242440 - 0.121380) = 192.20 tok/s` (+4.38). The composed baseline
already holds 91.6 of this.
(c) A/B: any further elision is a no-copy arm, which violates the written-input
contract; it needs a lifetime/alias proof (the P2b standard:
`nv-p2b-owned-invocation-input-record-20260805.md`) plus an exact-output wall
A/B. No mechanism is open today.
(d) Contract: the copy and its stream-ordered dependency edge are load-bearing;
P1/P5 preserved both or removed the write entirely with `shadows [0,0]` and
full-logit SHA-256 `31e5cc2c...46ed`.
(e) CPU now: none; any reopen is GPU-parked with a topology/census gate.

### 4. Scalar copyout + `Tensor.item()` — 83.247 us

(a) Mechanism: per token, `int(sampled.item())` walks
`Tensor.data() -> _buffer()` (cast/contiguous/realize/ensure_allocated) then
`Buffer.as_memoryview` (fresh `bytearray` + 4-byte NV D2H copyout + memoryview
index). The drain arm isolated 83.247 us of this after all device waiting.
(b) Arithmetic: `1000 / (5.3242440 - 0.083247) = 190.80 tok/s` (+2.98). This
is the largest remaining open host bucket.
(c) A/B: closed-default staging prototype — one preallocated pinned host
`bytearray`/memoryview reused across tokens, `Buffer.copyout` straight into
it, bypassing the tensor-layer churn; then reverse-bracket A/B/A with token +
full-logit equality. Packed greedy argmax (the device-side alternative) is
NO-GO: ordinary argmax 71.874 -> packed 142.647 us (+70.773),
`nv-packed-argmax-microgate-record-20260805.md`; do not reopen without a
different mechanism.
(d) Contract: sampled token equals full-logit argmax; stream order and the
public generator contract (yield `sampled.item()`) are preserved.
(e) CPU now: the staging prototype is pure Python and hermetic-testable on
CPU (the bytearray/memoryview mechanics do not need NV); wall credit and the
final A/B are GPU-parked.

### 5. Python yield tail — 3.096 us

(a) Mechanism: generator `yield` plus append and `_cached_tokens = tokens[:-1]`
after `item()`. (b) Arithmetic: `1000 / (5.3242440 - 0.003096) = 187.93 tok/s`
(+0.11); below the 25 us host promotion gate. (c) A/B: none warranted; any
change (return-vs-yield) alters the public streaming contract. (d) Contract:
the generator API is the contract. (e) CPU now: analyzed; no mechanism.

### 6. Redundant sync after `next()` — 4.358 us

(a) Mechanism: the diagnostic harness (`scratchpad/nv_decode_group_window_ledger.py`)
synchronizes after every `next(gen)`; `Tensor.item()` already synchronized, so
the extra sync is redundant. (b) Arithmetic: 0 remaining — the composed
baseline timer already syncs once per 32-token window
(`extra/llm_research/decode/nv_shared_q8_progressive_qualification.py`,
`_settled_continuous_windows`), so this harness cost is already absent at the
composed authority. (c) A/B: none; keep the windowed convention. (d) Contract:
synchronization is not a data dependency; token stream unchanged.
(e) CPU now: convention already fixed in the authority.

### 7. Pre-TinyJit generator/model work — 35.637 us

(a) Mechanism: per-token `v_start_pos.bind`, `v_toks.bind`,
`_route_should_use_flash_decode`, `_generation_input_slice`, feedback-slot
selection, `self(...)` dispatch, `out.realize()`, `decode_feedback_phase += 1`,
`start_pos += ntv`, `tokens.append`. (b) Arithmetic:
`1000 / (5.3242440 - 0.035637) = 189.09 tok/s` (+1.27), above the 25 us gate,
but no mechanism is named — the record calls this a bounded observation, not a
target. (c) A/B: none until a sub-instrumented profile names a row; a
CPU-safe cache of the settled flash-route decision is the only cheap piece
(`should_use_flash_decode` is constant once the flash threshold is crossed).
(d) Contract: route selection must stay the single
`FLASH_DECODE_THRESHOLD` authority. (e) CPU now: the flash-route decision cache
is hermetic-testable; wall credit is GPU-parked.

### 8. Settled graph-cache lookup — 1.112 us

(a) Mechanism: `graph_cache` hit inside `run_linear`. (b) Arithmetic:
`1000 / (5.3242440 - 0.001112) = 187.86 tok/s` (+0.04), far below gate.
(c) A/B: none. (d) Contract: n/a. (e) CPU now: nothing to do.

### 9. Per-token ping-pong contract re-check — UNMEASURED, new

(a) Mechanism: `Transformer.generate` calls `pingpong_capture_contract(pair)`
on **every token** when P5 is promoted — it walks both captures'
`expected_input_info`, return buffers, and shadow dicts, then compares
contracts (`tinygrad/llm/feedback_pingpong.py`). This host cost postdates the
host-partition record and is inside the composed baseline's serial path
between graph return and `item()`. (b) Arithmetic: unmeasured; the contract is
stable across settled tokens, so the per-token recheck is pure redundancy
after capture. (c) A/B: cache the admitted verdict keyed by both captures'
identity, revalidate only on capture boundaries; hermetic contract tests;
then a drain-style measurement on GPU. (d) Contract: the cached verdict must
be exactly the oracle's, including the `written_input_shadows [0,0]` and
`input_contract_mismatch` rejections. (e) CPU now: the caching change and its
hermetic tests are CPU-safe and default-off; the wall number is GPU-parked.

### 10. Micro host rows — each ≤ a few us, CPU-analyzed

`float(temperature.item())` in `decode_with_logits` per token (CPU tensor,
hoistable when `temperature` is an immutable CPU scalar); per-token
`self._cached_tokens = tokens[:-1]` O(depth) slice; `out.realize()` is already
a no-op on graph returns (buffer identity). None reaches the 25 us gate;
listed for completeness of the exhaustive enumeration.

### 11. Closed routes relevant to the host buckets

`packed_argmax`: NO-GO (+70.773 us, `nv-packed-argmax-microgate-record-20260805.md`).
`p2b_owned_invocation_input`: TOPOLOGY NO-GO, 841 -> <=804 gate failed, zero
credit (`nv-p2b-owned-invocation-input-record-20260805.md`); the second
one-per-fused-block copy class stays UNPROVEN
(`nv-p2b-second-copy-class-static-audit-20260805.md`). No-copy predispatch
arms: closed by the written-input contract
(`nv-p5-sampler-feedback-tail-record-20260805.md`).

## Arithmetic summary (marginal at the composed baseline; each row assumes
only that bucket is recovered, not a stacking table)

| item | us/token | tok/s after | +delta |
| --- | ---: | ---: | ---: |
| host outside-window delta (full) | 239.805 | 196.68 | +8.86 |
| full-logit predispatch A/B (banked via P1) | 69.166 | 190.29 | +2.47 |
| defensive copy/rebind (banked P1+P5) | 121.380 | 192.20 | +4.38 |
| JIT input/signature reconstruction | 77.927 | 190.61 | +2.79 |
| scalar copyout + `Tensor.item()` | 83.247 | 190.80 | +2.98 |
| pre-TinyJit | 35.637 | 189.09 | +1.27 |
| Python yield tail | 3.096 | 187.93 | +0.11 |
| redundant sync (absent at composed) | 4.358 | 187.97 | +0.15 |
| graph-cache lookup | 1.112 | 187.86 | +0.04 |
| remaining at composed baseline (P1+P5 banked) | 81.506 | 190.74 | +2.92 |

Composed-additional claimable (beyond P1+P5, record-justified caps): JIT
27.924 (+0.99), item 83.247 (+2.98), pre-TinyJit 35.637 (+1.27), copy 1.371
(+0.05), cache 1.112 (+0.04), yield 3.096 (+0.11). Partition rows are **not
additive**; the 239.805 delta is the hard ceiling. The ledger tool flags any
claim set whose total plus already-banked P1+P5 passes it (non-additivity
warning) and fails closed on per-item over-claims.

## Recovery ordering

Host is scheduled last, after overlap and fusion, whose GPU arms are parked
behind their own stops (`nv-decode-exposure-overlap-host-forward-scope-20260805.md`).
Within the host workstream:

1. CPU now (this turn): exhaustive scope, ledger tool, hermetic tests,
   accounting label fix; no defaults changed.
2. CPU-dev-able next (default-off, hermetic): ping-pong contract-verdict cache
   (item 9), staged `item()` copyout prototype (item 4), flash-route decision
   cache (item 7). All three still need a GPU wall A/B before booking.
3. GPU-parked, ranked by marginal tok/s: item copyout 83.247 (+2.98) > JIT
   remaining 27.924 (+0.99) > pre-TinyJit 35.637 (+1.27, needs a named
   mechanism first) > copy residual 1.371 (+0.05).
4. Closed: yield/sync/cache rows (below gate), packed argmax, P2b.

## HARD STOP gates

- No GPU arm for any host item unless the full-logit SHA-256 equality contract
  (off/on/off arms, same token stream, `sample == argmax`, position-advance
  check) is part of the A/B, per the P1 admission rule.
- No wall A/B books a host recovery unless wall moves by at least 25 us on a
  reverse bracket with exact tokens and unchanged program census.
- No no-copy or copy-elision arm without a bounded writer-to-reader slot and
  lifetime proof (P2b standard); the copy and its dependency edge are
  load-bearing.
- Packed greedy argmax stays closed; any new sampler mechanism must clear an
  included-cost microgate first (the packed route regressed +70.773 us).
- No claim in the ledger tool may exceed an item cap, reference a NO-GO route,
  be unknown or negative, or claim an already-banked item; the tool fails
  closed on each of those. Claim sets that push banked + claimed past the
  239.805 delta are refused as bookings and emitted with a non-additivity
  warning instead.
- The campaign GPU ban is in force: nothing above may run on NV until the ban
  is lifted by the campaign owner.

## References

- `nv-decode-native-d512-host-partition-record-20260804.md` (outside partition)
- `nv-decode-native-d512-predispatch-breakdown-record-20260805.md` (pre-first rows)
- `nv-decode-native-d512-predispatch-ab-record-20260805.md` (69.166 diagnostic)
- `nv-decode-p0-p1-prefill-and-predispatch-record-20260805.md` (P1 booking, oracle)
- `nv-decode-feedback-pingpong-record-20260805.md` (P5 booking)
- `nv-packed-argmax-microgate-record-20260805.md` (NO-GO)
- `nv-decode-final-accounting-audit-20260805.md` (Findings 1, 4)
- `nv-decode-final-composed-same-session-record-20260805.md` (composed baseline)
- `nv-decode-exposure-overlap-host-forward-scope-20260805.md` (parent scope)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booked total)
- `nv-decode-parity-p6-residual-priority-ledger-20260804.md` (P6 host rows)
- `nv-p5-sampler-feedback-tail-record-20260805.md`, `nv-p2b-owned-invocation-input-record-20260805.md`,
  `nv-p2b-second-copy-class-static-audit-20260805.md`, `nv-decode-native-d512-device-window-record-20260804.md`,
  `nv-decode-cuda-d512-group-span-ledger-record-20260804.md`
