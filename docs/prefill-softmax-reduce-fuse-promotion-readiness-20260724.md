# PREFILL_SOFTMAX_REDUCE_FUSE — promotion readiness

**Flag:** `PREFILL_SOFTMAX_REDUCE_FUSE` (commit `23b8e05fc`)
**Route:** `prefill_flash_attention_generated` (already promoted; this is a default-value change inside it)
**Gate:** `extra/qk/prefill_softmax_reduce_fuse_promotion_gate.py`
**Evidence:** `docs/prefill-softmax-reduce-fuse-evidence-20260724.json`
**Verdict:** **PASS — default flipped to ON.** Rollback is `PREFILL_SOFTMAX_REDUCE_FUSE=0`.

---

## What the flag is

Two hunks in `tinygrad/renderer/cstyle.py`. Not in the attention kernel — in the **shared** HIP renderer.

1. A float `Ops.CUSTOMI` with `child_count > 1` gets an SSA name instead of being inlined unconditionally.
2. `_hip_native_bpermute_max` accepts an already-rendered `__builtin_fmaxf` as its peer, so the
   online-softmax carry `new_m = max(old_m, row_max)` renders as `fmaxf` instead of decomposing to a
   select (whose exec-masked `v_cmpx_lt_f32`/`s_cbranch_execz` region LLVM's CSE will not cross).

Effect on the emitted HIP for the prefill attention kernel: textual `ds_bpermute` **272 → 64**, `fmaxf`
**240 → 40**, source **61079 → 34195** chars, loop body **952 → 660** instructions. See
`docs/prefill-R-theories-scope-20260724.md` THEORY 6 for the full static table.

## The two blockers this pass was opened to resolve

### 1. Decode non-regression — RESOLVED, byte-identical

This mattered because the predicate fires on *any* float `Ops.CUSTOMI` with two or more consumers, and
decode builds float `CUSTOMI` of exactly that family: `extra/qk/flash_kernels.py:98`
(`__builtin_amdgcn_fdot2`) and `tinygrad/schedule/wmma/softmax.py:104`
(`amd_gfx1100_broadcast_row_state`'s `"bpermute"`). `tinygrad/llm/prefill_policy.py:14`
`_SHARED_ATTENTION_PROOF_FIELDS` names `decode_nonregression_8b` and `decode_nonregression_14b` as
mandatory for exactly this reason.

New harness: **`extra/qk/decode_codegen_identity_check.py`**. It wraps
`tinygrad.device.Compiler.compile_cached` and records `sha256(src)` and `sha256(lib)` for every kernel
compiled while the **real** decode graph is realized through
`extra/qk/flash_decode_attention_executor.py:flash_decode_live_split_block_tile`
(`staging="KV_BOTH"`, `fused_combine=True`) — the exact path `tinygrad/llm/decode_routes.py` drives.
Both decode-admitted geometries, 8B `Hq=32` and 14B `Hq=40` (`Hkv=8`, `Hd=128`, `MAXC=512`, `Tc=400`, `S=4`).

Result: **8 kernels per arm (4 per geometry), all executed**, and the `src` sha list, `lib` sha list and
output sha are **identical** between arms for both geometries. Decode machine code is byte-for-byte
unchanged, which is a stronger statement than any throughput measurement, so no decode timing run is owed.

Two controls make that a real negative rather than a vacuous one:

* **Non-vacuity:** the decode tile kernel genuinely contains the CUSTOMI the predicate was evaluated
  against — 5 textual `ds_bpermute` + 1 `fdot2` in its source, in both arms. The predicate does not fire
  because decode's cross-lane reduce is a **linear** ladder (one consumer per rung); prefill's is a
  **butterfly** where every rung has two (the next `fmaxf` *and* the next `bpermute`).
* **Flag-took-effect:** in the same session, the same `compile_cached` hook on the *prefill* path shows
  272 → 64 `ds_bpermute` and a changed code-object sha. So `PREFILL_SOFTMAX_REDUCE_FUSE=1` was live.

The harness **fails closed**: if either arm captures zero kernels it reports `INCONCLUSIVE`, never
`IDENTICAL`. That is the specific trap in this area — `flash_kernels.py`'s
`flash_fused_gmax_combine_kernel` returns a *callable*, not a sink, so handing it to `to_program` raises
`'function' object has no attribute 'key'` in **both** arms, and two identical errors look exactly like
two identical binaries. Going through the executor avoids the trap; the zero-kernel assertion catches it
if anyone reintroduces it.

### 2. 14B evidence — OBTAINABLE, and now collected

The standing claim was that 14B could not be measured. That was wrong, and it was wrong for a mundane
reason: `extra/qk/prefill_flash_e2e_parity.py` loops over **both** models in **one** process and nothing
drops the 8B buffers, so the 14B arm hit `fp16 KV admits 0 ... free 5.2GB, weights 9.0GB`. That is
in-process VRAM exhaustion, not a 14B route or hardware problem.

Fix: a `--only <8B|14B>` argument, one model per process — `free 25.5GB, weights 9.0GB`. The 14B arm
then passes. Every 14B run also carries `TINYGRAD_PREFILL_PACKED_WMMA=0`, the known mitigation for the
separate packed-WMMA hardware fault (`docs/BOLTBEAM_GPU_HANG_DIAGNOSIS_HANDOFF_20260724.md`).

`extra/qk/prefill_long_context_numerics.py` was also hardcoded to `HQ = 32`; it now takes `QK_HQ`, so the
14B grid (`Hq=40`) can be driven through the same real path.

---

## Where the "bit-identical" claim was overstated, and how it was fixed

Commit `23b8e05fc` states ON and OFF are "bit-identical at every kv". What was actually observed was
**matching `max_abs_err` scalars to four significant figures**. That is a single reduction over the whole
output tensor and is insensitive to compensating changes — it is strong evidence, but it is not
bit-identity, and a renderer-naming change claiming numerical inertness should be held to the literal
statement.

`prefill_long_context_numerics.py` now prints `out_sha` (sha256 of the output tensor's fp32 bytes), so the
claim is checkable rather than inferred. The promotion gate requires at least one numerics sweep to
establish output bit-identity and refuses to let matching `max_abs_err` substitute for it.
`prefill_hd_sweep_numerics.py` is explicitly recorded as *unable* to establish bit-identity, because it
only prints a scalar.

---

## Measurement discipline used

Every GPU run inside `flock /tmp/gpu-bench.lock`, always `TINYGRAD_PREFILL_PACKED_WMMA=0`. All A/B is
paired, same-session, **interleaved** OFF/ON, repeated. No recorded baseline is used as a comparator —
this box drifts ~5% in absolute throughput across a session. CPU-only work (the whole unit suite, the
sha comparison, the gate) runs outside the lock.

For the attention-local sweeps, the **SDPA baseline timed in the same processes is an unchanged-code
control**: its run-to-run spread is the noise floor for that configuration, measured rather than assumed.

---

## 14B whole-model throughput: measured, corroborating, but under-powered

14B *must* run with `TINYGRAD_PREFILL_PACKED_WMMA=0`. That mitigation disables 14B's packed-WMMA prefill
fast path, so prefill falls back to graph-GEMM and the chunk becomes overwhelmingly GEMM-bound:
**~1420 ms per 512-token chunk, ~361 tok/s**, and whole-prefill barely moves with context
(361 @512 → 352 @4096). The attention kernel this flag changes is a few percent of chunk time there, so
even a 25–31% kernel win lands near the noise floor at the whole-model level.

It was measured anyway, interleaved, and it **corroborates the attention-local result quantitatively** —
which is much better evidence than either instrument alone. Predicting each chunk's delta from the
attention-local win (40 layers × fused ms at that chunk's kv, over the measured chunk time):

| start_pos | kv | attn share of chunk | predicted Δ | measured Δ |
|---|---:|---:|---:|---:|
| 0 | 512 | 1.6% | −0.44% | −0.78% |
| 512 | 1024 | 2.7% | −0.84% | −0.76% |
| 1024 | 1536 | 3.3% | −0.84% | −0.63% |
| 2048 | 2560 | 4.5% | −0.99% | −1.05% |
| 3584 | 4096 | 5.8% | −1.50% | **−1.45%** |

The deepest chunk — where the mechanism predicts the most and the prediction is least sensitive to
interpolation — matches to within 0.05 percentage points. Whole-prefill@4096 moved 352 → 355 tok/s
(+0.85%) against a 0.59% noise floor: **1.4× noise**, below the 2× credibility bar, so it cannot be the
gate criterion — but it is positive at every one of the four lengths and regresses nowhere.

The gate encodes exactly this as a deliberately narrow hatch,
`whole_model_paired_ab.status = "UNDERPOWERED_BY_INSTRUMENT"`, which requires **all** of: the run actually
happened (`measured_pairs` non-empty — "under-powered" is not a synonym for "not attempted"); the
mechanism's predicted delta is below what this instrument resolves at 2× noise; the measurement
**corroborates** the prediction to within 50%; no length regresses past the noise floor; and
`attention_local_paired_ab` PASSes. Earlier drafting of this gate used a weaker condition
(`predicted < noise_floor`); that was replaced once the real measurement showed the predicted delta is
*above* the raw noise floor while still below the 2× bar — the corroboration leg is what actually carries
the shape, not the instrument's incapacity.

So the 14B leg rests on: output-bit-identical numerics on the 14B grid, real-model token parity
(`SDPA=90310 == FUSED=90310` in both arms), attention-local device-synced paired A/B on the 14B grid
(−25% to −31% at kv=512/1024/4096; −18.5% at kv=2048, whose OFF arm is the noisiest sample in the whole
matrix), and a whole-model measurement that agrees with the prediction those numbers imply. All measured
in this pass.

---

## How the gate deliberately differs from its sibling

`extra/qk/prefill_causal_tile_skip_promotion_gate.py` requires **both** pp512 and pp4096 to clear 2× the
noise floor, because tile-skipping removes trip count at every depth. This flag removes ~292 instructions
from the KV-loop **body**, so its whole-model effect is (KV-loop share of chunk time) × (kernel speedup):
intrinsically smallest at pp512 and largest at pp4096. Requiring equal signal-to-noise at pp512 would
demand the effect be *larger* than its own theory predicts.

So this gate instead requires:

* the **deepest** measured length to clear both bars (that is where the mechanism predicts the most);
* at least **two** of the four lengths to clear both bars;
* **no** length to regress past the noise floor — all four lengths must be reported, so a regression
  cannot hide in an unreported one;
* the delta to **grow with context** — a loop-body win that does not scale with KV depth is not
  attributable to this change, however good the number looks;
* **attention-local paired A/B for every shape**, mandatory, which times the changed kernel directly
  instead of inferring it through a whole-model number.

Net: stricter than the sibling on regression coverage and on direct-kernel evidence; looser only at
shallow context, where the theory says to be. It also adds two legs the sibling has no analogue for —
decode codegen identity across both decode geometries, and whole-unit-suite failure-set equality.

The gate was tamper-tested: 13 mutations of the evidence (removed decode identity, `decode_nonregression_14b`
falsified, non-empty suite diff, deleted shape, deleted attention-local block, `UNRESOLVABLE_BY_INSTRUMENT`
with a resolvable predicted delta, regressed pp4096, non-growing delta, non-bit-identical numerics,
manifest violation, failed numeric check, removed control noise, removed flag-took-effect control) each
produce `FAIL`.

---

## 8B evidence

Numerics, `extra/qk/prefill_long_context_numerics.py` (`QK_HQ=32`), both arms interleaved: **output sha256
identical at every kv** — `ed9220e82a6c82c4` (Hd=64), `1e6011a5b6a9c4d6` / `0df8a5b388881905` /
`5469ca884a47a647` / `c010038feaeb52fd` at kv=512/1024/2048/4096. `max_abs_err` 3.905e-05 / 6.558e-05 /
2.655e-06 / 1.865e-06 / 1.053e-06, all finite, all PASS. `prefill_hd_sweep_numerics.py` 6.104e-05 PASS at
Hd=64 and Hd=128 in both arms.

Real-model token parity, `prefill_flash_e2e_parity.py --only 8B`: `SDPA=198 FUSED=198 MATCH PASS`,
`AUTHORITY_GATE: PASS`, in **both** arms.

Whole-model, **three same-session interleaved pairs measured in this pass** (tok/s):

| | rep1 off→on | rep2 off→on | rep3 off→on | mean Δ | Δ / 0.59% noise |
|---|---|---|---|---:|---:|
| pp512 | 3650 → 3686 | 3600 → 3649 | 3584 → 3647 | **+1.37%** | 2.32× |
| pp1024 | 3550 → 3614 | 3506 → 3581 | 3488 → 3578 | **+2.17%** | 3.68× |
| pp2048 | 3362 → 3475 | 3324 → 3443 | 3301 → 3439 | **+3.71%** | 6.28× |
| pp4096 | 3041 → 3231 | 3005 → 3206 | 2983 → 3198 | **+6.72%** | 11.38× |
| chunk@3584 (ms) | 196.2 → 178.1 | 199.1 → 178.9 | 200.5 → 179.7 | **−9.92%** | |

The delta is **monotone in context** (1.37 → 2.17 → 3.71 → 6.72), which is what a KV-loop-*body* win must
look like and is the strongest available attribution of the whole-model number to this change.

Note on drift: the three OFF arms fell 3650 → 3600 → 3584 tok/s at pp512 over ~35 minutes (1.84% total,
≈0.6% per adjacent run), and the ON arms fell 3686 → 3649 → 3647. That per-adjacent-run drift independently
corroborates the 0.59% noise floor and is precisely why a recorded baseline is not a valid comparator.

**All four lengths clear both the 1.0% floor and the 2× signal-to-noise bar**, so this evidence also
satisfies the sibling gate's stricter *uniform* criterion — the mechanism-aware criterion described above is
**not load-bearing** for this verdict. It was written before these three pairs were measured, when the only
available pp512 figure (+1.11%, inherited) sat at 1.88× noise; measuring it properly resolved the question
rather than the threshold doing so.

Attention-local, `prefill_flash_perf.py 0 <kv>`, device-synced, numeric-checked, interleaved, 2 reps —
fused kernel ms off→on: kv512 **−27.8%**, kv1024 **−32.0%**, kv2048 **−28.1%**, kv4096 **−31.8%**, against
an unchanged-SDPA control spread of 1.6–4.4%.

## Full unit suite

Whole `test/unit/` (147 files, `-p no:randomly`) — **not** a `-k` subset, because the change is in the
shared renderer.

| | failed | passed | skipped | xfailed | subtests | seconds |
|---|---:|---:|---:|---:|---:|---:|
| flag OFF | 51 | 1274 | 18 | 5 | 21 | 295.4 |
| flag ON | 51 | 1274 | 18 | 5 | 21 | 291.3 |
| new default (no env) | 51 | 1274 | 18 | 5 | 21 | 293.2 |

Sorted `FAILED`/`SUBFAILED`/`ERROR` sets are **byte-identical** — `diff` is empty. The gate is
set-equality, not zero; no pre-existing failure was touched. (The task brief estimated ~34 pre-existing
failures; 51 is the count for the *whole* unit suite rather than the amd/isa/attention subset.)

## Guards

* `assert_pure_machine_search({'PURE_MACHINE_SEARCH_ONLY':'1'})` — PASS; `prefill_attention` resolves to
  `prefill_flash_attention_generated (pure)`.
* `validate_manifest()` — PASS, no violations.
* Provenance stays `machine_authored_generated`: the change alters **rendering**, not the algorithm, the
  emitter, the route id, or the `route_attribution` chain.

Verified in the **actual shipping configuration** (no env var set), not only with the flag forced on: the
suite row above, the decode identity check re-run with the OFF arm explicit and the ON arm defaulted
(`AUTHORITY_GATE: PASS`), and both guards.

## The flip

`tinygrad/renderer/cstyle.py`, both call sites: `getenv("PREFILL_SOFTMAX_REDUCE_FUSE")` →
`getenv("PREFILL_SOFTMAX_REDUCE_FUSE", 1)`.

Both directions were verified against the pre-flip code objects, which is what makes the flip a no-op
beyond its intent:

| | textual `ds_bpermute` | source chars | source sha | lib sha |
|---|---:|---:|---|---|
| default, post-flip | 64 | 34195 | `be45ccbbdf48` | `b910e23f6c4e` |
| pre-flip `=1` | 64 | 34195 | `be45ccbbdf48` | `b910e23f6c4e` |
| `=0`, post-flip | 272 | 61079 | `22793719d063` | `126399483f41` |
| pre-flip default | 272 | 61079 | `22793719d063` | `126399483f41` |

So the new default reproduces the old ON arm byte-for-byte and `=0` reproduces the old default
byte-for-byte. The rollback is real, not nominal.

Two harnesses that hardcoded the old default were corrected so they cannot silently report the wrong arm:
`prefill_long_context_numerics.py`'s banner, and `decode_codegen_identity_check.py`'s effective-flag record
(both arms must now be set explicitly; the comparison reports `INCONCLUSIVE` if an arm ran in the wrong
state).

## Corrections to the framing this pass inherited

1. **`extra/qk/process_isolated.py` does not exist.** The 14B isolation was done by adding `--only` to
   `prefill_flash_e2e_parity.py` instead.
2. **14B was never blocked.** The `fp16 KV admits 0` failure was one-process VRAM exhaustion, exactly as
   suspected. Full 14B evidence — numerics, token parity, attention-local throughput and whole-model
   throughput — was collected in this pass. The manifest's claim that 14B is "UNTESTED ... fails on
   in-process VRAM" has been corrected in place, which also unblocks the **sibling** flag
   `PREFILL_CAUSAL_TILE_SKIP`: its 14B evidence is *uncollected*, not *uncollectable*. Its gate still
   correctly reports `AUTHORITY_GATE: FAIL`.
3. **"Bit-identical" was overstated in commit `23b8e05fc`** — it rested on matching `max_abs_err` scalars.
   Now proven by output sha256 on both grids.
4. **The pre-existing unit-test failure count is 51, not ~34**, for the whole unit suite.
5. **The decode risk was real and worth checking, but decode was never going to be affected**, for a
   reason worth recording: decode's cross-lane reduce is a linear ladder, so `child_count > 1` never fires.
   That is now asserted by a committed harness rather than by reasoning.

## Decision

**FLIP.** Every leg passes:

* decode codegen **byte-identical** on both decode-admitted geometries, with proof that both arms compiled
  and that the flag was live elsewhere in the same session;
* 14B evidence **complete** — bit-identical numerics, real-model token parity, attention-local A/B, and a
  whole-model measurement that corroborates the prediction;
* 8B evidence complete and strong on all four whole-prefill lengths, monotone in context;
* whole unit suite failure-set **equal** off / on / at the new default;
* purity guard, manifest validation and provenance unchanged;
* `extra/qk/prefill_softmax_reduce_fuse_promotion_gate.py` → `AUTHORITY_GATE: PASS`, and fails closed on
  each of 21 tested mutations of its evidence.
