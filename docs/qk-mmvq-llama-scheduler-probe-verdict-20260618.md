# Q4_K MMVQ llama-scheduler probe — VERDICT: C (scheduler built, not the lever); gap is per-thread codegen (2026-06-18)

Built llama's exact MMVQ decomposition (128 threads/row, K-blocks parallel, `ds_bpermute` warp-shuffle reduce +
LDS cross-warp, one write) and measured it. **The build REFUTES the scheduler-audit hypothesis: the work
decomposition is correctly expressible but is NOT the speed lever.** Every *correct* variant lands 36-48% (≤ the
8-thread coop); the one 55% result was a correctness artifact (unsigned dot4 on signed q8). RX 7900 XTX. No
model/default changes; probes only.

## What was built (the scheduler shape — Phase 1, CORRECT)
128-thread workgroup (4 warps × 32) computes 1 row; `gid=special(rows)`, `tid=special(128)`;
`kblock=tid//8, sublane=tid%8, lane=tid%32, warp=tid//32`; each thread one (kblock, sublane) partial (NO serial
block loop); within-warp reduce via `extra/amd_warp_reduce.warp_reduce_sum` (`ds_bpermute`, 5 butterfly steps,
disasm-confirmed); cross-warp via 4-elem LDS + barrier; one output write. `flat_work_group_size(1,128)`. **fp
version correct (rel 6e-7) — the scheduler shape works.**

## Phase 3 — microkernel speed (the refutation)

| 128-thread/row variant | correct? | % HBM peak |
|---|---|---|
| fp dequant | ✓ rel 6e-7 | **48** |
| plain-int signed dot (4 int-mul + sign-fix/grp) | ✓ rel 0.006 | **36** |
| unsigned dp4a + (+128)-bias correction (Family-A math) | ✓ rel 0.006 | **37** |
| `_sdot4` / v_dot4_i32_iu8 (no bias) | ✗ **rel 5.4** | 55 |
| (ref) fp coop 8-thread | ✓ | 43-48 |
| (ref) opaque asm 8-thread (unsigned+bias) | ✓ | 52 |
| llama / READRAW | — | 70 |

**Every correct 128-thread variant (36-48%) is ≤ the 8-thread coop (43-52%).** The 128-thread structure does NOT
beat 8-thread — more reduction/occupancy overhead per row, less work/thread. The 55% "win" was `v_dot4_i32_iu8`
computing **unsigned×unsigned** (verified by a known-input test: got a_u·b_u, not a·b_signed) → garbage on signed
q8 (rel 5.4). The lowering test had only checked the instruction is *emitted*, never its *value*.

## Root cause of 52→70 (revised, proven by build)
1. **No accessible signed dot4 on RDNA3 via tinygrad's compile.** `v_dot4_i32_iu8` (the only native v_dot4 the
   asm gives) is unsigned; `__builtin_amdgcn_sdot4` *compiles* (with `target("dot1-insts")`) but **scalar-
   fallbacks** (no v_dot4 emitted). llama's hipcc build emits a true signed dot4; tinygrad's `compile_hip`
   doesn't. So tinygrad's correct dot4 must use **unsigned + (+128)-bias correction** → scalar overhead.
2. **Even a true signed dot4 gains only ~+3%** (the wrong-but-fast 55% vs the correct unsigned+bias 52% 8-thread)
   — far from 70%.
3. **So the 52→70 gap is NOT the decomposition and NOT just the dot — it is per-thread CODE QUALITY**: clang's
   register allocation / instruction scheduling / ILP on llama's hand-written C inner loop vs tinygrad's
   custom_kernel codegen. Building llama's *exact* decomposition gave 36-48% — proving the structure isn't the
   lever; the per-thread codegen is.

## Verdict: C → D (scheduler correct but not the lever; remaining gap = per-thread codegen/backend)
The scheduler-audit's "128-thread decomposition is the missing lever" hypothesis is **refuted by build**. The
decomposition is correctly expressible in tinygrad (warp-shuffle + LDS + one write all work) but does not win.
The wall is the per-thread codegen quality (clang-on-hand-C vs tinygrad-custom_kernel) plus the unavailable
native signed dot4 — backend/codegen-internals, now PROVEN (not inferred) by building llama's structure.

**Model route: NOT earned** (no correct variant beats opaque 52%; full-linear gate ≥55% unmet). Not routed.

## Correctness caveat on committed code
The `_sdot4` renderer helper (cstyle.py) + `test_sdot4_lowering.py` are **mislabeled "signed dot4"** — the
emitted `v_dot4_i32_iu8` computes unsigned×unsigned (a-operand signedness unverifiable, but signed q8 gives wrong
results both operand orders). The test only validated *emission*, not *value*. The helper is used by NO shipped
path (probes only). Should be annotated/renamed or backed by a true signed dot4 if one is found. **Lesson: a
dot4 lowering test MUST check the computed value, not just instruction emission.**

## Next within current target (no 14B pivot per standing preference)
- **Signed-dot4 compile-flag investigation** (concrete, small EV ~+3%): find how llama's hipcc emits native
  signed dot4 on gfx1100 and replicate the flag in `tinygrad/runtime/support/compiler_amd.py compile_hip`. Even
  if it works, ~55% not 70%.
- **Accept the ~52% Q4_K MMVQ ceiling** and bank the durable findings (the full quadrant map; the per-thread-
  codegen wall now proven by building llama's structure; the v_dot4_iu8-is-unsigned ISA fact).
- The 52→70 gap is per-thread codegen (register/scheduler/ILP) — a tinygrad-codegen-internals investment, very
  high risk.

## Files / commits
`[docs]` this + `qk-mmvq-llama-scheduler-probe-20260618.md`; `bench/qk-mmvq-llama-scheduler-probe/baseline.json`.
Probes were transient (not committed to model/primitive). No `[codegen]`/`[nn]`, no defaults.
