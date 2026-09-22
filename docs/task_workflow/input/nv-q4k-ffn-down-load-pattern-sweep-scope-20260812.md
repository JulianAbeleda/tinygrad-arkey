# NV Q4 FFN-down 4096x12288 load-pattern sweep - exhaustive scope (audit -> arithmetic -> implement)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `48b8fa696`)
Status: **scope document** for the standing pipeline (audit -> arithmetic -> implement,
process `0515f2539`). Authorizes the MC2-pattern q4k-down diagnostic microbench with the
in-loop census gate and the exact-output AB + +50 us wall promotion contract. No route
admission, no emitter change, no promotion without the wall bar.

**Repo-state note.** The harness extension ([test] `01bbb87c0`) and the executed sweep
record (`6cf7149b8`, `nv-q4kd-load-pattern-measurement-record-20260812.md`) are already on
this branch: the sweep was run in-session and closed NO-GO under every offset convention
(section 6). This scope is the pre-execution contract the audit called for (audit section
7: "a diagnostic microbench under its own scope"); section 6 records the executed outcome
against the gates defined here, so the row is not silently re-opened.

## 1. Why this scope exists

The quant GEMV audit (`nv-quant-gemv-llama-audit-20260812.md`) left exactly one untried
load-pattern surface on the quant row: the Q4 FFN-down shape (4096x12288), the largest
per-shape ratio deficit in the per-shape cross-reference (2.27x llama at audit census) and
the largest untouched kernel family by total mass (18 kernels, 481.50 us/token at the
audit census). MC2 swept the partial Q6 shape, the coop-down Q6 shape, and the q4k
gate/up shape on 08-03 but never the Q4 down shape. This scope defines the missing
diagnostic microbench step: the same MC2 surface family (vec width / rows-per-block /
smem x-staging / prefetch / quad-lane u128) on the down geometry, gated against the
llama-class floor, with the standing in-loop census gate and promotion contract.

## 2. Audit (fresh same-session trace)

### 2.1 The q4k down family in the fresh prime trace

Source: `/tmp/tg_debug_probe_20260812.log` (DEBUG=2 prime-token trace, kernels 1545-2126,
window per `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`).
Family `q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` appears **18 times** (kernel IDs
1620-2029), summing to **484.9 us/token**:

| kernel id | us |
| ---: | ---: |
| 1620 | 31.84 |
| 1638 | 26.34 |
| 1673 | 27.23 |
| 1691 | 26.21 |
| 1726 | 26.53 |
| 1744 | 27.20 |
| 1776 | 26.72 |
| 1794 | 26.82 |
| 1829 | 26.40 |
| 1847 | 26.37 |
| 1879 | 26.40 |
| 1894 | 26.78 |
| 1924 | 27.04 |
| 1939 | 26.46 |
| 1969 | 26.40 |
| 1984 | 26.69 |
| 2014 | 26.43 |
| 2029 | 27.01 |

Aggregates: sum 484.9 us, mean 26.94 us, median 26.61 us, min 26.21 us, max 31.84 us
(the 31.84 outlier is the first node, 1620; the remaining 17 run 26.21-27.23).

### 2.2 Llama floor and ratio (fresh session)

llama's Q4 down node (`mul_mat_vec_q<Q4_K,1,has_fusion>`, d512, Qwen3-8B-Q4_K_M) is
**11.776 us/node** in-loop (same-session fresh floors, `nv-decode-gap-attribution-same-session-20260812.md`).
18 nodes x 11.776 = **211.97 us (~212.0 us)** at the floor.

| side | us/token | us/node | ratio vs floor |
| --- | ---: | ---: | ---: |
| llama floor (fresh) | 212.0 | 11.776 | 1.00x |
| tinygrad fresh trace | 484.9 | 26.94 mean / 26.61 median | **2.29x** |
| tinygrad audit census | 481.50 | 26.75 median | 2.27x |

The fresh-trace ratio (2.29x) confirms the audit census (2.27x) within ~1%; the row is the
largest open quant GEMV ratio and the only never-credited surface at audit time.

### 2.3 Cross-reference: control census

Control arm of the M1 cost-gate campaign (`/tmp/m1_costgate_ab_fixed.json`):
`q4k_g3_lanemap_gemv_epi_ffnresadd_4096_12288` = 18 kernels, 26.75 us median, 594-kernel
control arm total 5494.76 us. The audit's 481.50 us row (18 x 26.75) is the census the
sweep control must reproduce in-loop.

## 3. Arithmetic (sweep space and expected recovery)

### 3.1 Geometry constants

| constant | value | source |
| --- | ---: | --- |
| rows | 4096 | installed shape |
| K | 12288 | installed shape |
| Q4_K block | 144 B = 36 u32 words, 8 groups, 0 mod 16 | `l2_q6k_partial_sweep.cu`, MC2 record |
| k_blocks per row | **48** (K / 256) | audit section 7 |
| blocks per group (BPG) | 12 (48 / 4 groups) | probe constants |
| x stage size (XHALVES) | 12288 halfwords = 24 KB smem | probe constants |
| weight set | 4096 rows x 48 x 144 B = **28.31 MB** | same set as q4k gate/up |
| BW ceiling (k_bw_read) | 5.07 us / 5.58 TB/s standalone | executed sweep |

The down shape shares the gate/up kernel bodies and set size; the only differences are
48 vs 16 blocks/row and 4096 vs 12288 rows (256 vs 768 full-grid blocks at the winning
quad geometry).

### 3.2 Sweep surface

Fixed `k_blocks = 48`. Knobs: vec width (U16 / U32 / u128; Q4_K is 0 mod 16 so u128 is a
pure LDG.128 candidate), rows-per-block (1/4/8/16/32 with 32/128/256/512/1024 threads),
smem x-staging (off / on, 24 KB, staged once per launch outside the timed loop), prefetch
(pf1 / pf2 on u128 rows), lane map (installed G3 vs quad-lane group-of-2 u128). 13-row
CFGS2 table (shape=3) in the probe:

| config | blocks | thr/block | knob deltas |
| --- | ---: | ---: | --- |
| q4kd_legacy (installed control replica) | 4096 | 32 | baseline |
| q4kd_legacy_u16 | 4096 | 32 | vec u16 |
| q4kd_legacy_u128 | 4096 | 32 | vec u128 |
| q4kd_legacy_u128_pf2 | 4096 | 32 | u128 + prefetch 2 |
| q4kd_legacy_xsmem | 4096 | 32 | smem x |
| q4kd_4row_128thr_u32 | 1024 | 128 | 4 rows, u32 |
| q4kd_4row_128thr_u32_xsmem | 1024 | 128 | 4 rows, u32, smem x |
| q4kd_4row_128thr_u128_xsmem | 1024 | 128 | 4 rows, u128, smem x |
| q4kd_4row_128thr_u128_xsmem_pf2 | 1024 | 128 | + prefetch 2 |
| q4kd_8row_256thr_u128_xsmem | 512 | 256 | 8 rows |
| q4kd_16row_512thr_u128_xsmem | 256 | 512 | 16 rows |
| q4kd_32row_1024thr_u128_xsmem | 128 | 1024 | 32 rows |
| q4kd_16row_128thr_u128_quad_xsmem | 256 | 128 | quad-lane u128, 8 lanes/row |

### 3.3 Expected recovery and tok/s translation

Baseline: production HEAD **5.2031 ms/token = 192.19 tok/s** (d512, all promoted
policies). Llama floor node-sum = 212.0 us/token, so the max census recovery at floor is
**484.9 - 212.0 = 272.9 us (~273 us)** per token.

| recovery | us/token saved | new ms/token | tok/s | note |
| --- | ---: | ---: | ---: | --- |
| 0 (installed) | 0 | 5.2031 | 192.19 | baseline |
| promotion bar minimum | 50 | 5.1531 | 194.06 | +1.9 tok/s gate floor |
| 25% of row | 68.2 | 5.1349 | 194.74 | partial win |
| 50% of row | 136.5 | 5.0666 | 197.39 | partial win |
| floor at 1.3x in-loop offset | 210.0 | 4.9931 | 200.28 | attribution ladder step 4 |
| full floor at 1:1 | 272.9 | 4.9302 | **202.84 (202.8)** | ceiling, +10.6 tok/s |

Rule of thumb in the 190-205 band: ~25-27 us/token saved ~= +1 tok/s (derivative 26.9 us
at 192 tok/s, 25.4 us at 200 tok/s, per the attribution doc). The in-loop offset
(standalone 1.26-1.7x faster) keeps partial wins positive: a standalone winner of ~9.0 us
maps to ~11.3-15.3 us in-loop, and any in-loop median below the installed 26.75 is a net
positive regardless of the exact offset.

Standalone floor target: llama 11.776 us/node in-loop => **~6.9-9.4 us standalone**
(11.776 / 1.7 to 11.776 / 1.26; audit shorthand ~8-9 us at the same offset).

## 4. Implement plan (MC2-pattern harness extension)

### 4.1 Tooling steps (as executed in [test] `01bbb87c0`)

1. Extend `extra/llm_research/microbench/l2_q6k_partial_sweep.cu` in place: template the
   q4k family (`k_q4k_legacy`, `k_q4k_v`, `k_q4k_qv`) on `(NKB, BPG, XHALVES)` so the
   identical bodies serve the down shape with `NKB=48, BPG=12, XHALVES=12288`; gate/up
   rows keep `NKB=16, BPG=4, XHALVES=4096`. Rename the template parameter `KB -> NKB` to
   dodge the file's `#define KB 16`. All call sites pass explicit template args (CUDA
   13.2 rejects defaults on `__global__` templates).
2. Add `--shape q4kd` (shape=3) selecting the 13-row CFGS2 table above. The control
   replica is `k_q4k_legacy<48,12>` (the installed `q4k_g3_lanemap_gemv_epi_ffnresadd_
   4096_12288` body); the quad row is `k_q4k_qv<16,true,48,12,12288>`.
3. Reproduce the installed control: **26.75 us in-loop median** (18 kernels, control
   census) with the standalone replica landing in the 1.26-1.7x offset band, i.e.
   **15.7-21.2 us standalone** (26.75 / 1.7 to 26.75 / 1.26). Measured 18.47-18.78 us =>
   offset 1.42-1.45, within band.
4. Numerics: every non-control row spot-checked against the control replica output (row
   totals), one pass; max relative error bound ~2.9-3.4e-3 (fp32 reassociation noise on
   magnitudes ~1e6-1e8).
5. Purity: `--ptxas-options=-v` on the full binary; all 84 entry points 0 spills, 0 stack
   frames; SASS check (cuobjdump/nvdisasm) for 0 LDL/STL; smem x staging (24 KB) sits
   outside the timed loop by construction.
6. BW ceiling: `--bw` streams the exact Q4_K set (144 B/block, 4096 x 48 = 28.31 MB) with
   `k_bw_read` as the L2-resident achievable-bandwidth control (5.07 us / 5.58 TB/s).
7. Timing: best-of-N back-to-back passes inside one launch (`cudaEventElapsedTime`),
   `--iters 2000 --reps 5` for all rows, `--reps 10` re-confirmation on the control and
   the two best rows. Every GPU run serialized with `flock /tmp/nv_gpu.lock -c "<cmd>"`
   (executed run used `/tmp/gpu-bench.lock`; same discipline, 0% util at acquisition,
   lock file untouched).

### 4.2 Go/no-go contract

| target | bar |
| --- | --- |
| standalone floor | ~6.9-9.4 us (11.776 / 1.7 to 11.776 / 1.26) |
| in-loop census | median < 11.776 us/node over the 18 down kernels in the d512 prime trace |
| ranking | standalone winner must clear the standalone floor FIRST (a winner that only clears in-loop estimates is not evidence) |

### 4.3 Promotion contract (post-clearance)

If a row clears the floor in-loop, it becomes an additive-route candidate and must pass the
full gate: exact-output AB (control/candidate token-stream SHA-256 identical; production
HEAD bit-exact token hash `9e6664fd`, fixed-depth pins `9d6b3787...` / `0721c16f...`,
first token 151936, census candidate_set `1b8ea95d...`), census confinement (18 down
bodies 1:1, no side-effect shift), pg3 render-equality for the new kernel (NV sm_120
admission only; AMD/Metal keep the 10 legacy pg3 hashes), isolated same-session d512 wall
bracket with the +50 us promotion bar: candidate median >= **+50 us/token faster than
BOTH bracketing controls** (house bar, `nv-reduce-output-wall-bracket-record-20260809.md`),
then the promotion record + policy update.

## 5. Gates (summary)

| gate | requirement | evidence class |
| --- | --- | --- |
| G1 standalone control | replica in the 1.26-1.7x offset band of the 26.75 us in-loop median | OBSERVED |
| G2 standalone floor | best row <= ~6.9-9.4 us standalone | OBSERVED |
| G3 in-loop census | median < 11.776 us/node, 18 kernels | OBSERVED (decode harness) |
| G4 exact-output AB | token-stream hashes identical | OBSERVED |
| G5 wall promotion | >= +50 us/token vs BOTH bracketing controls | OBSERVED (like-for-like) |
| G6 landing | promotion record + policy, pg3 hash for the new kernel | OBSERVED |

A standalone winner is NOT evidence until G3 passes (MC2 lesson: the gate/up quad won
standalone at 10.00-10.15 us but fused in-loop at 49.2 us, -5% wall, and closed NO-GO
in-loop).

## 6. Executed outcome (repo state at HEAD)

The sweep defined by sections 3-4 was executed on 2026-08-12 (probe `[test] 01bbb87c0`,
record `6cf7149b8`) and closed **NO-GO**:

| config | blocks | thr | us/pass standalone | TB/s |
| --- | ---: | ---: | ---: | ---: |
| q4kd_legacy (control) | 4096 | 32 | 18.47-18.78 | 1.51-1.53 |
| q4kd_legacy_u16 | 4096 | 32 | 21.12 | 1.34 |
| q4kd_legacy_u128 | 4096 | 32 | 22.17 | 1.28 |
| q4kd_legacy_u128_pf2 | 4096 | 32 | 22.03 | 1.29 |
| q4kd_legacy_xsmem | 4096 | 32 | 23.65 | 1.20 |
| q4kd_4row_128thr_u32 | 1024 | 128 | 19.41 | 1.46 |
| q4kd_4row_128thr_u32_xsmem | 1024 | 128 | 11.81-11.88 | 2.38-2.40 |
| q4kd_4row_128thr_u128_xsmem | 1024 | 128 | 13.20 | 2.15 |
| q4kd_4row_128thr_u128_xsmem_pf2 | 1024 | 128 | 12.84 | 2.21 |
| q4kd_8row_256thr_u128_xsmem | 512 | 256 | 13.95 | 2.03 |
| q4kd_16row_512thr_u128_xsmem | 256 | 512 | 13.24 | 2.14 |
| q4kd_32row_1024thr_u128_xsmem | 128 | 1024 | 12.45-12.47 | 2.27 |
| q4kd_16row_128thr_u128_quad_xsmem | 256 | 128 | 11.43-11.45 | 2.47-2.48 |
| bw read (ceiling) | 512 | 256 | 5.07 | 5.58 |

Best row (quad u128 smem) 11.43-11.45 us standalone = 1.23-1.65x above the standalone
floor, in-loop estimate 11.43-11.45 x 1.26-1.7 = **14.4-19.5 us, above the 11.776 us
floor under every offset convention** (at the shape's own measured offset 1.42-1.45:
16.2-16.6 us = 1.38-1.41x above). No row clears the standalone floor, so per the G2/G3
ordering there is no winner to census; the Q4 down row closes NO-GO with this floor table
and the quant row is exhausted except for closed mechanisms. Ledger impact: the installed
family stays at 484.9 us/token (fresh trace) / 481.50 us/token (audit census); the
202.84 tok/s ceiling in section 3.3 is not booked.

## 7. Evidence

- `/tmp/tg_debug_probe_20260812.log` (raw fresh prime trace; family rows parsed into
  `docs/task_workflow/evidence/nv-q4k-ffn-down-prime-family-20260812.json`)
- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json` (parsed prime
  table, window [1545, 2126], sum 5424.9 us, replay factor 0.883)
- `/tmp/m1_costgate_ab_fixed.json` (control census: q4k down 26.75 us x18, 594 kernels,
  5494.76 us)
- `docs/task_workflow/input/nv-quant-gemv-llama-audit-20260812.md` (rank-1 row, gate,
  k_blocks=48)
- `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md` (fresh
  llama floors, tok/s ladder, 5.2031 ms baseline)
- `docs/task_workflow/input/mc2-load-pattern-measurement-record-20260803.md` (sweep
  surface, controls, offset discipline, q4k gate/up results)
- `docs/task_workflow/input/l2-q6k-partial-singlepass-measurement-record-20260803.md`
  (floors, offset, the 466.6 us anomaly explanation)
- `docs/task_workflow/input/nv-q4kd-load-pattern-measurement-record-20260812.md`
  (executed sweep, NO-GO closure, all rows and controls)
- Probe: `extra/llm_research/microbench/l2_q6k_partial_sweep.cu` (extended, 84/84 kernels
  0-spill)
- `nv-reduce-output-wall-bracket-record-20260809.md` (the +50 us promotion bar and
  exact-output contract)
