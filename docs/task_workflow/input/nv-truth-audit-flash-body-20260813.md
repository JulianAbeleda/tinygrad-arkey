# NV ground-truth audit - the flash-score "flat" verdict is wrong (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `8b1acc998`)
Status: **audit + decisive measurement (read-only).** Re-derives the decode gap
from raw evidence JSONs rather than accumulated prose, flags every number that
disagreed, and settles the one open empirical question (flash body) with
device-side CUPTI timing. It supersedes the "flash body flat" verdict in
`nv-flash-vec-llama-first-principles-record-20260813.md`.

## 0. Why this audit exists

Two committed verdicts could not both be true, and they were driving the plan:

1. `nv-flash-vec-llama-first-principles-record-20260813.md` concluded the
   flash-score **kernel body is flat** and "the win is overlap", citing a
   65.49 us vs 65.47 us microgate delta.
2. The node ledgers (`nv-tinygrad-d512-node-ledger-20260813.json` vs the pinned
   llama ledger) show a real flash-score delta: **+68.1 us** device-side
   (5.06 vs 3.16 us/node), and the 08-12 DEBUG=2 attribution showed **+166.5 us**
   (7.79 vs 3.16 us/node).

This audit does not trust either verdict. It re-reads the raw JSONs and runs the
missing measurement (true GPU kernel duration, not Python-loop wall).

## 1. Pinned ground truth (unchanged by this audit)

| quantity | value | source |
| --- | ---: | --- |
| tinygrad production tok/s | 193.54 | `nv-fp32-qk-wall-reconfirm-20260813.json` (DEV=NV, candidate arm) |
| llama tok/s (d512, fresh) | 247.98 | `nv-llama-bench-fresh-20260812.json` (n_gen=10, avg_ns 40.366 ms) |
| llama tok/s (d512, pair) | 245.45 | same file, nsys pair run |
| wall gap | +1130 us/token | 5.167 ms (tinygrad) - 4.036 ms (llama fresh) |
| llama GPU span | 3835.2 us | `nv-llama-d512-node-ledger-20260812.json` |
| tinygrad GPU span (DEV=CUDA) | 5380.5 us | `nv-tinygrad-d512-node-ledger-20260813.json` |
| llama overlap mass | 946.4 us | llama ledger (19.7% span discount, 1 stream) |
| tinygrad overlap mass | 0.0 us | tinygrad ledger (span > node-sum, 7 streams) |

These are the authoritative same-metric pins. The tinygrad span number is the
DEV=CUDA route (the only route nsys can profile at node level); production is
DEV=NV and has the promoted norm fusion, so it is not a production number, but
it is the only apples-to-apples GPU-span number that exists.

## 2. The flash body is NOT flat - decisive device-side measurement

The old microgate (`nv_flash_vec_llama_microgate.py`) timed
`fn(q,cache).realize()` in a Python loop and then one `Device.synchronize()`.
That measures Python dispatch + graph launch (~65 us/graph), not the GPU kernel
body (~3-8 us). A 2-5 us body difference is invisible under 65 us of dispatch.
That alone invalidates the "flat" conclusion.

The second, independent flaw is worse: the microgate compared the tile at
`S=4, L=144` and the vec at `S=4`, but **production decode runs the tile at
`S=48`**. So it never measured the production body at all.

Fresh device-side measurement (this audit): `nsys --trace=cuda`,
CUPTI KERNEL duration, 400 back-to-back launches, isolated score kernel,
DEV=CUDA (the same metric as both node ledgers).

| kernel | structure | device us/node | vs llama 3.16 |
| --- | --- | ---: | ---: |
| llama `flash_attn_ext_vec` (in-situ) | single-pass, S=4, 128 blocks | 3.16 | 1.00x |
| tinygrad production tile (isolated) | tiled, S=48, 384 blocks | **4.19** | 1.33x |
| tinygrad tile, the microgate's "legacy" | tiled, S=4, L=144, 32 blocks x 9 tiles | **25.41** | 8.04x |
| tinygrad vec, the microgate's "candidate" | single-pass, S=4, NCHUNK=9 | **39.61** | 12.54x |
| tinygrad vec, loop bound fixed (NCHUNK=2) | single-pass, S=4, 128 blocks | **10.20** | 3.23x |

Evidence: `docs/task_workflow/evidence/nv-flash-body-device-timing-20260813.json`.
Probe: `extra/llm_research/decode/nv_flash_body_device_timing.py`.

Three things follow from this one table:

1. **The body gap is real.** Production tile is 4.19 us isolated / 5.06 us
   in-situ vs llama 3.16 us. That is +1.03 us/node isolated (+37 us total) or
   +1.90 us/node in-situ (+68 us total). It is not flat, and it is not 166 us
   (the 08-12 +166.5 us conflates host launch+sync with device body).
2. **The vec rewrite is not a drop-in win.** It loops `NCHUNK = ceil(MAXC/512)`
   = 9 chunks, fixed to the full 4608 context, so at Tc=513 it computes ~4.5x
   the columns it needs. Fixed to NCHUNK=2 it is still 10.20 us, 2.4x slower
   than the production tile and 3.2x slower than llama. Routing it as-is would
   regress the flash row by ~6 us/node (~-8 tok/s).
3. **The microgate compared two non-production bodies** (25.4 us tile vs
   39.6 us vec) under 65 us of Python dispatch. Its "flat" is an artifact on
   two independent counts.

## 3. Resolving the two "flash score" numbers that disagreed

The 08-12 attribution (+166.5 us) and the 08-13 CUPTI ledger (+68.1 us) are
both real measurements of different things on different routes:

| route | what it measures | flash score us/node | delta vs llama |
| --- | --- | ---: | ---: |
| DEV=NV DEBUG=2 (08-12) | host launch + per-kernel sync + exec | 7.79 | +166.5 us |
| DEV=CUDA CUPTI in-situ (08-13) | device kernel duration in full graph | 5.06 | +68.1 us |
| DEV=CUDA CUPTI isolated (this audit) | device kernel duration alone | 4.19 | +37.0 us |

The honest device-side gap is **+68 us in-situ** (the number that maps to wall
tok/s). The +166.5 us figure is launch/sync overhead, not recoverable flash
body. The +37 us isolated figure is the pure body ceiling if contention is
removed, which it will not be in the serialized graph.

## 4. Corrected parity ladder

The 08-12 ladder ranked flash score at +166.5 us / +6.4 tok/s (its step 3).
That is wrong on two counts: the mass is +68 us (+2.7 tok/s), and the fix that
was assumed ready (the vec kernel) is 10.2 us, slower than production. The
flash row therefore moves from "structure done, just route it" to "open, needs
a 3.2x faster vec before it is a win".

Corrected flash row:

| claim | before (wrong) | after (this audit) |
| --- | --- | --- |
| body gap | flat (0) or +166.5 us | **+68 us device-side in-situ** |
| vec kernel status | correct + done | correct, but 10.2 us (NCHUNK=2) vs 4.19 us tile |
| ceiling tok/s | +6.4 | **~+2.7** (and only after a real vec speed-up) |

The other ladder rows are unaffected by this audit:

- reduce-output epilogue: 392 us (rank 1) still stands; fp32 q/k route already
  booked +83.5 us wall (`nv-fp32-qk-wall-reconfirm-20260813.json`).
- vocab aux chain: 57.3 us; P1 (packed max,index epilogue) landed in `45b59b4a3`.
- norms/rope/kv: already ahead of llama (fused).
- Q4 FFN-down GEMV body: closed NO-GO 08-12 (q4kd sweep).

## 5. What is actually true about the launch-hiding layer

Re-checked, not re-litigated blindly: llama's 946 us overlap decomposes into
quantize_q8_1 391 us + rms_norm 156 us + rope 33 us (all already fused by
tinygrad) + flash 143 us (the only exposed hidden mass). The graph scan found
flash has zero dependency-independent partners, so the transferable
launch-hiding ceiling is the scan's 17.9-33 us, not 946 us and not the
~0.48 ms the ladder's step 8 labels "launch hiding". This part of
`nv-launch-hiding-substrate-exhaustive-account-20260813.md` holds; the flash
half of it is the only row this audit changes.

## 6. Verdict

- "Flash body flat" is **wrong**; the body gap is +68 us in-situ.
- "Vec kernel done, just route it" is **wrong**; it is 10.2 us (NCHUNK=2) and
  39.6 us (NCHUNK=9), slower than the 4.19 us production tile.
- The flash row is a real but modest lever (~+2.7 tok/s), and it needs a real
  vec speed-up (10.2 -> ~3.2 us) before any routing decision.
- The dominant recoverable mass is unchanged: reduce-output epilogue (392 us),
  residual/plumbing (472 us), vocab aux (57.3 us).

## Evidence

- fresh device timing: `docs/task_workflow/evidence/nv-flash-body-device-timing-20260813.json`
- probe: `extra/llm_research/decode/nv_flash_body_device_timing.py`
- superseded verdict: `nv-flash-vec-llama-first-principles-record-20260813.md`
- llama ledger: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
- tinygrad ledger: `docs/task_workflow/evidence/nv-tinygrad-d512-node-ledger-20260813.json`
- 08-12 attribution: `nv-decode-gap-attribution-same-session-20260812.md`
