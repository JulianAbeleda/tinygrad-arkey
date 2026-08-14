# NV GEMV-body parity audit - the "at parity" claim is true in aggregate, closed per-shape (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `e97293cc4`)
Status: **audit (read-only, no GPU change).** Answers "are the GEMV bodies at
parity, or is there recoverable mass" by re-deriving the three conflicting
GEMV totals from committed evidence plus fresh per-kernel extraction from the
two CUPTI sqlite traces. The answer: the aggregate is at parity; the per-shape
deficits are real but closed with sound mechanisms, not premature like the
flash "flat" verdict was.

## 0. The question behind the question

The parity ladder calls the GEMV row "at parity" and therefore not a lever.
Three different committed numbers are attached to that claim, which is exactly
the kind of drift this audit is meant to kill:

1. `+3.1 us` - 08-12 attribution: tinygrad q4k+q6k (3724.3) vs llama
   mmq+quantize (3721.2).
2. `+105.6 us` - 08-13 CUPTI ledger anchor: tinygrad gemv (4130.3) vs llama
   mmq+quantize (4024.7).
3. per-shape ratios 0.85x-3.65x - quant GEMV audit: the aggregate is not
   uniform.

All three are true; they measure different routes and different windows.

## 1. The accounting that makes "parity" true

llama runs two kernels per matmul: `mul_mat_vec_q` (MMVQ) plus a separate
`quantize_q8_1` activation quantizer. tinygrad folds the quantizer into the
GEMV body, so it has no separate quantize node. The only honest comparison is
the **sum**:

| side | GEMV body | activation quant | total matmul+quant |
| --- | ---: | ---: | ---: |
| llama | ~3579 us (per-shape) / 3239 us (DEBUG=2 window) | ~552 us / 482 us | ~4131 us / 3721 us |
| tinygrad | ~4054 us (census) / 3724 us (DEBUG=2 window) | 0 (folded) | ~4054 us / 3724 us |

On the same DEBUG=2 prime window, tinygrad 3724.3 vs llama 3721.2 = **+3.1 us**.
On the full census, tinygrad 4053.7 vs llama 4130.6 = **-77 us** (tinygrad
ahead). On CUPTI (DEV=CUDA), tinygrad 4130.3 vs llama 4024.7 = **+105.6 us**
(tinygrad behind, and mostly the vocab GEMV, which is slower on the CUDA route
than on production). The three routes disagree on sign because they measure
different windows, node sets, and backends, but all three sit inside a
**~2.5% band**: total matmul+quant work is at parity to within about +/-100 us,
not the ~400 us a naive per-shape reading suggests. That band is ~4 tok/s, so
the GEMV row is not a lever to 240 either way.

## 2. The aggregate masks real per-shape variation

In-loop device numbers, both sides measured in the decode graph (llama CUPTI,
tinygrad DEBUG=2 DEV=NV), from `nv-quant-gemv-llama-audit-20260812.md`:

| shape | llama us/node | tinygrad us/node | ratio | tinygrad total us | delta vs llama |
| --- | ---: | ---: | ---: | ---: | ---: |
| gate/up 12288x4096 (36) | 37.86 | 38.74 | 1.02x | 1394.5 | +30 |
| attention O 4096x4096 (36) | 11.78 | 9.83 | 0.83x | 353.7 | -65 |
| attention Q 4096x4096 (36) | 9.54 | ~9.6 | ~1.0x | ~338 | ~-5 |
| attention K/V Q4 1024x4096 (54) | 3.33 | 4.88/3.87 | 1.47/1.16x | 237.2 | +44 |
| attention V Q6 1024x4096 (18) | 4.90 | 17.89 | 3.65x | 178.9 | +90 |
| FFN-down Q4 4096x12288 (18) | 11.78 | 26.75 | 2.27x | 481.5 | +135 |
| FFN-down Q6 4096x12288 (18) | 28.75 | 35.17 | 1.22x | 633.1 | +112 |
| vocab 151936x4096 (1) | 303.62 | 330.21 | 1.09x | 330.2 | +27 |

tinygrad wins big on attention O/Q and loses big on FFN-down and V Q6; the
sum roughly cancels, which is why the aggregate looks like parity. The three
large deficits (FFN-down Q4/Q6, V Q6) are the only real GEMV mass, ~337 us at
floor. They are all closed.

## 3. Why the three big deficits are closed (and this time it is sound)

**FFN-down Q4 (2.27x, the biggest).** Two independent mechanisms failed:

- Load-pattern sweep (`nv-q4kd-load-pattern-measurement-record-20260812.md`):
  best row 11.43 us standalone, but the measured in-loop offset for this shape
  is 1.42-1.45x, so 16.2-16.6 us in-loop vs llama's 11.78 us. No row clears the
  floor. The shape is bandwidth-bound: llama is at ~2.4 TB/s on the 28.31 MB
  set, and the best sweep row is 2.48 TB/s, so llama is not doing anything the
  load-pattern surface can copy.
- llama's actual algorithm (Q8_1 activation + DP4A) was **built and is a device
  win** (-6.37 us/layer, -28.9%): `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md`.
  It fails the production gate on two independent counts: precision (relative
  L2 0.0020 > 0.001 from Q8_1 activation quantization) and wall (the passing
  layer-8 singleton regresses +6.2 us/token because the two-kernel
  provider+consumer split re-adds launch cost). Closed FAIL_CLOSED, not swept
  under the rug.

**FFN-down Q6 (1.22x).** MC2 coop-down sweep: installed control is the local
optimum; all 17 variants worse. Closed NO-GO.

**Attention V Q6 (3.65x).** MC2/L2 partial sweep: installed split-4 is the
local optimum; all 14 variants worse. Closed NO-GO.

The K/V Q4 remainder (1.47x on the un-admitted 28/54 blocks) is closed by the
shared-Q8 lease boundary; tail expansion was NO-GO.

## 4. Fresh CUPTI cross-check (DEV=CUDA, this audit)

Extracted per-kernel device durations from the committed traces
(`/tmp/tg_node_20260813.sqlite` and `/tmp/llama_tg10_node_20260812.sqlite`):

| tinygrad kernel (DEV=CUDA) | avg us/node |
| --- | ---: |
| gate/up 12288x4096 | 19.10 (unpromoted, no w1w3 fusion) |
| FFN-down Q6 4096x12288 | 47.80 |
| FFN-down Q4 4096x12288 | 24.80 |
| vocab 151936x4096 | 401.28 |
| V Q6 1024x4096 | 15.83 |
| Q/O 4096x4096 | 7.97 |
| K/V Q4 1024x4096 | 3.67 |

The DEV=CUDA route disables the DEV=NV-only promotions (w1w3 fusion, shared-Q8
leases), so these are an unpromoted baseline, not production. They are useful
for one thing: they confirm the **vocab GEMV is the whole CUPTI anchor delta**
(401 vs llama 303.6 = +97 us of the +105.6 us), and that the vocab kernel is
slower on the CUDA route than on production DEV=NV (330 us). The vocab row is
1.09x on production and closed near-parity.

## 5. Verdict

- "GEMV at parity" is **true in aggregate** (total matmul+quant), but it is an
  accounting fold: llama's separate quantize is folded into its GEMV total.
- The aggregate **masks** real per-shape deficits (FFN-down Q4 2.27x, V Q6
  3.65x, FFN-down Q6 1.22x, ~337 us at floor).
- Those deficits are **closed with sound mechanisms** - the flash-style
  premature closure does not repeat here: llama's DP4A algorithm was actually
  built and measured, and failed on precision and wall, not on a broken probe.
- The GEMV row is therefore **not a path to 240**. The recoverable mass is the
  fusion rows (reduce-output 392 us, residual/plumbing 472 us, vocab aux 57 us)
  plus the corrected flash row (68 us). Summed at 1:1 that is ~239 tok/s;
  at the realistic 0.6 body map ~219 tok/s. Full 240-245 additionally needs a
  launch-hiding layer that does not transfer (33 us scan ceiling).

## Evidence

- `nv-quant-gemv-llama-audit-20260812.md` (per-shape llama floors + native census)
- `nv-q4kd-load-pattern-measurement-record-20260812.md` (Q4 down load-pattern NO-GO)
- `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md` (DP4A device win, precision+wall NO-GO)
- traces: `/tmp/llama_tg10_node_20260812.sqlite`, `/tmp/tg_node_20260813.sqlite`
- `nv-decode-gap-attribution-same-session-20260812.md` (+3.1 us accounting)
- `nv-tinygrad-node-ledger-gap-record-20260813.md` (+105.6 us CUPTI anchor)
