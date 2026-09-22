# NV packed Q4_K x Q8 IMMA executable checkpoint

## Correctness PASS

`extra/llm_research/prefill/nv_q4k_imma_fragment_microgate.py` is a
research-only composite provider/kernel.  It reads canonical GGML Q4_K uint32
words, selects unsigned low/high nibbles, packs the exact m16n8k32 lane
fragments, executes signed-int8 IMMA, applies Q8/Q4 scale and min correction,
and writes guarded FP32 output.  It never creates an expanded weight tensor or
group-partial allocation.

The adversarial `(16,8,256)` fixture passes:

- all eight group int32 dot products bit-exact, nonzero, maximum error zero;
- complete corrected output finite and allclose to the independent oracle;
- maximum absolute error 0.0001220703125, mean 0.0000170916;
- `IMMA.16832.S8.S8` SASS, no HMMA or IDP.4A;
- complete kernel: 40 registers/thread, stack/local/spill zero.

This also identified the correct tinygrad/PTX A-fragment map:
`row=lane/4+8*(reg&1)`, `k=4*(lane%4)+16*(reg/2)+byte`.  The earlier alternate
map was rejected by the numerical gate.

## Production-shaped timing FAIL

The same bounded-final-output kernel executes `(512,12288,4096)` directly from
28.312 MB packed weights and writes only the 25.166 MB FP32 result.  Hot R9:

`4.0186, 4.0149, 4.0339, 4.0439, 4.0350, 4.0433, 4.0403, 4.0355, 4.0434 ms`

Minimum is 4.015 ms and median 4.035 ms per projection.  This is numerically
and structurally executable but fails the invest timing gate: corrected FP16 is
about 0.570 ms and llama's packed main is about 0.167 ms.

CTA packing alone does not fix it.  Grouping the same 16x8 warps into 2/4/8
warp CTAs measured minima 4.045/4.894/4.497 ms.  The kernel repeats packed
weight unpack across 32 M tiles and performs scalar byte construction around
each IMMA.  The next required geometry is a shared packed/unpacked weight stage
feeding a 128-row multi-warp tile (plus wider N ownership), not more launch
grouping.  An initial shared-stage spelling hung and was removed; the passing
non-shared kernel remains the checkpoint.

Production routing and admission remain closed.

## Faithful topology and bounded Stream-K

The final research revision implements the audited 128N x 128M x K256,
eight-warp service tile with 76-dword Q4 rows, 36-dword Q8 rows, and a
57,856-byte dynamic shared arena. Warp pairs own N32 and partners own
alternating M8 bands.

The legal 128x128x256 fixture passes the intended llama half2 coefficient and
Q8 metadata roundpoint: maximum absolute error 0.0001220703125 and mean error
0.0000092825. All eight raw IMMA group dots are bit exact. The separately
reported delta from the older unrounded FP32 algebra oracle is maximum
0.340637 and mean 0.040661.

For M512/N12288/K4096, 6,144 tile/K256 units are partitioned over 170
persistent CTAs. Interior tiles write final output directly. Each CTA emits at
most two boundary partials, and a deterministic two-slot map lets the fixup
write boundary tiles without clearing the output.

Hot R9 complete main plus fixup is:

0.499296, 0.499808, 0.498592, 0.498560, 0.500000, 0.499072,
0.499584, 0.498880, 0.498208, 0.498272, 0.498464, 0.498816 ms.

Minimum is 0.498208 ms and median 0.498816 ms. Adding the independently
qualified 3.584 us compact Q8 producer gives approximately 0.501792 ms
minimum, about 68 us faster than the 0.570 ms FP16 control.

Native runtime qualification found a loader constraint: dynamic-shared main,
fixup, and zero entry points must be separate cubins. Combining sibling entry
points or transitioning zero-to-dynamic-shared can watchdog. The ownership
fixup removes the zero launch.

Admission remains closed pending real-model nonzero combined-chain guards,
readonly hashes, and resource/SASS qualification.

The executed work is 6,291,456 m16n8k32 IMMA instructions, or approximately
51.54 tera integer operations when a multiply-add counts as two operations.
At the 0.498208 ms minimum this is about 103.5 TOP/s. Useful traffic is about
113.25 MB of Q4 data (four M tiles), 201.33 MB of Q8 data (96 N tiles),
25.17 MB of final output, and 41.42 MB of boundary partial writes plus reads,
or roughly 765 GB/s at the measured minimum. The kernel therefore remains
well away from a pure packed-byte bandwidth roof; instruction issue and tensor
duty are material. The winning specialized SASS uses 135 registers/thread,
zero stack/local spill, 33 static IMMA sites, and no HMMA or DP4A.

## LDSM and capturable lifecycle checkpoint

The Q4 operand was transposed into the faithful llama orientation and loaded
as the IMMA A fragment with `ldmatrix.sync.aligned.m8n8.x4.b16`.  The isolated
lane-map gate is bit exact.  Full real `blk.0.ffn_gate` qualification also
passes over all 6,291,456 outputs: maximum absolute error is 1.6689301e-6,
mean absolute error is 1.1045216e-8, direct tiles are bit exact, and the split
tile maximum is 1.6689301e-6.  The R1 complete chain is 504.48 us (3.616 us
producer and 489.44 us main).  SASS has four static LDSM and 32 static IMMA
sites, uses 167 registers/thread, and has no spill.

Direct `NVProgram` calls could not participate in the model TinyJit lifecycle.
A finalized `Ops.PROGRAM` adapter now carries the one-symbol cubin, exact
buffer dependency ABI, fixed scalar values, launch geometry, and 58,880-byte
runtime shared allocation.  An isolated producer/main/fixup TinyJit capture
and replay gate passes with finite exact-zero output; host-call plus device
synchronize R9 is 0.6241 ms minimum and 0.6270 ms median.  This wall number is
not the kernel-only projection number.  Production admission remains closed
until the real nonzero model route passes capture/replay qualification.

## Qualified v4 deep unroll

The remaining two K128-panel and four K32-group loops were swept separately
and together. Full eight-way unroll won: 256 static IMMA sites, 32 LDSM sites,
255 registers, and no LDL/STL spill instructions. The absolute full-real R9
gate passes all 6,291,456 outputs and readonly activation/weight hashes:
chain minimum/median 470.112/470.976 us, main minimum 455.040 us, maximum
absolute error 1.6689301e-6, direct tiles bit exact, split maximum 1.6689301e-6,
and the independent scalar tile exact.

The 72-real-weight synchronized proxy improves from 39.24045 ms to 37.03472
ms (5.62%). In the actual partial-redirect model, native main time falls from
42.283648 ms to 39.842176 ms and R9 wall improves from 95.3 ms to 93.2 ms.
This remains a NO_GO versus the 83.793 ms FP16 model wall. Provider cache v4
is the qualified research artifact; production admission remains closed.
