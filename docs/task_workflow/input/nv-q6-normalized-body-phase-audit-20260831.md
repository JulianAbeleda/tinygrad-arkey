# NV Q6 normalized direct-body phase audit

## Scope

This audit compares one direct `128x128xK256` body from the pinned llama Q6
cubin with the generated broad rolling-prefetch CTA.  It expands the generated
producer's runtime row loop before comparing instruction counts.  That
normalization supersedes the earlier static-census comparison: the generated
loop body is static once but executes sixteen times.

Pinned inputs:

- llama cubin SHA-256: `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca`;
- generated saturated cubin SHA-256: `f671af44721251d41179dda925b0a3b3c4cd1ba4b7937974b55dee3c521c86f7`;
- llama direct K256 body: `0x0d80..0xeb50`;
- generated Q6 row loop: `0x0210..0x06d0`, sixteen trips; and
- generated compute end is `0xd990`; the following 64 `STG` instructions are
  output writeback and are excluded.

## Phase alignment

| logical phase | generated PCs | llama PCs |
|---|---|---|
| Q6 plus initial Q8 production | setup `0x0000..0x0200`, Q6 loop `0x0210..0x06d0` x16, Q8 `0x06e0..0x0e90` | `0x0d80..0x383f` |
| initial publication barrier | `0x0b60`, `0x0ea0` | `0x3840` |
| K128 half 0 plus next-Q8 prefetch | `0x0eb0..0x5e8f` | `0x3850..0x861f` |
| half-0 barrier | `0x5e90` | `0x8620` |
| next-Q8 publication | `0x5ea0..0x602f` | `0x8630..0x888f` |
| next-Q8 barrier | `0x6030` | `0x8890` |
| K128 half 1 | `0x6040..0xd990` | `0x88a0..0xeb3f` |
| lifecycle end | absent in the screening microkernel | `0xeb40`, branch `0xeb50` |

The equal whole-kernel `BAR=4` count hid different placement.  The exact
combined-publication A/B corrected the placement while holding all useful work
fixed.  It added 65 `MOV`, kept 255 registers and zero spills, and regressed
the saturated median from `18.848 us` to `19.168 us`.  Commit `d5793a93c`
therefore closes barrier placement as a performance lever.

## Control-flow-expanded counts

The generated estimate counts every instruction in the sixteen-trip row loop.
The gated scale/D paths have active lanes and reconverge inside each warp, so
their path instructions are charged.  Hardware executed counters would be the
stronger authority if counter permission becomes available.

| family | generated producer | llama producer | delta |
|---|---:|---:|---:|
| all instructions | 1,390 | 685 | +706 |
| `LDG` | 244 | 87 | +157 |
| `STS` | 114 | 53 | +61 |
| `IMAD` | 91 | 55 | +36 |
| `IADD` | 131 | 85 | +46 |
| `LEA` | 105 | 59 | +46 |
| `SHF` | 121 | 64 | +57 |
| `LOP3` | 202 | 201 | +1 |
| `PRMT` | 69 | 64 | +5 |
| `BRA / BRX / BSSY / BSYNC` | 80 / 32 / 16 / 16 | 0 / 0 / 0 / 0 | +144 control instructions |

Within the full body, generated Q6 alone issues an estimated 208 loads and 96
stores.  Llama's fully unrolled Q6 publisher uses 69 loads and 35 stores.
Both bodies add 36 Q8 loads and 36 Q8 stores, so the full-body differences are
139 global loads and 61 shared stores.

| family | generated consumer | llama consumer | signed delta |
|---|---:|---:|---:|
| all instructions | 3,246 | 2,864 | +382 |
| `IMMA / LDSM / I2FP` | 256 / 32 / 512 | 256 / 32 / 512 | 0 |
| `FMUL` | 512 | 0 | +512 |
| `FFMA` | 512 | 640 | -128 |
| `IMAD` | 1,027 | 1,028 | -1 |
| `LDS` | 168 | 176 | -8 |
| `MOV` | 66 | 35 | +31 |

The normalized generated body is therefore about 4,636 issued instructions
versus 3,549 for llama, a 1,087-instruction excess before dynamic scheduler
effects.  Producer excess is the largest region.

## Dependency and lifetime facts

- Generated Q6 rows use a serial chain `LDG -> bit extraction -> signed pack ->
  STS -> loop backedge`, repeated sixteen times.  Llama emits a straight-line,
  fully unrolled publisher and interleaves independent rows.
- Generated panel-1 Q8 values are loaded at `0x0850..0x0b30` and retained until
  stores ending at `0x5fc0`, a 1,400-instruction static span.  Llama loads them
  at `0x80e0..0x8290` and stores them by `0x8870`, a 122-instruction span.
- The pinned llama source accumulates four K32 terms as
  `tmp += int_dot * dB`, then applies `sum += tmp * dA` once per output and
  K128 half.  This produces 512 inner plus 128 final `FFMA` instructions.
  Generated originally applied `dA*dB` inside every K32 update, producing 512
  `FMUL` plus 512 `FFMA` instructions.

The first two bullets are binary facts.  The claim that their shorter chains
explain latency is an inference until a causal A/B changes only that chain.

## Ranked next tests

1. **Oracle Q6 publisher topology.** Completed. Replace the sixteen-trip generated Q6
   publisher with a straight-line publisher that produces byte-identical
   76-word shared rows using exactly 69 Q6 `LDG` and 35 Q6 `STS`.  Keep Q8,
   consumer, arithmetic and lifecycle fixed.  Require exact output, 256
   `IMMA`, 32 `LDSM`, no `LDL/STL`, and R31 one/saturated timing.  Invest only
   if projected recovery is at least `23.5 us`. The causal arm passed with
   `168.852 us` matched-control projected recovery, but its absolute screening
   projection is `372.929 us`, so the broad route is not integrated.
2. **Late Q8 prefetch lifetime.** Constrain the 18 panel-1 loads to the tail of
   half 0 and require load-to-store span below 160 instructions, without
   changing counts.  This tests register lifetime/schedule rather than memory
   volume.
3. **Scale factoring.** Completed below.  It is causal and positive but misses
   the investment bar, so do not integrate it independently.
