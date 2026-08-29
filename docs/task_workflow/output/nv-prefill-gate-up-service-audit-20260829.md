# NVIDIA prefill gate/up service audit — 2026-08-29

## Decision

**COUNTER BRIDGE PASS; invest only in issue scheduling.** The exact pp512
tinygrad cubin and the first llama gate/up Q4 body now have a matched NCU
protocol. They execute the same useful IMMA count and request similar L2 and
shared traffic. Tinygrad's hot-body loss is lower issue/tensor duty, fewer
eligible warps, more non-tensor instructions, and higher long-scoreboard
pressure. This authorizes a narrow scheduling/latency-hiding discriminator,
not a broad memory, cp.async, TMA, or overlap campaign.

## Test 1: exact prefill primitive

Ran `nv_compiler_q4k_production_gate.py` against the canonical real
`blk.0.ffn_gate.weight` with shape `(M=512,N=12288,K=4096)`, compact-Q8
records, read-only carriers, and an independent FP32 reference. `tile-k=32`
fails closed during compilation (`current atomic staging requires at least
two tensor-core K steps`). The smallest legal `tile-k=64` passes correctness:
finite/full output, zero unwritten sentinels, max-abs `2.136e-4`, signed IMMA,
no expanded global weight, and exact compiler identity. It is nevertheless
slower than the qualified v4 chain: generated median `483.920 us` versus
`464.352 us` (+4.22%; min `474.312` versus `464.192`). This is a measured
negative primitive result, so it cannot advance to population or investment.
The retained machine-readable result is `/tmp/prefill_gate64.json` (source
`/tmp/prefill_gate64.cu`, SASS `/tmp/prefill_gate64.sass`).

## Physical launch / counter discriminator

The earlier driver wrapper used an incomplete `(96,1,1)` grid and is
retracted. The generated body is `(96,4,1)` with block `(32,2,4)`, or 384
CTAs. The corrected real-weight CUDA bridge passes full write coverage and
read-only inputs. Fresh matched cache-hot NCU rows report:

| counter | tinygrad K64 | llama Q4 MMQ |
|---|---:|---:|
| duration | 409.312 us | 219.200 us |
| IMMA instructions | 6,291,456 | 6,291,456 |
| all instructions | 194,359,296 | 156,188,112 |
| L2 requested bytes | 412.892 MB | 386.984 MB |
| shared read / write bytes | 452.992 / 228.905 MB | 512.365 / 209.618 MB |
| tensor active | 14.65% | 31.71% |
| issue active | 37.44% | 52.38% |
| eligible warps / cycle | 0.554 | 0.807 |
| long scoreboard / issue-active cycle | 0.535 | 0.324 |

The identical IMMA count rejects extra matrix work. The similar L2/shared
traffic rejects a first-order on-chip byte-volume explanation. Tinygrad is
1.87x slower while tensor duty is only 46% of llama's, total instructions are
24.4% higher, and long-scoreboard pressure is 65% higher. The selected next
test is a register-safe K-step/fragment load-to-use scheduling change that
must reduce instructions or long scoreboard and raise tensor duty without a
spill. It must then win the native 72-role population before investment.

## Evidence inspected

The existing promotion record
`nv-gateup-fourwarp-vector-typed-promotion-result-20260824.md` reports an
isolated decode kernel reduction from 21.968 us to 21.021 us and a typed
decode wall recovery of 53.329 us/token. Its geometry is 128 threads per row,
47 registers, and 32 B shared memory. The associated profile and wall files
are retained under
`evidence/nv-ranked-parity-campaign-20260824/02-gateup-*`.

The retained CUDA/NCU JSON files
`nv-s4-g32-p256-o-gate2-ncu.json` and
`nv-s4-g64-p256-o-gate2-ncu.json` are residual-O experiments, not gate/up.
They therefore cannot be used as gate/up physical-counter evidence. They do
show that the CUDA-launchable NCU route is available, but do not close this
workstream's discriminator.

## Gate ledger

| required gate | result | reason |
|---|---|---|
| Current-generated Q4 prefill body vs matched llama Q4 body | STOP | current primitive is +4.22% versus qualified v4 chain; no directly launchable matched llama prefill body |
| Real prefill full-output oracle | PASS | exact real weight, compact-Q8, full output; max-abs `2.136e-4` |
| Hot and rotated-cold prefill R9 | STOP | no exact 72-role prefill bracket |
| Executed gate/up NCU counters | PASS | exact real tinygrad cubin and fresh llama gate/up captured under one cache-hot metric protocol |
| One-variable prefill sweep | READY | issue scheduling / K-step interleave is selected; memory-topology speculation is excluded |
| 72-role population winner | STOP | absent |
| Whole-model prefill R9/deep replay with regenerated cut | STOP | prerequisite population winner absent |

The successful decode promotion remains valid evidence for the decode route
and is not reverted. This report deliberately books zero prefill recovery and
does not start cp.async, TMA, overlap, or a new compiler substrate project.

## Q4-down recheck (pp512)

The requested new shape is `(M=512,N=4096,K=12288)` using
`blk.0.ffn_down.weight`. The existing prefill typed-Q4 gate cannot express
this role: its source-level constants are `M,N,K = 512,12288,4096`, its
carrier is constructed as `PackedWeightTransform("Q4_K", N, K)`, and its
fixture validation is hard-coded to `blk.0.ffn_gate.weight` with reversed
dims `(12288,4096)`. Consequently invoking that gate for down would either
reject the canonical weight or test the wrong ABI/transpose; it cannot be
called Q4-down evidence. A dedicated down binding/shape contract is the
missing substrate. No down primitive, hot/cold R9, or 18-role result is
claimed until that contract exists.

## Re-entry condition

Resume with one register-safe K64 scheduling candidate using the new bridge.
Require the full output oracle, canonical/read-only inputs, no local spill,
higher tensor duty, lower long scoreboard or total instructions, and hot plus
rotated-cold R9. Only a candidate that then wins the exact 72-role native
population should proceed to whole-model replay.
