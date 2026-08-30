# Genuine llama gap audit

## Endpoint

The fresh three-arm tinygrad production median is 4.170389 ms/token, or
239.786 tok/s.  The retained llama official authority is 4.021721 ms/token,
or 248.711 tok/s.  The wall gap is therefore 148.668 us/token.

This audit corrects the earlier table by requiring like-for-like ownership.
Llama's separate activation quantization, RoPE, cache store, and host sampler
work cannot be omitted when tinygrad performs the same work inside a fused GPU
body.

## Ten-row causal ledger

| rank | row | corrected reading | why llama appears faster | first-pass decision |
|---:|---|---|---|---|
| 1 | flash combine | genuine GPU body deficit | tinygrad's 32-lane combine is about 102.016 us/token versus llama's 37.056 us/token | width is not the lever |
| 2 | flash score | production-conditioning deficit | the isolated bodies were near parity, but tinygrad exposes about 1.7 us/call more command service across 36 serialized launches | body arithmetic and readiness are closed; boundary removal only |
| 3 | Q projection | genuine but smaller after quant accounting | tinygrad is 307.392 us; llama Q plus Q-quant is 272.609 us before allocating tinygrad's shared provider | service geometry gives only a small ceiling |
| 4 | O projection | genuine but smaller after quant accounting | tinygrad is 306.752 us; llama O plus O-quant is 284.993 us | service geometry gives only a small ceiling |
| 5 | 4096 norm | genuine residual body deficit | tinygrad is 239.584 us versus llama 203.778 us | current primitive remains best; tested geometry changes are closed |
| 6 | Q head norm | tied, not a genuine gap | tinygrad's fused norm+RoPE is 73.152 us; llama norm+RoPE is 72.641 us | remove from target list |
| 7 | K head norm | tinygrad faster like-for-like | tinygrad fused K norm/RoPE/cache plus residual store work is about 73.696 us; llama's separate norm+RoPE+store is 102.817 us | remove from target list |
| 8 | vocabulary | comparison-domain mismatch plus a small main-body deficit | tinygrad's GPU row includes transforms and argmax; llama's GPU ledger omits its D2H/host sampler.  The main GEMV difference alone is about 14 us | do not claim the full row as recoverable llama debt |
| 9 | gate/up | tied after quant accounting | tinygrad is 1294.784 us; llama gate/up plus G-quant is 1291.116 us | remove from target list unless a same-clock rate regression appears |
| 10 | K/V projection | tied within shared-provider allocation | llama K/V plus K/V quant is 263.712 us; tinygrad is 237.312 us before allocating its 33.792 us provider shared with Q | remove from target list; boundary constructions already exhausted |

## Double-back results

### Flash combine

The prior width-128 test ran an older non-composed token path and was close to
noise, so it was not full information.  A fresh composed reps=9
control/candidate/control bracket initially recovered 4.955 us against the
midpoint but lost to the faster control.  The reverse hot-clock
candidate/control/candidate bracket closed the ambiguity:

```text
candidate A        4.169484 ms/token
control            4.155057 ms/token
candidate C        4.164373 ms/token
candidate midpoint 4.166928 ms/token
candidate loss       11.871 us/token
```

All hashes match.  Wider combine reduces the isolated body but loses in the
production graph.  Width is a true closed wall; reopen only with a different
algorithm or a removed score/combine boundary.

### Flash score

Retained cold counters show tinygrad executes about 2.7x llama's warp
instructions, but the isolated bodies were approximately 4.16--4.19 us versus
4.10 us.  Neither body is DRAM-bound.  The much larger production ledger delta
is therefore launch/conditioning exposure, not a 60-us arithmetic pool.
Geometry and Q/K-readiness PDL tests could not crack it because they did not
remove the repeated production boundary.

### Q and O

Llama uses grid 4096 with block 32x4; tinygrad's installed vector kernel uses
grid 4096 with one 32-thread warp.  A new exact four-warp vector microgate tests
that principle directly:

```text
installed vector median  4.861653 us/launch
four-warp vector median  4.808747 us/launch
ratio                    0.989118
```

The deterministic nonzero output matches exactly.  The combined Q+O ceiling
is only about 3.8 us/token, below a safe invasive-production investment gate
and far smaller than the raw ledger rows.  Four-warp ownership is real but is
not the missing Q/O explanation.  The earlier vector-load promotion already
captured the major fixed-byte achieved-rate win; the remaining raw delta mixes
provider ownership and command-domain differences.

### 4096 norm

The promoted attention completion recovered 62.995 us/token.  Two follow-ups
are now causally closed:

```text
one warp instead of 16      +2241.744 us/token
warp-0 merge and broadcast    +21.451 us/token
```

One warp serializes 128 elements per lane instead of eight.  Warp-0 merge
keeps parallel service but adds a shuffle phase and second block barrier; that
cost exceeds the redundant shared-memory reads it removes.

### Q/K heads

The original gap was an apples-to-oranges command-wall comparison.  Pure GPU
Q/K norm bodies were already approximately 1.18--1.20 us, near llama, and the
current tinygrad bodies additionally own RoPE and K-cache work.  Native head
norm replacement lost 1.792 us/token because it broke the useful fusion.

### Vocabulary

The native fp32/int32 argmax already removed the old reduction chain and
booked 56.386 us/token.  The fresh tail is the 314.976-us Q6 main body, two
small transforms totaling 8.928 us, and a 9.120-us native argmax.  Llama's GPU
ledger does not include the equivalent host scan/D2H sampler work, so only the
roughly 14-us main-body difference is a clean GPU service target.  The prior
in-GEMV packed-u64 top-1 path failed because it wrote and reread a cold key
buffer and used a more expensive 64-bit reduction.  That failure does not
support retrying fusion without a native pair-reduction/output-boundary change.

### Gate/up and K/V

Gate/up's vector four-warp typed-output route already booked 53.329 us/token.
After including llama's quantization it is now tied.  K/V's pair, direct, and
full-grid producer campaign removed completion work, but device wins that
changed the program/output boundary failed to reach token wall.  After shared
provider accounting K/V is also tied; neither row is a current llama body win.

## Corrected action set

1. Do not optimize Q/K heads, gate/up, or K/V as llama gaps.
2. Do not retry combine width, native norm geometry, or launch-ahead alone.
3. Reopen flash only with a construction that removes a production boundary.
4. Treat Q/O four-warp vector as a small cleanup ceiling, not the parity lever.
5. For vocabulary, compare complete sampler lifecycle before assigning llama's
   host work to a tinygrad GPU deficit.
6. The missing parity lever is now a cross-boundary construction or a remaining
   compulsory-byte/service-rate difference, not ten independent slow kernels.

## Evidence

- `docs/task_workflow/evidence/nv-genuine-llama-gap-audit-20260826/`
- `docs/task_workflow/evidence/nv-parity-final-stretch-20260826/`
- `docs/task_workflow/evidence/nv-flash-counter-ab-20260821/`
- `docs/task_workflow/evidence/nv-vector-load-reopen-20260824/`
- `docs/task_workflow/evidence/nv-lifecycle-recovery-tests-20260826/`
