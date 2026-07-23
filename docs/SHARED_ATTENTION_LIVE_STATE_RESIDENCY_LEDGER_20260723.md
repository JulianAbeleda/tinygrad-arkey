# Shared attention live-state and residency ledger

Date: 2026-07-23

## Scope and evidence boundary

This is the production single-wave `gfx1100` attention kernel, captured for the 8B overlay first chunk (`Q=512`, `KV=512`, `Hq=32`, `Hkv=8`, `Hd=128`). The same resource result is present in all four proven production captures.

| Capture | VGPR | SGPR | LDS | Scratch | highest ISA VGPR | WMMA |
|---|---:|---:|---:|---:|---:|---:|
| 8B overlay, first/prefix | 254 | 26 | 512 B | 0 | 235 | 16 |
| 14B bounded, first/prefix | 254 | 26 | 512 B | 0 | 235 | 16 |

Exact machine-readable evidence is `artifacts/shared-attention-m10e1-20260723/8b-overlay-first.json`: `hip_resources.vgpr=254`, `sgpr=26`, `lds_bytes=512`, both spill counts zero, and `wavefront_size=32`. Its source and ISA SHA-256 values are respectively `522333faebe0d75f324d7900ad56ff53736ed099adb2ea7438a12bc49ed9ebe4` and `20f33aa106bc6c0f8ba05c8d7daf0cce68cf26912ce72cfd6fa32bfc4cb73398`.

`highest ISA VGPR=235` is an observation of explicit textual register use, not the allocation. Residency decisions must use the compiler metadata allocation of 254 VGPRs, not register-number placement or the 235-value scan.

## Fixed live-state ledger

The following roles have exact source/ISA attribution. Counts are 32-bit VGPR lanes per thread. They are a lower bound on simultaneous live state at the QK-to-PV transition, not a claim that every temporary below is simultaneously live.

| Role | VGPRs | Persistent across KV tiles? | Exact evidence |
|---|---:|---|---|
| PV output accumulators | 64, `v8:v71` | Yes | Fixed ABI; final normalization/store consumes these registers in the ISA footer. HIP source lines 637-644 seeds `wmma8..15` from prior accumulator values, one `float8` fragment per 16-wide output slice. |
| Online maximum `m` | 8, `v72:v79` | Yes | Fixed ABI; captured ISA uses `v72` in QK reduction. Eight elements match wave32 `16x16` C-fragment ownership. |
| Online normalization `l` | 8, `v80:v87` | Yes | Fixed ABI; ISA footer reciprocates `v80:v87` to normalize the 64 PV accumulator values. |
| QK C fragment | 8, `v88:v95` | Tile-local, overlaps PV state | ISA emits `v_wmma_f32_16x16x16_f16(v[88:95], v[200:207], v[208:215], v[88:95])`. |
| Alpha/rescale | 8, `v96:v103` | Tile-local, overlaps PV state | Fixed ABI; one alpha per owned score-row fragment is needed for `acc = alpha*acc + P@V`. |
| WMMA A fragment | 8, `v200:v207` | Tile-local | Same QK WMMA instruction. |
| WMMA B fragment | 8, `v208:v215` | Tile-local | Same QK WMMA instruction. |
| **Exact fixed subtotal** | **112** | **PV + m/l survive; remaining 32 overlap them** | Directly attributable fixed-register roles. |

The ABI and fragment mapping are stated in `SHARED_ATTENTION_HANDOFF_20260723.md`. The compiler capture proves eight QK and eight PV WMMA roles (`static_wmma_count=16`), so the ledger does not infer tensor-core use from source spelling alone.

## Compiler-temporary and address ledger

| Class | Visible register region | What is known | What is deliberately not claimed |
|---|---|---|---|
| Address/index state | principally `v0:v7`, with reused address temporary `v151` | ISA has global addresses, lane/index arithmetic, and loop control; source has the `Ridx9600 < 32` KV loop. | The region is not a single live interval and must not be summed as permanently live state. |
| Generated load/conversion/reduction temporaries | `v104:v199`, `v216:v234` | Source materializes Q/K/V fragment loads, masks, exp/reduction values, LDS P publication/reload, and PV operands. ISA textual maximum is `v234` (reported as `highest_vgpr=235`). | These 123 numbered registers are not proven simultaneously live. |
| Allocation-only margin | 19 (`254 - 235`) | Compiler metadata allocates 254 VGPRs although the ISA scan reaches 235. | It is not assigned to a semantic object; it may include allocation granularity, ABI/reserved state, or registers absent from the text scan. |

This prevents the invalid conclusion that the 254 allocation is `112 + 123 + 19` simultaneously live. It establishes the actionable fact: long-lived output/softmax state occupies at least 80 VGPRs before QK, alpha, operands, addresses, and generated temporaries are admitted.

## Residency facts and thresholds

Facts obtained from the live `rocminfo` device record, rather than architecture assumptions:

| Device fact | Value | Consequence |
|---|---:|---|
| GPU | AMD Radeon RX 7900 XTX, `gfx1100` | Matches the capture target. |
| Wavefront size | 32 | One workgroup is one wave: capture records `workgroup_waves=1`. |
| SIMDs per CU | 2 | CU-level wave limit divides across two execution partitions. |
| Max waves per CU | 32 | Absolute CU ceiling; with two SIMDs this is 16 waves/SIMD before resources. |
| Max work-items per CU | 1024 | Same 32-wave ceiling for wave32 work. |
| Group/LDS pool | 64 KiB | Captured 512 B LDS is 0.78% of this pool, so LDS capacity cannot explain a one-wave limit. |
| Compiler VGPR allocation | 254/thread | Current residency comparison must use this allocation. |
| Compiler SGPR allocation | 26/wave | Recorded, but the device record exposes no SGPR-file capacity. |

The device record does **not** expose VGPR-file bytes, VGPR allocation granularity, or register-limited waves for this kernel. Therefore this ledger does not fabricate a numerical "2 waves/SIMD" result from the 254 count. The only exact calculated upper bound from available facts is:

```text
resident wave32/CU <= min(32 max waves, 1024 max work-items / 32) = 32
resident wave32/SIMD <= 32 / 2 = 16
```

The relevant compiler allocation thresholds are nevertheless exact and testable: 254 is within two registers of the 256-VGPR per-thread interface cap exposed by installed HIP (`hipDeviceAttributeMaxAvailableVgprsPerThread`) and rocRoller (`maxVGPRs=256`). A candidate at `<=128` crosses the half-capacity allocation bucket; `<=64` crosses the quarter-capacity bucket. Neither bucket is claimed to imply a residency count until a profiler or device API reports it.

## Falsifiable target: query-row ownership variant

The next production-shaped variant must reduce query-row ownership while retaining all 128 output dimensions and avoiding an output reduction. It passes this ledger gate only if its same-profile compiler capture proves all of:

1. `vgpr <= 128` (crosses the half-capacity allocation threshold), with zero VGPR/SGPR spills and zero scratch bytes.
2. The exact live subtotal falls below 112 because owned PV and `m/l` rows are reduced, rather than merely moving their fixed register numbers.
3. It retains 8 attributed QK plus 8 attributed PV WMMA roles for each equivalent 16-row logical unit, or explicitly records the changed work unit and compensating count.
4. A residency counter/device API demonstrates more resident waves than the baseline, or this register-residency theory is rejected for the tested partition. Allocation alone is not success.

If the candidate remains above 128 VGPR, or allocation falls but measured resident waves do not increase, do not proceed to K/V-sharing or software pipelining under the claim that PV state was the limiting resource. Record the negative result and test the output-dimension diagnostic instead.
