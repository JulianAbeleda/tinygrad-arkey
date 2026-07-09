# Prefill Route Census Starting Point

Date: 2026-07-08

## Purpose

This is the clean starting line for comparing machine-code routes:

```text
route -> final AMD instruction stream -> same counters -> optional timing
```

The tool is:

```bash
extra/qk/prefill/prefill_route_census.py
```

It normalizes generated and hand routes into one table:

```text
WMMA count
inst/WMMA
wait/WMMA
global_load_b128/WMMA
ds_store_b128/WMMA
ds_load_b128/WMMA
between-WMMA global staging regions
future-slot-before-compute
DBUF D7 readiness
```

## 2x2 Timed Starting Table

Command:

```bash
DEV=AMD:ISA PYTHONPATH=. \
python3 extra/qk/prefill/prefill_route_census.py \
  --shapes '2,2' \
  --routes generated-direct,generated-kmajor,hand-lds2 \
  --pin-clock
```

Result:

| route | status | TFLOPS | WMMA | inst/WMMA | wait/WMMA | global/WMMA | ds_store/WMMA | ds_load/WMMA | between-global regions | future-slot | D7 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| generated-direct-2x2 | ok | 14.01 | 8 | 36.375 | 1.375 | 4.0 | 0.0 | 0.0 | 7 | true | false |
| generated-kmajor-2x2 | ok | 11.91 | 16 | 31.875 | 2.562 | 2.0 | 2.0 | 2.0 | 0 | false | false |
| hand-lds2-2x2 | structural | n/a | 32 | 12.844 | 0.562 | 1.0 | 1.0 | 2.0 | 3 | true | true |

## Active-Shape Structural Starting Table

Command:

```bash
DEV=AMD:ISA PYTHONPATH=. \
python3 extra/qk/prefill/prefill_route_census.py \
  --structural-only \
  --shapes '2,2;4,2;2,4' \
  --routes generated-kmajor,hand-lds2
```

Result:

| route | WMMA | inst/WMMA | wait/WMMA | global/WMMA | ds_store/WMMA | ds_load/WMMA | between-global regions | future-slot | D7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| generated-kmajor-2x2 | 16 | 31.875 | 2.562 | 2.0 | 2.0 | 2.0 | 0 | false | false |
| hand-lds2-2x2 | 32 | 12.844 | 0.562 | 1.0 | 1.0 | 2.0 | 3 | true | true |
| generated-kmajor-4x2 | 32 | 27.0 | 1.781 | 1.5 | 1.5 | 1.5 | 0 | false | false |
| hand-lds2-4x2 | 64 | 10.547 | 0.281 | 0.75 | 0.75 | 1.5 | 3 | true | true |
| generated-kmajor-2x4 | 32 | 27.5 | 1.781 | 1.5 | 1.5 | 1.5 | 0 | false | false |
| hand-lds2-2x4 | 64 | 10.547 | 0.281 | 0.75 | 0.75 | 1.5 | 3 | true | true |

## Starting Diagnosis

Generated K-major is correct, but it is not hand-like:

```text
no between-WMMA global staging regions
no future-slot-before-compute cadence
D7 is false
~2.5-3.0x instruction density gap
~6.3-9.1x wait density gap
~2x global/LDS-store stage-amortization gap
```

This makes the next primitive measurable:

```text
future changes must increase between-global regions and reduce global/store per WMMA without raising ds_load per WMMA
```

## 2026-07-08 Pivot: Direct/Register-Resident Is The Banked Route

The LDS K-major route remains useful as a structural oracle, but it is not the immediate performance path. A clean
schedule scan showed the direct register-resident route is correct and materially faster once the existing warmstart
knobs are used:

| route | status | TFLOPS | WMMA | inst/WMMA | wait/WMMA | global_b128/WMMA |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| direct 2x2 loc=4 unr=16 | ok | 32.46-33.64 | 64 | 12.5 | 1.047 | 4.0 |
| direct 4x2 loc=8 unr=16 | ok | 28.4 | 128 | 11.969 | 1.023 | 4.0 |
| direct 2x4 loc=8 unr=16 | ok | 29.5 | 128 | 12.0 | 1.023 | 4.0 |

The scoped table gate passes for the production `5120x5120` shape:

```bash
DEV=AMD:ISA PYTHONPATH=. \
python3 extra/qk/prefill_v2_schedule_table_gate.py --shapes 5120x5120 --run-amd
```

Result:

```text
verdict: PREFILL_V2_SCHEDULE_TABLE_APPLIES_PASS
5120x5120 opts: 2x2, LOCAL=4, UNROLL=16
measured: 33.64 TFLOPS, status=ok
```

Current interpretation:

```text
direct/register-resident is the primitive route to bank now
LDS K-major is not the main route until it can amortize global/stores without correctness or LDS-resource failures
the remaining direct-route gap to hand is global operand amortization: direct still pays 4 global_b128 per WMMA
```
