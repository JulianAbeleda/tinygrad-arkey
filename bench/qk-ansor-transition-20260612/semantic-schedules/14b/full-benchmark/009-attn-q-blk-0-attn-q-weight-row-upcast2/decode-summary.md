# QK Decode Summary

| label | samples | avg tok/s | drop1 | last64 | last32 | last16 | stdev | min | max | Q4 install | Q6 install | storage MB | policy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `explicit1` | 128 | 34.38 | 34.60 | 34.62 | 29.63 | 36.84 | 9.67 | 6.10 | 43.46 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit2` | 128 | 39.65 | 39.91 | 39.47 | 39.38 | 39.09 | 4.43 | 6.08 | 43.45 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `explicit3` | 128 | 40.36 | 40.63 | 39.47 | 38.74 | 38.44 | 3.71 | 6.09 | 43.26 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/search/14b/policies/current.policy.json` |
| `generated1` | 128 | 36.44 | 36.68 | 36.00 | 35.60 | 35.34 | 3.32 | 6.07 | 38.80 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated2` | 128 | 35.29 | 35.52 | 34.97 | 35.64 | 35.52 | 4.53 | 5.83 | 38.91 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
| `generated3` | 128 | 36.70 | 36.94 | 36.02 | 35.60 | 35.44 | 2.93 | 6.06 | 38.81 | 240 | 40 | 0.00 | `bench/qk-ansor-transition-20260612/semantic-schedules/14b/full-benchmark/009-attn-q-blk-0-attn-q-weight-row-upcast2/policy.json` |
