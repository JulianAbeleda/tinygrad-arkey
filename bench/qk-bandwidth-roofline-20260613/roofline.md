# QK Bandwidth Roofline

This report is generated from committed shared-storage QK decision artifacts.
It does not run benchmarks. `full-file GB/s` is a logical decode roofline
proxy: GGUF file bytes times tokens/sec. It is useful for comparing tinygrad
and llama.cpp on the same model, but it is not a hardware-counter HBM read
measurement.

- device: `AMD Radeon RX 7900 XTX / gfx1100`
- peak memory assumption: `960.0 GB/s`
- verdict: `memory_load_efficiency_gap`

## Model Rows

| model | generated tok/s | llama tok/s | generated file GB/s | llama file GB/s | generated % peak | llama % peak | generated % llama | primitive source GB/s | A/B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `8B` | 52.07 | 101.20 | 261.82 | 508.81 | 27.27% | 53.00% | 51.46% | 206.77 | `True` |
| `14B` | 40.55 | 65.80 | 365.04 | 592.32 | 38.03% | 61.70% | 61.63% | 321.11 | `True` |
| `32B` | 17.23 | 30.80 | 340.47 | 608.67 | 35.47% | 63.40% | 55.94% | 321.79 | `True` |

## Interpretation

- tinygrad generated path reaches `27.27-38.03%` of the 960 GB/s peak by the full-file proxy.
- llama.cpp reaches `53.00-63.40%` by the same proxy.
- tinygrad is `51.46-61.63%` of llama.cpp by this same byte model.
- largest file-bandwidth gap to llama.cpp: `268.21 GB/s`.

The result supports treating the remaining decode gap as a memory-load
efficiency/codegen problem before adding more local schedule knobs. A future
hardware-counter pass can replace this logical proxy, but the current
decision is already strong enough to freeze the exhausted schedule surfaces.
