# Current Decode Benchmark (2026-06-24 11:52 AM EDT)

Command:

```bash
QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
```

Artifacts:

- `bench/qk-current-decode-benchmark/decode-current-20260624-115219.json`
- `bench/qk-current-decode-benchmark/current.json`

Promoted defaults active:

- `Q4K_GEMV_WARP=default-on`
- `Q4K_GEMV_WARP_DOWN=default-on`
- `Q4K_GEMV_WARP_PROJ=default-off`

Result:

| ctx | tok/s | ms/token | dispatch ceiling | host sync |
|---:|---:|---:|---:|---:|
| 512 | 102.6 | 9.74 | 100.8 | 0.0% |
| 1024 | 100.8 | 9.92 | 99.2 | 0.0% |
| 2048 | 98.4 | 10.17 | 97.0 | 0.0% |
| 4096 | 93.9 | 10.65 | 92.4 | 0.0% |
