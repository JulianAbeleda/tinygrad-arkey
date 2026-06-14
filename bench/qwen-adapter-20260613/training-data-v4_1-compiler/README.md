# Adapter JSON Dataset V4.1 Compiler

This compiler-only artifact isolates the Phase 4.2 prompt/data redesign.
The expected answers are stable concept keys such as `qk_gemv`, not
row-specific keys such as `train_qk_gemv_005`.

- SFT rows: `102`
- train rows: `68`
- held-out eval rows: `34`
- category: `compiler`
- schema: `{"answer": "qk_<concept>"}`
- disjointness: train/eval prompts and template instances are checked
- answer overlap: intentional, because the stable concept key is the task target

## Stable Keys

| concept | stable key | definition |
|---|---|---|
| `wide_load` | `qk_wide_load` | loads multiple adjacent packed words at once |
| `coalesced_read` | `qk_coalesced_read` | maps neighboring lanes to neighboring addresses |
| `wavefront` | `qk_wavefront` | names the SIMD execution group on AMD GPUs |
| `dequant` | `qk_dequant` | converts packed quantized values toward floating point values |
| `gemv` | `qk_gemv` | multiplies a matrix by one vector |
| `q4_block` | `qk_q4_block` | stores a group of four-bit quantized weights |
| `q6_block` | `qk_q6_block` | stores a group of six-bit quantized weights |
| `uop` | `qk_uop` | names tinygrad's internal operation node |
| `beam` | `qk_beam` | searches schedule choices in tinygrad |
| `policy` | `qk_policy` | records which lowering choice to use for a tensor family |
| `suffix_cache` | `qk_suffix_cache` | stores the frozen prefix hidden state before adapter blocks |
| `json_axis` | `qk_json_axis` | separates parse, schema, type, and value scoring |
