# NV Q6 const/restrict RegionLoad panel1 Gate13 decision

## Verdict

`REJECT_SOURCE_SASS`. `commit_ready=false`. Do not run trusted correctness, bit-exactness, fixup validation, or R31 timing.

## Frozen controls

- Main anchor: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`.
- All-partials fixup: `483de2ee3eed3597932a8632f9892377ce054e77bfe34c2420fe5a5d54ff5514`.
- Timing reference: `256.256 us`.
- Target and launch remain `sm_120` and `__launch_bounds__(256)`.

## Focused source gate

The complete focused suite passed `26/26` in `25.93s`. Exactly `data2_1769472` changed to `const unsigned int *__restrict__`; normalizing that token makes the full generated candidate source byte-identical to the frozen Gate12 RegionLoad source (`206ebe0ea6214fccfa6c389c19e6b4e6f1d9e0fcc38557495552710555e90017`). The source retains 18 direct assignments, four existing barriers, and zero Q8 destination stores.

## Fresh static result

- Anchor rebuild SHA: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`.
- Candidate source SHA: `5bfae489edd748d376c3e134d2b0eaf44296bbfc2dd1928c2f62f38c5cd44c2d`.
- Candidate cubin SHA: `9639ffce4be40b6bc736c96ecac2709e00e52de8e67b82901937ae6b52e7c390`.
- Fresh compile time anchor/candidate: `0.876175s` / `0.688606s`.
- Panel copy: 18 `LDG.E.CONSTANT`, 18 `STS`, exact offsets.
- First load/store: PC `0x6410` ordinal `1601` / PC `0xaaa0` ordinal `2730`.
- Span: `1129` instructions; required `<=160`.
- Barriers: `0x2f60`, `0xaa40`, `0xabc0`, `0x109f0`; load-before-overwrite/store-after-overwrite shape passes.
- Census: `IMMA/LDSM/LDS/LDG/STS/STG/BAR = 256/32/184/109/73/64/4`.
- Arithmetic: `I2FP/FMUL/FADD/FFMA = 1024/1544/1024/0`.
- Instructions: `5152`; required `<=5144`.
- LOP3: `212`; anchor `211`.
- Registers/stack/local/LDL/STL: `255/0/0/0/0`.
- `MEMBAR/ATOM = 0/0`.

## Causal conclusion

`const __restrict__` is sufficient to select `LDG.E.CONSTANT` and hoist the panel1 loads before the overwrite barrier. It is not sufficient to reproduce llama's short live interval: ptxas hoisted the loads across 1129 instructions. It eliminated Gate12's stack and local spill traffic, but retained the extra eight LDS, the extra LOP3, and an instruction-count regression.

Correctness and R31 were not run because the static gate failed. The lock was held for focused and compile phases, no GPU workload started, and `flock -n /tmp/nv-q6-oracle-gpu.lock true` returned `0` afterward.

Evidence: `docs/task_workflow/evidence/nv-q6-const-restrict-region-panel1-gate13-20260901/result.json`.
