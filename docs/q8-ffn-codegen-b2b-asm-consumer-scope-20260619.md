# q8 FFN B2b AMD DSL/ASM consumer scope (2026-06-19)

This is the remaining decode ownership path after:

- A4 proved the q8 route in-model: W==D `1.051-1.063x`, dNLL `+0.002887`;
- B0/B1 showed both fast and slow consumers already emit `v_dot4_i32_iu8`;
- B2a closed the COMGR fused-C sublane: correct but `146.88us` vs `<=60us`.

The goal is a tinygrad-owned fused Q4_K x q8_1 `ffn_gate/up` consumer using the AMD DSL / `Ops.PROGRAM` assembler path,
with no hipcc/LLD artifact and no COMGR C source.

## B2b0 smoke result

Executed:

- probe: `extra/q8_ffn_asm_gateup_smoke.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_dot4_smoke.json`.

Result: **PASS**.

The probe emits a tiny AMD DSL kernel with:

- `s_load_b64` kernarg pointer load;
- `v_dot4_i32_iu8`;
- `global_store_b32`;
- HCQ launch through `Ops.PROGRAM`;
- no C compiler, hipcc, LLD, or HIP runtime.

It writes `4` from `dot4([1,1,1,1], [1,1,1,1])`.

This proves the mechanical assembler path can host the required instruction family. It does not prove the full MMVQ
consumer yet.

## B2b1 exact contract

Build a standalone fused gate/up consumer with the same buffer order as the A4 artifact:

1. `dst_gate: float[12288]`
2. `dst_up: float[12288]`
3. `gate_words: uint32[7077888]`
4. `up_words: uint32[7077888]`
5. `q8: uint8[4608]`

Launch:

- global: `(12288, 2, 1)`;
- local: `(32, 4, 1)`;
- one row per `gidx0`;
- `gidx1 == 0` selects gate, `gidx1 == 1` selects up.

Work decomposition:

- 128 threads per row;
- `tid = ly * 32 + lx`;
- `kb = tid / 8`, fixed `16` Q4_K blocks for `K=4096`;
- `sub = tid & 7`;
- 8 dot4 operations for `sumi`;
- 8 dot4 operations for `sumq`;
- wave reduction plus 4-way workgroup reduction;
- one final store by `tid == 0`.

Correctness gate:

- compare against the existing q8 proxy on real GGUF `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`;
- gate/up max_abs `<=2e-3`;
- no default route.

Performance gate:

- fused gate/up consumer `<=60us`;
- no hipcc/LLD;
- no COMGR C source;
- no HIP runtime in process.

Executed first slice:

- probe: `extra/q8_ffn_asm_gateup_skeleton.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_gateup_address_skeleton.json`;
- verdict: **PASS**.

This proves the real five-buffer fused gate/up signature works through `Ops.PROGRAM`, with `gidx0` as row and `gidx1`
as gate/up selector. It writes `gate[row] = row` and `up[row] = row + 12288` across the full `(12288,2,1)` grid with
zero mismatches.

Executed second slice:

- probe: `extra/q8_ffn_asm_q8_load_skeleton.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_q8_load_skeleton.json`;
- verdict: **PASS**.

This proves global byte loads from the q8 side-channel buffer work under the same five-buffer contract. The kernel loads
`q8[(row & 127) * 36 + 4 + role]` and stores diagnostics to gate/up with zero mismatches.

Executed third slice:

- probe: `extra/q8_ffn_asm_q4_load_skeleton.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_q4_load_skeleton.json`;
- verdict: **PASS**.

This proves gate/up Q4 weight pointer selection and the `576`-word row stride work under the same contract. The kernel
loads the first synthetic Q4 word for each row and role with zero mismatches.

Executed fourth slice:

- probe: `extra/q8_ffn_asm_q4_field_skeleton.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_q4_field_skeleton.json`;
- verdict: **PASS**.

This proves Q4_K intra-block byte addressing for scale bytes and `qs` words. Coverage uses `kb=row&15` and
`sub=(row>>4)&7` for both gate/up pointers, with zero mismatches against CPU.

Executed fifth slice:

- probe: `extra/q8_ffn_asm_one_subblock_dot.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_one_subblock_dot.json`;
- verdict: **PASS**.

This proves the first MMVQ math slice: fixed `sub=0`, `kb=row&15`, eight `v_dot4_i32_iu8` operations, low-nibble Q4
extraction, positive q8 payload, and `sumi/sumq` accumulation for both gate/up pointers. It matches CPU exactly.

Executed sixth slice:

- probe: `extra/q8_ffn_asm_signed_high_dot.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_signed_high_dot.json`;
- verdict: **PASS**.

This closes the main dot4 correctness trap: odd-sub high-nibble extraction plus signed q8 payload. The AMD DSL
`v_dot4_i32_iu8(..., neg=2)` matches the oracle's `neg_lo:[0,1,0]` signed-q8 modifier exactly.

Executed seventh slice:

- probe: `extra/q8_ffn_asm_scaled_subblock.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_scaled_subblock.json`;
- verdict: **PASS**.

This proves the scaled Q4_K affine for one sub-block: half `d`, half `dmin`, scale byte `sc`, min byte `mn`, half `d8`,
signed q8, high nibble, and float output. The first attempt exposed `v_sub_f32_e32` operand ordering by producing an
exact sign inverse; after fixing operand order, max_abs is `0.0`.

Executed eighth slice:

- probe: `extra/q8_ffn_asm_local_id_probe.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_local_id_probe.json`;
- verdict: **PASS with local=(128,1,1)**.

Important descriptor finding: under `assemble_linear`, `local=(32,4,1)` safely exposed `v0` as local-x only; using `v1`
as local-y caused an MMU fault. The viable path is `local=(128,1,1)` with flattened `tid=v0`.

Executed ninth slice:

- probe: `extra/q8_ffn_asm_thread_partials.py`;
- artifact: `bench/q8-ffn-codegen-transfer/asm_thread_partials.json`;
- verdict: **PASS**.

This proves the real 128-thread decomposition without reduction: `kb=tid>>3`, `sub=tid&7`, q4 low/high nibble selection,
signed q8 dot4, and `sumi/sumq` accumulation. It writes one diagnostic partial per thread and matches CPU for the first
128 rows across all 128 lanes for both gate/up pointers.

## B2b2 implementation order

Do not write the whole consumer in one jump. Build it in slices:

1. **Address skeleton:** load five buffer pointers, compute row/which/tid/sub/kb, and store a deterministic value per
   row/which. Gate: output pattern matches CPU expectation. **DONE/PASS.**
2. **Q8 load skeleton:** load `q8` scale and one packed q8 lane, store a diagnostic value. Gate: matches CPU extraction.
   **DONE/PASS for byte load and q8 block addressing.**
3. **Q4 load skeleton:** load Q4_K scale/min/qs for one sub-block, store diagnostic scale/min/nibble values. Gate:
   matches CPU extraction. **DONE/PASS for pointer selection, row stride, scale byte, and `qs` word addressing.**
4. **One-block dot:** compute one `kb` contribution for one row. Gate: matches CPU partial reference.
   **DONE/PASS for signed q8, low/high nibble, scale/min affine, and all 128 per-thread partials.**
5. **Full-row dot:** loop/emit all 16 `kb` lanes, reduce across wave/workgroup. Gate: one row matches reference.
   **NEXT: workgroup reduction/full-row output.**
6. **Full fused gate/up:** run all rows and both roles. Gate: real correctness + `<=60us`.

Each slice must bank a JSON artifact. Stop at the first slice that proves this is becoming a broad assembler project
rather than a bounded primitive.

## Kill criteria

Stop B2b and classify decode ownership as project-level renderer/ASM work if any of these are true:

- address/control skeleton cannot be represented cleanly with current AMD DSL;
- full-row correctness requires copying large chunks of compiler disassembly without understandable ownership;
- consumer remains far above `60us` after matching the oracle's work decomposition;
- register pressure or manual scheduling requires a general scheduler rather than a local hand kernel.

## Decision

B2b is now **GO to B2b1/B2b2**, but with strict gates.

The smoke proves the assembler can host the primitive. The next meaningful work is the address/control skeleton, not more
COMGR C variants and not producer work. If the fused consumer cannot clear `<=60us`, stop decode ownership; the q8 route
stays a research artifact, and native parity becomes compiler project work.
