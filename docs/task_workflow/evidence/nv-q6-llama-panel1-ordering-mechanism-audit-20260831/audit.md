# NV Q6 llama panel-1 ordering mechanism audit

Status: read-only source and binary audit complete. No Q6 PTX artifact was available.

## Frozen inputs

| item | identity |
|---|---|
| llama checkout | `/home/ubuntu/env/llama.cpp`, commit `ac4cddeb0dbd778f650bf568f6f08344a06abe3a` |
| Q6 source | `ggml/src/ggml-cuda/mmq.cuh`, SHA-256 `6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f` |
| MMA source | `ggml/src/ggml-cuda/mma.cuh`, SHA-256 `c7e0f3332da182e203b4b953f9fa8535ffca3767d2b7d4d7dbf7ce486262d1af` |
| cubin | `docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin`, SHA-256 `04eb9bcb2edef62c672b5496d743a98c57e3236558b88f2ff117964b7fbb91ca` |
| normalized SASS | `docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/llama.nvdisasm`, SHA-256 `c97e22ee7b3fa81c7322eb52f07111f6c7aef3b7b1391c4f559ccf675e9a4802` |
| function | `_Z15dense_mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiPfS5_5uint3iiiiiS6_S6_iiiS6_S6_iiiS6_` |

The direct and partial bodies are two inlined paths in this one function and cubin, not separate symbols. Their extents are `0x0d80..0xeb50` and `0x12880..0x20780` respectively.

The only `.ptx` files under the pinned checkout and current Q6 evidence tree are CMake compiler-identification outputs. No PTX corresponding to this function is available. Therefore the audit cannot distinguish CUDA front-end/NVVM scheduling from ptxas scheduling.

## Source mechanism

`mmq.cuh:3374..3380` binds Q6 to `load_tiles_q6_K` and `vec_dot_q6_K_q8_1_mma`. The K-loop is in `mul_mat_q_process_tile` at `mmq.cuh:3447..3521`.

The relevant source order is:

```cuda
for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);
    const int * by0 = y + ncols_y * (kb0 * qk / ne_block) * sz;
    tile_y[l] = by0[l];
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);
    __syncthreads();
    const int * by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz);
    tile_y[l] = by0[l];
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
}
```

Exact locations are the loop at line 3485, panel-0 store at line 3493, first `vec_dot` at line 3499, panel-1 store at line 3509, and second `vec_dot` at line 3515. Complete-tile/direct and final-partial calls instantiate the same function body with `fixup=false` at lines 3710..3713 and `fixup=true` at lines 3779..3782. `fixup` changes final publication, not either Q8 copy.

### Address expression

The panel-1 global address is affine and independent of phase-0 arithmetic:

```text
by0 = y + ncols_y * ((kb0 * qk / ne_block) * sz + sz)
address(word l) = by0 + l
```

Its inputs are the kernel pointer `y`, loop/control value `kb0`, `ncols_y`, lane-derived `l`, and compile-time `qk`, `ne_block`, and `sz`. It does not consume `sum`, a `vec_dot` result, an IMMA result, or any accumulator.

The source has a lexical/effect boundary: the second copy statement follows phase-0 `vec_dot` and `__syncthreads`. It does not have a value/data dependency on phase-0 arithmetic.

## Binary mechanism

### Direct path

- One 64-bit base is formed by `LEA` at `0x80b0` and `LEA.HI.X` at `0x80c0` into `R4.64`.
- Eighteen `LDG.E.CONSTANT` instructions occur at `0x80e0..0x8290` from `desc[UR8][R4.64 + offset]`, with offsets `0x0000..0x4400` in steps of `0x400`.
- The loads are interleaved with the final phase-0 IMMA, integer accumulation, conversion, FFMA, and shared loads.
- The overwrite barrier is `BAR.SYNC.DEFER_BLOCKING 0x0` at `0x8620`, 83 static instructions after the first LDG. That interval contains 6 IMMA and 9 scalar LDS.
- The 18 STS instructions occur at `0x86e0..0x8870`; publication barrier is `0x8890`.

### Partial path

- One 64-bit base is formed by `LEA` at `0x19cd0` and `LEA.HI.X` at `0x19ce0` into `R4.64`.
- Eighteen `LDG.E.CONSTANT` instructions occur at `0x19d00..0x19e20` from the same base with offsets `0x0000..0x4400` in steps of `0x400`. Issue order is permuted but the set is exact.
- The overwrite barrier is `BAR.SYNC.DEFER_BLOCKING 0x0` at `0x1a230`, 82 static instructions after the first LDG. That interval contains 7 IMMA and 8 scalar LDS.
- The 18 STS instructions occur at `0x1a2f0..0x1a470`; publication barrier is `0x1a4d0`.

Thus the compiler split each source `tile_y[l] = by0[l]` copy. It hoisted the independent global read before the source-level shared-memory barrier while retaining the shared write after the machine barrier. The LDGs are late relative to the phase body, but not ordered after a specific accumulator instruction.

## Exact loaded-register liveness

Every loaded register has zero textual occurrence and zero redefinition between its LDG and matching STS. The STS is its first use, last use, and death point.

| body | word | LDG PC | register | STS PC | span |
|---|---:|---:|---|---:|---:|
| direct | 0 | 0x80e0 | R49 | 0x86e0 | 96 |
| direct | 1 | 0x8100 | R51 | 0x8770 | 103 |
| direct | 2 | 0x8120 | R52 | 0x8780 | 102 |
| direct | 3 | 0x8150 | R53 | 0x8790 | 100 |
| direct | 4 | 0x8160 | R54 | 0x87a0 | 100 |
| direct | 5 | 0x8170 | R55 | 0x87b0 | 100 |
| direct | 6 | 0x8180 | R64 | 0x87c0 | 100 |
| direct | 7 | 0x81b0 | R65 | 0x87d0 | 98 |
| direct | 8 | 0x81e0 | R177 | 0x87e0 | 96 |
| direct | 9 | 0x8210 | R178 | 0x87f0 | 94 |
| direct | 10 | 0x8220 | R179 | 0x8800 | 94 |
| direct | 11 | 0x8230 | R47 | 0x8810 | 94 |
| direct | 12 | 0x8240 | R50 | 0x8820 | 94 |
| direct | 13 | 0x8250 | R45 | 0x8830 | 94 |
| direct | 14 | 0x8260 | R46 | 0x8840 | 94 |
| direct | 15 | 0x8270 | R44 | 0x8850 | 94 |
| direct | 16 | 0x8280 | R40 | 0x8860 | 94 |
| direct | 17 | 0x8290 | R42 | 0x8870 | 94 |
| partial | 0 | 0x19d70 | R51 | 0x1a3a0 | 99 |
| partial | 1 | 0x19d00 | R55 | 0x1a2f0 | 95 |
| partial | 2 | 0x19d60 | R59 | 0x1a380 | 98 |
| partial | 3 | 0x19d80 | R96 | 0x1a3c0 | 100 |
| partial | 4 | 0x19d90 | R97 | 0x1a3e0 | 101 |
| partial | 5 | 0x19da0 | R98 | 0x1a3f0 | 101 |
| partial | 6 | 0x19db0 | R99 | 0x1a400 | 101 |
| partial | 7 | 0x19dc0 | R100 | 0x1a410 | 101 |
| partial | 8 | 0x19dd0 | R101 | 0x1a420 | 101 |
| partial | 9 | 0x19de0 | R102 | 0x1a430 | 101 |
| partial | 10 | 0x19df0 | R103 | 0x1a440 | 101 |
| partial | 11 | 0x19d20 | R46 | 0x1a300 | 94 |
| partial | 12 | 0x19d30 | R47 | 0x1a330 | 96 |
| partial | 13 | 0x19d40 | R44 | 0x1a340 | 96 |
| partial | 14 | 0x19d50 | R45 | 0x1a360 | 97 |
| partial | 15 | 0x19e00 | R50 | 0x1a450 | 101 |
| partial | 16 | 0x19e10 | R48 | 0x1a460 | 101 |
| partial | 17 | 0x19e20 | R49 | 0x1a470 | 101 |

SASS dependency-control fields also pair LDG write barriers with STS request/read masks. No loaded register is an intervening LDSM or IMMA operand. The registers are pure global-to-shared transport values. After STS, llama immediately reuses several physical registers: direct R49 is redefined at `0x86f0`, while partial R55 and R59 are redefined at `0x1a310` and `0x1a3b0`.

## Causal classification

| proposed mechanism | classification | evidence |
|---|---|---|
| Explicit accumulator/value dependency orders panel-1 LDG | disproven | Source address has no accumulator input; all 36 binary load registers have no intervening arithmetic use. |
| Explicit shared-memory/control boundary introduces panel 1 | proven | Second copy block follows phase-0 `vec_dot` and `__syncthreads`; machine BAR remains before STS. |
| Compiler schedules the LDG component before that boundary | proven | Both binaries place LDG before BAR and matching STS after BAR. |
| Exact compiler heuristic selects the 82/83-instruction lead | unknown | No matching PTX or compiler scheduling trace exists. |
| Short lifetime permits physical-register reuse | proven | Liveness ends at STS and redefinitions begin immediately afterward. |
| Register pressure causes the exact late placement | inferred, not proven | The outcome is compatible with pressure minimization, but SASS does not expose the cost-model decision. |
| A physical C++ template boundary limits the initial scheduling region | proven as source structure; causal strength unknown | Panel 1 is a second inlined copy block. SASS proves some cross-boundary motion, so the boundary is not a hard LDG order edge. |

The exact answer to "why late" is therefore: source structure makes the panel-1 copy available only in the second copy block, and the compiler hoists its independent LDGs a short distance into the phase-0 tail while leaving shared publication barrier-correct. No explicit arithmetic dependency creates the late point. The exact hoist distance is a compiler scheduling decision whose cost model is unavailable.

## Gate 9 contrast

Gate 9 encoded:

```python
panel1_base = (q8_epoch + Q8_WORDS + lid).strict_after(live_accumulator_token)
panel1_raw = tuple(q8_record[panel1_base + i*256].load() for i in range(18))
```

That makes the live phase-0 arithmetic token a structural input of the address UOp. The unrepaired compile remained in initial symbolic rewriting at `simplify_valid -> valid.backward_slice -> UOp.toposort`. The single scalar-DEFINE_REG repair still timed out at 240 seconds in `_commutative_key -> norm(u.tuplize)`. Candidate CUDA and SASS were never emitted.

Llama does the opposite: its Q8 address remains a small affine value graph with no phase-0 arithmetic input. Ordering comes from a source effect boundary plus downstream scheduling, not from merging the address and compute DAGs.

## Minimal opaque-dependency design implication

The compiler needs an order edge that is semantically reachable but excluded from value-expression recursion:

```text
ordered_base = strict_after(base_index, token_ref)

value identity/hash/canonicalization(ordered_base) == value identity/hash/canonicalization(base_index)
token_ref -> producer is retained in a side-edge table, not recursively embedded in INDEX validity/tuplize
late scheduler edge: producer -> one ordered_base materialization -> all 18 derived LDG addresses
backend must lower the edge or reject compilation
```

Minimum requirements:

- `STRICT_AFTER` dependency source is treated as an opaque leaf by symbolic validity, hashing, `tuplize`, range simplification, and address canonicalization.
- A separate scheduler/effect edge retains the actual producer, prevents DCE, and enforces producer-before-base-materialization.
- One ordered base is shared by all 18 constant-offset indices; the arithmetic DAG is not copied into each INDEX.
- Late lowering may create one zero-net address dependency, but must not add BAR, MEMBAR, global/shared traffic, local memory, or spills.
- A backend unable to preserve the edge must fail compilation rather than erase it.

The prerequisite test must attach `strict_after` to a deliberately large arithmetic DAG and prove bounded compile scaling versus the unconstrained address, then prove in SASS `producer < ordered-base materialization < LDG < STS`. Only after that test passes should the unchanged Q6 integration be retried.

## Proven, inferred, unknown

### Proven

- The exact source statement and affine address that introduce panel 1.
- No phase-0 arithmetic value participates in that address.
- The source copy is after phase-0 `vec_dot` and a barrier.
- Both SASS bodies hoist 18 LDGs before the overwrite BAR and keep all 18 STS after it.
- All 36 registers are transport-only, remain untouched until STS, die there, and are then reusable.
- Gate 9 failed because its strict dependency entered recursive initial-symbolic value processing.

### Inferred

- The short lifetime likely reduces peak register-role interference and is probably attractive to the scheduler.
- The 82/83-instruction distance likely balances memory latency hiding against register pressure.

### Unknown

- Whether CUDA front-end/NVVM or ptxas performed the split/hoist.
- The exact scheduling cost model and whether register pressure was decisive.
- Dynamic overlap, scoreboard-stall duration, and timing contribution.
- Exact C-variable mapping of every LEA source register without PTX/debug metadata.

## Reproduction commands

```bash
git -C /home/ubuntu/env/llama.cpp rev-parse HEAD
sha256sum \
  /home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh \
  /home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mma.cuh \
  docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin

git -C /home/ubuntu/env/llama.cpp show \
  ac4cddeb0dbd778f650bf568f6f08344a06abe3a:ggml/src/ggml-cuda/mmq.cuh |
  sed -n -e '2388,2650p' -e '3368,3383p' -e '3440,3538p'

.venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm -c \
  docs/task_workflow/evidence/nv-packed-q4k-q8-llama-extracted-20260830/q6k-mmq-dense.sm_120a.cubin

rg -n '/\\*(7f|80|81|82)[0-9a-f]0\\*/.*(LEA|LDG.E.CONSTANT)|/\\*(19b|19c|19d|19e)[0-9a-f]0\\*/.*(LEA|LDG.E.CONSTANT)' \
  docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/llama.nvdisasm

PYTHONPATH=. .venv/bin/python \
  docs/task_workflow/evidence/nv-q6-q8-panel1-binary-liveness-audit-20260831/parse_panel_liveness.py
```
