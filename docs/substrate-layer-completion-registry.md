# Substrate Layer Completion Registry

Canonical tracking doc for the pure-generated prefill WMMA GEMM substrate (recover the ~4413 tok/s hand kernel
with NO handwritten kernels). The layer model is vendor-agnostic; the coverage is per-GPU.

**Completion axis (what the % measures):** how complete tinygrad-arkey's ability is to *generate* this layer's
output — at hand-kernel quality, for the prefill WMMA GEMM — **without a handwritten kernel and without depending on
a closed vendor compiler to own the perf-critical layers.** 100% per layer is defined objectively below.

Confidence: **AMD = census-backed** (file:line audited 2026-07-06). **NVIDIA / Metal = architecture-reasoned
estimates**, not yet censused — flagged ⚠. Run the same census to harden them.

---

## 1. Main table (agnostic) — the layer stack across vendors

| # | Layer (agnostic) | AMD gfx1100 | NVIDIA | Metal (Apple) |
|---|---|---|---|---|
| 1 | Tensor / high-level IR | UOp (`uop/`) | UOp (shared) | UOp (shared) |
| 2 | Schedule / tiling | OptOps (`codegen/opt`) | OptOps (shared) | OptOps (shared) |
| 3 | Instruction selection | HIP / LLVM / **native ISA** | CUDA C / **PTX** / NAK(NIR) | MSL |
| 4 | Instruction scheduling / **software pipelining** | LLVM *or* tinygrad-ISA | **ptxas** (closed) | **Apple** (closed) |
| 5 | Register allocation | LLVM *or* tinygrad-ISA | **ptxas** (closed) | **Apple** (closed) |
| 6 | Async-memory sync (**waitcnt**) | `s_waitcnt` (LLVM/ISA) | scoreboard/`cp.async` (**ptxas**) | fences (**Apple**) |
| 7 | Tensor-core instruction | WMMA 16×16×16 | mma/wgmma | simdgroup_matrix 8×8×8 |
| 8 | Assembly / object | `assemble_linear`→**ELF (open)** | ptxas→cubin (**closed**) | →AIR/metallib (**closed**) |
| 9 | Hardware | gfx1100 RDNA3 | SM (Hopper/Ada…) | AGX |

**Shared / agnostic:** layers 1–2 and the `TensorCore`/`Ops.WMMA`/`OptOps` abstraction are ONE codebase for all
backends (each vendor supplies only its tc dims + renderer leaf). **Vendor-locked:** how *low tinygrad can descend*
to own layers 4/5/6 — only AMD (open ISA + native ISA renderer + open assembler) and x86 CPU have a native-ISA path.
NVIDIA/Metal bottom out at a *virtual* ISA (PTX) / source (MSL); the closed vendor compiler owns 4/5/6/8.

---

## 2. Per-layer completion definitions (objective 100% criterion)

| # | Layer | **100% =** |
|---|---|---|
| 1 | Tensor/IR | the GEMM math is expressible as device-independent UOps |
| 2 | Schedule/tiling | OptOps express the tile/wave/unroll/pad shape the hand kernel uses; searched schedules injectable |
| 3 | Instruction selection | every UOp **including `Ops.WMMA`** lowers to target instructions on a path that can own layers 4/6 |
| 4 | Instr. scheduling / pipelining | can emit a **software-pipelined (double-buffered) K-loop** with real load/compute overlap at hand quality |
| 5 | Register allocation | allocates **WMMA fragment tuples (contiguous VGPR ranges)** + handles register pressure / spill |
| 6 | Async-mem sync | emits **counter-targeted `vmcnt(n)`/`lgkmcnt(n)`** (not just full-drain) to gate the pipeline |
| 7 | Tensor-core instruction | emits the matrix instruction (`v_wmma`) from `Ops.WMMA` on the goal path |
| 8 | Assembly/object | assembles a finalized instruction list into a launchable binary |
| 9 | Hardware | the target part exists and is characterized |

---

## 3. AMD gfx1100 checklist (census-backed) — the project target

Framing: coverage is for the **native-ISA path** (`renderer/isa/amd.py`), i.e. the Track-B goal path — because the
default HIP path, while 100%-functional, **structurally caps at ~40 TFLOPS** (LLVM owns 4/6, [seb-v/census]) and
cannot reach the hand kernel's ~58. So a "done" HIP layer that can't own pipelining is NOT 100% for our goal.

| # | Layer | Coverage | Rationale (file:line) |
|---|---|---:|---|
| 1 | Tensor/IR | **100%** | device-independent UOp graph; done |
| 2 | Schedule/tiling | **100%** | `OptOps` TC/UPCAST/LOCAL/UNROLL/PADTO (`opt/__init__.py:6-9`, `postrange.py`); warmstart injects searched schedules (`postrange.py:541-564`) |
| 3 | Instruction selection (ISA) | **100%** | scalar/VALU/LDS/global/loops/index and `Ops.WMMA` lower on the native ISA path; default-on b128 fragment loads fold route-shaped prefill `a @ b.T` into 16 `global_load_b128`, 0 packs, 0 scalar half loads |
| 4 | Instr. scheduling / pipelining | **30%** | list scheduler is span-aware and `v_wmma` latency is modeled; remaining blocker is explicit two-phase fragment pipeline scope in `docs/native-isa-l4-software-pipeline-scope.md` |
| 5 | Register allocation (ISA) | **100%** | multi-output WMMA uses low contiguous accumulator/A/B fragment windows and reclaims `v1..v7` scalar scratch for epilogues; generated 4x4 remu/GPU passes |
| 6 | Async-mem sync / waitcnt (ISA) | **55%** | `_insert_waitcnt` default remains full-drain; opt-in targeted `vmcnt(n)` is 4x4-correct and scalar-pack waits are coalesced, but it is still not a performance-valid default |
| 7 | Tensor-core instruction (ISA) | **100%** | `Ops.WMMA` emits `v_wmma_f32_16x16x16_f16`; rolled any-K and 4x4 generated harnesses are bit-correct |
| 8 | Assembly / object | **100%** | `assemble_linear`→ELF complete (`amd/elf.py:15`); autogen encodes every needed instr; **proven by the hand kernel end-to-end** |
| 9 | Hardware | **100%** | gfx1100 characterized (WMMA units, vmcnt/lgkmcnt, VGPR≥238 trap known) |

**AMD rollup:** floor (8), top (1–2), native-ISA instruction selection (3), tensor-core emit (7), and WMMA register
allocation (5) are done on the native ISA path. The remaining handtrace-parity gap is concentrated in performance-valid
L6 targeted waitcnt and L4 load/compute overlap. Cooperative-B ownership remains a HIP/postrange medium-stage issue, but
it is no longer a native-ISA b128 eligibility blocker. The current measured table-local generated path remains below the
hand trace class.

---

## 4. NVIDIA checklist ⚠ (estimated — architecture-reasoned, not censused)

Different regime: the vendor compiler (`ptxas`) is excellent and `cp.async`/`mbarrier` express pipelining, so **peak
GEMM is achievable WITHOUT tinygrad owning 4/5/6** — but tinygrad also *cannot* own them (no SASS assembler; floor =
PTX, a virtual ISA). "Peak achievable?" ≠ "tinygrad-generated ownership."

| # | Layer | tinygrad-gen coverage | Peak achievable? | Note |
|---|---|---:|:--:|---|
| 1 | Tensor/IR | 100% | ✅ | shared UOp |
| 2 | Schedule/tiling | 100% | ✅ | shared OptOps; `cuda_sm75/80/89` tc (`tc.py:131-135`) |
| 3 | Instruction selection | ~90% ⚠ | ✅ | `CUDARenderer`/`PTXRenderer` (`cstyle.py:9`, `ptx.py:134`); mma via PTX |
| 4 | Instr. scheduling / pipelining | 0% (vendor) ⚠ | ✅ | ptxas owns it; expressed via `cp.async` intrinsics |
| 5 | Register allocation | 0% (vendor) ⚠ | ✅ | ptxas owns it |
| 6 | Async-mem sync | 0% (vendor) ⚠ | ✅ | scoreboard/control-codes owned by ptxas |
| 7 | Tensor-core instruction | ~85% ⚠ | ✅ | mma/wgmma emittable via PTX |
| 8 | Assembly/object | 0% (closed) ⚠ | ✅ | ptxas→cubin; no open path |
| 9 | Hardware | 100% | — | SM |

**NVIDIA takeaway:** the descent-to-own-pipelining move (AMD's Track B) is **neither needed nor available** — you
delegate 4/5/6/8 to ptxas and reach peak via async-copy intrinsics at layers 3/7.

---

## 5. Metal (Apple) checklist ⚠ (estimated — architecture-reasoned, not censused)

Same regime as NVIDIA: Apple's compiler owns 4/5/6/8; floor = MSL source.

| # | Layer | tinygrad-gen coverage | Peak achievable? | Note |
|---|---|---:|:--:|---|
| 1 | Tensor/IR | 100% | ✅ | shared UOp |
| 2 | Schedule/tiling | 100% | ✅ | shared OptOps; `metal` tc 8×8×8 (`tc.py:181`) |
| 3 | Instruction selection | ~85% ⚠ | ✅ | MSL via cstyle |
| 4 | Instr. scheduling / pipelining | 0% (vendor) ⚠ | ✅ | Apple compiler owns |
| 5 | Register allocation | 0% (vendor) ⚠ | ✅ | Apple compiler owns |
| 6 | Async-mem sync | 0% (vendor) ⚠ | ✅ | Metal fences owned by Apple |
| 7 | Tensor-core instruction | ~80% ⚠ | ✅ | `simdgroup_matrix` via MSL |
| 8 | Assembly/object | 0% (closed) ⚠ | ✅ | →AIR/metallib; no open path |
| 9 | Hardware | 100% | — | AGX |

---

## 6. Rollup & the one insight

- **Layers 1–2 + tc/WMMA/OptOps abstraction are agnostic** — any work there helps every backend at once (the truly
  portable investment).
- **AMD is the outlier:** its vendor compiler underperforms for this GEMM (40 vs 58 TFLOPS) AND it uniquely allows a
  native-ISA descent → owning layers 4/5/6 (Track B) is both *needed* and *possible*. On NVIDIA/Metal it is neither.
- **The AMD gap is concentrated and defined:** 5 items in `renderer/isa/amd.py` (L7 WMMA, L5 fragment regalloc, L4
  pipelining, L6 targeted waitcnt, L3 wide-mem). Everything above and below is at 100%.

Sources for the agnostic framing: see `docs/prefill-substrate-layer-census-20260706.md` (LLVM backend phases, MLIR
progressive lowering, RDNA3 ISA, SIInsertWaitcnts, CUTLASS pipelining, seb-v gfx11 50-TFLOPS proof).
</content>
