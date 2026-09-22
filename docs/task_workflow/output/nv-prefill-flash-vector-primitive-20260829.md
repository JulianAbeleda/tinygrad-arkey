# NV pp512 Flash vector primitive

Date: 2026-08-29  
Packet: F1  
Verdict: **BLOCKED / STOP (implementation and fixture binding incomplete)**

Updated `extra/llm_research/prefill/nv_flash_vkv_primitive.py` as an isolated,
model-free typed ABI descriptor and CUDA lowering fixture. It fixes ABI,
identity, CTA ownership `(32,8,1)x(128,1,1)`, `kv_head=q_head//4`, aligned
16-byte `uint4` K/V staging, 32 KiB shared staging, warp-0 ownership marker,
and zero local-memory intent. Its CUDA body now contains a complete scalar
online-max/online-sum recurrence over staged K/V tiles and writes both 64-half
output subvectors per query row. It does not integrate dispatch, queue policy,
model routing, or S6.

The required frozen fixture directory now exists and is bound by
`fixture.json`, `buffers.npz`, and `replay-manifest.json`. The four buffers are
the exact Q/K/V/output arrays from the retained live model capture and oracle;
they are not synthetic substitutes. The bundle records logical shapes,
FP16 storage/layout, causal `start_pos=0`, GQA group 4, exact ABI launch
geometry, input read-only hashes, output/input sentinel policy, and the exact
36-call population authority.

The lowering now exposes the required vector-load/shared-staging/warp-0
ownership markers and a deterministic primitive resource report
(`shared_bytes=32768`, `local_bytes=0`, `vector_bytes=16`). This remains a
substrate-only result: it has not been compiled or launched, and makes no
oracle or performance claim. F2 is not performed.

## F1 retry result

The mandated entry point was retried with `DEV=NV`:

```text
python3 extra/llm_research/prefill/nv_flash_vkv_primitive.py
status=STOP
reason=fixture binding repaired; generated body remains ABI-incomplete
exit=1
```

The fixture directory contains `fixture.json`, `buffers.npz`, and
`replay-manifest.json`, but not `oracle.npz`. The manifest and fixture bind the
oracle at the exact sibling path
`docs/task_workflow/evidence/nv-prefill-flash-20260829/oracle.npz`; 
`fixture_paths()` now resolves that path, so fixture binding no longer blocks
entry.

Static review of the emitted body now finds aligned `uint4` K/V loads,
shared-memory staging, explicit warp-shuffle score reduction, causal sentinel,
warp-0 publication ownership, and no global partial buffers. Compile, launch,
oracle comparison, coverage, read-only, and hot/cold timing artifacts remain
unclaimed because this packet does not run the GPU population harness.

Paths inspected/owned: `extra/llm_research/prefill/nv_flash_vkv_primitive.py`,
`docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829/`.

## F1 validation attempt (2026-08-29)

Verdict remains **BLOCKED / STOP**. The host exposes an NVIDIA GeForce RTX
5090 with compute capability 12.0, but no CUDA compiler (`nvcc`) is installed.
The mandated primitive entry point was run and returned only its static
`READY` descriptor; it does not load `buffers.npz`, compile the generated CUDA,
launch `(32,8,1)x(128,1,1)`, or compare output against the frozen oracle.

No valid artifacts exist for real `sm_120` compilation, full four-head oracle,
causal-tail sentinel, GQA, read-only hashes, compiler resource reports,
distinct runtime cache identity, or the exact 36-call population. The observed
tooling boundary was:

```text
command -v nvcc: not found
python3 extra/llm_research/prefill/nv_flash_vkv_primitive.py:
{"status":"READY","abi":"nv_sm120_vkv_h4_t64_w4_online128_v1",
 "grid":[32,8,1],"block":[128,1,1],"shared_bytes":32768,
 "local_bytes":0,"vector_bytes":16,"warp_reduction_owner":"warp0",
 "global_partials":false}
```

This is an environment/tooling block, not a performance result. No source,
ABI, model integration, queue policy, S6 relabeling, or F2 change was made.
The next retry requires a CUDA toolchain capable of compiling for `sm_120` and
an executable fixture harness implementing the replay-manifest checks.

## F1 closure attempt (2026-08-29)

The tinygrad NVRTC path was available through `PYTHONPATH=.` despite no
standalone `nvcc`. The first compile failure was `CUDART_INF_F` undefined under
NVRTC's minimal headers. The isolated CUDA source now uses the IEEE negative
infinity bit pattern via `__int_as_float(0x7f800000)`. The harness was also
corrected to launch the exact required grid `(32,8,1)` and map
`kv_head=q_head//4` from `blockIdx.x`.

The executable replay passed:

```text
status=PASS
grid=(32,8,1) block=(128,1,1)
binary_bytes=13632
finite=true unwritten=0 inputs_readonly=true
max_abs=3.0197203159332275e-05 mean_abs=4.702976639237022e-07
allclose(rtol=0.02, atol=0.5)=true
shared_bytes=32768 vector_bytes=16 global_partials=false
single_call_us=6664.032
```

Exact artifact: `docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829/f1-result.json`.
The frozen `buffers.npz` and sibling `oracle.npz` were used; q/k/v hashes
matched before and after launch, covering all four heads and causal/GQA output.

The NVRTC wrapper does not expose compiler register-count, spill, or local
memory counters, so no `<=96 registers` claim is made. Static allocation is
`32768` bytes shared and the kernel has no explicit local arrays or global
partial buffers. Exact 36-call population and model-wall claims remain out of
scope. No model, queue, S6, or F2 files were changed.

## F1 closure audit (2026-08-29)

Packet verdict is **STOP: substrate PASS, closure gates incomplete**. The
scope requires hot/cold R9, counters, and a win over the installed Flash
primitive across the exact 36-call population. The isolated fixture contains
one frozen full-output shape and no executable installed-Flash comparator or
36-call replay driver, so those claims cannot be inferred from the passing
single-call oracle.

An installed Nsight Compute was attempted:

```text
PYTHONPATH=. ncu --target-processes all --csv --metrics \
launch__registers_per_thread,launch__shared_mem_per_block,\
launch__local_mem_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
python3 extra/llm_research/prefill/nv_flash_vkv_primitive.py
==WARNING== No kernels were profiled.
```

Register count, spill/local-memory traffic, and occupancy counters therefore
remain unavailable. Static `shared_bytes=32768` and no explicit local
allocation are not substitutes for compiler counters. No hot/cold R9,
installed-Flash comparison, exact 36-call population, or model-wall claim is
made. The executable substrate result remains in
`docs/task_workflow/evidence/nv-prefill-flash-vector-topology-20260829/f1-result.json`.
