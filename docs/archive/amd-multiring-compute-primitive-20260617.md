# AMD multi-compute-ring primitive — scope & audit (Phase 0, 2026-06-17)

Goal: make **same-process compute overlap** possible by giving `AMDDevice` more than one real hardware compute
ring, behind an opt-in env, default unchanged. This is the lever gated in the decode arc (cross-process probe
showed ~+32% hardware overlap; same-process tinygrad overlap is ~1.0× because everything funnels to one ring).
`[runtime]`/`[test]` only — NO model.py, NO decode scheduler in this arc (stop after the Phase-3 overlap probe).

## Current single-ring behavior

- `AMDDevice.__init__` creates exactly one compute ring: `self.compute_queue = self.create_queue(COMPUTE_AQL
  if is_aql else COMPUTE, ...)` (ops_amd.py:1001). On gfx1100 `xccs==1` → `is_aql=0` → **PM4 (non-AQL)**.
- `self.hw_compute_queue_t = functools.partial(AMDComputeAQLQueue if is_aql else AMDComputeQueue, self)`
  (:1010) — every `hw_compute_queue_t()` is just a *command builder*; they all `_submit` into the one ring.
- `AMDQueueDesc` (:673) is a **self-contained** ring descriptor: own `ring`, `read_ptr`, `write_ptr`,
  `doorbell`, `put_value`, `params`. `signal_doorbell` (:681) writes only its own write_ptr/doorbell. So N
  independent descriptors = N independent rings; nothing in the descriptor is global.

## Why same-process queues serialize (root cause)

Both compute submit paths **hardcode `dev.compute_queue`**:
- PM4 `AMDComputeQueue._submit` (:409–422): writes `dev.compute_queue.ring`, bumps `dev.compute_queue.put_value`,
  rings `dev.compute_queue.signal_doorbell`.
- AQL `AMDComputeAQLQueue._submit` (:456–465): same, into `dev.compute_queue.ring`.

So no matter how many `hw_compute_queue_t()` builders exist, every dispatch lands on the **one** hardware ring
→ the GPU's command processor executes them in ring order → serial. (Cross-*process* overlaps because each
process opens its own KFD queue = its own ring.)

## The copy queue already proves the pattern

`AMDCopyQueue._submit` (:526) does `sdma_queue = dev.sdma_queue(self.queue_idx)` and targets *that* descriptor;
`dev.sdma_queue(idx)` (:1065) lazily creates SDMA rings in a `self.sdma_queues` dict; `hw_copy_queues` (:1117)
exposes them. **Compute simply lacks the `queue_idx` + per-idx descriptor that copy already has.** The refactor
mirrors this.

## KFD vs AM/PCI/USB (where a 2nd ring is safe)

- **KFD (local, this box):** `KFDIface.create_queue` (:796) calls `AMDKFD_IOC_CREATE_QUEUE`; **KFD assigns each
  queue its own `doorbell_offset`** (the `idx` arg is unused here). The doorbell mmap region (`doorbells_base`,
  0x2000) is shared but each queue's doorbell sits at a distinct offset. So a 2nd compute ring = one more
  `create_queue(COMPUTE)` call; KFD gives an independent ring + doorbell. **Low risk.**
- **AM / PCI / USB (`is_am()`):** these manage doorbells/rings themselves and thread `idx` (:877, :942); a 2nd
  ring needs a non-colliding `idx` and the recovery path. **Out of scope initially** — gate the 2nd ring to
  KFD-local first; fail clearly elsewhere.

## AQL vs non-AQL

PM4 (gfx1100) has **no shared per-queue state** in the submit path (the indirect-buffer trick at :412 is for
`xccs>1` only; the IB lives in the queue's own ring). **AQL is different:** `create_queue` stores a SINGLE
`self.aql_gart`/`self.aql_desc` (:1051–1052) and scratch setup rewrites `self.aql_desc` (:1090–1103) — a 2nd
AQL ring would clobber it and needs a per-queue `aql_desc`. So **start with PM4/non-AQL** (the gfx1100 default);
explicitly refuse a 2nd ring when `is_aql` until per-queue AQL desc is built.

## Timeline / signals

`timeline_signal` + `timeline_value` are **device-global** (one monotonic timeline). Independent workloads on
two rings (Phase 3) don't need a shared timeline. Cross-ring dependencies (Phase 4) need explicit wait/signal;
the global timeline value must be incremented carefully so a wait on ring 1 sees ring 0's signal. The HCQ
`wait`/`signal` ops already encode (signal, value) pairs, so cross-ring ordering should compose — to be PROVEN
in Phase 4, not assumed.

## PMC / SQTT / recovery assumptions (must preserve)

- PMC/SQTT build their queues via `unwrap(self.hw_compute_queue_t)()` (:615, :618, :624, :1032) → ring 0 by
  default. Fine (profiling stays on ring 0); the refactor must keep `hw_compute_queue_t()` defaulting to idx 0.
- Recovery `on_device_hang` (AM path, :877–909) resets only `d.compute_queue` (idx 0). A 2nd ring would not be
  reset → on a hang with rings>1, the box may need a manual reset. Since 2nd ring is KFD-gated and KFD recovery
  differs (`can_recover=self.is_am()`), Phase 3 treats a hang as a **kill condition to report**, not patch.

## Refactor shape (Phases 1–2)

- **Phase 1 (NFC):** `AMDComputeQueue`/`AMDComputeAQLQueue` gain `queue_idx=0`; `_submit` resolves a
  per-idx descriptor (`dev.compute_queue_desc(self.queue_idx)`); `dev.compute_queue` stays as the idx-0
  descriptor (alias for PMC/SQTT/recovery). Only one ring exists; default path byte-identical.
- **Phase 2:** `AMD_COMPUTE_RINGS` (default 1, opt-in 2). On request + KFD + non-AQL, lazily create ring 1
  (`compute_queue_desc(1)` via `create_queue(COMPUTE)`); a builder for idx 1 (`hw_compute_queue_t(queue_idx=1)`).
  Default call unchanged.

## Measurement method (Phase 3)

GPU-kernel `time_sum_s` SUMS per-kernel times and does NOT reveal overlap. Overlap is a **wall-clock** property:
dispatch two large independent workloads A,B and compare `wall(A on r0 ‖ B on r1)` vs `wall(A;B serial on r0)`.
Mitigate the wall-clock trap (the prior LDS lesson): use big workloads (≫ dispatch overhead), `System` fence +
read the timeline/EOP signal to confirm completion before stopping the clock, and take the min over many
repeats. Cases: A‖B, A‖A, B‖B; single-ring control must stay ~1.0×.

## Kill conditions

Two real rings still serialize (~1.0×); doorbell / read-write-pointer conflict; device fault/hang; timeline
signals don't order across rings. On any of these: **stop and report; do not patch around it in model code.**

## Plan / stop point

Phase 0 (this doc) → Phase 1 (NFC queue-idx) → Phase 2 (opt-in 2nd ring) → **Phase 3 (overlap probe) → STOP**.
Phases 4–7 (dependency guards, DAG prototype, decode design, guarded decode) only if Phase 3 proves >1.2×.
