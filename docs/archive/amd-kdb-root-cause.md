# AMD PSP KDB "BL not ready" — Root Cause

Closed 2026-06-10. This documents the root cause, the bisection that found it,
and the verification, so the May 22 → June 10 investigation has a single
authoritative ending.

## Symptom

First PSP bootloader command (KDB load): driver writes `C2PMSG36=fw_pri>>20`,
`C2PMSG35=0x80000`; PSP consumes the command (`35 -> 0`) and never sets ready
(`0x80000000`). 10s timeout, no error bits, no MMHUB fault. Linux amdgpu on the
same boot completes the same command in ~0.65 ms.

## Root cause

`_load_remote_discovery_profile` (the `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c`
workaround added 2026-05-22 in `b18b8ffed` because reading the discovery table
through the indirect VRAM aperture wedged the TinyGPU bridge) hardcoded the
wrong IP versions for this card:

| Field | Profile (wrong) | Real (discovery table) |
|---|---|---|
| MP0 (PSP) | (13,0,10) | **(13,0,0)** |
| MP1 (SMU) | (13,0,10) | **(13,0,0)** |
| NBIO | (7,9,0) | **(4,3,0)** |

PSP firmware is selected by MP0 version (`psp_{ver}_sos.bin`, `amdev.py`), so
every profile-driven run extracted the KDB from `psp_13_0_10_sos.bin` — a
differently-signed sOS bundle for a different ASIC. The PSP bootloader
consumes a wrongly-signed KDB and silently never raises ready. That is the
entire failure. The wrong values appear nowhere in the codebase before the
profile commit; they were most likely cribbed from the GC 11.0.3 /
PSP 13.0.10 variant IP set (plausibly seeded by the "RX 7900 GRE [XFX]"
subsystem string). This card is a standard Navi31: GC 11.0.0 / PSP 13.0.0.

Because the profile was part of the experiment harness (forced in
`run_remote_kdb_attempt.sh`'s base env on Mac and Ubuntu alike), every one of
the ~40 KDB variants of May 24 – June 9 ran with the wrong firmware. The
equivalence audits (payload sha, GART PTEs, MMHUB state, mailbox ordering)
all "matched Linux" because the driver-visible state really was equivalent —
the input firmware was not.

## How it was found (2026-06-10, one day, 7 cold boots)

Cold-boot-gated bisection, one variable per boot, each boot sandwiched by
`linux_amd_state.sh` NORMAL_HEALTHY gates and a boot-ID freshness check:

| Cycle | Config | Result |
|---|---|---|
| 1 | stock local (no bridge, no env) | PASS |
| 2 | stock through serve.py | PASS |
| 3 | + `AM_REMOTE_SKIP_RESIZE_BAR` + `AM_REMOTE_SMALL_BAR_DISCOVERY` | PASS |
| 4 | + `AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c` | **FAIL — BL not ready** |
| 5 | ipver trace (`AM_PSP_TRACE_REGS=1`, real discovery) | real versions: `mp0=(13,0,0) nbio=(4,3,0) gc=(11,0,0)` |
| 6 | Cycle-4 config + fixed profile | KDB ready in **0.618 ms**; died later at `KeyError: 26` |
| 7 | + NBIF alias fix | **FULL PASS** — `[2, 3, 4]`, rc=0 |

Key captures: `kdb-stock-local-20260610-143743`, `kdb-stock-remote-20260610-152602`,
`kdb-smallbar-remote-20260610-164228`, `kdb-discovery-profile-remote-20260610-165833`
(the isolating FAIL), `kdb-ipver-trace-20260610-171825` (ground truth),
`kdb-profile-fixed-20260610-173436`, `kdb-profile-fixed2-20260610-174517` (verification).

## Fixes

- `ca3a0d816` — correct profile IP versions (MP0/MP1 13.0.0, NBIO 4.3.0); also
  fixes the same wrong 13.0.10/11.0.3 assumption baked into
  `capture_linux_psp_good_trace.sh`'s firmware hash list.
- `b033d47a7` — populate the `NBIF_HWIP` alias (real discovery fills both
  NBIO/NBIF keys, the only HWIP alias pair; `AMDDevice.__init__` reads NBIF).

## Hypotheses killed by this result

GART PTE flags/cacheability, MMHUB context/invalidate/CID2 semantics, msg1
placement (VRAM vs sysmem-GART, low vs top table), mailbox ordering/timing,
sysmem allocation modes, KDB payload slicing/sizing — none were ever the
blocker. The 0x1d40-vs-0x1d50 KDB size mismatch chased by the slice variants
was the two different source binaries.

## Lessons

1. Run the stock configuration as a control before instrumented variants. One
   boot on day 2 would have saved ~2.5 weeks.
2. Any constant written down without verification is part of the system under
   test — especially constants inside workarounds that silence an error.
3. A regression introduced by the debugging harness reads exactly like a
   deeper layer of the disease it was built to investigate. When a new failure
   appears within a day of a new workaround, suspect the workaround first.
4. PSP silent hang (command consumed, no ready, no error) is the signature of
   firmware signature rejection — verify firmware identity before memory
   semantics.

## Open / next

- Re-test the Mac/TinyGPU path with the fixed profile (the only environment
  that actually needs it). Consider re-testing whether the indirect-aperture
  discovery read still wedges the patched TinyGPU build — if not, delete the
  profile entirely.
- The original problems the fork exists for remain: slowness (RPC round trips
  per MMIO access) and dropouts under sustained DMA load (16MB PrepareDMA
  trigger mitigated by `AMD_REMOTE_ALLOC_CAP_MB=2`, mechanism unexplained).

## 2026-06-10 evening addendum: Mac path closed too

Same day, the fixed profile was validated on the original Mac/TinyGPU path.
Full AM boot (`boot done`, all IP blocks including PSP) completed in 11.4s /
26,263 roundtrips, and `Tensor([1,2,3])+1 -> [2, 3, 4]` executed on the GPU —
the first successful Mac boot and computation on this card.

Two additional root causes found and fixed along the way:

- `ensure_app` downloaded the pinned upstream TinyGPU release mid-session,
  pkilled the running patched app, and installed over /Applications. The
  resulting mismatched stack (upstream app + arkey dext) made single u32 BAR0
  writes close the bridge — masquerading as a transport fragility for several
  hours. Fixed in 9eb0b042b (never clobber an arkey-signed app). All write
  sizes up to 4KB pass on the matched stack.
- MMHUB register reads hang the fabric (and drop the PCIe tree) when the card
  is in a stale gated state. Recovery rule: a TRUE power-off of the GPU resets
  it; a cable replug does not.

Operational rules for the Mac path: check serve.py's age before reuse (a
stale bridge runs stale protocol code), verify /Applications/TinyGPU.app is
the arkey build, and power-cycle (not replug) after any bridge death.

The "Open / next" items above are superseded by the addendum: the Mac path is
validated. Remaining: re-test whether the patched TinyGPU survives a real
discovery-table read (if yes, delete the profile), retest Qwen under load
(the original dropout investigation's open item), and relax
`REMOTE_MMIO_CHUNK`/`REMOTE_MMIO_FENCE_EVERY` for speed once stability holds.
