# Decode Native Renderer DNR-3C8 Tooling Inventory Result - 2026-06-20

## Verdict

`SCOPE_DNR3C8_TOOLING_INVENTORY_READY_PARTIAL_ATTRIBUTION_TOOLS`

DNR-3C8 inventories the tools available for q8 issue/resource attribution after DNR-3C6 refuted local static-count
matching.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c8_tooling_inventory.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c8_tooling_inventory_result.json
```

## Tool Status

| tool | status | use |
|---|---|---|
| static instruction grouping | ready | opcode class/count deltas |
| same-harness timing ladder | ready | verify actual latency movement |
| native PROGRAM resource descriptor | ready for native programs | LDS, private/scratch, kernarg, rsrc fields |
| native register scanner | ready for native programs | max/unique VGPR and SGPR use |
| oracle resource metadata | partial | launch size, group/private segment, kernarg |
| PMC counters | partial runnable | memory/cache/VALU/SALU/LDS direction |
| SQTT body timeline | blocked | capture exists, body mapping/decode unusable |
| issue/interleaving model | missing | overlap vs serialization attribution |
| search objective | blocked | needs attribution before BEAM/search |

## Resource Probe

The native PROGRAM path can already assemble and expose useful resource metadata.

| row | max VGPR | unique VGPR | max SGPR | private | LDS |
|---|---:|---:|---:|---:|---:|
| native DNR-2 | `55` | `34` | `22` | `0` | `16` |
| DNR-3C4 | `96` | `48` | `22` | `0` | `16` |

DNR-3C4's static-shape rewrite adds the `v[80:95]` preload band. It closes load/LDS counts, but it also raises the
max VGPR footprint from `55` to `96`, which is a plausible reason the count win does not translate into oracle-like
timing.

Oracle metadata currently has only partial resource data:

| field | value |
|---|---:|
| local size | `[32, 4, 1]` |
| group segment | `16` |
| private segment | `0` |
| kernarg size | `40` |

It does not yet include oracle VGPR/SGPR or live-range metadata.

## What This Means

We have enough tooling to start DNR-3C7A:

1. build a native/C4/oracle static resource ledger;
2. add live-range bands and occupancy estimates;
3. decide whether DNR-3C4's extra VGPR pressure explains why the static-count win barely moves timing.

We do not have enough tooling for full attribution yet:

- PMC is runnable but not yet a same-harness native-vs-C4-vs-oracle counter ladder.
- SQTT body timeline remains blocked.
- The issue/interleaving model is missing.
- BEAM/search remains blocked because there is no objective beyond timing.

## Next

DNR-3C7A should be the next tool to build: a resource ledger with register bands, descriptor data, launch shape,
occupancy estimate, and oracle metadata gaps clearly marked.

No renderer defaults changed.
