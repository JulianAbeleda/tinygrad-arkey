# AMD WMMA resource evidence adapter

`extra.qk.amd_wmma_resource_adapter` joins final AMD evidence to the existing
fail-closed `check_mmq_resource_evidence` gate.

The AMD code-object notes are authoritative for allocated VGPR/SGPR, fixed LDS,
private scratch, spill counts, workgroup size, and wavefront size. Generated
assembly is authoritative for barrier and WMMA/MFMA sites. No number is
reconstructed from register references, LDS declarations, geometry, or device
limits.

Occupancy is not emitted by the code-object notes or assembly parser. Callers
must supply an explicit measured/compiler occupancy value; omission remains
missing and the existing gate rejects the candidate. This keeps successful
compilation separate from resource admission.
