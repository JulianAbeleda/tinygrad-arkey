# MMQ resource evidence contract

`extra/qk/mmq_resource_checks.py` is the resource-evidence boundary. It
consumes final code-object metadata and emitted-ISA facts; it does not emit
code or choose a route.

Admission requires explicit VGPR, LDS, scratch, spill, workgroup, wavefront,
and occupancy fields. Scratch and both spill counts must be zero; VGPR/LDS and
occupancy must satisfy caller-supplied bounds. Barrier and MFMA site counts
must be explicit, and a multi-wave workgroup requires a barrier. Missing facts
are rejected, never replaced with zeros or geometry-derived estimates.
