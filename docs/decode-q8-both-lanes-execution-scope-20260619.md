# Decode q8 both-lanes execution scope - 2026-06-19

Purpose: execute the user's "do both" decision after the two-lane closeout.

## Lane 1 - Accept Research Artifact Route

Decision:

- keep `Q8_FFN_HANDWRITTEN=1`;
- keep default off;
- accept the external hipcc/LLD HSACO dependency for research-flag use only;
- do not generalize beyond the documented Qwen3-8B/gfx1100/Q4_K gate-up shape without a new scope.

Done means:

- the flag is already wired;
- artifact rebuild command and hashes are documented;
- W==D and dNLL evidence remains the promotion gate;
- policy boundary is explicit.

## Lane 2 - Charter Native Transfer Project

Decision:

- fund as project-level AMD backend work, not a q8-specific bounded patch;
- use the artifact route as oracle;
- do not implement compiler changes until a bounded feature has `>=30us` movement or the broader backend project is
  explicitly funded.

Done means:

- native project phases are defined;
- start gate is explicit;
- current native failures remain recorded;
- artifact route remains the oracle.

## Non-Goals

- no default behavior change;
- no artifact dependency promotion beyond research;
- no native compiler implementation in this phase;
- no imported Q4 route resurrection.
