# Q6-V V-only tile sweep

Added `compile_research_v_schedule`, which compiles only the explicit attention-V role and derives expected global geometry from the requested output tile. The `64x64` output tile compiles with an exact candidate identity. `128x32` and `128x64` fail closed with no matching compiler PROGRAM. No nonbaseline runtime result is claimed yet.
