# S9 resource-group isolation

The requested commits were checked for route relevance before attempting an
A/B:

- `e31f340d4` changes generated two-buffer pipeline lowering and postrange
  state;
- `92d8908d6` changes typed resource-capture diagnostics;
- `774ab015b` changes the pure single-buffer evaluator gate.

The reproduced historical S9 authority is `prefill_pipe_role_selective_generated`
with `prefill_route_rolled_back=true`; it uses the external handwritten
backend-atom path. None of these commits changes that S9 emitter, route
selector, or backend atom implementation. Replaying them as isolated S9 A/B
would therefore test unrelated diagnostics/evaluator code, not a causal
performance boundary. They remain relevant to the pure generated track, not
the current S9 regression.

Result: no valid causal S9 delta from this group; do not promote or attribute
the 11.3 ms S9 gap to these commits. The next causal boundary must be in the
S9 route/atom/runtime history itself.
