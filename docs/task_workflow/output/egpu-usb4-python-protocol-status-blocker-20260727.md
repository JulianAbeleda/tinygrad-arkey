# eGPU USB4 Python protocol/status blocker

Date: 2026-07-27

Status: blocked on native protocol restoration

Scope: Python-side status integration for `APLRemotePCIDevice` in
`tinygrad/runtime/support/system.py`.

## Finding

Python cannot safely implement a keepalive capability or status query against
the currently installed TinyGPU server. `RemoteCmd` already reserves values
`0..14` in Python, including `PING`, `HEALTH`, and `SYSMEM_SYNC`, while the
last retained native `Shared/server.c` only implements values `0..11` and has
no protocol version negotiation. Its unknown-command response is a generic
error, so a Python client cannot distinguish an old server from a future
server that assigns a different meaning to an uncoordinated command number.

`REMOTE_KEEPALIVE_S` is not consumed by `APLRemotePCIDevice`. It must remain
non-authoritative until a native capability handshake exists; Python must not
claim that the environment variable has enabled protection.

## Required native-first contract

Before changing Python behavior, restore the native source and define one
versioned schema shared by `server.c` and `system.py` (generated, or checked
by numeric/layout tests). Preserve existing command numbers. The schema needs:

- a side-effect-free protocol-version/capability query;
- a `KEEPALIVE_STATUS` capability and status command;
- explicit `unsupported_protocol` / `unsupported_capability` errors;
- a fixed JSON or binary status payload with a schema identifier and exact
  field units;
- payload-length validation on both sides.

For a legacy server, the handshake must return an explicit unsupported result
without sending a status request, hanging, or treating a generic RPC failure
as an active keeper.

## Python integration after the contract lands

1. Add `APLRemotePCIDevice.keepalive_status()` that performs the handshake,
   capability check, status request, and strict payload validation.
2. Return an explicit unsupported/inactive result; do not infer support from
   `REMOTE_KEEPALIVE_S` or native log output.
3. Surface the effective interval, enabled state, identity, driver generation,
   tick counters, consecutive failures, timestamps, maximum success gap, and
   timer error exactly as provided by the native status schema.
4. Treat `REMOTE_KEEPALIVE_S` as either a negotiated development policy
   request or a fail-loud deprecated setting. Do not leave it as a silent no-op.
5. Add CPU-only tests for command numeric stability, old-server fallback,
   malformed/truncated status rejection, and schema validation.

No Python runtime code was changed in this pass because assigning a provisional
wire command before the native protocol source of truth exists would create a
compatibility hazard on the installed app.
