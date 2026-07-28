# eGPU USB4 Python protocol/status blocker

Date: 2026-07-27

Status: blocked on the P1 wire-specification deliverable and compatible native implementation

Scope: Python-side status integration for `APLRemotePCIDevice` in
`tinygrad/runtime/support/system.py`.

## Finding

Python cannot safely implement a keepalive capability or status query against
the currently installed TinyGPU server. `RemoteCmd` already reserves values
`0..14` in Python, including `PING`, `HEALTH`, and `SYSMEM_SYNC`, while the
recovered native `Shared/server.c` implements the older command set through
`0..11` and has no protocol version negotiation. Its unknown-command response
is a generic error, so a Python client cannot distinguish an old server from a
future server that assigns a different meaning to an uncoordinated command
number.

The native TinyGPU source has been recovered locally under `extra/usbgpu/`,
but remains untracked. The DriverKit keepalive prototype builds. Neither fact
establishes a safe Python-facing protocol: the native handshake and status
operations have not been implemented, and the required
`extra/usbgpu/protocol/tinygpu-wire-v1.md` specification/fixtures do not yet
define their command numbers, payload layouts, errors, or compatibility behavior.

`REMOTE_KEEPALIVE_S` is not consumed by `APLRemotePCIDevice`. It must remain
non-authoritative until a native capability handshake exists; Python must not
claim that the environment variable has enabled protection.

## Required protocol contract

Before changing Python behavior, add and freeze
`extra/usbgpu/protocol/tinygpu-wire-v1.md` plus machine-readable fixtures for
`server.c` and `system.py`. Preserve existing command numbers. Per the fork
principle, client and server wire identifiers remain independently implemented;
they are not generated from a shared declaration. The specification is the
wire-encoding authority, the main task scope freezes status semantics, and
conformance tests must check numeric identifiers, payload layouts, and
compatibility behavior at both endpoints.

The specification and compatible native implementation need:

- a side-effect-free protocol-version/capability query on new servers;
- a `KEEPALIVE_STATUS` capability and status command;
- explicit `unsupported_protocol` / `unsupported_capability` errors;
- the scope's fixed `tinygpu.keepalive.v1` JSON semantic fields, including exact
  field units, cadence counters, saturation state, and active resource counts;
- payload-length validation on both sides;
- a native handshake implementation and status implementation that conform to
  the frozen specification;
- explicit compatibility rules for the installed legacy server and all future
  protocol versions.

An unmodified legacy server cannot return a new typed handshake error. The
client sends only the bounded handshake probe selected by the wire
specification. A complete legacy 17-byte response with `status=RESP_ERR`,
`resp0=0`, and `resp1=0`, or clean EOF before any response byte, maps locally to
`unsupported_protocol`; the client closes and sends no status request. Timeout,
partial response, a nonzero legacy error length, or other bytes are protocol
errors. New servers return typed unsupported-version/capability errors.

## Python integration after the contract lands

1. Add `APLRemotePCIDevice.keepalive_status()` that performs the handshake,
   capability check, status request, and strict payload validation.
2. Return an explicit unsupported/inactive result; do not infer support from
   `REMOTE_KEEPALIVE_S` or native log output.
3. Surface the effective interval/leeway, enabled state, identity, provider
   generation, tick/gap counters, saturation, consecutive failures, timestamps,
   maximum success gap, timer error, and active resource counts exactly as
   provided by the native status schema.
4. Treat `REMOTE_KEEPALIVE_S` as either a negotiated development policy
   request or a fail-loud deprecated setting. Do not leave it as a silent no-op.
5. Add CPU-only tests for command numeric stability, legacy-server unsupported behavior,
   malformed/truncated status rejection, and schema validation.

No Python runtime code changed in this pass. Assigning a provisional wire
command before the frozen specification, native handshake/status implementation,
and compatibility rules exist would create a compatibility hazard with the
installed app.
