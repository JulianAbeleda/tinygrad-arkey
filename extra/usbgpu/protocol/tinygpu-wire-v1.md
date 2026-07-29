# TinyGPU wire protocol v1

This document and `fixtures/` are the encoding authority for the TinyGPU Unix
socket protocol and its DriverKit selector bridge. They deliberately do not
generate C, Swift, or Python declarations: each endpoint owns an independent
codec and constants, verified against these fixtures.

Each endpoint implements its own independent codec; fixtures, rather than a
shared runtime declaration, keep those implementations conformant.

The fixtures are `fixtures/legacy-rpc-v1.json`,
`fixtures/handshake-v1.json`, `fixtures/error-v1.json`, and
`fixtures/keepalive-status-v1.json`, and
`fixtures/power-residency-status-v2.json`.

## Scope and byte order

All integer fields are unsigned little-endian unless explicitly called signed.
The legacy request is exactly 33 bytes, `struct.pack("<BIIQQQ", command,
device_id, bar, arg0, arg1, arg2)`. The legacy response is exactly 17 bytes,
`struct.pack("<BQQ", status, resp0, resp1)`. New-command responses and all
responses carrying an error payload use `resp0` as the byte length of the
immediately following payload and use `resp1` only as specified below.
Legacy success responses retain their command-specific `resp0`/`resp1` meanings;
they are not reinterpreted as a new envelope. No implementation may use native
C struct layout as the wire definition.

Response status codes are fixed: `OK=0`, `LEGACY_ERROR=1`,
`UNSUPPORTED_VERSION=2`, `UNSUPPORTED_CAPABILITY=3`,
`MALFORMED_REQUEST=4`, `INVALID_STATE=5`, `BUSY=6`, `INTERNAL_ERROR=7`,
`PROVIDER_UNAVAILABLE=8`, `DEVICE_LOST=9`, and `NATIVE_ERROR=10`.
Codes 2..7 carry the typed JSON error payload below; code 1 preserves opaque
legacy error bytes or an empty payload.

Legacy commands are frozen:

| ID | Name | Meaning | Successful response |
|---:|---|---|---|
| 0 | `PROBE` | probe compatible devices | `resp0=payload bytes`, `resp1=0`, UTF-8 device-list payload |
| 1 | `MAP_BAR` | map PCI BAR | `resp0=mapped address`, `resp1=mapped bytes` |
| 2 | `MAP_SYSMEM_FD` | allocate DMA memory, return SCM_RIGHTS fd | `resp0=mapped bytes`, `resp1=allocation index`, exactly one fd |
| 3 | `CFG_READ` | PCI configuration read | `resp0=value`, `resp1=0` |
| 4 | `CFG_WRITE` | PCI configuration write | both zero |
| 5 | `RESET` | reset device | both zero |
| 6 | `MMIO_READ` | bulk BAR read | `resp0=payload bytes`, `resp1=0`, exact binary payload |
| 7 | `MMIO_WRITE` | bulk BAR write | both zero |
| 8 | `MAP_SYSMEM` | legacy system-memory map | `resp0=physical-address payload bytes`, `resp1=handle`, exact binary payload |
| 9 | `SYSMEM_READ` | legacy system-memory read | `resp0=payload bytes`, `resp1=0`, exact binary payload |
| 10 | `SYSMEM_WRITE` | legacy system-memory write | both zero |
| 11 | `RESIZE_BAR` | legacy BAR resize/no-op | both zero |

IDs `12` (`PING`), `13` (`HEALTH`), and `14` (`SYSMEM_SYNC`) are reserved by
the current Python endpoint. A v1 server must either implement their existing
semantics or reject them as reserved/unsupported; it must not allocate them.

## Negotiation commands

New commands retain the legacy 33-byte request and 17-byte response envelope.
Their payload, successful or error, follows the response and has exactly
`resp0` bytes. Payloads are at most 65,536 bytes. A receiver rejects a payload length above that limit,
partial header/payload, invalid reserved bits, or disconnect with a typed local
protocol error; it must not dispatch the request to hardware.

| ID | Name | Request fields | Success payload / response fields |
|---:|---|---|---|
| 15 | `HANDSHAKE` | `arg0=client_major`, `arg1=client_minor`, `arg2=required_capabilities`; `device_id=bar=0` | UTF-8 JSON capability object; `resp1=server_major` |
| 16 | `LEASE_ACQUIRE` | `device_id=bar=arg0=arg1=arg2=0` | no payload; `resp1=lease_id` |
| 17 | `LEASE_RELEASE` | `device_id=bar=arg1=arg2=0`, `arg0=lease_id` | no payload |
| 18 | `KEEPALIVE_STATUS` | `device_id=bar=arg0=arg1=arg2=0` | UTF-8 `tinygpu.keepalive.v1` JSON |
| 19 | `KEEPALIVE_SET_POLICY` | reserved unless negotiated capability is present | defined only by a future compatible extension |
| 20 | `POWER_RESIDENCY_STATUS` | `device_id=bar=arg0=arg1=arg2=0` | UTF-8 `tinygpu.power-residency.v2` JSON |

Clients send exactly one `HANDSHAKE` per connection before any new command.
A server accepts a new command only after that successful handshake; a second
handshake is `INVALID_STATE`. Major-version
mismatch is `unsupported_version`; unavailable required capability is
`unsupported_capability`. `HANDSHAKE` has no hardware, lease, BAR, DMA,
configuration, reset, or power side effect.

On a negotiated v1 connection, legacy hardware commands require an active
workload lease except `RESET`. `RESET` is accepted only before lease
acquisition, only as an explicit operator action, and only while the provider
reports zero workload leases. It is never sent automatically after a keeper
failure.

The handshake JSON object has these exact keys and types:

```json
{"schema":"tinygpu.handshake.v1","protocol_major":1,"protocol_minor":0,"capabilities":11,"server_build_id":"ascii-build-id"}
```

`schema` is exactly `tinygpu.handshake.v1`; major/minor are unsigned 16-bit;
`capabilities` is an unsigned 64-bit bitset; and `server_build_id` is 1..128
ASCII bytes matching `[A-Za-z0-9._+-]+`. Capability bit 0 is
`KEEPALIVE_STATUS`, bit 1 is `WORKLOAD_LEASE`, and bit 2 is
`KEEPALIVE_SET_POLICY`. Bit 3 is `POWER_RESIDENCY_STATUS`. Unknown capability
bits are ignored unless required by the client; bits required by `arg2` must
all be present. v11 workload clients require bits 0, 1, and 3; bit 2 remains
reserved and unset.

## Legacy handshake probe

The v1 legacy probe is command 15 with `device_id=0`, `bar=0`, `arg0=1`,
`arg1=0`, and `arg2=0`, encoded exactly as in `fixtures/handshake-v1.json`.
Against an unmodified recovered legacy server, only either of these outcomes
maps to `unsupported_protocol`:

1. one complete 17-byte legacy response with `status=1`, `resp0=0`, and
   `resp1=0`; or
2. clean EOF before any response byte.

The client immediately closes after either outcome and sends no status or
policy command. A timeout, partial response, nonzero error length, malformed
response, or any other byte sequence is `protocol_error`, never absence of
capability. The probe timeout is 3000 ms and is not a keepalive claim.

## Typed local errors

`unsupported_protocol`, `unsupported_version`, `unsupported_capability`,
`malformed_header`, `malformed_payload`, `payload_too_large`,
`invalid_reserved_field`, `invalid_enum`, `invalid_range`, `partial_read`,
`partial_write`, `disconnect`, and `timeout` are distinct endpoint-local
errors. Legacy server error strings remain opaque bytes; they do not alter
these classifications.

New-server non-OK responses use UTF-8 JSON with no BOM or duplicate keys,
bounded to 1024 bytes, and this exact schema:

```json
{"schema":"tinygpu.error.v1","code":"unsupported_version","message":"protocol major 2 is unsupported"}
```

`schema` is exactly `tinygpu.error.v1`; `code` is exactly one of
`unsupported_version`, `unsupported_capability`, `malformed_request`,
`invalid_state`, `busy`, or `internal_error`; and `message` is 1..512 UTF-8
bytes. Status codes 2..7 respectively require the corresponding `code`.
`LEGACY_ERROR` is deliberately not typed. A typed error has `resp1=0` and
`resp0` equal to its encoded byte length, exactly like a success payload.

## DriverKit selector bridge

DriverKit selector IDs are local to the app-to-dext bridge, but frozen here to
prevent server drift. Every selector has scalar-only arguments unless stated
otherwise. Selectors 0..3 preserve the existing ABI:

| Selector | Name | Input | Output | Role |
|---:|---|---|---|---|
| 0 | `CFG_READ` | offset:u32, width:u32 | value:u32 | workload lease |
| 1 | `CFG_WRITE` | offset:u32, width:u32, value:u32 | none | workload lease |
| 2 | `RESET` | none | none | explicit diagnostic/operator action; provider must have zero workload leases |
| 3 | `PREPARE_DMA` | structure descriptors | structure output | workload lease |
| 4 | `HANDSHAKE` | client_major:u64, client_minor:u64, required_caps:u64 | protocol_major:u64, capabilities:u64 | diagnostic |
| 5 | `KEEPALIVE_STATUS` | none | fixed status JSON bytes | diagnostic |
| 6 | `LEASE_ACQUIRE` | none | lease_id:u64 | diagnostic to workload |
| 7 | `LEASE_RELEASE` | lease_id:u64 | none | workload |
| 8 | `MMIO_READ` | bar:u32, offset:u64, width:u32 | value:u32 | workload lease |
| 9 | `MMIO_WRITE` | bar:u32, offset:u64, width:u32, value:u32 | none | workload lease |
| 10 | `POWER_RESIDENCY_STATUS` | none | fixed power-residency JSON bytes | diagnostic |

BAR mapping and `CopyClientMemoryForType` require an active lease. Diagnostic
connections may call selectors 4, 5, and 10; selector 2 is additionally admitted
only as an explicit operator action while the provider has zero workload
leases. Width is one of 1, 2, or 4;
configuration offsets must be aligned to width and within PCI configuration
space. A provider serializes every selector that can touch PCI with the timer
and provider lifecycle gate.

The app's diagnostic `keepalive handshake` command converts selector 4's two
scalars into the wire handshake JSON. `keepalive status` and `power status`
return selectors 5 and 10 respectively. These commands do not open the Unix
socket server, acquire a lease, map a BAR, or allocate DMA/shared memory.

## Keepalive status

`KEEPALIVE_STATUS` returns UTF-8 JSON with no BOM, no duplicate keys, and no
unknown required fields. The encoded payload is <= 4096 bytes. Whitespace and
key ordering are not semantically significant. The exact schema string is
`tinygpu.keepalive.v1`; required keys and JSON types are:

| Key | Type/range |
|---|---|
| `schema`, `state`, `policy_id`, `expected_identity`, `last_identity_dword` | string |
| `provider_generation`, `attempts`, `successes`, `failures`, `consecutive_failures`, `last_attempt_monotonic_ns`, `last_success_monotonic_ns`, `success_gap_over_leeway_count`, `max_success_gap_ms` | unsigned 64-bit JSON integer |
| `interval_ms`, `maximum_timer_leeway_ms`, `active_workload_leases`, `active_bar_mappings`, `active_dma_allocations` | unsigned 32-bit JSON integer |
| `timer_error` | signed 32-bit JSON integer |
| `enabled`, `counter_saturated` | boolean |

States are exactly `unsupported`, `inactive`, `active_healthy`,
`active_degraded`, `quiescing`, or `stopped`. `expected_identity` is
`1002:744c`; `last_identity_dword` is lower-case `0x` plus eight lower-case
hex digits. The admitted policy is `usb4_amd_744c_v1`, interval 1000 ms, and
maximum timer leeway 100 ms. Counters saturate instead of wrapping, and
`attempts == successes + failures`. `timer_error=0` means no timer error.

The canonical valid examples are in `fixtures/keepalive-status-v1.json`.

## Power-residency status

`POWER_RESIDENCY_STATUS` is a separately negotiated diagnostic payload; it does
not extend or reinterpret `tinygpu.keepalive.v1`. The encoded payload is <=4096
bytes and has the exact schema `tinygpu.power-residency.v2` with these fields:

| Key | Type/range |
|---|---|
| `schema`, `policy_id`, `last_canary_identity_dword` | string |
| `provider_generation`, `power_request_attempts`, `last_power_request_monotonic_ns`, `transition_count`, `unexpected_downgrade_count`, `last_transition_monotonic_ns`, `last_canary_success_monotonic_ns` | unsigned 64-bit JSON integer |
| `desired_power_flags`, `last_observed_power_flags`, `stop_busy_leases`, `stop_busy_bars`, `stop_busy_dma` | unsigned 32-bit JSON integer |
| `override_probe_prejoin_error`, `override_probe_postjoin_error`, `power_request_error`, `power_release_error` | signed 32-bit JSON integer |
| `full_power_requested`, `power_request_accepted`, `power_request_confirmed`, `power_release_attempted`, `publishable` | boolean |

The v11 policy is exactly `driverkit_full_power_v1`. Its desired and healthy
observed flags are `kIOServicePowerCapabilityOn` (`2`). A healthy active payload
has the expected failing pre-join probe and successful post-join probe, at least
one accepted full-power request, an On notification after that request, a later
successful identity canary, zero request/release errors, zero unexpected
downgrades, zero recorded Stop resource counts, and `publishable=true`.
`power_request_accepted` records only that the API accepted the argument; it is
not confirmation. The payload proves only DriverKit request and callback state;
physical tunnel continuity remains an A0/A1 hardware requirement.

The canonical valid and invalid examples are in
`fixtures/power-residency-status-v2.json`.
