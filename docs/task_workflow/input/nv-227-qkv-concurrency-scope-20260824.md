# NV 227 push: Q versus K/V producer concurrency scope

Date: 2026-08-24

## Question

Can the independent Q and K/V projection branches shorten their complete
producer-to-join span under production-cold weight streaming, and does any
unrealized overlap remain beyond the installed two-GPFIFO ready placement?

The installed checkpoint is `4515.395719 us/token = 221.464532 tok/s`. The
227 target is `4405.286344 us/token`, leaving `110.109375 us/token`.

## Gates

1. Render the installed shared-Q8 RMSNorm/Q8 provider, Q projection, and Q4/Q4
   paired K/V producer into a CUDA-stream fork/join microgate. Evict at least
   the device L2 before every timed span and require bit-exact Q/K/V outputs.
2. Repeat with exact cubins, live production buffers and weight VAs, and two
   native HCQ GPFIFOs. Use unique QMD/kernarg state for every sample.
3. Audit the production graph placement. If the current scheduler does not
   already express the fork, test a harness-only placement override before
   editing runtime code.
4. If placement is already identical, test the one physical difference left
   by the positive microgate: auxiliary-queue-first replay submission.
5. Book no isolated recovery. Promotion requires identical token streams and
   a fresh-process A/B/A candidate below both controls.

All production renderer, scheduler, runtime, and route files remain unchanged
until the wall gate passes.
