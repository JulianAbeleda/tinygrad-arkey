# Q4-V raw launch debug (2026-08-29)

- Serialized asset: `/tmp/q4v-asset/program.cubin`; manifest ABI `(grid=(32,8,1), block=(32,2,2), globals=(0,1,2), outs=(0,), ins=(1,2))`.
- Raw serialized V main launched directly through `native_nv_program` + `call_native` with output `(512*1024,f32)`, record `RECORD_U32` (`uint32`), packed Q4-K words (`uint32`), zero-filled inputs. It completed and returned all-zero output.
- The same producer cubin/source launched directly via `NVProgram` with `(grid=(512,8,1), block=(128,1,1))` and buffers `(fp16[512,4096], uint32[RECORD_U32])`; it completed in ~4 microseconds.
- The producer launched through `native_nv_program` + `call_native` hangs before completion, with either argument ordering and with `outs=(1,),ins=(0,)` or both inputs.

## Current conclusion

The serialized Q4-V main ABI itself is valid below the model boundary. The blocker is the native UOp/call path for the Q8 producer, not the V cubin or its main launch metadata. Keep the asset unchanged. Next isolate `call_native`/`run_linear` handling for this producer (or use the proven `NVProgram` producer wrapper) before testing producer→main.
