import pathlib

src = pathlib.Path('docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_guarded_k64.cu').read_text()
start, end = src.index('    unsigned int val0 ='), src.index('    __syncthreads();', src.index('    unsigned int val0 ='))
body = src[start:end]
lines = body.splitlines(True)
groups = {
  'fragment_load_to_use_reorder': (lines[0:12], lines[12:]),
  'metadata_load_reorder': (lines[0:7], lines[7:12] + lines[12:]),
}
for name, (a, b) in groups.items():
  pathlib.Path(f'docs/task_workflow/evidence/nv-prefill-gateup-schedule-discriminator-20260829/{name}.cu').write_text(src[:start] + ''.join(b+a) + src[end:])
