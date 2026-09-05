# Focused kernel correctness investigations

These standalone harnesses call the production kernels. The workflow builds a baseline
and a causal control using isolated header copies with the accompanying patches.
The source hypotheses are not runtime-confirmed. A successful compilation or host layout
observation is not a GPU correctness result.

The audited baseline is `59e3a3338d516ca6ce0e073af8da65289678a35c`.
Build with CUDA13.1.1 from the repository root:

```sh
nvcc -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr -gencode=arch=compute_90a,code=sm_90a \
  -Iinclude -Itools/util/include test/regression/fp8_scale_wave.cu -o fp8_scale_wave
nvcc -std=c++17 -O3 -lineinfo --expt-relaxed-constexpr -gencode=arch=compute_100a,code=sm_100a \
  -Iinclude -Itools/util/include -Iexamples/93_blackwell_low_latency_gqa \
  test/regression/gqa_max_scratch.cu -o gqa_max_scratch
```

The fork workflow compiles both variants, preserves PTX/SASS, and runs only the host
layout observation. Its standard runner has no GPU. The container image and Actions
revisions are pinned in `.github/workflows/kernel-correctness.yml`.
Compressed binaries, patches, checksums and generated code are retained in the job log.

## FP8 scale lifetime

```sh
./fp8_scale_wave --layouts  # CPU layout observation; no CUDA kernel executes
./fp8_scale_wave            # Minimal numerical case; requires H100/H200
./fp8_scale_wave --all      # Opposite branches and tile/stage/addressing boundaries
```

The minimal GEMM uses CTA256x128x128, cluster1x1x1, automatic stage selection, FP8 E4M3 A/B,
FP32 scales/output, scale granularity128x64x128, and MN-major scales on both sides.
A=B=1, A scales are2/4 across the two row bands, B scales are1/2 across the two column
bands. The exact expected 128x64 blocks are `[[256,512],[512,1024]]`.
The hypothesis predicts `[[256,512],[1024,2048]]` because the B register views alias
across M waves. The executable reports actual mismatches without accepting this
prediction as the expected answer.

`--all` checks one M wave, one B scale, multiple A scales per wave, unit A scales,
multiple K tiles, stage wrap, K tail, automatic stages with two carveouts, partial M/N, padded leading
strides, batching, and alpha/beta epilogue behavior. It compares every logical output
and allocated D row-padding element with an independent integer block-product oracle.
It does not yet cover other scale majors, clustering, pointer-array dispatch, or FP8 formats.

The control removes the in-place B-scale multiplication in both the steady-state and
drain paths and calls the existing two-scale accumulation overload. It preserves the
broadcast B register value across M waves. This isolates the suspected dependency;
its wider numerical and performance suitability still requires separate review.

## GQA maximum scratch lifetime

```sh
./gqa_max_scratch                         # One CTA per cluster, two KV tiles
./gqa_max_scratch --all --repetitions 100  # Boundary controls; requires B200
compute-sanitizer --tool racecheck --error-exitcode 3 ./gqa_max_scratch --repetitions 1
```

The default case has64 Q heads,8 KV heads, q length1, head dimension64, BF16 Q/K/V/O,
FP32 accumulation, KV length256, tile128, three DMA stages, and one split/reduction CTA.
Q has a single1 per head; K produces alternating base2 logits0/1 by 128-token tile.
V varies across KV tiles, heads and features in exactly representable quarter steps.
A separate CPU scalar softmax computes the expected output in double precision and
rounds to BF16. The absolute comparison tolerance is1/256, one BF16 step at0.5.
The `--all` sweep retains V=1 as a separate exact normalization control.
Sinks, sliding windows, and PDL are disabled. A single CTA per cluster removes a possible
cross-CTA initialization confounder.

Every launch poisons O, executes the original wrapper, waits for completion, and checks
every output. Default repetition count is100; a case stops on its first discrepancy.
The sweep includes the shipped eight-split KV2048 path, one-tile controls, equal logits,
stage wrap, partial tail, two/three stages, and head dimension128. It does not yet cover
paged KV, sliding windows, sinks, or other layouts.

An ordinary passing run does not disprove a race. Record hardware/driver/toolkit, the
exact revision, first failing iteration, coordinates, actual/expected values, and
sanitizer findings. Confirmation must distinguish the internal maximum-scratch hazard
from other ordering failures. Compare the original with a narrowly placed read-completion
barrier, then revert it; delay-instrumented-only failures are diagnostic evidence.
The supplied control patch adds only that read-completion barrier at the end of
`cta_reduce`, after every participating warp has consumed the shared scratch.

Both executables return0 only when all requested numerical checks pass. Numerical
mismatch returns1; invalid invocation, unsupported hardware, and execution errors fail with a nonzero exit.
`--layouts` is a separate host-only diagnostic. No benchmark or full-library build is needed.
