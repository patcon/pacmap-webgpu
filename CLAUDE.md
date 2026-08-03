# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
npm install
npm run dev      # vite dev server
npm run build    # tsc --noEmit && vite build
npm run preview
```

There is no test suite and no linter. `npm run build` (i.e. `tsc --noEmit`) is the only automated check. Strict mode is on; `@webgpu/types` is loaded globally via `tsconfig.json` `types`.

Running the app requires a WebGPU-capable browser (Chrome/Edge 113+, Firefox 141+ on Windows / 145+ on Apple silicon, Safari 26). If the page reports no adapter, check `navigator.gpu` in the console. Nothing can be verified headlessly — the whole pipeline runs in the browser.

## Architecture

Single-page Vite demo: MNIST → randomized PCA → PaCMAP dimensionality reduction → animated scatter plot. The organizing constraint is **once setup finishes, the 2D positions never leave GPU memory**.

Pipeline, in order (`src/main.ts` `go()` wires it all):

1. `src/mnist.ts` — fetches the TF.js-hosted 784×65000 sprite PNG and decodes it through a canvas in 2500-row chunks (the full decode is ~200MB of RGBA transiently). Labels are stored one-hot, 10 uint8 per example.
2. `src/pca.ts` — `pcaProject` reduces 784→100 on the CPU via a randomized range finder (no SVD). Output **spans** the top-k principal subspace but is not rotated to the principal axes, so per-axis variance ordering is meaningless. That's acceptable because the only consumer is a distance computation. An SVD of `B` would be needed to get true axis-ordered components (e.g. for PC1/PC2 init).
3. `src/pacmap-webgpu.ts` — the library. No DOM dependencies, reusable.
4. `src/main.ts` — demo wiring: bounds-reduce compute pass, instanced point renderer, playback transport, DOM/status.

### Playback history (`main.ts`)

The timeline scrubber replays banked frames, so it holds the same invariant as the live path: nothing is read back. During the run each captured iteration is copied into a slot of one big `posHistory` buffer (`VERTEX | COPY_DST`), and the bounds reduce for that iteration is copied into `boundsHistory` alongside it — replay is then two `copyBufferToBuffer` calls and a render pass, with no compute and no `mapAsync`. Banking the bounds rather than recomputing them on scrub is what makes a replayed frame framed exactly as it was live.

A frame costs `N*8` bytes, so `HISTORY_BUDGET_BYTES` (128MB), not the iteration count, decides how many fit; above the budget the trace is captured every `stride` iterations rather than truncated. That is also why the run loop steps in `stride`-sized chunks and only presents every `stepsPerFrame`-worth of them.

### The CPU/GPU split in `pacmap-webgpu.ts`

CPU, once at setup: sigma scaling, pair sampling (near / mid-near / further), CSR build.
GPU: brute-force kNN (`knnGPU`, one thread per query, bounded insertion sort in registers), then 450 iterations of gradient accumulation + Adam, fully resident.

`bruteForceKnn` is the CPU reference for the same computation, kept as a correctness oracle and selectable via the `knn: "cpu"` option. The two are not bit-identical (JS accumulates in f64, WGSL in f32), so near-ties swap order; recall, not exact-order agreement, is the metric. The demo's `?knncheck=1` runs both over the same input and reports it (`?knn=cpu` forces the CPU path).

Three design decisions carry most of the file:

- **Each pair is duplicated into both endpoints' CSR lists** (`buildCSR`). This lets the gradient kernel gather instead of scatter, which is what makes a float-atomic-free WGSL shader possible. The sign works out because the displacement flips on the reverse entry while the scalar coefficient (a function of |d|² only) does not.
- **Pair type is packed into the top 2 bits of each adjacency entry** (`T_NB`/`T_MN`/`T_FP`, mask `0x3FFFFFFF`) so one `u32` array carries both neighbor and kind.
- **Per-iteration weights live in one uniform buffer, 256-byte-aligned slot per iteration**, addressed by dynamic offset. This is what allows the entire loop to be encoded into a single command buffer with no host round-trip. `weightsAt()` implements the 3-phase schedule (w_MN 1000→3, then local refine, then attract-repel).

`PacmapRun.positions` is created with `VERTEX` usage and bound directly as a vertex buffer by the renderer — no `mapAsync`, no pipeline stall, so per-iteration animation costs nothing extra. `read()` exists but is only for host readback, not the render path. Same reason the bounds/autoscale reduce in `main.ts` is a single-workgroup GPU pass: the render loop must never read positions back.

Known deviations from the reference PaCMAP, deliberate and documented at their sites: Gaussian init rather than PCA init (so layouts vary by `seed` — change it in the `pacmapWebGPU` call), and no PCA-to-100d inside the library (the demo does it upstream).

### Performance shape

The optimizer has never been the bottleneck — all 450 GPU iterations stay well under a second. With kNN moved to the GPU, the remaining cost is the CPU PCA (~4·n·d·k MACs of plain JS, tens of seconds at 65k); porting those matmuls to WGSL is the next real win. Above ~100k points O(N²) kNN stops being viable at any throughput and NN-Descent would be needed.

Long CPU loops in `pca.ts` yield to the event loop on a time slice (`Pacer`, `MessageChannel` rather than `setTimeout(0)` to dodge the 4ms nested clamp) so the page keeps painting progress. This is why `pcaProject` is `async`.
