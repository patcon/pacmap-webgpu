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
4. `src/main.ts` — demo wiring: bounds-reduce compute pass, instanced point renderer, playback transport, Tweakpane view controls, DOM/status.

The pane has two folders with different lifetimes. `view` is live — an edit rewrites the current run's uniform. `pacmap` (kNN backend, n_neighbors, MN/FP pair ratios, seed) is setup-time: the kNN graph is built and pairs sampled before the first iteration, so those values are read at the top of `go()` and the folder is disabled while a run is in flight. The `kNN algo` dropdown is seeded from `?knn=` and owns the value after that. `nNeighbors` is sent as `undefined` while "auto neighbors" is on, which is what lets the library's own default apply; the slider mirrors `defaultNeighbors(N)` so the value is visible rather than implicit.

The transport bar is always present. It never hides — it goes inert (controls disabled, readout `—`) so a run doesn't reflow the canvas.

Tweakpane (the only runtime dependency) is mounted into `#pane`, an absolutely-positioned child of `#stage`, rather than its default fixed top-right, which would land under the header. Its bindings mutate the module-level `view` object; a run installs `onViewChange` so an edit rewrites *that* run's view uniform immediately, including mid-run when nothing else is touching it. Point size is a CSS-px radius, scaled by dpr and floored at 1.5 framebuffer px.

### Playback history (`main.ts`)

The timeline scrubber replays banked frames, so it holds the same invariant as the live path: nothing is read back. During the run each captured iteration is copied into a slot of one big `posHistory` buffer (`VERTEX | COPY_DST`), and the bounds reduce for that iteration is copied into `boundsHistory` alongside it — replay is then two `copyBufferToBuffer` calls and a render pass, with no compute and no `mapAsync`. Banking the bounds rather than recomputing them on scrub is what makes a replayed frame framed exactly as it was live.

A frame costs `N*8` bytes, so `HISTORY_BUDGET_BYTES` (128MB), not the iteration count, decides how many fit; above the budget the trace is captured every `stride` iterations rather than truncated. That is also why the run loop steps in `stride`-sized chunks and only presents every `stepsPerFrame`-worth of them.

### The CPU/GPU split in `pacmap-webgpu.ts`

CPU, once at setup: sigma scaling, pair sampling (near / mid-near / further), CSR build.
GPU: brute-force kNN (`knnGPU`, one thread per query, bounded insertion sort in registers), then 450 iterations of gradient accumulation + Adam, fully resident.

`bruteForceKnn` is the CPU reference for the same computation, kept as a correctness oracle and selectable via the `knn: "cpu"` option. The two are not bit-identical (JS accumulates in f64, WGSL in f32), so near-ties swap order; recall, not exact-order agreement, is the metric.

There is a third backend, `nndescentGPU` (`knn: "nndescent"`), approximate rather than exact. It is **slower than both brute-force paths at every size this demo reaches** — it exists for the asymptotics (past ~100k nothing else is viable) and as something to compare against, not as an optimization. Measured recall is ~99.9% at N=2000–5000 with k=60. Two things about it are worth knowing before touching it:

- It is the one path that is **not deterministic**. The reverse-neighbor list is capped at `revCap` entries and filled first-writer-wins, which is a race; repeated runs at a fixed seed agree to about 1e-5 of recall. The cap is what bounds worst-case work — MNIST has hub points that thousands of rows point at, and an uncapped reverse list would hand one thread all of them.
- Its join kernel reads rows other threads are concurrently writing. That is safe *by construction*, not by luck: only indices are read across rows, never distances, so a torn read yields a stale or duplicate candidate index (which the dedup scan handles) and never a mismatched index/distance pair. Don't "fix" it by double-buffering.

`?knncheck=1` runs every backend over one identical input, scoring each against the CPU oracle, and reports recall / exact-order / max rel Δd² / ms. `?knn=gpu|nnd` picks a backend, as does the pane's `kNN algo` dropdown. The demo defaults to the CPU oracle — exact and slow — so the GPU paths are opt-in even though they are the faster ones.

Note that the demo cannot be verified headlessly, but the *library* can: `npx esbuild src/pacmap-webgpu.ts --bundle --format=esm --platform=neutral` produces a bundle that runs under `@kmamal/gpu` (Dawn bindings for Node), which compiles the real WGSL and executes the kernels. Both kNN shaders shipped broken once because they were committed without ever being compiled — a WGSL parse error takes out every pipeline at once and the failure looks like a plausible blob rather than an error. Compile shaders before committing them.

Three design decisions carry most of the file:

- **Each pair is duplicated into both endpoints' CSR lists** (`buildCSR`). This lets the gradient kernel gather instead of scatter, which is what makes a float-atomic-free WGSL shader possible. The sign works out because the displacement flips on the reverse entry while the scalar coefficient (a function of |d|² only) does not.
- **Pair type is packed into the top 2 bits of each adjacency entry** (`T_NB`/`T_MN`/`T_FP`, mask `0x3FFFFFFF`) so one `u32` array carries both neighbor and kind.
- **Per-iteration weights live in one uniform buffer, 256-byte-aligned slot per iteration**, addressed by dynamic offset. This is what allows the entire loop to be encoded into a single command buffer with no host round-trip. `weightsAt()` implements the 3-phase schedule (w_MN 1000→3, then local refine, then attract-repel).

`PacmapRun.positions` is created with `VERTEX` usage and bound directly as a vertex buffer by the renderer — no `mapAsync`, no pipeline stall, so per-iteration animation costs nothing extra. `read()` exists but is only for host readback, not the render path. Same reason the bounds/autoscale reduce in `main.ts` is a single-workgroup GPU pass: the render loop must never read positions back.

Known deviations from the reference PaCMAP, deliberate and documented at their sites: Gaussian init rather than PCA init (so layouts vary by `seed` — change it in the `pacmapWebGPU` call), and no PCA-to-100d inside the library (the demo does it upstream). A third applies only under `knn: "nndescent"`: that path is not reproducible at a fixed seed (see above). Note the exact kNN is itself a deviation in the other direction — upstream uses ANNOY, so it is approximate too.

One more, easy to miss: omitting `nNeighbors` falls back to `defaultNeighbors(N)`, the `10 + 15*(log10(N) - 4)` rule. Upstream that rule is opt-in — `PaCMAP.__init__` defaults `n_neighbors=10` and only consults the rule when the caller passes `None` explicitly — so the library's implicit default is *not* the reference's. The demo passes 10 unless "auto neighbors" is ticked.

### Performance shape

The optimizer has never been the bottleneck — all 450 GPU iterations stay well under a second. With kNN moved to the GPU, the remaining cost is the CPU PCA (~4·n·d·k MACs of plain JS, tens of seconds at 65k); porting those matmuls to WGSL is the next real win.

Above ~100k points O(N²) kNN stops being viable at any throughput, which is what `nndescentGPU` is for. It does not help below that: it measures several times slower than the brute-force kernel at N=2000–5000, narrowing as N grows but not crossing over anywhere near 65k. The reason is memory, not arithmetic — see its docstring. Making it competitive would mean attacking the scattered candidate reads (a workgroup per point with a cooperative distance reduction, rather than a thread per point) and adding the new/old flag sampling this variant drops.

Long CPU loops in `pca.ts` yield to the event loop on a time slice (`Pacer`, `MessageChannel` rather than `setTimeout(0)` to dodge the 4ms nested clamp) so the page keeps painting progress. This is why `pcaProject` is `async`.
