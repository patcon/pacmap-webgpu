# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
npm install
npm run dev      # vite dev server
npm run build    # tsc --noEmit && vite build
npm run preview
npm run check:shaders   # compile every WGSL source under Dawn
```

There is no test suite and no linter. `npm run build` (i.e. `tsc --noEmit`) and `npm run check:shaders` are the two automated checks; CI runs both. Strict mode is on; `@webgpu/types` is loaded globally via `tsconfig.json` `types`, and `scripts/` is type-checked alongside `src/`.

Running the app requires a WebGPU-capable browser (Chrome/Edge 113+, Firefox 141+ on Windows / 145+ on Apple silicon, Safari 26). If the page reports no adapter, check `navigator.gpu` in the console. Nothing can be verified headlessly — the whole pipeline runs in the browser.

## Architecture

Single-page Vite demo: MNIST → randomized PCA → PaCMAP (or LocalMAP) dimensionality reduction → animated scatter plot. The organizing constraint is **once setup finishes, the 2D positions never leave GPU memory**.

Pipeline, in order (`src/main.ts` `go()` wires it all):

1. `src/mnist.ts` — fetches the TF.js-hosted 784×65000 sprite PNG and decodes it through a canvas in 2500-row chunks (the full decode is ~200MB of RGBA transiently). Labels are stored one-hot, 10 uint8 per example.
2. `src/pca.ts` — `pcaProject` reduces 784→100 on the CPU via a randomized range finder (no SVD). Output **spans** the top-k principal subspace but is not rotated to the principal axes, so per-axis variance ordering is meaningless. That's acceptable because the only consumer is a distance computation. An SVD of `B` would be needed to get true axis-ordered components (e.g. for PC1/PC2 init).
3. `src/pacmap-webgpu.ts` — the library. No DOM dependencies, reusable.
4. `src/main.ts` — demo wiring: bounds-reduce compute pass, instanced point renderer, playback transport, Tweakpane view controls, DOM/status.

The pane has two folders with different lifetimes. `view` is live — an edit rewrites the current run's uniform. `algorithm` (variant, low_dist_thres, kNN backend, n_neighbors, MN/FP pair ratios, seed) is setup-time: the kNN graph is built and pairs sampled before the first iteration, so those values are read at the top of `go()` and the folder is disabled while a run is in flight. The `algorithm` and `kNN algo` dropdowns are seeded from `?algo=` and `?knn=` and own their values after that. `nNeighbors` is sent as `undefined` while "auto neighbors" is on, which is what lets the library's own default apply; the slider mirrors `defaultNeighbors(N)` so the value is visible rather than implicit.

The `algorithm` options map is annotated `Record<string, Variant>` rather than inferred. Tweakpane types `options` as plain strings, so a typo in a *value* type-checks, and the library's `opts.variant ?? "pacmap"` would then quietly run PaCMAP while the pane claimed LocalMAP. The annotation is what makes that a build error. The `kNN algo` dropdown still has that hole.

The transport bar is always present. It never hides — it goes inert (controls disabled, readout `—`) so a run doesn't reflow the canvas.

Tweakpane (the only runtime dependency) is mounted into `#pane`, an absolutely-positioned child of `#stage`, rather than its default fixed top-right, which would land under the header. Its bindings mutate the module-level `view` object; a run installs `onViewChange` so an edit rewrites *that* run's view uniform immediately, including mid-run when nothing else is touching it. Point size is a CSS-px radius, scaled by dpr and floored at 1.5 framebuffer px.

### Playback history (`main.ts`)

The timeline scrubber replays banked frames, so it holds the same invariant as the live path: nothing is read back. During the run each captured iteration is copied into a slot of one big `posHistory` buffer (`VERTEX | COPY_DST`), and the bounds reduce for that iteration is copied into `boundsHistory` alongside it — replay is then two `copyBufferToBuffer` calls and a render pass, with no compute and no `mapAsync`. Banking the bounds rather than recomputing them on scrub is what makes a replayed frame framed exactly as it was live.

A frame costs `N*8` bytes, so `HISTORY_BUDGET_BYTES` (128MB), not the iteration count, decides how many fit; above the budget the trace is captured every `stride` iterations rather than truncated. That is also why the run loop steps in `stride`-sized chunks and only presents every `stepsPerFrame`-worth of them.

### The CPU/GPU split in `pacmap-webgpu.ts`

CPU, once at setup: sigma scaling, pair sampling (near / mid-near / further), CSR build, and the initial further-pair forward/reverse arrays (`buildFpReverse`).
GPU: brute-force kNN (`knnGPU`, one thread per query, bounded insertion sort in registers), then 450 iterations of gradient accumulation + Adam, fully resident — plus, under `variant: "localmap"`, the further-pair redraw and reverse-CSR rebuild, in that same pass.

`bruteForceKnn` is the CPU reference for the same computation, kept as a correctness oracle and selectable via the `knn: "cpu"` option. The two are not bit-identical (JS accumulates in f64, WGSL in f32), so near-ties swap order; recall, not exact-order agreement, is the metric.

There is a third backend, `nndescentGPU` (`knn: "nndescent"`), approximate rather than exact. It is **slower than both brute-force paths at every size this demo reaches** — it exists for the asymptotics (past ~100k nothing else is viable) and as something to compare against, not as an optimization. Measured recall is ~99.9% at N=2000–5000 with k=60. Two things about it are worth knowing before touching it:

- It is the one path that is **not deterministic**. The reverse-neighbor list is capped at `revCap` entries and filled first-writer-wins, which is a race; repeated runs at a fixed seed agree to about 1e-5 of recall. The cap is what bounds worst-case work — MNIST has hub points that thousands of rows point at, and an uncapped reverse list would hand one thread all of them.
- Its join kernel reads rows other threads are concurrently writing. That is safe *by construction*, not by luck: only indices are read across rows, never distances, so a torn read yields a stale or duplicate candidate index (which the dedup scan handles) and never a mismatched index/distance pair. Don't "fix" it by double-buffering.

`?knncheck=1` runs every backend over one identical input, scoring each against the CPU oracle, and reports recall / exact-order / max rel Δd² / ms. `?knn=gpu|nnd` picks a backend, as does the pane's `kNN algo` dropdown. The demo defaults to the CPU oracle: it is exact, and measured, it is also the fastest of the three at the sizes this demo runs — which is not what its O(N²·D) in plain JS would suggest. Treat any speed claim about these backends as something to re-measure with `?knncheck=1` rather than derive.

Note that the demo cannot be verified headlessly, but the *library* can: bundling with esbuild (`--format=esm --platform=neutral`) produces something that runs under `@kmamal/gpu` (Dawn bindings for Node), which compiles the real WGSL and executes the kernels. Both kNN shaders shipped broken once because they were committed without ever being compiled — a WGSL parse error takes out every pipeline at once and the failure looks like a plausible blob rather than an error.

That mechanism goes further than compilation, and the LocalMAP work leaned on it heavily: the same bundle runs a full `pacmapWebGPU` end to end, so a change that is supposed to preserve behavior can be **measured** against a bundle of the previous commit rather than eyeballed. Moving the further pairs out of the CSR was verified that way (max|Δ| / layout extent = 2.9e-5, against 0.50 for a deliberately broken version), as was LocalMAP's bit-exact reproducibility across processes. Three things to know if you rebuild that harness: `@kmamal/gpu` exposes `GPUBufferUsage`/`GPUShaderStage`/`GPUMapMode` as module properties rather than globals, it holds an interval per instance so the process needs an explicit `process.exit`, and `$?` after a pipe is the *last* command's status — which will happily hide a timeout. Always include a no-change control run; a confounded comparison otherwise looks like a passing one.

`npm run check:shaders` is that mechanism, standing: `scripts/check-shaders.ts` compiles all six WGSL sources under Dawn and builds the real pipeline for every entry point. Pipelines matter as much as modules — a renamed entry point, an incompatible bind-group layout, or a bad vertex/blend state compiles clean and fails only at `createPipeline`. This is why `src/shaders.ts` exists (`main.ts` touches the DOM at module scope, so its shaders had to move somewhere importable) and why `pacmap-webgpu.ts` exports `shaderSources`. Add a case there whenever you add a shader.

Three design decisions carry most of the file:

- **Each pair is duplicated into both endpoints' lists.** This lets the gradient kernel gather instead of scatter, which is what makes a float-atomic-free WGSL shader possible. The sign works out because the displacement flips on the reverse entry while the scalar coefficient (a function of |d|² only) does not. Near and mid-near pairs get this via the CSR (`buildCSR`); further pairs get it via a separate pair of arrays, because LocalMAP rebuilds them mid-run — see below.
- **Pair type is packed into the top 2 bits of each adjacency entry** (`T_NB`/`T_MN`, mask `0x3FFFFFFF`) so one `u32` array carries both neighbor and kind. Only two kinds are packed now; `T_FP` is gone with the further pairs.
- **Per-iteration weights live in one uniform buffer, 256-byte-aligned slot per iteration**, addressed by dynamic offset. This is what allows the entire loop to be encoded into a single command buffer with no host round-trip. `weightsAt()` implements the 3-phase schedule (w_MN 1000→3, then local refine, then attract-repel). The slot is 48 bytes: weights + lr, the two Adam bias corrections, LocalMAP's `(nnA, nnB)` pair, and the iteration index.

### LocalMAP (`variant: "localmap"`)

LocalMAP (Wang et al., AAAI 2025) ships in the same upstream repo as the PaCMAP reference (`source/pacmap/pacmap.py`, `localmap()` around line 1542). Upstream models it as a subclass reusing `PaCMAP.fit`, so it is a variant here rather than a second entry point; `localmapWebGPU()` is a one-line wrapper. Everything before the optimizer — kNN, sigma scaling, pair sampling, the weight schedule — is shared and unchanged. It differs only inside phase 3, in two ways.

**The near-pair coefficient gains a `(low_dist_thres/2)/√(1+|d|²)` factor** — i.e. `inverseSqrt(dd)`, where `dd` is the same `1 + |d|²` the rest of the gradient uses, so it stays finite at zero separation. Upstream branches to a whole second gradient kernel for this. Here it rides as an affine `(nnA, nnB)` pair in the params slot: `c = w_NB · 20/(10+dd)² · (nnA + nnB·inverseSqrt(dd))`, with `(1, 0)` for PaCMAP and for LocalMAP up to `n1+n2`, and `(0, low_dist_thres/2)` after. `(1, 0)` reproduces the PaCMAP coefficient *exactly* — `dd ≥ 1` always, so the added term is exactly zero and the multiply by 1.0 is exact in IEEE754 — which is what lets both variants share one code path with no branch and no second entry point.

**The further pairs are redrawn against the embedding.** This is the "locally adjusted graph" the name refers to, and it is the reason further pairs had to leave the CSR: the pair set changes 24 times mid-run (iterations 210, 220 … 440 at the default phases — note the guard is a strict `it > n1+n2`, so not 200). Since the draw reads `Y`, it has to run on the GPU, and so does the rebuild of the structure the gradient gathers from. `fpShaderSource` is that chain, six entry points appended to the same compute pass as `grad`/`adam`:

`fp_resample` redraws each point's `n_FP` partners, rejecting self, a partner already drawn this round, the point's own near partners (`nbFwd`, hence that array), and anything past `low_dist_thres` in the embedding. Then `fp_clear → fp_count → fp_scan → fp_scatter → fp_sort` rebuilds the reverse CSR around the new forward set. `buildFpReverse` is the CPU equivalent that seeds it at setup and stays the oracle the kernels were checked against.

Four things about it are worth knowing before touching it:

- **`runRange` still submits one command buffer and reads nothing back.** That was the whole point of doing the rebuild on the GPU. Dispatches inside a pass are synchronized by WebGPU, so the chain needs no explicit barrier between stages any more than `grad → adam` does.
- **The scan is a single serial invocation.** N is at most a few hundred thousand and it runs two dozen times a run, so one thread walking N counters is a fraction of a millisecond. A two-level parallel scan would buy microseconds for a page of subtle WGSL — the same trade the demo's single-workgroup bounds reduce makes.
- **`fp_sort` is load-bearing, not tidiness.** `fp_scatter` claims slots with an atomic cursor, so which writer lands where is a race; sorting each reverse list afterwards is what keeps the gradient's f32 summation order reproducible. Measured: skipping it puts ~75% of the index region in a different order. With it, two LocalMAP runs at a fixed seed agree bit-for-bit — unlike `nndescentGPU`, this path is fully reproducible, and that is deliberate.
- **The draw budget bounds every rejection path, which upstream's does not.** In the reference, the self-hit and distance-failure branches `continue` past the `if count > 100` escape, so a point with no eligible partner inside `low_dist_thres` never terminates. On a CPU that is a rarely-hit hang; on a GPU it takes out the device. The escape is plainly the intent, so it applies to all four rejection paths here.

Under `variant: "pacmap"` the whole `fp` object is `null`: no extra buffers, no kernels encoded. The PaCMAP path is untouched rather than merely equivalent.

The gradient shader sits at **8 storage buffers**, the default `maxStorageBuffersPerShaderStage`, with no headroom. If a ninth is ever needed, interleave `M` and `V` — same length, same access pattern, and `adam_main` already walks both in lockstep. The `check-shaders.ts` layout is what will catch the overflow.

`PacmapRun.positions` is created with `VERTEX` usage and bound directly as a vertex buffer by the renderer — no `mapAsync`, no pipeline stall, so per-iteration animation costs nothing extra. `read()` exists but is only for host readback, not the render path. Same reason the bounds/autoscale reduce in `main.ts` is a single-workgroup GPU pass: the render loop must never read positions back.

Known deviations from the reference PaCMAP, deliberate and documented at their sites: Gaussian init rather than PCA init (so layouts vary by `seed` — change it in the `pacmapWebGPU` call), and no PCA-to-100d inside the library (the demo does it upstream). A third applies only under `knn: "nndescent"`: that path is not reproducible at a fixed seed (see above). A fourth applies only under `variant: "localmap"`: the resample's draw budget bounds every rejection path, where upstream's bounds only two and can therefore not terminate (see the LocalMAP section). Note the exact kNN is itself a deviation in the other direction — upstream uses ANNOY, so it is approximate too.

One more, easy to miss: omitting `nNeighbors` falls back to `defaultNeighbors(N)`, the `10 + 15*(log10(N) - 4)` rule. Upstream that rule is opt-in — `PaCMAP.__init__` defaults `n_neighbors=10` and only consults the rule when the caller passes `None` explicitly — so the library's implicit default is *not* the reference's. The demo passes 10 unless "auto neighbors" is ticked.

### Performance shape

The optimizer has never been the bottleneck — all 450 GPU iterations stay well under a second. With kNN moved to the GPU, the remaining cost is the CPU PCA (~4·n·d·k MACs of plain JS, tens of seconds at 65k); porting those matmuls to WGSL is the next real win.

Above ~100k points O(N²) kNN stops being viable at any throughput, which is what `nndescentGPU` is for. It does not help below that: it measures several times slower than the brute-force kernel at N=2000–5000, narrowing as N grows but not crossing over anywhere near 65k. The reason is memory, not arithmetic — see its docstring. Making it competitive would mean attacking the scattered candidate reads (a workgroup per point with a cooperative distance reduction, rather than a thread per point) and adding the new/old flag sampling this variant drops.

Long CPU loops in `pca.ts` yield to the event loop on a time slice (`Pacer`, `MessageChannel` rather than `setTimeout(0)` to dodge the 4ms nested clamp) so the page keeps painting progress. This is why `pcaProject` is `async`.
