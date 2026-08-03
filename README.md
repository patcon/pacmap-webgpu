# PaCMAP on WebGPU — MNIST demo

```
npm install
npm run dev
```

Open the URL Vite prints, drag the sample slider (up to the full 65,000), hit
**Run**. Needs a WebGPU-capable
browser (Chrome/Edge 113+, Firefox 141+ on Windows / 145+ on Apple silicon,
Safari 26). Check `navigator.gpu` in the console if the page reports no adapter.

## What runs where

| Stage | Where | Notes |
|---|---|---|
| MNIST decode | CPU | TF.js sprite PNG, chunked through a canvas |
| PCA 784 → 100 | CPU | randomized range finder, `src/pca.ts` — the setup bottleneck |
| kNN | CPU by default | the exact `bruteForceKnn` reference; a GPU brute-force kernel (`knnGPU`) and GPU NN-Descent are both selectable |
| Sigma scaling + pair sampling | CPU | near / mid-near / further, CSR build |
| 450 optimizer iterations | **GPU** | gradient + Adam, no host round-trip |
| Bounds / autoscale | **GPU** | single-workgroup reduce |
| Rendering | **GPU** | `pm.positions` bound directly as a vertex buffer |

The point of the split: once setup is done, positions never leave GPU memory.
The render pass binds the same buffer the optimizer is writing, so per-iteration
animation costs nothing extra — no `mapAsync`, no pipeline stall. Set
**iters/frame** to 1 to watch the three phases separately; set it to 450 to
submit everything in one go and see the wall-clock number.

## Where the time goes

Optimization has never been the bottleneck — all 450 iterations finish well
under a second at every size the slider offers. Setup is the whole cost, and the
slider shows a rough estimate of it before you commit to a run.

kNN dominates that on the default backend: O(N²·D) on the CPU is ~60s at 10k and
roughly 40 minutes at 65k. Switching the `kNN algo` dropdown to the GPU kernel — one
thread per query with a bounded insertion sort in registers — is what makes the
top of the slider reachable at all. Behind it sits the CPU PCA — ~4·n·d·k MACs
of plain JS, tens of seconds at 65k. Porting those matmuls to WGSL is the next
real win.

## Three kNN backends

Pick one from the `kNN algo` dropdown in the pane, or with `?knn=`:

| Backend | | Notes |
|---|---|---|
| brute force (CPU) | `?knn=cpu` | default; exact, the oracle — but ~40 minutes at 65k |
| brute force (GPU) | `?knn=gpu` | exact, one thread per query |
| NN-Descent (GPU) | `?knn=nnd` | approximate, ~99.9% recall |

**NN-Descent is slower than brute force here, and that is the honest result.**
Past ~100k points O(N²) stops being viable at any throughput and NN-Descent is
the only one of the three left standing — but 65k is not past 100k. It measures
several times slower than the brute-force kernel at N=2000–5000, narrowing as N
grows without coming close to crossing over. The cause is memory rather than
arithmetic: brute force has every thread in a workgroup reading the same
candidate row at the same instant, which broadcasts and caches, while NN-Descent
has each thread walking its own scattered candidate set. It's here for the
asymptotics and for comparison. Being approximate does put it closer to the
reference PaCMAP than the exact paths, which use ANNOY.

It's also the one backend that isn't reproducible run-to-run: its reverse-
neighbor lists are capped and filled first-writer-wins, which is a race. Fixed
seed, repeated runs, recall agrees to about 1e-5.

`?knncheck=1` runs every backend over one identical input and scores each
against the CPU oracle, reporting recall, exact-order agreement, max relative
Δd², and wall clock.

Nothing agrees bit-for-bit even among the exact backends (JS accumulates
distances in f64, WGSL in f32), so near-ties legitimately swap order. Recall is
the metric that would catch a broken kernel. Read the other two per backend: for
an exact backend a low exact-order score means a bug, while for NN-Descent both
are *expected* to move, and max rel Δd² becomes a measure of how much worse the
substituted neighbor is.

## Files

- `src/pacmap-webgpu.ts` — the library. No DOM dependencies, reusable.
- `src/mnist.ts` — sprite loader
- `src/pca.ts` — randomized PCA
- `src/main.ts` — demo wiring, bounds reduce, point renderer

## Caveats

- Gaussian init rather than PCA init, so layouts vary by seed. Change `seed` in
  the `pacmapWebGPU` call to reshuffle.
- Under `?knn=nnd` the seed doesn't fully determine the layout — see above.
- The PCA output spans the top-100 principal subspace but isn't rotated to the
  principal axes. Distances are unaffected; per-axis variance ordering is not
  meaningful.
- Decoding the full sprite costs ~200MB of RAM transiently regardless of how
  many samples you keep.
- `pcaProject` is called with `inPlace: true`, which centers the MNIST array
  where it sits rather than allocating a second copy (204MB at 65k). It destroys
  its input; the demo never reads `X` again.
