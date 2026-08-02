# PaCMAP on WebGPU — MNIST demo

```
npm install
npm run dev
```

Open the URL Vite prints, pick a sample count, hit **Run**. Needs a WebGPU-capable
browser (Chrome/Edge 113+, Firefox 141+ on Windows / 145+ on Apple silicon,
Safari 26). Check `navigator.gpu` in the console if the page reports no adapter.

## What runs where

| Stage | Where | Notes |
|---|---|---|
| MNIST decode | CPU | TF.js sprite PNG, chunked through a canvas |
| PCA 784 → 100 | CPU | randomized range finder, `src/pca.ts` |
| kNN + pair sampling | CPU | brute force, the setup bottleneck |
| 450 optimizer iterations | **GPU** | gradient + Adam, no host round-trip |
| Bounds / autoscale | **GPU** | single-workgroup reduce |
| Rendering | **GPU** | `pm.positions` bound directly as a vertex buffer |

The point of the split: once setup is done, positions never leave GPU memory.
The render pass binds the same buffer the optimizer is writing, so per-iteration
animation costs nothing extra — no `mapAsync`, no pipeline stall. Set
**iters/frame** to 1 to watch the three phases separately; set it to 450 to
submit everything in one go and see the wall-clock number.

## Expected timings

Rough, on a mid-range discrete GPU:

- 3,000 points: PCA ~3s, kNN+pairs ~4s, optimization well under a second
- 10,000 points: setup climbs to ~60s, optimization still fast

The asymmetry is the honest result. Optimization is not the bottleneck at these
sizes; the O(N²·D) CPU kNN is. That's the piece to move next — either a tiled
WGSL distance kernel with a per-thread bounded heap, or NN-Descent past ~20k.

## Files

- `src/pacmap-webgpu.ts` — the library. No DOM dependencies, reusable.
- `src/mnist.ts` — sprite loader
- `src/pca.ts` — randomized PCA
- `src/main.ts` — demo wiring, bounds reduce, point renderer

## Caveats

- Gaussian init rather than PCA init, so layouts vary by seed. Change `seed` in
  the `pacmapWebGPU` call to reshuffle.
- The PCA output spans the top-100 principal subspace but isn't rotated to the
  principal axes. Distances are unaffected; per-axis variance ordering is not
  meaningful.
- Decoding the full sprite costs ~200MB of RAM transiently regardless of how
  many samples you keep.
