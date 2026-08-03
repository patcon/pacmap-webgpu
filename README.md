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
| kNN | **GPU** | brute force, one thread per query, `knnGPU` |
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

kNN used to dominate that: O(N²·D) on the CPU is ~60s at 10k and roughly 40
minutes at 65k. It now runs as a WGSL kernel, one thread per query with a
bounded insertion sort in registers, which is what makes the top of the slider
reachable at all. What's left is the CPU PCA — ~4·n·d·k MACs of plain JS, tens
of seconds at 65k. Porting those matmuls to WGSL is the next real win. Past
~100k points O(N²) stops being viable at any throughput and NN-Descent would be
needed.

The CPU implementation is still there as the correctness oracle. Two URL
switches:

- `?knn=cpu` — force the CPU path
- `?knncheck=1` — run both over the same input and report recall, exact-order
  agreement, and the measured speedup on your machine

They aren't bit-identical (JS accumulates distances in f64, WGSL in f32), so
near-ties legitimately swap order. Recall is the metric that would catch a
broken kernel; exact-order agreement is reported only as colour.

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
- `pcaProject` is called with `inPlace: true`, which centers the MNIST array
  where it sits rather than allocating a second copy (204MB at 65k). It destroys
  its input; the demo never reads `X` again.
