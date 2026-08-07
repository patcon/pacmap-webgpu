# PaCMAP and LocalMAP on WebGPU — MNIST demo

https://github.com/user-attachments/assets/109fd4bd-dd54-4e50-b2e8-6f7993794ac1

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
| Local graph adjustment | **GPU** | LocalMAP only — redraw + reverse-CSR rebuild, same pass |
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

The bulk of it is the CPU PCA — ~4·n·d·k MACs of plain JS, tens of seconds at
65k. Porting those matmuls to WGSL is the next real win.

LocalMAP's graph adjustment is the one cost here that has **not** been measured
on real hardware. It adds six dispatches at 24 of the 450 iterations, and the
reverse-CSR rebuild is linear in the pair count, so there is no reason to expect
it to matter next to setup — but that is a prediction, not a measurement, and
this file's whole point is that those come apart. Watch the ms/iter figure in
the status line.

kNN is the other half, and which backend is fastest is not what the asymptotics
predict — the CPU brute force measures fastest in practice, which is why it is
the default. Don't reason about the three from their complexity; measure with
`?knncheck=1`.

## Two or three dimensions

`components` in the pane, or `?dims=3`. Default is 2. It is setup-time, like
everything else in that folder: the buffers, the generated shaders and the
vertex layout are all sized from it before the first iteration, so the folder
is disabled while a run is in flight.

Both engines honour it — ours as `nComponents`, druid as its own `n_components`
— and the 2D path is bit-identical to what predates the switch.

In 3D, **left-drag orbits and right-drag pans** (2D keeps left-drag panning,
since there is nothing to rotate). The `occlusion` checkbox in the `view` folder
decides how depth is painted:

- **On** (default): points are opaque discs drawn against a depth buffer, so
  near clusters genuinely hide far ones. No sorting and no readback — which
  matters, because the usual answer, sorting an index buffer on the CPU each
  frame, is exactly the per-frame readback this renderer is built to avoid.
- **Off**: everything blends in buffer order, so density reads as opacity and
  depth comes only from parallax as you drag.

The checkbox is greyed in 2D, where the points are coplanar and a depth buffer
would resolve ties by whatever order the buffer happens to be in.

A 3D frame is 50% larger, so at the same 128MB history budget a 3D run banks
fewer frames — or captures every *n*th iteration instead. The status line says
which.

## Seeing the pairs

PaCMAP's whole mechanism is three sets of pairs pulling and pushing on each
other, and the animation shows only the result. Tick `show edges` in the
`pair graph` folder to draw the pairs themselves:

| | colour | per point | what it does |
|---|---|---|---|
| neighbors | green | `n_neighbors` — 10 by default | pull, throughout |
| mid-near | yellow | `round(n_neighbors * MN_ratio)` — 5 | pull, but only for the first 200 of 450 iterations |
| far | red | `round(n_neighbors * FP_ratio)` — 20 | push, throughout |

Every pair drawn is a pair that acts: the optimizer walks all three sets on
every iteration and never subsamples them. The selection happens once, when the
pairs are drawn at setup — mid-near, for instance, samples six candidates and
keeps the second-closest, and the five losers are never stored.

Two things are easy to misread:

- **Those counts are what a point *draws*, not its degree.** Each pair acts on
  both of its endpoints, so a point is also pulled by everyone who picked *it*.
  At 100% you will see roughly twice the number above touching any given point.
  That is why `% far` defaults to 5: 20 drawn each, doubled, is fog.
- **Yellow goes inert.** `w_MN` is 0 for the whole of phase 3, so for the last
  250 iterations the mid-near pairs still exist and still draw but exert no
  force at all. Nothing on screen distinguishes those two states — edge opacity
  is deliberately not tied to weight.

The three percentages are **render-time**. The index buffer holds every pair, so
a slider moves a draw count: no buffer is rewritten and no run restarts, and
because each set is shuffled once at setup, a percentage is a uniform sample
spread through the cloud rather than a prefix of the sample order.

Two limitations, both visible rather than subtle:

- Under `?algo=localmap` the red edges are the further pairs **as first drawn**.
  LocalMAP redraws them against the embedding 24 times during phase 3, on the
  GPU, and those redraws never come back to the host — so the red edges sit
  still while the points move. The folder title says so when LocalMAP is
  selected.
- The overlay needs the pair graph, and DruidJS builds its own neighbour graph
  internally without exposing it. The folder is greyed under both CPU engines.

## Two algorithms

Pick one from the `algorithm` dropdown, or with `?algo=`:

| Variant | | |
|---|---|---|
| PaCMAP | `?algo=pacmap` | default |
| LocalMAP | `?algo=localmap` | adds the locally adjusted graph |

[LocalMAP](https://doi.org/10.1609/aaai.v39i20.35436) is the same authors'
successor to PaCMAP, and against it the delta is small and confined to phase 3.
Two changes:

**The near-pair attraction is scaled by `(low_dist_thres/2)/√(1+|d|²)`**, so it
falls off with distance instead of staying flat. Points that are already
together pull harder. (The `1+` is why it stays finite at zero separation.)

**The further pairs are redrawn every ten iterations against the *embedding***,
keeping only partners that are already close in 2D. This is the "locally
adjusted graph" the name refers to: repulsion stops being a global scatter and
starts pushing apart specifically the clusters that are currently adjacent,
which is what sharpens the gaps between them. `low_dist_thres` (default 10, and
a slider in the pane) is the radius that counts as "already close" — it sets
both effects, so it is one knob and not two.

Everything else is shared: same kNN, same pair sampling, same weight schedule,
same 450 iterations — so the setup stage, which is where all the time goes, does
the same work either way. The run itself does the extra redraw and rebuild 24
times, at iterations 210, 220 … 440. What that costs on real hardware has not
been measured; see below.

The redraw reads the embedding, so it runs on the GPU, and so does the rebuild
of the structure the gradient reads. Nothing is read back — a LocalMAP run holds
the same invariant as a PaCMAP one, and the timeline scrubber works the same way.

Unlike upstream, a fixed seed fully determines a LocalMAP layout here: the draw
uses a counter-based hash rather than a global RNG, and the rebuilt adjacency is
sorted so the gradient sums in a reproducible order. Two runs at the same seed
agree bit-for-bit.

## Three kNN backends

Pick one from the `kNN algo` dropdown in the pane, or with `?knn=`:

| Backend | | Notes |
|---|---|---|
| brute force (CPU) | `?knn=cpu` | default; exact, the oracle, and the fastest here in practice |
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
- `src/edges.ts` — the pair-graph overlay's index buffer. Its own module, with
  no DOM, so it can be checked headlessly — see `CLAUDE.md`.
- `src/main.ts` — demo wiring, bounds reduce, point renderer

## Caveats

- Gaussian init rather than PCA init, so layouts vary by seed. Change `seed` in
  the `pacmapWebGPU` call to reshuffle.
- Under `?knn=nnd` the seed doesn't fully determine the layout — see above.
- Under `?algo=localmap` the further-pair redraw gives up after 100 tries per
  slot and keeps the partner it had. Upstream intends the same bound but only
  applies it to two of its four rejection paths, so a point with nothing inside
  `low_dist_thres` never terminates there. That is a hang on a CPU and a lost
  device on a GPU, so the bound here covers every path.
- The PCA output spans the top-100 principal subspace but isn't rotated to the
  principal axes. Distances are unaffected; per-axis variance ordering is not
  meaningful.
- Decoding the full sprite costs ~200MB of RAM transiently regardless of how
  many samples you keep.
- `pcaProject` is called with `inPlace: true`, which centers the MNIST array
  where it sits rather than allocating a second copy (204MB at 65k). It destroys
  its input; the demo never reads `X` again.
