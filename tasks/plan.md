# Plan: draw the pair graph — near, mid-near and further edges

## What this is

PaCMAP's whole mechanism is three sets of pairs pulling and pushing on each other, and
right now none of them are visible. The demo animates the *result* of the forces without
ever showing the forces. This adds a line renderer for the three pair sets, coloured by
kind:

| kind | colour | count per point | what it does |
| --- | --- | --- | --- |
| near (`T_NB`) | green | `nNeighbors` (10, or `defaultNeighbors(N)`) | attract |
| mid-near (`T_MN`) | yellow | `round(nNB * mnRatio)` = 5 | attract, on a schedule — see below |
| further (`fpFwd`) | red | `round(nNB * fpRatio)` = 20 | repel |

**Every pair drawn is a pair that acts.** Checked against upstream rather than assumed:
`pacmap_grad` (`pacmap/pacmap.py:272`) loops over the whole of `pair_neighbors`, `pair_MN`
and `pair_FP` on every call — there is no per-iteration subsampling anywhere in the
optimizer. The subsetting people expect happens once, at sampling time: `sample_MN_pair`
draws 6 candidates per slot and keeps the 2nd-closest, and the five losers are never
stored. So the arrays this overlay draws are exactly the force set, not a superset of it.

The one caveat, and it is a real one: `find_weight` sets **`w_MN = 0.0` for the whole of
phase 3** — the last 250 of 450 iterations. The mid-near pairs still exist and will still
be drawn yellow, but they exert no force for most of the run. That is the single place
where "an edge means this pair exists" reads as a stronger claim than it is, so the pane
and `CLAUDE.md` say so. It is a fact about the schedule, not a reason to encode weight
into opacity.

Deliberately **not** in scope, per the request:

- No opacity or width varying with the pair's weight or the schedule's phase. A drawn
  edge means "this pair exists", nothing more. The weights swing three orders of
  magnitude over a run (`w_MN` 1000→3), so encoding them is its own design problem.
- No re-reading LocalMAP's **resampled** further pairs. Under `variant: "localmap"` the
  further set is redrawn on the GPU 24 times during phase 3, and the drawn red edges are
  the *initial* set throughout. This is a real inaccuracy and gets a note in the pane, not
  a fix. Doing it properly means the index buffer becoming GPU-written, which is a much
  larger change than everything else here combined.

## Where it plugs in

Two facts about the existing code decide almost the entire design.

**The pairs already exist on the CPU at setup and are already dense.** `samplePairs`
builds `nbFwd` (N × nNB, row-major) and `fpFwd` (N × nFP) as dense arrays — `nbFwd`
because LocalMAP's resample needs it as a reject list, `fpFwd` because LocalMAP redraws
it. Mid-near is the only one of the three that exists *only* inside the CSR's interleaved
`(i, j, t)` triples. So the library-side work is: build `mnFwd` symmetrically with the
other two, and expose all three. There is no new sampling and no new randomness — `mnFwd`
records the `second` that the mid-near loop already picks, so the RNG stream is untouched
and `check:ab` must stay at exactly 0.

**An edge is two points, and points are already a vertex buffer.** So edges are an
**indexed** draw over the *same* position buffer the point renderer binds, with topology
`line-list` and an index buffer of endpoint pairs. This is what makes the feature cheap:

- No storage binding for positions, so no `maxStorageBufferBindingSize` question about
  `posHistory` (which sits at the 128MB budget, exactly the default limit).
- No dynamic offset into `posHistory`, which would be illegal anyway — a storage binding's
  dynamic offset must be 256-byte aligned and `frameBytes = N*4*d` generally is not
  (520,000 at N=65k, d=2).
- **Playback interpolation comes for free.** Bind slot `a` to vertex buffer 0 and slot `b`
  to vertex buffer 1 and `mix(p, pB, V.lerpT)` in the edge vertex shader is the same line
  the point shader already runs. Same for the two `Bounds` uniforms and the same
  data→world normalization, so an edge cannot disagree with its endpoints about framing.

The one thing the indexed form costs is that **the pair kind cannot be a vertex
attribute** — an indexed draw's `@builtin(vertex_index)` is the fetched index value, not
the position in the index buffer, so it cannot be compared against a boundary. The kind
therefore comes from *which draw* is being encoded: one index buffer holding three
contiguous ranges, three `drawIndexed(count, 1, firstIndex)` calls, and the colour arriving
through a **dynamic-offset uniform** with a 256-byte slot per kind. That is the same
mechanism the optimizer already uses for its per-iteration weights, so it is an idiom in
this codebase rather than a new one.

### The proportion slider

Each of the three ranges is **shuffled once at setup**, so any prefix of a range is a
uniform random sample of that kind. The slider then moves an instance count and nothing
else — `drawIndexed(pct/100 * total * 2, ...)` — with no buffer rebuilt, no run restarted,
and edges appearing spread through the cloud rather than walking along the sample order.
This is exactly the argument the `digit %` slider already rests on, and it is why the
index buffer holds *every* pair rather than a sampled subset.

Cost of holding all of them: 8 bytes per edge. At N=65k with auto neighbours (nNB≈13,
nMN≈7, nFP≈26) that is ~3M edges and ~24MB — against `posHistory`'s 128MB and the digit
atlas's 51MB, it is the small one. Three sliders rather than one, because the three counts
differ by 4x and the useful percentage differs with them.

### Defaults

Edges default **off**. The point cloud is the thing being demonstrated and 3M lines over
it is a wash of colour, so it is opt-in. When ticked: near 100%, mid-near 100%, further
5% — the far pairs are both the most numerous and the least legible (they connect
everything to everything, by construction), and 5% of them is enough to read as "outward
pressure" without covering the canvas. Alpha is a fixed 0.35, not a slider, until it is
shown to need one.

### Depth, and why the depth pipeline is not optional

If occlusion is on, the render pass carries a depth attachment, and **a pipeline without
depth state cannot be used in a pass that has one** — it is a validation error, not a
visual difference. So the edge pipeline needs the same blended/occluded pair the point
pipeline has, built from one descriptor, selected by the same `occludingNow()`. That
function stays the single source of truth for all four things that must now agree: which
point pipeline, which edge pipeline, whether the pass has a depth attachment, and the
`occlude` flag in the view uniform.

Line width in WebGPU is always one device pixel; there is no `lineWidth`. Nothing to do
about that short of drawing quads, which is not worth it here.

### Draw order

Edges are encoded **first**, in the same render pass, so points sit on top of them. A
second pass with `loadOp: "load"` would work too and is strictly worse — one more pass per
frame for nothing.

### The CPU engine has no graph

DruidJS builds its own neighbour graph internally and exposes nothing, so edges are a
GPU-engine feature. `graph` is therefore **optional on `EmbeddingRun`** and simply absent
under `druid-cpu.ts`; the pane greys the whole `edges` folder when it is missing. Same
shape as `kNN algo` greying out for the CPU engines already.

## What the headless checks can and cannot see

- `check:kernels` gains a `graph/*` section. This is a genuine gap being closed, not
  ceremony: `nbFwd` and `fpFwd` have never been checked as *data*, only used. The
  invariants are cheap and sharp — every row is the right length, no entry is out of
  range, no point is its own partner, `nbFwd` rows are duplicate-free, `fpFwd` rejects
  everything in the matching `nbFwd` row, and `mnFwd` agrees with the `T_MN` entries the
  CSR carries. That last one is the check that `mnFwd` is *the* mid-near set and not a
  second draw from the RNG.
- `check:shaders` gains the edge pipeline at both dimensionalities and both depth modes,
  and the existing occlusion-draw case gains an `drawIndexed` of edges under both. That
  covers the two failures that blank the canvas without raising anything: an index buffer
  format or a vertex layout the pipeline disagrees with, and a pipeline/pass depth-state
  mismatch.
- `check:ab` must print **0** throughout. Nothing here touches the optimizer.
- **Nothing headless renders a pixel**, so whether an edge lands on the points it belongs
  to — the one mistake that matters most and the easiest to make, since it is an off-by-one
  in an index buffer — is visible only in a browser. Every task carries a browser step and
  they are the author's to run.

## Dependency graph

```
1 library exposes the graph (mnFwd, PacmapRun.graph)
      │
      └── 2 edge renderer: shader, both pipelines, index buffer, draw  ── checkpoint (browser)
                │
                └── 3 pane: three sliders, greying, LocalMAP note      ── checkpoint (browser)
                          │
                          └── 4 docs
```

Strictly serial. There is no useful parallel slice: task 2 is the whole vertical path and
tasks 1 and 3 are the ends of it.

---

## Task 1 — The library exposes its pair graph — S

`src/pacmap-webgpu.ts`, `scripts/check-kernels.ts`

Add `mnFwd` to `Pairs` and `samplePairs` (N × nMN, row-major, filled from the `second`
the mid-near loop already selects), and expose all three dense arrays on the run:

```ts
export interface PairGraph {
  nbFwd: Uint32Array;  // N x nNB
  mnFwd: Uint32Array;  // N x nMN
  fpFwd: Uint32Array;  // N x nFP — the *initial* set; LocalMAP redraws it on the GPU
  nNB: number; nMN: number; nFP: number;
}
```

`EmbeddingRun.graph?: PairGraph` (optional — the druid engine has none), narrowed to
required on `PacmapRun`. The doc comment on `fpFwd` must say plainly that it is not
refreshed by LocalMAP's resample, because that is the thing a reader will otherwise assume.

**Acceptance**

- `mnFwd[i * nMN + m]` is the same partner the CSR's `m`-th `T_MN` entry for `i` carries.
- No RNG call is added, moved or removed anywhere in `samplePairs`.
- Arrays are the ones already allocated — `nbFwd` and `fpFwd` are handed out, not copied.

**Verify**

- `npm run build`
- `npm run check:kernels` — new `graph/*` section: row lengths, in-range, no self-pairs,
  `nbFwd` duplicate-free within a row, `fpFwd ∩ nbFwd = ∅` per row, and the three sets at
  the rank quantiles their names imply.
- **Prove it can fail:** point `mnFwd` at a uniform draw; drop the `!nbSet.has(c)`
  rejection.
- `npm run check:ab -- HEAD` prints **0** on `--variant=pacmap` and `--variant=localmap`.
  Run the no-change control on a clean tree first.

**Landed — two departures from the plan above, both for the better**

- **The `mnFwd`-vs-CSR comparison became a construction rather than a check.** The plan
  wanted an oracle walking the run's own CSR, which would have meant exposing the CSR
  purely for a test. Instead the mid-near loop writes one local into both `mnFwd` and the
  CSR triple, so there are no longer two things that *can* disagree. The `best`-vs-`second`
  mutation the plan proposed is consequently not expressible — the right outcome for it.
- **Distance was too blunt a statistic; the check is in rank space.** Mean |d| over the
  three sets is 4.58 / 19.79 / 20.49 at D=16 — a 3% mid-near-vs-further gap, because
  pairwise distances concentrate in 16 dimensions. A `mnFwd` wired to a uniform draw
  *passed* an ordering test on those numbers, caught only incidentally by the self-pair
  check. Rank quantiles do not concentrate: measured 0.013 / 0.281 / 0.499 against
  ~0 / 2÷7 / ½ predicted from how each set is drawn. The mid-near band is bounded on both
  sides, and the uniform-draw mutation trips it at 0.501.

### ✅ Checkpoint: the graph is available and checked

Nothing user-visible. `build`, `check:shaders` (11), `check:kernels` (55), `check:druid`
(20) green; `check:ab` bit-identical on both variants, control run first.

---

## Task 2 — The edge renderer — L

`src/shaders.ts`, `src/main.ts`, `scripts/check-shaders.ts`

The whole vertical path, behind one `edges` checkbox with the proportions hard-coded to
their defaults. Sliders are task 3; this task is about the draw landing correctly.

**`edgeWGSL(d)` in `shaders.ts`**, alongside `renderWGSL`:

- Bindings 0/1/2 are `Bounds` A, `View`, `Bounds` B — the same three the point shader
  reads, so the data→world normalization, the box mix and `lerpT` are copied verbatim
  rather than re-derived. Binding 3 is the per-kind colour, `hasDynamicOffset: true`.
- Vertex buffers 0 and 1 are the two keyframes, `stepMode: "vertex"` (the point pipeline
  binds the same buffers at `stepMode: "instance"` — different pipeline, different layout,
  same memory).
- `pos = mix(p, pB, V.lerpT)` → world → `V.viewProj`. No quad expansion, no radius.
- Fragment returns `vec4(colour, ALPHA)` with `ALPHA = 0.35`. Under `occlude` it must
  behave the way the point shader's occluded branch does — a line that writes depth while
  semi-transparent is the dark-streak failure again. Simplest correct answer: when
  occluding, return alpha 1.0.

**`main.ts`**

- Build the index buffer at setup from `pm.graph`: three contiguous ranges (near, mid-near,
  further), each shuffled with a generator seeded from the run's seed so two runs at one
  seed draw the same sample — same argument as the digit ranks. Record
  `{ first, count }` per range.
- One `edgeBGL` + `edgeBG`; a 3-slot dynamic-offset uniform holding the three colours
  (`vec4`, 256-byte stride).
- `edgePipe` and `edgePipeDepth` from one descriptor, the depth one only in 3D, exactly
  as the point pair is built.
- In `encodeRender`, before the point draw and inside the same pass: set the edge pipeline
  chosen by the same `occlude` local, set both position vertex buffers at the same two
  offsets the point draw uses, `setIndexBuffer`, then three `drawIndexed` calls skipped
  when their count is 0.
- `view.edges = false` and three percentage constants for now.

**Acceptance**

- Ticking `edges` mid-run changes the next drawn frame with no restart.
- Scrubbing moves edges with their endpoints, including between keyframes with
  `interpolation` on — an edge must never lag its points by a frame.
- Unticking gives back a canvas bit-identical to what predates this: the point draw is
  untouched and no edge draw is encoded.
- 3D with `occlusion` both on and off draws without validation errors.

**Verify**

- `npm run build`
- `npm run check:shaders` — new `edge-{2,3}d{,-depth}` pipeline cases, and the existing
  `render-3d-occlusion-draw` bundle case gains the edge pipeline and a `drawIndexed`
  under both modes. **Prove it can fail** by handing the edge pipeline a `uint16`
  index format against a `uint32` buffer, and by dropping the depth state from
  `edgePipeDepth` while the bundle still declares a depth format.
- `npm run check:ab -- HEAD` prints 0 — `main.ts` and `shaders.ts` are not in the library.
- **Browser (yours):** 2D, 2k points, PaCMAP. Edges on — green mesh inside clusters,
  yellow reaching between them, red spanning the whole cloud. Scrub the timeline and
  confirm edges track points exactly. Zoom right in on one point and confirm its green
  edges terminate *on* it, not near it.
- **Browser (yours):** 3D at 2k — same, then occlusion on and off.
- **Browser (yours):** 65k with `further` at its default. Note the frame time; if it is
  bad enough to matter, say so and the defaults move rather than the design.

### ⬜ Checkpoint: edges render

End-to-end on the GPU engine, both dimensionalities, both depth modes, live and on
playback — confirmed by hand.

---

## Task 3 — The pane — M

`src/main.ts`

- New `edges` folder in the pane, below `rendering` and live like it (an edit rewrites the
  current run's state; no restart). Contents: `show edges` checkbox, then `near %`,
  `mid-near %` and `further %` sliders (0–100, defaults 100 / 100 / 5).
- Each slider resolves against its range's total on the CPU and moves a draw count. The
  shader learns nothing new.
- The whole folder is disabled when `pm.graph` is absent — i.e. under both CPU engines —
  driven from the same `syncAlgorithm` that already greys `kNN algo`.
- The mid-near slider's label carries the `w_MN = 0` fact — the yellow edges stop pulling
  at iteration `n1+n2` and are inert for the rest of the run.
- Under `variant: "localmap"`, the further-pair sliders' label or the folder's title says
  the red edges are the initial draw and not LocalMAP's resampled set. A one-line pane
  title suffix is enough; the reason lives in `CLAUDE.md`.

**Acceptance**

- Moving a slider redraws at the new count with no buffer written and no run restarted.
- 0% encodes no draw for that kind rather than a zero-count draw.
- Selecting a CPU engine greys the folder; selecting a GPU engine un-greys it, and the
  state survives switching back and forth without a run in between.

**Verify**

- `npm run build`; `check:shaders`, `check:kernels`, `check:druid` green; `check:ab` at 0.
- **Browser (yours):** all three sliders sweep 0→100 smoothly at 10k. Switch to
  `PaCMAP (CPU - druid)` and confirm the folder greys rather than throwing.
- **Browser (yours):** LocalMAP, watch the red edges across the phase-3 boundary and
  confirm they are visibly static while the points move — which is the documented
  limitation looking exactly like itself.

### ⬜ Checkpoint: full feature

All four checks green. Every control exercised by hand under both engines.

---

## Task 4 — Documentation — S

`CLAUDE.md`, `README.md`

- New `CLAUDE.md` section, `The pair graph overlay`: why indexed line-list over the
  existing position buffer rather than a storage buffer (the alignment and binding-size
  arguments above), why the kind rides in a dynamic-offset uniform rather than a vertex
  attribute (`vertex_index` is the fetched index), why the ranges are shuffled at setup,
  and why the depth pipeline is mandatory rather than a nicety.
- The LocalMAP section gains a sentence: the overlay's further pairs are the initial draw,
  the resample is not reflected, and what it would take.
- The coverage-gaps paragraph gains edges — pipelines and index format are checked,
  whether an edge lands on its own endpoints is browser-only.
- The pane paragraph gains the `edges` folder and its lifetime (live, like `rendering`).
- `README.md` gains a short `Seeing the pairs` section: the three colours, what each set
  does to the layout, and that the percentages are render-time.
