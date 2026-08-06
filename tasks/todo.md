# TODO: draw the pair graph — near (green), mid-near (yellow), further (red)

Full detail, rationale and acceptance criteria in [plan.md](plan.md).
One commit per task, straight to `main`. Gates after every task: `npm run build`, plus the
checks named on the task. `npm run check:ab` must print exactly **0** throughout — nothing
here touches the optimizer, and a non-zero would mean it does.

Out of scope by decision: opacity/width varying with pair weight; reflecting LocalMAP's
mid-run further-pair resample (red edges are the initial draw — documented, not fixed).

## Phase 1 — The library exposes its graph

- [x] **1. `mnFwd` + `PacmapRun.graph`** — S — deps: none
      `src/pacmap-webgpu.ts`, `scripts/check-kernels.ts`
      - [x] `Pairs.mnFwd` (N x nMN, row-major) — **no new RNG call, none moved**
      - [x] The mid-near loop writes **one local** into both `mnFwd` and the CSR triple,
            so the two cannot disagree. This replaces the planned oracle: the alternative
            was exposing the CSR purely for a test, and construction beats comparison
      - [x] `PairGraph { nbFwd, mnFwd, fpFwd, nNB, nMN, nFP }`; `EmbeddingRun.graph?`
            optional (druid has none), required on `PacmapRun`
      - [x] `PairGraph`'s doc records what upstream actually does: `pacmap_grad` walks all
            three arrays every call, so these are the force set exactly — and `w_MN = 0`
            through phase 3, so a drawn mid-near edge is not a pulling one
      - [x] `fpFwd`'s doc says plainly it is the *initial* set and LocalMAP's GPU resample
            does not refresh it
      - [x] Arrays handed out, not copied
      - [x] `check:kernels` gains `graph/*` (14 checks): row lengths, in-range, no
            self-pairs, `nbFwd` duplicate-free per row, `fpFwd ∩ nbFwd = ∅` per row, and
            the three sets at their expected **rank quantiles**
      - [x] Rank, not distance: mean |d| is 4.58 / 19.79 / 20.49 at D=16 — a 3%
            mid-near-vs-further gap that a uniform `mnFwd` *passed*. Rank quantiles come
            out 0.013 / 0.281 / 0.499 against ~0 / 2÷7 / ½ predicted, and the mid-near
            band is bounded on both sides
      - [x] Demonstrated failing first: uniform `mnFwd` trips `mn-rank-is-mid` at 0.501;
            dropping `!nbSet.has(c)` trips `fp-avoids-near-partners` with 208 overlaps
      - [x] `check:ab` no-change control printed 0, then `-- HEAD` bit-identical on
            `--variant=pacmap` and `--variant=localmap`

### ✅ Checkpoint: graph available and checked
- [x] `build`, `check:shaders` (11), `check:kernels` (55), `check:druid` (20) green;
      `check:ab` bit-identical on both variants
- [x] Nothing user-visible yet

## Phase 2 — The renderer

- [x] **2. Edge shader, both pipelines, index buffer, draw** — L — deps: 1
      `src/shaders.ts`, `src/edges.ts` (new), `src/main.ts`, `scripts/check-shaders.ts`,
      `scripts/check-kernels.ts`
      - [x] `edgeWGSL(d)`: bindings 0/1/2 are `Bounds` A / `View` / `Bounds` B; binding 3
            is the per-kind colour, `hasDynamicOffset`
      - [x] The two uniform structs and the data→world mapping are **shared text**
            (`BOUNDS_STRUCT`, `VIEW_STRUCT`, `worldStatements`) spliced into both shaders,
            so an edge cannot be placed by different arithmetic than its endpoints.
            Verified `renderWGSL`'s output is unchanged: every non-comment line identical
            at d=2 and d=3 against HEAD
      - [x] Vertex buffers 0/1 are the two keyframes at `stepMode: "vertex"`;
            `mix(p, pB, V.lerpT)` → world → `V.viewProj`. No quad, no radius
      - [x] Fragment alpha 0.35 blended, **1.0 when occluding**
      - [x] Index buffer built at setup: three contiguous ranges, each shuffled from the
            run's seed so a prefix is a uniform sample and two runs at one seed agree
      - [x] 3-slot dynamic-offset colour uniform — near green, mid-near yellow, further red
      - [x] `pipe` / `pipeDepth` from one descriptor, depth only in 3D, selected by the
            existing `occludingNow()` — **mandatory:** a pipeline without depth state
            cannot be used in a pass that has a depth attachment
      - [x] Encoded **first**, same pass as the points, same two vertex-buffer offsets;
            three `drawIndexed` calls, skipped at count 0
      - [x] `view.edges = false` plus a `show edges` checkbox in `rendering` (the plan's
            "behind one checkbox" — without it none of the browser steps are reachable);
            `view.edgePct` hard-coded to 100 / 100 / 5 until Task 3
      - [x] **`buildEdgeIndices` moved to its own DOM-free `src/edges.ts`.** It is the one
            piece of the overlay whose mistakes are off-by-ones rather than validation
            errors, and stranding it in `main.ts` would have made it permanently
            browser-only. Now checked as data (11 new `edges/*` checks)
      - [x] `check:shaders`: `edge-{2,3}d{,-depth}` pipeline cases plus an
            `edge-3d-occlusion-draw` bundle case (11 → 15)
      - [x] Demonstrated failing first — five mutations:
            shuffle swapping single indices rather than whole pairs (3 checks, 3874/4000
            wrong); no shuffle at all (3 checks, 0% descending); returning the whole
            allocation rather than the subarray; drawing the pad entries; a WGSL error in
            `edgeWGSL` (all 4 shader cases). Plus, on the draw case: no `setIndexBuffer`,
            depth state dropped while the bundle declares a depth format, and a
            misaligned dynamic offset
      - [x] **Two planned mutations do *not* fail, checked rather than assumed:**
            `setIndexBuffer` with `uint16` against a u32 buffer (the format is a
            byte-width declaration, not a claim about contents) and a `float32x2`
            attribute against a `vec3<f32>` input (WebGPU fills the missing components).
            The check-shaders comment now says so rather than claiming coverage it lacks
      - [x] `check:ab` bit-identical on both variants
      - [ ] **Browser (yours):** 2D, 2k, PaCMAP — green mesh in clusters, yellow between
            them, red spanning. Zoom in on one point: its green edges terminate *on* it
      - [ ] **Browser (yours):** scrub with `interpolation` on — edges never lag their
            points by a frame
      - [ ] **Browser (yours):** unticking gives back the pre-existing canvas exactly
      - [ ] **Browser (yours):** 3D at 2k, occlusion on and off
      - [ ] **Browser (yours):** 65k at the defaults — note the frame time; if it is bad,
            the defaults move, not the design

### ⬜ Checkpoint: edges render
- [ ] End-to-end on the GPU engine, both dims, both depth modes, live and on playback

## Phase 3 — The pane

- [ ] **3. `edges` folder: three sliders, greying, LocalMAP note** — M — deps: 2
      `src/main.ts`
      - [ ] New `edges` folder below `rendering`, live like it — no restart on edit
      - [ ] `show edges` checkbox; `near %`, `mid-near %`, `further %` (0–100,
            defaults 100 / 100 / 5)
      - [ ] Each slider resolves against its range total on the CPU and moves a draw
            count; the shader learns nothing new
      - [ ] 0% encodes no draw rather than a zero-count draw
      - [ ] Folder disabled when `pm.graph` is absent (both CPU engines), driven from the
            same `syncAlgorithm` that greys `kNN algo`
      - [ ] The `mid-near %` label carries the `w_MN = 0` fact — yellow edges are inert
            from iteration `n1+n2` on, so a drawn one is not necessarily a pulling one
      - [ ] Under LocalMAP, the pane says the red edges are the initial draw
      - [ ] `build`, `check:shaders`, `check:kernels`, `check:druid` green; `check:ab` at 0
      - [ ] **Browser (yours):** all three sliders sweep 0→100 at 10k
      - [ ] **Browser (yours):** `PaCMAP (CPU - druid)` greys the folder rather than
            throwing; switching back and forth keeps the state
      - [ ] **Browser (yours):** LocalMAP across the phase-3 boundary — red edges visibly
            static while points move, which is the limitation looking like itself

### ⬜ Checkpoint: full feature
- [ ] All four checks green; every control exercised by hand under both engines

## Phase 4 — Documentation

- [ ] **4. `CLAUDE.md` + `README.md`** — S — deps: 3
      - [ ] New `The pair graph overlay` section: indexed line-list over the existing
            position buffer (and why not a storage buffer — 256-byte dynamic-offset
            alignment vs. `frameBytes`, and `posHistory` already at the 128MB binding
            limit); why the kind rides in a dynamic-offset uniform (`vertex_index` is the
            *fetched index* on an indexed draw, so it cannot be compared to a boundary);
            why the ranges are shuffled at setup; why the depth pipeline is mandatory
      - [ ] LocalMAP section: the overlay shows the initial further pairs, not the
            resampled ones, and what it would take to fix
      - [ ] Coverage gaps: pipelines and index format are checked; whether an edge lands
            on its own endpoints is browser-only
      - [ ] Pane paragraph gains the `edges` folder and its lifetime
      - [ ] `README.md` gains `Seeing the pairs`: the three colours, what each set does,
            and that the percentages are render-time
