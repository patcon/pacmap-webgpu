# Implementation Plan: CPU comparison algorithms via DruidJS

## Context

The demo offers one implementation of each algorithm — our own WebGPU PaCMAP and LocalMAP,
selected by the pane's `algorithm` dropdown. There is nothing to check the layouts
*against*. `check:kernels` proves the kernels compute what a CPU oracle says they should,
and `check:ab` proves a refactor didn't move the embedding, but neither answers "does this
look like what an *independent* PaCMAP implementation produces?"

DruidJS 0.9.0 ships `PaCMAP` and `LocalMAP` classes — verified in the published tarball
(`src/dimred/PaCMAP.js`, `src/dimred/LocalMAP.js`), written against the same upstream
reference we track, `YingfanWang/PaCMAP`. Running them beside ours, in the same viewer,
over the same PCA output, at the same seed, turns that question into something you can
scrub through.

Outcome: the `algorithm` dropdown offers four entries — `PaCMAP (GPU - custom)`,
`LocalMAP (GPU - custom)`, `PaCMAP (CPU - druid)`, `LocalMAP (CPU - druid)` — and all four
drive the *same* renderer, history buffer, and transport.

## Architecture Decisions

- **The CPU backends present themselves as a `PacmapRun`.** Everything downstream of
  `pacmapWebGPU()` in `main.ts` — the bounds reduce, the `posHistory`/`boundsHistory`
  banking, the scrubber, the phase readout — is written against `{ positions, runRange,
  totalIters, destroy }`. A worker-backed run satisfying that shape reuses all of it. The
  one concession: `runRange` becomes `void | Promise<void>` and the run loop awaits it.
  The loop is already `async`, so this is a one-word change and the GPU path is untouched.

- **The invariant survives, in the only form it can.** Positions still never *leave* GPU
  memory on the render path. The CPU run's `positions` is a real `GPUBuffer`
  (`STORAGE | VERTEX | COPY_DST | COPY_SRC`) that the worker's snapshot is written *into*
  via `queue.writeBuffer`. Direction is the whole point — an upload per captured frame,
  never a `mapAsync` readback — so banking, scrubbing and replay stay exactly as they are.

- **Druid gets the demo's PCA output, not raw pixels.** `pcaProject` already produces the
  100-d `Z` the GPU path consumes; the worker receives that same array (transferred, not
  copied) with `apply_pca: false`. Same input, same seed, same `num_iters: [100, 100,
  250]` ⇒ `totalIters` 450 for every backend, so the history/stride math needs no case.

- **Warn, never block.** Druid's neighbour search is exact O(N²·D) and its optimizer is JS
  f64, so 65k points is minutes-to-hours. That stays selectable; the cost is surfaced in
  the existing `~Ns setup` hint, and the run becomes abortable so a long one is
  recoverable without closing the tab.

- **One expected difference, stated up front.** Druid initializes from a scaled PCA
  embedding (`PaCMAP.js:515`); ours from a scaled Gaussian, a deviation already documented
  in `CLAUDE.md`. The two will not converge to the same layout at the same seed and
  shouldn't be expected to. What is comparable is cluster structure, not orientation.

## Task List

### Phase 1: Foundation

- [ ] Task 1: Dependency + headless check (`check:druid`)

### Checkpoint: Foundation
- [ ] `check:druid` green, and demonstrated failing when broken

### Phase 2: The backend

- [ ] Task 2: Worker + adapter, wired end to end
- [ ] Task 3: Four-way dropdown and the cost hint

### Checkpoint: Core
- [ ] All four algorithms animate at N=2000
- [ ] GPU output unchanged from `main` (`check:ab`, no-change control first)

### Phase 3: Polish

- [ ] Task 4: Abort, so a long CPU run is recoverable
- [ ] Task 5: Document it

### Checkpoint: Complete
- [ ] `build`, `check:shaders`, `check:kernels`, `check:druid` all clean

---

## Task 1: Dependency + headless check (`check:druid`)

**Description:** Add `@saehrimnir/druidjs` and land the automated gate *first* — it is the
only check this work gets, and unlike the demo it needs no browser and no Dawn, because
the druid path is pure CPU JS.

**Acceptance criteria:**
- [ ] `@saehrimnir/druidjs` in `dependencies`; `check:druid` script in `package.json`
- [ ] `scripts/check-druid.ts` builds a synthetic 4-blob dataset (~400 points, 20-d) and
      asserts: output is N×2 and finite for both classes; a fixed seed reproduces
      bit-for-bit; PaCMAP and LocalMAP at one seed differ; two runs differing only in
      `n_neighbors` differ (this is the assertion that catches a typo'd option name, which
      druid would otherwise silently ignore); mean intra-blob 2-d distance < inter-blob
- [ ] Each assertion demonstrated failing before being trusted green

**Verification:**
- [ ] `npm run check:druid` passes
- [ ] `npm run build`
- [ ] Misspell `n_neighbors` in the param object; the params assertion trips

**Dependencies:** None
**Files:** `package.json`, `scripts/check-druid.ts`
**Scope:** S

---

## Task 2: Worker + adapter, wired end to end

**Description:** The vertical slice — a CPU run that renders. `src/druid-worker.ts` owns
the druid instance (no DOM, no GPU); `src/druid-cpu.ts` owns the `GPUBuffer` and hands
back something `main.ts` already knows how to drive.

Worker protocol, modelled on the reference `druidWorker.ts` (command in, events out, error
event on throw):
- `init` — receive transferred `Z` (`Float32Array`, N×100) + params. Widen to
  `Float64Array[]` row views once. Construct `new druid.PaCMAP(rows, p)` or
  `new druid.LocalMAP(rows, p)`. Call `check_init()` *explicitly*, so the expensive kNN and
  pair sampling happens here and can be reported, rather than lazily inside the first step.
  Hold `gen = dr.generator(450)`. Post `ready`.
- `step { to }` — pull `gen.next()` until the counter reaches `to`, flatten `dr.Y.values`
  (flat `Float64Array`, `Matrix.d.ts:445`) into a `Float32Array(N*2)`, post it
  **transferred**. Only capture boundaries pay the conversion.

Drive `generator()` rather than calling `next()` on the instance: a completed or `break`-ed
generator releases druid's WASM buffers, a hand-driven `next()` loop does not.

Param mapping, confirmed against the shipped `ParametersPaCMAP` / `ParametersLocalMAP`:
`n_neighbors`, `MN_ratio`, `FP_ratio`, `seed`, `d: 2`, `num_iters: [100, 100, 250]`,
`apply_pca: false`, `knn: null`, plus `low_dist_thres` for LocalMAP only.

**Acceptance criteria:**
- [ ] `druidCPU(device, Z, N, opts): Promise<PacmapRun>` spawns the worker via
      `new Worker(new URL("./druid-worker.ts", import.meta.url), { type: "module" })`
- [ ] `runRange(from, to)` posts, awaits the frame, and `writeBuffer`s it — no readback
- [ ] `destroy()` terminates the worker; worker errors surface through `onStatus`
- [ ] `PacmapRun.runRange` widened to `void | Promise<void>`; the run loop awaits it

**Verification:**
- [ ] `npm run build`
- [ ] `?algo=pacmap-cpu` at N=2000 animates and lands on a recognisable digit layout
- [ ] Scrubbing and replay behave exactly as on the GPU path

**Dependencies:** Task 1
**Files:** `src/druid-worker.ts`, `src/druid-cpu.ts`, `src/pacmap-webgpu.ts`, `src/main.ts`
**Scope:** M

---

## Task 3: Four-way dropdown and the cost hint

**Description:** Replace `params.variant: Variant` with `params.algorithm: AlgoKey`
(`"pacmap-gpu" | "localmap-gpu" | "pacmap-cpu" | "localmap-cpu"`), keeping the
`Record<string, AlgoKey>` annotation on the options map — that annotation is what makes a
typo'd *value* a build error rather than a silently wrong run, and it matters more with
four entries than it did with two.

**Acceptance criteria:**
- [ ] Labels exactly: `PaCMAP (GPU - custom)`, `LocalMAP (GPU - custom)`,
      `PaCMAP (CPU - druid)`, `LocalMAP (CPU - druid)`
- [ ] `?algo=` accepts the four keys; `pacmap`/`localmap` still work as GPU aliases
- [ ] `go()` branches to `druidCPU` or `pacmapWebGPU` on the derived engine, nothing else
- [ ] `kNN algo` disabled under a CPU engine (druid runs its own exact search);
      `low_dist_thres` stays enabled for `localmap-cpu`, which reads it
- [ ] `estimateSetupSecs` gains a CPU branch; the hint refreshes on dropdown change

**Verification:**
- [ ] `npm run build`
- [ ] All four exercised at N=2000; label text matches
- [ ] Hint jumps by orders of magnitude on switching to a CPU option — at N=65000 it
      should read in the hours. That number *is* the warning.

**Dependencies:** Task 2
**Files:** `src/main.ts`
**Scope:** S

---

## Task 4: Abort, so a long CPU run is recoverable

**Description:** With no cap, a 65k CPU run is otherwise only escapable by closing the
tab. During a run the Start button becomes a Stop that bumps `runGen`, calls
`pm.destroy()` (terminating the worker), and restores the controls. Cheap, and it is what
makes "warn, never block" a safe default.

**Acceptance criteria:**
- [ ] Stop is present for every engine, and is the only control enabled mid-run
- [ ] Stopping a CPU run at N=20000 returns the UI to idle within a frame or two
- [ ] The last banked frame stays on screen; the transport remains scrubbable over it

**Verification:**
- [ ] `npm run build`
- [ ] Start a CPU run at N=20000, stop it mid-flight, start a GPU run — no stale canvas,
      no orphaned worker (check the browser task manager)

**Dependencies:** Task 3
**Files:** `src/main.ts`, `index.html`
**Scope:** S

---

## Task 5: Document it

**Description:** `CLAUDE.md` is the map of this repo; a fourth check and a second engine
belong in it.

**Acceptance criteria:**
- [ ] `check:druid` in the commands block and in the paragraph distinguishing the checks —
      it sits at a level none of the other three cover (an independent implementation,
      not our own kernels)
- [ ] The CPU backend documented with its deviations: PCA init vs our Gaussian, druid's
      own exact kNN, f64 vs f32, upload-per-frame rather than readback
- [ ] The LGPL-3.0-or-later note on the dependency

**Verification:**
- [ ] `npm run build && npm run check:shaders && npm run check:kernels && npm run check:druid`

**Dependencies:** Task 4
**Files:** `CLAUDE.md`
**Scope:** XS

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Druid's per-iteration cost makes even N=2000 sluggish | Med | Measured in Task 1's check before any UI exists; if bad, the hint is the honest answer |
| Bundling druid's WASM through a Vite worker misbehaves | Med | Task 2 lands the worker alone; druid falls back to a JS neighbour path when WASM is unavailable |
| LGPL-3.0 dependency in a repo with no LICENSE file | Low | Flagged, not resolved here |

## Open Questions

- The LGPL-3.0-or-later license on `@saehrimnir/druidjs` is the first copyleft dependency
  here, and the repo has no LICENSE of its own. Noted for a separate decision; it does not
  block the work.
