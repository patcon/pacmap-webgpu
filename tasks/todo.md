# TODO: CPU comparison algorithms via DruidJS

Full detail, rationale, and acceptance criteria in [plan.md](plan.md).
One commit per task, straight to `main`. Gates after every task: `npm run build`, plus
`npm run check:druid` once Task 1 lands.

## Phase 1 — Foundation

- [x] **1. Dependency + headless check (`check:druid`)** — S — deps: none
      `package.json`, `scripts/check-druid.ts`, `.github/workflows/pages.yml`
      - [x] `@saehrimnir/druidjs` added; `check:druid` script (esbuild → node, no Dawn)
      - [x] Synthetic 4-blob dataset (400 points, 20-d)
      - [x] Asserts: N×2 and finite, both classes
      - [x] Asserts: fixed seed reproduces bit-for-bit
      - [x] Asserts: PaCMAP ≠ LocalMAP at one seed
      - [x] Asserts: params are *read* — `n_neighbors` and `seed` each moved separately
            (the one that catches an option name druid silently ignores)
      - [x] Asserts: mean intra-blob 2-d distance < inter-blob
      - [x] Each assertion demonstrated failing before being trusted green
      - [x] Added to the CI `gpu` job — a behaviour check, so it reports rather than
            gating the page deploy, same as the other two

### ✅ Checkpoint: Foundation
- [x] `npm run check:druid` green — 11 checks
- [x] Misspelling `n_neighbors` trips the params assertion (`agree to 0.00e+0`)

Measured while landing this, and worth carrying into Task 3's cost estimate:
450 iterations at N=400 cost **156ms** (PaCMAP) / **209ms** (LocalMAP), and the
blob separation ratio sits at **0.055 / 0.122** against a 0.5 threshold, with a
scrambled layout scoring 1.004 — the gate has room on both sides.

## Phase 2 — The backend

- [x] **2. Worker + adapter, wired end to end** — M — deps: 1
      `src/druid-protocol.ts`, `src/druid-worker.ts`, `src/druid-cpu.ts`,
      `src/pacmap-webgpu.ts`, `src/main.ts`, `scripts/check-druid.ts`
      - [x] Worker: `init` builds a **`Matrix`** (not `Float64Array[]` — see below),
            constructs the class, calls `check_init()` explicitly, holds
            `generator(450)`, posts `ready`
      - [x] Worker: `step { to }` pulls to `to`, flattens `dr.Y.values` → `Float32Array`,
            posts it **transferred**
      - [x] Worker: drives `generator()`, not bare `next()` — that is what releases the
            WASM buffers
      - [x] Params mapped in `druid-protocol.ts`, shared with the check so a typo there
            fails CI rather than the browser
      - [x] `druidCPU()` owns a `STORAGE | VERTEX | COPY_DST | COPY_SRC` buffer;
            `runRange` uploads via `writeBuffer` — never a readback
      - [x] `EmbeddingRun` added as the async-capable supertype; `PacmapRun` extends it
            and re-narrows to `void`, so the GPU contract is unchanged rather than
            widened (deviation from the plan, which said to widen `PacmapRun` itself)
      - [x] Run loop awaits `pm.runRange(...)`
      - [x] `destroy()` terminates the worker; worker `onerror`/`onmessageerror`/thrown
            errors all reach `onStatus` instead of hanging the first await
      - [x] `?algo=pacmap-cpu` / `?algo=localmap-cpu` select the engine; bare
            `pacmap`/`localmap` still mean the GPU pair

**Two findings worth keeping.** Druid's `generator()` yields `this.projection`, and
for a `Float64Array[]` input that getter calls `to2dArray()` — N fresh arrays on
every one of the 450 iterations, all discarded, ~29M allocations at 65k. Passing a
`Matrix` makes `projection` return `this.Y` by reference. And `check:druid` now
proves the two decisions this rests on are behaviour-preserving:
`worker/chunked-generator-matches-transform` (uneven strides, bit-for-bit against
`transform()`) and `worker/matrix-input-matches-rows`.

- [ ] **3. Four-way dropdown and the cost hint** — S — deps: 2
      `src/main.ts`
      - [ ] `params.algorithm: AlgoKey` replaces `params.variant`, options map still
            annotated `Record<string, AlgoKey>` (a typo'd value must stay a build error)
      - [ ] Labels: `PaCMAP (GPU - custom)`, `LocalMAP (GPU - custom)`,
            `PaCMAP (CPU - druid)`, `LocalMAP (CPU - druid)`
      - [ ] `?algo=` takes the four keys; `pacmap`/`localmap` remain GPU aliases
      - [ ] `kNN algo` disabled under a CPU engine; `low_dist_thres` live for `localmap-cpu`
      - [ ] `estimateSetupSecs` CPU branch; hint refreshes on dropdown change
      - [ ] Status line names the engine alongside the variant

### Checkpoint: Core
- [ ] All four algorithms animate at N=2000
- [ ] Scrubbing and replay identical across engines
- [ ] `npm run check:ab` prints exactly 0 on a clean tree *first*, then confirms the GPU
      output is unchanged from `main`

## Phase 3 — Polish

- [ ] **4. Abort, so a long CPU run is recoverable** — S — deps: 3
      `src/main.ts`, `index.html`
      - [ ] Start becomes Stop mid-run; bumps `runGen`, calls `pm.destroy()`
      - [ ] N=20000 CPU run stops within a frame or two, last banked frame stays up
      - [ ] No orphaned worker afterwards (browser task manager)

- [ ] **5. Document it** — XS — deps: 4
      `CLAUDE.md`
      - [ ] `check:druid` in the commands block and in the checks-at-different-levels
            paragraph — it covers a level the other three do not
      - [ ] CPU backend deviations: PCA init vs Gaussian, druid's own exact kNN, f64 vs
            f32, upload-per-frame rather than readback
      - [ ] LGPL-3.0-or-later note on the dependency

### Checkpoint: Complete
- [ ] `npm run build`, `check:shaders`, `check:kernels`, `check:druid` all clean
- [ ] Five commits on `main`
