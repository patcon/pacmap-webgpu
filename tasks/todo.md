# TODO: CPU comparison algorithms via DruidJS

Full detail, rationale, and acceptance criteria in [plan.md](plan.md).
One commit per task, straight to `main`. Gates after every task: `npm run build`, plus
`npm run check:druid` once Task 1 lands.

## Phase 1 — Foundation

- [ ] **1. Dependency + headless check (`check:druid`)** — S — deps: none
      `package.json`, `scripts/check-druid.ts`
      - [ ] `@saehrimnir/druidjs` added; `check:druid` script (esbuild → node, no Dawn)
      - [ ] Synthetic 4-blob dataset (~400 points, 20-d)
      - [ ] Asserts: N×2 and finite, both classes
      - [ ] Asserts: fixed seed reproduces bit-for-bit
      - [ ] Asserts: PaCMAP ≠ LocalMAP at one seed
      - [ ] Asserts: params are *read* — two runs differing only in `n_neighbors` differ
            (the one that catches an option name druid silently ignores)
      - [ ] Asserts: mean intra-blob 2-d distance < inter-blob
      - [ ] Each assertion demonstrated failing before being trusted green

### Checkpoint: Foundation
- [ ] `npm run check:druid` green
- [ ] Misspelling `n_neighbors` trips the params assertion

## Phase 2 — The backend

- [ ] **2. Worker + adapter, wired end to end** — M — deps: 1
      `src/druid-worker.ts`, `src/druid-cpu.ts`, `src/pacmap-webgpu.ts`, `src/main.ts`
      - [ ] Worker: `init` widens `Z` to `Float64Array[]` rows, constructs the class,
            calls `check_init()` explicitly, holds `generator(450)`, posts `ready`
      - [ ] Worker: `step { to }` pulls to `to`, flattens `dr.Y.values` → `Float32Array`,
            posts it **transferred**
      - [ ] Worker: drives `generator()`, not bare `next()` — that is what releases the
            WASM buffers
      - [ ] Params mapped: `n_neighbors`, `MN_ratio`, `FP_ratio`, `seed`, `d: 2`,
            `num_iters: [100,100,250]`, `apply_pca: false`, `knn: null`,
            `low_dist_thres` for LocalMAP only
      - [ ] `druidCPU()` owns a `STORAGE | VERTEX | COPY_DST | COPY_SRC` buffer;
            `runRange` uploads via `writeBuffer` — never a readback
      - [ ] `PacmapRun.runRange` widened to `void | Promise<void>`; run loop awaits
      - [ ] `destroy()` terminates the worker; worker errors reach `onStatus`

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
