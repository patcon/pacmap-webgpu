# TODO: LocalMAP on WebGPU

Full detail, rationale, and acceptance criteria in [plan.md](plan.md).
One commit per task, straight to `main`. Both gates after every task:
`npm run build` and `npm run check:shaders`.

## Phase 1 — LocalMAP gradient, end to end

- [x] **1. `variant` option + modified NN coefficient** — S — deps: none
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`
      - [x] `variant?: "pacmap" | "localmap"` and `lowDistThres?: number` (default 10) on `PacmapOptions`
      - [x] `localmapWebGPU()` wrapper exported
      - [x] params slot 32 → 48 bytes, carrying `(A, B)` and the iteration index; `minBindingSize` bumped in the layout *and* in `check-shaders.ts`
      - [x] `grad_main` near-pair coefficient becomes `w_NB * 20/(10+dd)² * (A + B*inverseSqrt(dd))`
      - [x] `(A,B) = (0, lowDistThres/2)` only for `it > n1+n2` under `localmap`; `(1,0)` everywhere else
      - [x] PaCMAP output unchanged from `main` — by construction: `(1,0)` makes the
            trailing factor exactly `1.0 + 0.0*inverseSqrt(dd)`, and `dd >= 1` always,
            so there is no NaN/Inf path and `x * 1.0` is exact in IEEE754

- [x] **2. Demo controls for the variant** — S — deps: 1
      `src/main.ts`
      - [x] `algorithm` dropdown in the folder (retitled `algorithm · next run`), seeded from `?algo=localmap`
      - [x] `low_dist_thres` slider (2–30, step 0.5, default 10), disabled under `pacmap`
      - [x] variant shown in the status line beside the kNN label
      - [x] phase readout reads `3 · attract-repel + local graph` under localmap
      - [x] dropdown option *values* type-checked (`Record<string, Variant>`) — Tweakpane
            types `options` as plain strings, so a typo there silently ran PaCMAP

### ✅ Checkpoint: Phase 1
- [ ] Both gates clean
- [ ] Both variants run at N=5000 and N=65000
- [ ] PaCMAP output verified unchanged from `main`

## Phase 2 — Local graph adjustment

- [ ] **3. Split FP out of the static CSR (refactor, no new behavior)** — M — deps: 1
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`
      - [ ] `samplePairs` returns FP partners as a separate `N × n_FP` array
      - [ ] `buildCSR` takes NB + MN only (tag bits unchanged)
      - [ ] `FpFwd` + reverse CSR `FpRev` (`[0,N+1)` offsets, then indices) built on the **CPU** for now
      - [ ] `NbFwd` (`N × n_NB`) emitted for Task 5's reject list
      - [ ] `grad_main` gains the FP forward + FP reverse loops
      - [ ] gradient shader at 8 storage buffers; M/V-interleave escape hatch noted in a comment
      - [ ] A/B against `main` at N=5000, seed 7, `knn: "cpu"` — visually identical

- [ ] **4. Reverse-CSR rebuild kernels** — M — deps: 3
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`
      - [ ] new `fpShaderSource(N, nNB, nFP, lowDistThres)`
      - [ ] `fp_clear` / `fp_count` / `fp_scan` (single-invocation serial exclusive scan, then reseat counters as cursors) / `fp_scatter` / `fp_sort`
      - [ ] `hash3` reused from `nndShaderSource`
      - [ ] registered in `shaderSources` + a `check-shaders.ts` case with its explicit layout
      - [ ] one-off readback confirms the chain reproduces the CPU-built reverse CSR exactly, then deleted

- [ ] **5. `fp_resample` — the draw** — S — deps: 4
      `src/pacmap-webgpu.ts`
      - [ ] ≤100 draws per slot from `hash3(seed, i*nFP + slot, round*128 + try) % N`
      - [ ] rejects self, this-round duplicates, `NbFwd[i]`, and `|y_i − y_j|² > lowDistThres²`
      - [ ] try counter increments on *every* draw and bails at `> 100`, matching the reference
      - [ ] exhausted slot keeps its previous partner

- [ ] **6. Wire the chain into `runRange`** — S — deps: 5
      `src/pacmap-webgpu.ts`
      - [ ] chain appended after `adam` for `it > n1+n2 && it % 10 == 0` under `localmap`
      - [ ] fires at 210, 220 … 440 for `(100,100,250)` — **not** at 200
      - [ ] still one command buffer per `runRange`, still zero readback
      - [ ] nothing extra encoded or allocated under `variant: "pacmap"`
      - [ ] `destroy()` releases every new buffer

### ✅ Checkpoint: Phase 2
- [ ] Both gates clean
- [ ] Two LocalMAP runs at fixed seed + `knn: "cpu"` give identical layouts (reproducibility)
- [ ] N=65000 completes with no measurable per-iteration regression

- [ ] **7. Documentation** — S — deps: 6
      `CLAUDE.md`, `README.md`
      - [ ] LocalMAP subsection: FP-out-of-CSR, the serial scan and why, `fp_sort`'s role
      - [ ] README: variant row in "What runs where", `?algo=` switch, deviations
      - [ ] the "add a `check-shaders.ts` case per shader" rule covers the new source
      - [ ] no unmeasured speed claims

### ✅ Checkpoint: Complete
- [ ] All acceptance criteria met, seven commits on `main`
