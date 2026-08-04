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

- [x] **3. Split FP out of the static CSR (refactor, no new behavior)** — M — deps: 1
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`
      - [x] `samplePairs` returns FP partners as a separate `N × n_FP` array
      - [x] `buildCSR` takes NB + MN only; `T_FP` dropped, only two kinds still packed
      - [x] `FpFwd` + reverse CSR `FpRev` (`[0,N+1)` offsets, then indices) built on the **CPU** for now
      - [x] `NbFwd` (`N × n_NB`, short rows padded with `N`) emitted for Task 5's reject list
      - [x] `grad_main` gains the FP forward + FP reverse loops, sharing one `fpForce`
      - [x] gradient shader at 8 storage buffers; M/V-interleave escape hatch noted in a comment
      - [x] A/B **measured headlessly under Dawn**, not by eye: N=800 D=20, seed 7,
            `knn: "cpu"`, against the pre-refactor bundle from git.
            iter 1 rel 1.9e-5 · iter 450 rel 2.9e-5 (max|Δ| over layout extent).
            Gate proven to have teeth: dropping the reverse-FP loop gives rel 0.50.

- [x] **4. Reverse-CSR rebuild kernels** — M — deps: 3
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`
      - [x] new `fpShaderSource(N, nFP)` — **not** the planned
            `(N, nNB, nFP, lowDistThres)`: those two only serve `fp_resample`, so
            they arrive with it in Task 5 rather than sitting unused here
      - [x] `fp_clear` / `fp_count` / `fp_scan` (single-invocation serial exclusive scan, then reseat counters as cursors) / `fp_scatter` / `fp_sort`
      - [ ] ~~`hash3` reused~~ — belongs to `fp_resample`, moved to Task 5
      - [x] registered in `shaderSources` + a `check-shaders.ts` case with its explicit layout
      - [x] chain checked against an **independently written** CPU oracle over 4
            cases (uniform, hub, empty-lists, ragged-N) — exact match on every
            word of both regions. Gate proven: an inclusive-vs-exclusive scan
            fails all 4; skipping `fp_sort` fails 3 of 4, confirming the atomic
            scatter really is unordered and the sort is load-bearing

- [x] **5. `fp_resample` — the draw** — S — deps: 4
      `src/pacmap-webgpu.ts`
      - [x] ≤100 draws per slot from `hash3(seed, i*nFP + slot, round*128 + try) % N`
      - [x] rejects self, this-round duplicates (held in registers, so the scan sees
            only this round's picks as the reference's `result[]` does), `NbFwd[i]`,
            and `|y_i − y_j|² > lowDistThres²`
      - [x] **deviation:** the try budget bounds *every* rejection path. Upstream's
            self-hit and distance-failure branches `continue` past its `count > 100`
            escape, so a point with no eligible partner inside `low_dist_thres` never
            terminates. Harmless-ish on a CPU, takes out the device on a GPU.
      - [x] exhausted slot keeps its previous partner
      - [x] behavioural test over line geometry (spacing 0.1, so "within thres" is
            exactly "within 10 indices") + one deliberately isolated point:
            all four rejection rules hold on 1939 redrawn slots, 61 exhausted slots
            kept, isolated point terminates and keeps its partners, same round
            reproduces bit-for-bit, different round differs.
            Gate proven: removing the distance filter → 1907 violations; removing
            the near-partner check → 287; ignoring `round` → caught by the
            different-round assertion.

- [x] **6. Wire the chain into `runRange`** — S — deps: 5
      `src/pacmap-webgpu.ts`
      - [x] chain appended after `adam` for `it > n1+n2 && it % 10 == 0` under `localmap`
      - [x] fires at 210, 220 … 440 for `(100,100,250)` — **not** at 200. That is
            **24** fires, not the 25 the plan said; comments corrected.
      - [x] still one command buffer per `runRange`, still zero readback
      - [x] nothing extra encoded or allocated under `variant: "pacmap"` — the whole
            `fp` object is `null` there
      - [x] `destroy()` releases every new buffer
      - [x] end-to-end, headless: pacmap unchanged vs the pre-refactor baseline
            (2.9e-5); **localmap bit-exact reproducible across processes (0.0e+0)**,
            which is the Phase 2 checkpoint criterion and what `fp_sort` exists for;
            localmap ≠ pacmap (0.10); `lowDistThres` 2 vs 30 differ (0.64).
            Gate proven with a 0.0e+0 sanity control: boundary `>=` → 0.109,
            chain never encoded → 0.084, every-iteration → 0.111.

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
