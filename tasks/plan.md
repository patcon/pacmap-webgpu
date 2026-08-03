# Implementation Plan: LocalMAP on WebGPU

## Context

`src/pacmap-webgpu.ts` implements PaCMAP with the optimizer fully GPU-resident. LocalMAP
(Wang et al., AAAI 2025) is the same authors' successor, and it ships in the same repo we
already track as the reference — `YingfanWang/PaCMAP`, `source/pacmap/pacmap.py` lines
1435–1830. We want it selectable here too.

Read against `pacmap()`, LocalMAP is a small delta, entirely inside phase 3:

1. **Modified NN gradient** (`pacmap_grad_nearby_recip_sqrt`, line 1493). For `itr > n1+n2`
   the near-pair coefficient is multiplied by `NN_coef_recip / sqrt(d_ij)`, where
   `NN_coef_recip = low_dist_thres / 2` and `d_ij = 1 + |y_i - y_j|²`. MN and FP
   coefficients are untouched. Everything else — the weight schedule (`find_weight` is
   byte-for-byte our `weightsAt`), Adam, the phase split, `n_neighbors`/`MN_ratio`/
   `FP_ratio` defaults — is identical to PaCMAP.
2. **Local graph adjustment** (`sample_FP_pair_nearby`, line 1467). At every `itr` with
   `itr > n1+n2 && itr % 10 == 0`, every point redraws its `n_FP` further partners
   uniformly at random, rejecting: itself, partners already drawn in this round, its own
   `n_NB` kNN partners, and any `j` with `‖y_i − y_j‖ > low_dist_thres` **in the
   embedding**. Up to 100 draws per slot; on exhaustion the previous partner is kept.
   `low_dist_thres` defaults to 10.

Change 1 is nearly free here. Change 2 is the whole difficulty: it mutates the pair set
25 times mid-run, and our CSR — with each pair duplicated into both endpoints so the
gradient kernel can gather instead of scatter (the thing that buys a float-atomic-free
shader) — is built once on the CPU before the first iteration.

**The intended outcome is that this stays true to the file's organizing constraint: once
setup finishes, positions never leave GPU memory.** The resample reads `Y`, so it must run
on the GPU, and the CSR rebuild must run there too. Every kernel below chains into the
same compute pass as `grad`/`adam`, so `runRange()` remains one command buffer with no
host round-trip and the demo's banked-frame playback is unaffected.

## Architecture Decisions

- **A variant option, not a fork.** `PacmapOptions.variant?: "pacmap" | "localmap"` plus
  `lowDistThres?: number` (default 10). Upstream `LocalMAP` subclasses `PaCMAP` and reuses
  its `fit`; mirroring that keeps one setup path, one kNN selection, one pair sampler. A
  thin `localmapWebGPU(...)` wrapper is exported for API parity.
- **The modified gradient needs no branch and no new entry point.** Fold it into the
  existing coefficient as `c = w_NB * 20/(10+dd)² * (A + B·inverseSqrt(dd))`, with
  `(A,B) = (1,0)` for PaCMAP and every LocalMAP iteration up to `n1+n2`, and
  `(0, lowDistThres/2)` for LocalMAP's `itr > n1+n2`. `A`/`B` ride in the already-unused
  `bc.z`/`bc.w` of the per-iteration params slot. One FMA, no divergence, one code path,
  and `weightsAt()`'s schedule is untouched.
- **FP moves out of the static CSR.** `buildCSR` keeps only NB and MN pairs (both still
  duplicated into both endpoints, tag bits unchanged). FP becomes two GPU-resident
  structures: a fixed `FpFwd` array of `N × n_FP` partner indices, and a rebuilt reverse
  CSR `FpRev` (`[0, N+1)` = offsets, then `N × n_FP` source indices) in one buffer. The
  gradient kernel gains two extra loops with the identical FP coefficient — the sign works
  out exactly as it does for the CSR duplication, since the displacement flips and the
  scalar (a function of `|d|²`) does not.
- **The reverse rebuild uses a single-thread serial prefix scan.** `N ≤ 65k` and the chain
  runs 25 times per run; a serial scan in one invocation is sub-millisecond at that size.
  A two-level parallel scan would save microseconds and cost a page of WGSL with real
  correctness subtleties. Same reasoning that makes the demo's bounds reduce a
  single-workgroup pass.
- **Reverse lists are sorted so runs stay reproducible.** The scatter uses an atomic
  cursor, so *which* slot each writer lands in is racy; a final per-point insertion sort
  over each reverse list removes that. Without it, f32 accumulation order in the gradient
  would vary run-to-run and LocalMAP would join `nndescentGPU` as a non-deterministic path.
  We do not want that — `knn: "cpu"` + `variant: "localmap"` must be bit-reproducible.
- **Storage-buffer budget.** `maxStorageBuffersPerShaderStage` defaults to 8. The gradient
  shader would go to `Y, Off, Adj, Grad, M, V, FpFwd, FpRev` = exactly 8. If a 9th is ever
  needed, interleave `M` and `V` into one buffer — same length, same access pattern.
- **The resample round index must be a dynamic-offset uniform, not a `writeBuffer`.**
  Queue writes are queue-ordered against the single `submit()`, so 25 pre-submit
  `writeBuffer` calls would all land before any dispatch and every resample would see the
  last value. The iteration index goes into the per-iteration params slot instead, which
  extends it from 32 to 48 bytes (`minBindingSize` 48; the 256-byte alignment already
  covers it).

## Task List

### Phase 1: LocalMAP gradient, end to end

#### Task 1: `variant` option and the modified NN coefficient
**Description:** Add `variant` and `lowDistThres` to `PacmapOptions`, export
`localmapWebGPU`, extend the params slot from 32 to 48 bytes with the `(A, B)` pair and
the iteration index, and change one line of `grad_main`'s near-pair branch. Honor the
reference's strict `itr > n1+n2` (with `(100,100,250)`, the modified gradient starts at
iteration 201, not 200).

**Acceptance criteria:**
- [ ] `pacmapWebGPU(..., { variant: "pacmap" })` produces the same embedding as today
      (`bc.z=1, bc.w=0` at every iteration reduces to the current expression exactly).
- [ ] `variant: "localmap"` applies `× (lowDistThres/2)·inverseSqrt(dd)` to near pairs
      only for `it > n1+n2`.
- [ ] `minBindingSize` updated to 48 in both the layout and `check-shaders.ts`'s
      `"uniform-dynamic"` case.

**Verification:**
- [ ] `npm run build`
- [ ] `npm run check:shaders`
- [ ] Manual: run the demo at N=5000 with both variants; LocalMAP shows visibly tighter,
      better-separated clusters in phase 3.

**Dependencies:** None. **Scope:** S (`src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`)

#### Task 2: Demo controls for the variant
**Description:** Add an `algorithm` dropdown and a `low_dist_thres` slider to the
`pacmap · next run` folder in `main.ts` (setup-time, so the folder-disabled-while-running
rule already covers them), seeded from `?algo=localmap` the same way `knnMethod` is seeded
from `?knn=`. Grey out `low_dist_thres` under `pacmap`. Fold the variant into the status
line next to the kNN label.

**Acceptance criteria:**
- [ ] Dropdown switches variant on the next run; `?algo=localmap` preselects it.
- [ ] `low_dist_thres` slider (range ~2–30, step 0.5, default 10) disabled under `pacmap`.
- [ ] Folder title and phase readout still accurate for both variants.

**Verification:** `npm run build`; manual — switch variants and re-run without reload.

**Dependencies:** Task 1. **Scope:** S (`src/main.ts`)

### Checkpoint: Phase 1
- [ ] `npm run build` and `npm run check:shaders` both clean
- [ ] Both variants run at N=5000 and N=65000; PaCMAP output unchanged from `main`
- [ ] Commit (one per task, straight to `main`)

### Phase 2: Local graph adjustment

#### Task 3: Split FP out of the static CSR — refactor, no new behavior
**Description:** `samplePairs` returns FP partners as a separate `N × n_FP` array (it
already generates exactly `n_FP` per point in order). `buildCSR` takes NB+MN only. Build
`FpFwd` and the reverse CSR **on the CPU** for now and upload both. Add the two FP loops
to `grad_main`. Also emit `NbFwd` (`N × n_NB`, i's own near partners) — the resample
kernel's reject list in Task 5.

**Acceptance criteria:**
- [ ] Embeddings match `main`'s to visual identity at a fixed seed (gradient accumulation
      order changes, so exact f32 equality is not expected; max coordinate delta should be
      small relative to the layout's extent).
- [ ] `FpRev` layout documented at its definition: `[0, N+1)` offsets, then indices.
- [ ] Gradient shader at 8 storage buffers; the M/V-interleave escape hatch noted in a
      comment rather than taken.

**Verification:** `npm run build`; `npm run check:shaders`; manual A/B against `main` at
N=5000 with `seed: 7`, `knn: "cpu"`.

**Dependencies:** Task 1. **Scope:** M (`src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`)

#### Task 4: The reverse-CSR rebuild kernels
**Description:** New `fpShaderSource(N, nNB, nFP, lowDistThres)` with the rebuild chain,
minus the draw itself: `fp_clear` (zero counts), `fp_count` (one thread per point,
`atomicAdd` on each partner's counter), `fp_scan` (one invocation, serial exclusive prefix
sum into the offsets region, then reseat the counters as write cursors), `fp_scatter`
(atomic cursor, write the source index), `fp_sort` (one thread per point, insertion-sort
its reverse list ascending). Reuse the counter-based `hash3` already in
`nndShaderSource`. Register in `shaderSources` and add a `check-shaders.ts` case with its
explicit layout — the CLAUDE.md rule.

**Acceptance criteria:**
- [ ] Running the chain over the CPU-built `FpFwd` reproduces the CPU-built reverse CSR
      exactly (verify once via a temporary readback, then delete it).
- [ ] `fp_sort` makes the reverse list order independent of scatter order.
- [ ] Every entry point has a pipeline built in `check:shaders`.

**Verification:** `npm run check:shaders` (this is the real gate — nothing here is
verifiable in a browser until Task 5); one-off readback comparison during development.

**Dependencies:** Task 3. **Scope:** M (`src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`)

#### Task 5: `fp_resample` — the draw itself
**Description:** One thread per point. For each of `n_FP` slots, up to 100 draws from
`hash3(seed, i*nFP + slot, round*128 + try) % N`; reject `j == i`, `j` already written to
a lower slot **this round**, `j` in `NbFwd[i]`, or `dot(y_i − y_j, ·) > lowDistThres²`
(square the threshold rather than taking a root). On exhaustion, leave the slot's previous
partner in place. Match the reference's try counter exactly — it increments on *every*
draw, including self- and duplicate-rejections, and bails at `count > 100`.

**Acceptance criteria:**
- [ ] Threshold compared in squared space; `lowDistThres` baked in as a WGSL constant.
- [ ] Deterministic at a fixed seed (the RNG is counter-based; the round comes from the
      iteration index in the params slot).
- [ ] A slot that exhausts its 100 draws keeps its old partner rather than writing garbage
      or `0`.

**Verification:** `npm run check:shaders`; manual — with `lowDistThres` set very small,
almost every slot should exhaust and the layout should barely differ from Task 3's.

**Dependencies:** Task 4. **Scope:** S (`src/pacmap-webgpu.ts`)

#### Task 6: Wire the chain into `runRange`
**Description:** Under `variant: "localmap"`, for each encoded iteration with
`it > n1+n2 && it % 10 == 0`, append `fp_resample → fp_clear → fp_count → fp_scan →
fp_scatter → fp_sort` to the same compute pass, **after** `adam` (matching the reference's
ordering inside its loop). No barriers needed — WebGPU synchronizes between dispatches in
a pass. Destroy the new buffers in `destroy()`.

**Acceptance criteria:**
- [ ] `runRange()` still submits exactly one command buffer and performs no readback.
- [ ] Under `variant: "pacmap"` not a single FP kernel is encoded, and no FP buffers are
      allocated beyond `FpFwd`/`FpRev`.
- [ ] Resampling fires at iterations 210, 220, … 440 for `(100,100,250)` — not at 200.
- [ ] `destroy()` releases every new buffer.

**Verification:** `npm run build`; `npm run check:shaders`; manual — run LocalMAP at
N=65000, scrub the timeline through phase 3 and watch clusters tighten at the resample
boundaries; confirm the banked-frame count and MB figure are unchanged from a PaCMAP run
at the same N.

**Dependencies:** Task 5. **Scope:** S (`src/pacmap-webgpu.ts`)

### Checkpoint: Phase 2
- [ ] Both automated checks clean
- [ ] Two runs at a fixed seed with `knn: "cpu"`, `variant: "localmap"` produce identical
      layouts (reproducibility — the thing `fp_sort` is there for)
- [ ] LocalMAP at N=65000 completes with no measurable regression in the reported
      per-iteration time

#### Task 7: Documentation
**Description:** Update `CLAUDE.md` (a LocalMAP subsection under the CPU/GPU split,
covering the FP-out-of-CSR change, the serial scan and why, and `fp_sort`'s role in
reproducibility) and `README.md` (a variant row in "What runs where", the `?algo=` switch,
and LocalMAP's place among the documented deviations — Gaussian init still applies).

**Acceptance criteria:**
- [ ] Both files describe the FP split and the resample chain accurately.
- [ ] The "add a case to `check-shaders.ts` whenever you add a shader" rule reflects the
      new source.
- [ ] No speed claims that were not measured.

**Verification:** Read back against the final code.

**Dependencies:** Task 6. **Scope:** S (`CLAUDE.md`, `README.md`)

### Checkpoint: Complete
- [ ] All acceptance criteria met, seven commits on `main`

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `fp_sort`'s insertion sort is O(deg²) and a dense-region point could attract a large reverse degree | Med | Reverse degree averages `n_FP` (≈20) and the low-distance filter spreads draws across a neighborhood rather than concentrating them the way kNN hub points do. Measure the phase-3 per-iteration time in Task 6; if it regresses, the documented fallback is to drop `fp_sort` and accept f32-order non-determinism. |
| 9th storage buffer needed in the gradient shader | Low | Interleave `M` and `V` — same length, same access pattern. Noted in a Task 3 comment. |
| `low_dist_thres = 10` is calibrated for the reference's PCA init; we use Gaussian init, so the embedding may sit at a different scale in phase 3 | Med | It is a slider, and phase 3 starts after 200 iterations of the same weight schedule, so the scale should be comparable. Compare layouts across `low_dist_thres` values in the Phase 2 checkpoint. |
| Nothing in phase 2 is verifiable headlessly beyond compilation | Med | `check:shaders` gates every entry point and its real bind-group layout — the failure mode CLAUDE.md warns about (a WGSL error that looks like a plausible blob). The one-off readback in Task 4 checks the rebuild against a CPU-built reference before it goes live. |

## Verification

Run after every task:

```
npm run build          # tsc --noEmit && vite build
npm run check:shaders   # compiles all WGSL under Dawn, builds every pipeline
```

End to end, in a WebGPU browser (`npm run dev`):

1. `?algo=pacmap` at N=5000, seed 7 — output must match `main`'s.
2. `?algo=localmap` at the same N and seed — clusters tighten and separate in phase 3.
3. Re-run 2 without changing anything — identical layout (reproducibility).
4. `?algo=localmap` at N=65000 — completes, timeline scrubs, per-iteration time in the
   status line comparable to PaCMAP's.
5. `?knncheck=1` still reports the three kNN backends unchanged (nothing in this plan
   touches that path).
