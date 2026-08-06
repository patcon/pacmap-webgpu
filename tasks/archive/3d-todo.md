# TODO: 3D embeddings — a `components` control (2D / 3D)

Full detail, rationale, and acceptance criteria in [plan.md](plan.md).
One commit per task, straight to `main`. Gates after every task: `npm run build`, plus the
checks named on the task. `npm run check:ab` on a clean tree must print exactly **0**
throughout — the 2D path is meant to come out bit-identical.

## Phase 1 — Foundation (headless, nothing user-visible)

- [x] **1. Widen bounds to a fixed 32 bytes** — S — deps: none
      `src/shaders.ts`, `src/main.ts`, `scripts/check-kernels.ts`, `tasks/`
      - [x] `boundsWGSL`: `sLo`/`sHi` become `vec4<f32>`; reduce writes `B[0..7]`, z/w = 0
      - [x] `struct Bounds { lo : vec4<f32>, hi : vec4<f32> }`; vertex reads `.xyz`
      - [x] `BOUNDS_BYTES = 32` replaces every literal `16` (buffers, history slot stride,
            staging readback, `copyBufferToBuffer` sizes)
      - [x] `readSeedBounds()` requires length 8; a stale 4-length entry is ignored
      - [x] **New:** `check:kernels` gains a `bounds/*` section — the reduce had no
            behaviour check at all, only compilation. Oracle is a CPU min/max; output
            buffer is poison-filled so an unwritten word is a mismatch
      - [x] Spike fixtures at indices 0/1/255/256/257/512/1999 rather than one random
            input: a wrong grid stride drops one index class and random extremes are
            almost never in it
      - [x] Both mutations demonstrated failing first — stride `256u`→`257u` trips
            `spike-at-256`; reduction tree `128u`→`64u` trips `spike-at-255`, `-1999`
            and `grid-stride`
      - [x] `check:shaders`, `check:kernels` (25), `check:druid` (17) green
      - [x] `check:ab` vs. `HEAD` prints 0 — bit-identical, the library is untouched
      - [ ] **Browser (yours):** a run looks identical, `auto zoom` on and off

- [x] **2. `nComponents` in the library** — L — deps: 1
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`, `scripts/check-kernels.ts`
      - [x] `PacmapOptions.nComponents?: 2 | 3`, default 2 (note: `D` is the *input* dim)
      - [x] `alias Pt` + `const DIM` + `ld(i)` in `shaderSource` and `fpShaderSource`;
            every `vec2<f32>(Y[...])` becomes `ld(...)`, zero-init becomes `Pt()`.
            **Not `V`** — the gradient shader already binds Adam's second moment under
            that name and WGSL reads the collision as a redeclaration
      - [x] Gradient stores emitted per component; `adam_main` loop bound `DIM`
      - [x] `Y0`, `gradBuf`, `mBuf`, `vBuf`, `read()` staging sized `N*d`
      - [x] `EmbeddingRun.positions` doc says `N x d`
      - [x] `check-shaders.ts` case table iterates `d ∈ {2, 3}` (8 cases)
      - [x] `check-kernels.ts` runs the resample at `d = 2` *and* `d = 3`; the line
            fixture now runs **diagonally** (0.1/√d per component per step) rather than
            along x with y = z = 0 — same "within thres = within ten indices" invariant,
            but every component is load-bearing, so a plane-only distance fails it
      - [x] New `e2e-3d/*` cases per variant: length, finite, reproducible, and
            `third-axis-is-optimized` (z spread within 5x of x/y — an unoptimized z is
            still finite and non-constant, just stuck at the init's 1e-4 scale)
      - [x] Host distance oracle uses all `d` components
      - [x] Both mutations demonstrated failing first — Adam looping to `2u` and `ld`
            zeroing z each trip `third-axis-is-optimized`; the latter also trips
            `fp-resample-3d/respects-low-dist-thres`
      - [x] `check:ab -- HEAD` prints 0 for `--variant=pacmap` *and* `--variant=localmap`
            — bit-identical, both

### ✅ Checkpoint: Foundation
- [x] `build`, `check:shaders` (8), `check:kernels` (41), `check:druid` (17) all green
- [x] `check:ab` proves the 2D path unmoved on both variants — bit-identical
- [x] Library computes in 3D; nothing draws it yet

Measured while landing this, and worth carrying into Task 3: the 3D layouts come out
near-isotropic — per-axis spreads 29.8 / 29.9 / 29.1 (pacmap) and 23.2 / 25.2 / 23.3
(localmap) at N=400 — so the single `span` the renderer maxes over frames a 3D run
sensibly and no per-axis scaling is needed. It also sets the threshold's headroom: the
check demands z > 0.2x the widest axis, against a real ratio of ~0.97 and an
unoptimized one of 3e-5.

## Phase 2 — The renderer

- [x] **3. Renderer + pane dropdown + camera mode** — L — deps: 2
      `src/shaders.ts`, `src/main.ts`, `scripts/check-shaders.ts`
      - [x] `renderWGSL(d)`: `@location(0) p : vec2/vec3<f32>`, world built per dim
      - [x] Quad expansion unchanged — radius stays screen-constant under zoom
      - [x] Pipeline `arrayStride: 4*d` / `float32x{d}`; `frameBytes = N * 4 * d`
      - [x] `N x 2` / 520KB history comment updated (a 3D frame is 780KB at 65k, so a
            3D run banks fewer frames or strides coarser at the same budget)
      - [x] `components` dropdown in `dimensional reduction`, options annotated
            `Record<string, Components>`; seeded from `?dims=3`; read at the top of `go()`
      - [x] `check-shaders` builds the render pipeline at both dims (9 cases) — the
            `float32x3` attribute and the 12-byte stride are validated, not assumed
      - [x] **`setCameraMode(d)` rebuilds the controller rather than reconfiguring it.**
            ogl captures `enableRotate` as a constructor closure variable, so
            `orbit.enableRotate = true` type-checks against `OrbitOptions` and does
            nothing. `remove()` detaches every listener and the new controller seeds its
            sphericals from the camera's current position, so the view carries across
      - [x] 3D restores ogl's `{ORBIT:0, ZOOM:1, PAN:2}`; 2D keeps the left-pan remap.
            Orbit suppresses the context menu itself, so right-drag pan is free
      - [x] The camera comments that described this as pending now describe what it does
      - [x] **CPU engine stays 2D-only until Task 5:** `syncAlgorithm` greys the
            dropdown and forces it back to 2, rather than letting a run be asked for 3D
            and quietly handed a plane
      - [ ] **Browser (yours):** 3D at 2k — left-drag orbits, right-drag pans, wheel
            zooms, double-click resets, scrubbing replays and still rotates
      - [ ] **Browser (yours):** back to 2D — left-drag pans again, old framing intact
      - [ ] **Browser (yours):** expect an unsorted haze in 3D — no depth buffer until
            Task 4. Structure should still be legible as you rotate

### ⬜ Checkpoint: 3D renders
- [ ] End-to-end 3D on the GPU engine, both mouse mappings confirmed by hand

## Phase 3 — Depth and the second engine

- [x] **4. Depth occlusion + toggle** — M — deps: 3
      `src/shaders.ts`, `src/main.ts`, `scripts/check-shaders.ts`
      - [x] `depth24plus` texture created in `resize()`, sized with the canvas, old one
            destroyed (dimensions are fixed at creation, so it is recreated not resized)
      - [x] Second pipeline with `depthWriteEnabled: true`, `depthCompare: "less"`,
            built only in 3D and sharing the 2D descriptor
      - [x] `occludingNow()` is the single source for all three things that must agree:
            which pipeline is set, whether the pass carries a depth attachment, and the
            uniform flag. A pipeline with depth state and a pass without an attachment
            is a validation error; the flag disagreeing with either is the dark streaks
      - [x] `occlude` flag took the `View` uniform's existing pad slot — no size change
      - [x] Fragment paints a solid disc when occluding, so nothing semi-transparent
            ever writes depth. The feathered edge cannot survive that, which is
            precisely why this is a toggle and not the only mode
      - [x] `view.occlusion` checkbox, default on, greyed in 2D via `syncComponents`
      - [x] `check:shaders` builds the depth pipeline (10 cases), and it was shown
            failing first by handing it a colour format for its depth state
      - [x] **Bug found in the browser, then covered:** both pipelines were built with
            `layout: "auto"`, which mints a *fresh* bind group layout per pipeline —
            never compatible with another's. The bind group built from `renderPipe`
            could not be set under `renderPipeDepth`, so the draw was dropped at
            validation and every point vanished the moment occlusion was ticked. Fixed
            with one explicit `GPUBindGroupLayout` shared by both
      - [x] New `render-3d-occlusion-draw` case encodes a real draw per mode via a
            **render bundle encoder** — it takes attachment formats rather than views,
            which validates bind-group-vs-layout and depth-state-vs-format, and is the
            only way to encode a draw under these bindings at all (@kmamal/gpu's
            `createView()` sends a component swizzle this adapter lacks, so no texture
            view can be made and no render pass begun). Restoring `layout: "auto"`
            reproduces the browser's exact error message
      - [ ] **Browser (yours):** occlusion on — near clusters hide far ones, no dark
            streaks; off — back to the blended haze; 2D unaffected and greyed

- [x] **5. 3D on the druid CPU engine** — M — deps: 2
      `src/druid-protocol.ts`, `src/druid-worker.ts`, `src/druid-cpu.ts`,
      `scripts/check-druid.ts`
      - [x] `nComponents` threaded through `DruidCpuOptions` → init command →
            `buildDruidParams` (replaces the hard-coded `d: 2`)
      - [x] Worker snapshot `Float32Array(N*d)`; `positions` buffer `N*4*d`; `read()`
            fallback likewise
      - [x] `check:druid` case at `d: 3`: output length `N*3`, finite, blob separation
            generalized to d — the case that proves `d` is *forwarded*, not just spelled
            right (20 checks)
      - [x] Demonstrated failing first: hard-coding `d: 2` back leaves 400 non-finite
            values and a NaN spread, tripping all three
      - [x] The Task 3 clamp is lifted — `components` is live under every engine now
      - [ ] **Browser (yours):** `LocalMAP (CPU - druid)` at 3D, 500 points, renders
            and scrubs

### ✅ Checkpoint: Full
- [x] All four checks green — build, shaders 11, kernels 41, druid 20
- [x] Confirmed in the browser by the author across the run of the work — including
      the two bugs the checks could not see (occlusion drawing nothing, rotation
      dragging 6.7x short), both since covered or fixed

## Phase 4 — Documentation

- [x] **6. `CLAUDE.md` + `README.md`** — S — deps: 5
      - [x] New `2D or 3D (components)` section: the `alias Pt`/`ld()` scheme vs. a
            component loop (and the `V` collision), the fixed 32-byte bounds, occlusion
            over sorting, the 50%-larger frame, and the shared explicit bind group layout
      - [x] Camera section rewritten: three mode-keyed differences rather than three
            things "keeping it 2D"; the controller-rebuild finding; both speeds needing
            the inertia compensation, not just pan
      - [x] `check:shaders` description covers both dims and the draw-encoding case,
            including why it uses a render bundle encoder
      - [x] Coverage-gaps paragraph now names rotation, `components` and `occlusion`
      - [x] Druid section notes `d` is forwarded and checked at 3
      - [x] Point-radius paragraph points at `TODO.md` for the depth-scaling idea
      - [x] README gains a `Two or three dimensions` section: `?dims=3`, the mouse map,
            what `occlusion` does and why it is not a sort, the history cost
