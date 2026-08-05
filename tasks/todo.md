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

- [ ] **2. `nComponents` in the library** — L — deps: 1
      `src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`, `scripts/check-kernels.ts`
      - [ ] `PacmapOptions.nComponents?: 2 | 3`, default 2 (note: `D` is the *input* dim)
      - [ ] `alias V` + `const DIM` + `ld(i)` in `shaderSource` and `fpShaderSource`;
            every `vec2<f32>(Y[...])` becomes `ld(...)`, zero-init becomes `V()`
      - [ ] Gradient stores emitted per component; `adam_main` loop bound `DIM`
      - [ ] `Y0`, `gradBuf`, `mBuf`, `vBuf`, `read()` staging sized `N*d`
      - [ ] `EmbeddingRun.positions` doc says `N x d`
      - [ ] `check-shaders.ts` case table iterates `d ∈ {2, 3}`
      - [ ] `check-kernels.ts` runs resample/reverse at `d = 3`; on-a-line fixture keeps
            y = z = 0 so its "within thres = within ten indices" invariant survives
      - [ ] New genuinely-3D kernel case: candidate inside thres in x/y, outside via z
      - [ ] Host distance oracle uses all `d` components
      - [ ] Each new assertion demonstrated failing first (e.g. drop z from `ld`)
      - [ ] `check:ab -- <pre-task-ref>` prints 0 for `--variant=pacmap` *and*
            `--variant=localmap`

### ⬜ Checkpoint: Foundation
- [ ] `build`, `check:shaders`, `check:kernels`, `check:druid` all green
- [ ] `check:ab` proves the 2D path unmoved on both variants
- [ ] Library computes in 3D; nothing draws it yet

## Phase 2 — The renderer

- [ ] **3. Renderer + pane dropdown + camera mode** — L — deps: 2
      `src/shaders.ts`, `src/main.ts`
      - [ ] `renderWGSL(d)`: `@location(0) p : vec2/vec3<f32>`, world built per dim
      - [ ] Quad expansion unchanged — radius stays screen-constant under zoom
      - [ ] Pipeline `arrayStride: 4*d` / `float32x{d}`; `frameBytes = N * 4 * d`
      - [ ] `N x 2` / 520KB history comment updated (frames are 50% larger in 3D)
      - [ ] `components` dropdown in `dimensional reduction`, options annotated
            `Record<string, 2 | 3>`; seeded from `?dims=`; read at the top of `go()`
      - [ ] `setCameraMode(d)`: `enableRotate = d === 3`, `mouseButtons` swaps between the
            2D remap and ogl's defaults; called from the dropdown and from `go()`
      - [ ] The two "the 3D step does X" comments on the camera updated to describe what
            it now does
      - [ ] Browser: 3D at 2k — left-drag orbits, right-drag pans, wheel zooms,
            double-click resets, scrubbing replays and still rotates
      - [ ] Browser: back to 2D — left-drag pans again, old framing intact

### ⬜ Checkpoint: 3D renders
- [ ] End-to-end 3D on the GPU engine, both mouse mappings confirmed by hand

## Phase 3 — Depth and the second engine

- [ ] **4. Depth occlusion + toggle** — M — deps: 3
      `src/shaders.ts`, `src/main.ts`
      - [ ] `depth24plus` texture created in `resize()`, old one destroyed
      - [ ] Second pipeline with `depthWriteEnabled: true`, `depthCompare: "less"`;
            `encodeRender` picks it and attaches the depth view (`clear`, 1.0)
      - [ ] `occlude` flag in the `View` uniform's existing pad slot; when set the
            fragment hard-discards outside the disc and returns alpha 1.0 — no
            semi-transparent fragment ever writes depth (the dark-streak failure mode)
      - [ ] `view.occlusion` checkbox, default on, greyed in 2D
      - [ ] `check:shaders` builds both render pipelines at both dims
      - [ ] Browser: occlusion on — near clusters hide far ones, no dark streaks; off —
            back to the blended haze; 2D unaffected

- [ ] **5. 3D on the druid CPU engine** — M — deps: 2
      `src/druid-protocol.ts`, `src/druid-worker.ts`, `src/druid-cpu.ts`,
      `scripts/check-druid.ts`
      - [ ] `nComponents` threaded through `DruidCpuOptions` → init command →
            `buildDruidParams` (replaces the hard-coded `d: 2`)
      - [ ] Worker snapshot `Float32Array(N*d)`; `positions` buffer `N*4*d`; `read()`
            fallback likewise
      - [ ] `check:druid` case at `d: 3`: output length `N*3`, finite, blob separation
            holds in 3-d — the case that proves `d` is *forwarded*, not just spelled right
      - [ ] Assertion demonstrated failing first (hard-code `d: 2` back)
      - [ ] Browser: `LocalMAP (CPU - druid)` at 3D, 500 points, renders and scrubs

### ⬜ Checkpoint: Full
- [ ] All four checks green
- [ ] One run per engine × dimensionality (8 combinations) confirmed in the browser
- [ ] Stop mid-run on a CPU 3D run keeps its partial trace

## Phase 4 — Documentation

- [ ] **6. `CLAUDE.md` + `README.md`** — S — deps: 5
      - [ ] New Architecture subsection: the `alias V`/`ld()` scheme vs. a component loop,
            the fixed 32-byte bounds and why `vec3` padding makes it free, occlusion over
            sorting (a CPU depth sort would mean a per-frame readback), the camera's three
            2D flags now mode-switched rather than fixed
      - [ ] Coverage-gaps paragraph updated: the camera gap now covers rotation and the
            occlusion toggle too
