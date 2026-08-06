# Plan — 3D embeddings: a `components` control (2D / 3D)

## Context

The demo embeds and renders in exactly two dimensions today, and that `2` is hard-coded
in about forty places across the library, the druid worker, the renderer and the checks.
The camera was already built for this step: it is a real ogl perspective camera uploading
a view-projection `mat4`, the WebGL→WebGPU clip-z remap is already applied in `resize()`
specifically so depth testing can be turned on later, and `main.ts:732` names the exact
three things holding the canvas to 2D (`enableRotate: false`, the shader's `0.0` world z,
the non-default `mouseButtons` remap).

The ask: a setup-time **`components`** dropdown (2D / 3D) in the `dimensional reduction`
folder, with a renderer that actually works in 3D. Depth is painted by **occlusion** —
a real depth buffer, chosen over fog and perspective size-falloff — with a toggle so the
occluded look can be compared against today's blended cloud. Mouse mapping in 3D goes
back to ogl's defaults (left orbit, right pan); 2D keeps its left-pan remap.

Reference read: the sibling project `~/scratch/marimo-pacmap-animation` renders 3D in
`app/app.js` with only two cues — perspective point shrink and a per-frame CPU sort of a
`Uint32` index buffer — with `depthTest: false`. That sort is exactly what we can't do
(it would mean reading positions back per frame, which the whole architecture forbids).
A depth buffer gets correct occlusion for free, with no sort and no readback. Its
`render_fpl.py:62-79` also records the failure mode to avoid: semi-transparent geometry
writing depth produces dark streaks that cull whole clusters. Hence opaque discs when
occlusion is on, rather than depth-writing the anti-aliased edge.

**Non-goal:** the 2D path must come out bit-identical. `nComponents` defaults to 2
everywhere, and `npm run check:ab` on a clean tree must keep printing exactly 0.

---

## Design decisions

**Layout: packed stride `d`.** Positions stay one flat `array<f32>` at `N*d`; the vertex
buffer becomes `arrayStride: 12` / `float32x3` in 3D. No `vec3` padding anywhere, because
nothing puts a `vec3` in a struct — the only place alignment bites is the bounds uniform,
handled below.

**WGSL stays vectorized, not looped.** Both generated shaders (`shaderSource`,
`fpShaderSource`) gain, at the top of the template:

```wgsl
alias V = vec${d}<f32>;
const DIM : u32 = ${d}u;
fn ld(i : u32) -> V { return V(${comps.map(c => `Y[DIM * i + ${c}u]`).join(", ")}); }
```

Every `vec2<f32>(Y[2u*j], Y[2u*j+1u])` becomes `ld(j)`, `vec2<f32>(0.0, 0.0)` becomes
`V()`. `dot(d, d)` and every coefficient are unchanged and stay vectorized. Only the two
gradient *stores* are emitted component-wise (a generated `Grad[DIM*i + c] = g.c;` line
per component), and `adam_main`'s loop bound `2u` becomes `DIM` — that loop is already
written component-wise.

**Bounds become a fixed 32 bytes at both dimensionalities.** `struct Bounds { lo: vec4,
hi: vec4 }`, with `lo.z = hi.z = 0` in 2D. A `vec3` uniform pads to 16 bytes anyway, so
the alternative saves nothing and costs a size that varies with the dropdown; a constant
`BOUNDS_BYTES = 32` replaces the four literal `16`s in `main.ts` and keeps
`boundsHistory`'s slot stride, the staging readback and the sessionStorage schema single-
valued. `readSeedBounds()` requires length 8, so a stale 4-length entry is simply ignored
for one run (it already falls back to auto-zoom cleanly).

**Occlusion is a second pipeline, not a runtime flag.** Depth state is baked into a
render pipeline, so 3D builds two (`renderPipe` / `renderPipeDepth`) and `encodeRender`
picks one. When occlusion is on, the fragment shader returns alpha `1.0` and hard-discards
outside the disc, so the depth it writes matches what it painted.

---

## Tasks

One commit per task, straight to `main`. Gate after every task: `npm run build`, plus the
checks named in that task.

### 1. Widen bounds to 32 bytes (2D only, no behaviour change)

`src/shaders.ts`, `src/main.ts`

- `boundsWGSL`: workgroup `sLo`/`sHi` become `vec4<f32>`, the reduce writes 8 floats
  (`B[0..7]`), z/w written as 0.
- `struct Bounds { lo : vec4<f32>, hi : vec4<f32> }`; vertex `ctr`/`span` read `.xyz`
  (z-extent is 0, so `span` is unchanged in 2D).
- `main.ts`: `const BOUNDS_BYTES = 32` replaces the literal `16` at `main.ts:241,245,274,
  455,457,483,555,560`; `readSeedBounds` requires length 8.
- Acceptance: `npm run check:shaders` green; `npm run check:ab` on the clean tree prints
  **0**, then prints 0 again against the pre-task ref (this is a pure-framing refactor, so
  a nonzero here is a bug). Browser: a run looks pixel-identical, auto-zoom on and off.

### 2. `nComponents` in the library

`src/pacmap-webgpu.ts`, `scripts/check-shaders.ts`, `scripts/check-kernels.ts`

- `PacmapOptions.nComponents?: 2 | 3` (default 2). Named for upstream/druid's
  `n_components`; note the existing `D` argument is the *input* dimension.
- Thread `d` into `shaderSource(N, nFP, d)` and `fpShaderSource(N, nFP, nNB, thres, seed,
  d)` via the `alias V` / `ld()` scheme above; size `Y0`, `gradBuf`, `mBuf`, `vBuf`,
  `read()`'s staging at `N*d`. Update the `N x 2` doc on `EmbeddingRun.positions`.
- `check-shaders.ts`: the case table iterates `d ∈ {2, 3}` for both generated sources and
  for `boundsWGSL` + the render pipeline (which gains the `float32x3` and depth variants
  in Task 4 — leave a case ready).
- `check-kernels.ts`: run the resample/reverse cases at `d = 3`. The on-a-line fixture
  (`check-kernels.ts:265`) generalizes by keeping y = z = 0, which preserves its
  "within `low_dist_thres` = within ten indices" invariant; **add one genuinely-3D case**
  where a candidate is inside the threshold in x/y but outside it via z alone, so the
  third component is load-bearing in the check. Host oracle at `:354` uses all `d` comps.
- Acceptance: `check:shaders` and `check:kernels` green at both dims; each new assertion
  demonstrated failing first (e.g. drop the z term from `ld`). `check:ab` clean-tree
  control **0**, and `npm run check:ab -- <pre-task-ref>` prints 0 for both
  `--variant=pacmap` and `--variant=localmap` — that is the proof the 2D path is untouched.

### 3. Renderer + pane dropdown + camera mode (end-to-end 3D, GPU engine)

`src/shaders.ts`, `src/main.ts`

- `renderWGSL` becomes `renderWGSL(d)`: `@location(0) p : vec2/vec3<f32>`, world built as
  `vec3(p - ctr.xy, 0.0) / (span*0.55)` or `(p - ctr.xyz) / (span*0.55)`. The screen-space
  quad expansion (`clip.xy + c*r*clip.w`) is dimension-independent and stays — point
  radius remains screen-constant, per the existing deliberate choice.
- Pipeline: `arrayStride: 4*d`, `format: "float32x" + d`; `frameBytes = N * 4 * d` (update
  the `N x 2` / 520KB comment at `main.ts:84`; frames are 50% larger in 3D, so at a fixed
  `HISTORY_BUDGET_BYTES` the trace strides coarser — expected, and already reported in the
  status line).
- Pane: `params.nComponents` bound in `pacmapFolder` as `components` with options
  `{ "2D": 2, "3D": 3 }`, annotated `Record<string, 2 | 3>` for the same reason
  `ALGO_OPTIONS` is (Tweakpane types option values as plain strings). Seeded from `?dims=`
  alongside `?algo=`/`?knn=`. Setup-time: read at the top of `go()`, folder already
  disables during a run.
- Camera: `setCameraMode(d)` sets `orbit.enableRotate = d === 3` and swaps `mouseButtons`
  between the current 2D remap `{ORBIT:2, ZOOM:1, PAN:0}` and ogl's defaults
  `{ORBIT:0, ZOOM:1, PAN:2}`. Called from the dropdown's `change` and at the top of `go()`.
  `resetCamera()` is unchanged — +z at `DEFAULT_DIST` is the front view in both modes.
  Update the two comments at `main.ts:732` and `:754` that describe this as pending.
- Acceptance (browser, required — this is the wiring the headless checks can't see):
  select 3D, run at 2k; the cloud rotates with left-drag, pans with right-drag, wheel
  zooms, double-click resets. Scrub the timeline: replayed frames rotate too. Switch back
  to 2D and confirm left-drag pans again and the framing is the old one.

### 4. Depth occlusion + toggle

`src/shaders.ts`, `src/main.ts`

- Depth texture (`depth24plus`, canvas-sized) created in `resize()`, old one destroyed;
  `encodeRender` attaches it with `depthClearValue: 1.0`, `loadOp: "clear"`.
- Second pipeline with `depthStencil: { format: "depth24plus", depthWriteEnabled: true,
  depthCompare: "less" }`. Fragment gains an `occlude` flag (a `View` uniform field in the
  existing `_pad` slot): when set, discard outside the disc and return alpha `1.0`, so
  no semi-transparent fragment ever writes depth — the dark-streak failure mode recorded
  in the reference project's `render_fpl.py:62-79`.
- `view.occlusion` checkbox in the `view` folder, default on, `disabled` in 2D (where a
  depth buffer over coplanar points buys nothing).
- Acceptance: `check:shaders` builds both render pipelines at both dims. Browser: with
  occlusion on, rotating a 3D MNIST run shows near clusters cleanly hiding far ones and
  no dark streaks; off, it returns to the blended haze. 2D is unaffected and the checkbox
  is greyed.

### 5. 3D on the druid CPU engine

`src/druid-protocol.ts`, `src/druid-worker.ts`, `src/druid-cpu.ts`, `scripts/check-druid.ts`

- `DruidRunOptions.nComponents` / `DruidCpuOptions.nComponents` → `DruidInitCommand` →
  `buildDruidParams` (`druid-protocol.ts:68` already emits `d: 2`; `d` is already in
  `EXPECTED_KEYS`). Worker snapshot `new Float32Array(N * d)` (`druid-worker.ts:86`),
  `positions` buffer `N * 4 * d` (`druid-cpu.ts:69`), `read()` fallback likewise.
- `check-druid.ts`: a `d: 3` case asserting the output is length `N*3` and finite, and
  that blob separation holds in 3-d (generalize the `:179` distance). This is the case
  that proves `d` is *forwarded* rather than constant — the existing key-set check only
  proves the key is spelled right.
- Acceptance: `npm run check:druid` green, new assertion shown failing first (hard-code
  `d: 2` back and watch the length check trip). Browser: `LocalMAP (CPU - druid)` at 3D,
  500 points, renders and scrubs.

### 6. Documentation

`CLAUDE.md`, `README.md`

- New subsection under Architecture for the components axis: the `alias V`/`ld()` scheme
  and why it beats a component loop; the fixed 32-byte bounds and why `vec3` padding makes
  it free; occlusion-over-sorting and why a CPU depth sort is unavailable here; the
  camera's three 2D flags now being mode-switched rather than fixed.
- Update the coverage-gaps paragraph: the camera gap now also covers rotation and the
  occlusion toggle.

---

## Checkpoints

- **After Task 2** — headless gate: `build`, `check:shaders`, `check:kernels`,
  `check:druid` all green, and `check:ab` proves 2D unmoved (0 on both variants). Nothing
  user-visible has changed yet; this is the point at which the library computes in 3D and
  nothing draws it.
- **After Task 3** — browser gate: 3D is selectable, renders and rotates end to end on the
  GPU engine. Both mouse mappings confirmed by hand.
- **After Task 5** — full gate: all four checks green, and one run per engine × per
  dimensionality (8 combinations) confirmed in the browser.

## Verification, end to end

```
npm run build          # tsc --noEmit && vite build
npm run check:shaders  # both dims, both render pipelines
npm run check:kernels  # resample/reverse at d=2 and d=3
npm run check:druid    # incl. the new d=3 forwarding case
npm run check:ab                          # clean tree must print exactly 0
npm run check:ab -- <ref> --variant=localmap   # 2D path unmoved by Task 2
npm run dev            # then, in a WebGPU browser:
```

Browser pass (the wiring no check sees): for each of the four algorithm entries, run at 2D
and at 3D — 2k points on GPU, 500 on CPU. Confirm per run: it completes, the transport
scrubs, `auto zoom` on/off both behave, `reset camera` and double-click restore the front
view, the mouse mapping matches the mode, and in 3D the occlusion checkbox visibly changes
the cloud while 2D leaves it greyed. Stop button mid-run on a CPU 3D run keeps its partial
trace.

## Task files

Task 1 also archives the current `tasks/plan.md` + `tasks/todo.md` (the completed DruidJS
plan) into `tasks/archive/` and writes this plan's task list in their place, matching the
existing checkbox-per-acceptance-criterion format.
