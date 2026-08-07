import {
  pacmapWebGPU,
  bruteForceKnn,
  knnGPU,
  nndescentGPU,
  defaultNeighbors,
  type EmbeddingRun,
} from "./pacmap-webgpu";
import { druidCPU } from "./druid-cpu";
import { loadMnist, IMAGE_SIZE, NUM_AVAILABLE } from "./mnist";
import { pcaProject } from "./pca";
import { boundsWGSL, renderWGSL, edgeWGSL } from "./shaders";
import {
  buildEdgeIndices,
  EDGE_KINDS,
  EDGE_COLORS,
  type EdgeKind,
} from "./edges";
import { Pane } from "tweakpane";
import { Camera, Orbit, Vec3, type OGLRenderingContext } from "ogl";

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const canvas = document.getElementById("view") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLElement;
const iterEl = document.getElementById("iter") as HTMLElement;
const phaseEl = document.getElementById("phase") as HTMLElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const sampleSel = document.getElementById("samples") as HTMLInputElement;
const sampleOut = document.getElementById("samplesOut") as HTMLOutputElement;
const sampleHint = document.getElementById("samplesHint") as HTMLElement;
const speedSel = document.getElementById("speed") as HTMLSelectElement;
const playBtn = document.getElementById("play") as HTMLButtonElement;
const scrub = document.getElementById("scrub") as HTMLInputElement;
const readoutEl = document.getElementById("readout") as HTMLElement;

const status = (m: string) => (statusEl.textContent = m);

// ---------------------------------------------------------------------------
// URL switches
//   ?algo=localmap          run LocalMAP rather than PaCMAP (default pacmap)
//   ?algo=pacmap-cpu        run DruidJS on the CPU rather than our WGSL
//   ?knn=gpu|nnd            pick the kNN backend (default cpu brute force)
//   ?dims=2                 embed and render in 2D (default 3)
//   ?knncheck=1             run every backend over one input and report how they compare
// ---------------------------------------------------------------------------

type KnnMode = "gpu" | "cpu" | "nndescent";
type Variant = "pacmap" | "localmap";
/** Which implementation runs the optimizer: our WGSL, or DruidJS off-thread. */
type Engine = "gpu" | "cpu";

/**
 * Algorithm and implementation as one choice, because that is how it is picked.
 *
 * The two axes are not independent in the UI — there is no reason to select
 * "LocalMAP" and then separately select "on the CPU" — and collapsing them into
 * a single key keeps the dropdown one control rather than two that can disagree.
 */
type AlgoKey = `${Variant}-${Engine}`;

/** Embedding width. The dropdown's values, and the library's `nComponents`. */
type Components = 2 | 3;

const VARIANT_OF: Record<AlgoKey, Variant> = {
  "pacmap-gpu": "pacmap",
  "localmap-gpu": "localmap",
  "pacmap-cpu": "pacmap",
  "localmap-cpu": "localmap",
};
const ENGINE_OF: Record<AlgoKey, Engine> = {
  "pacmap-gpu": "gpu",
  "localmap-gpu": "gpu",
  "pacmap-cpu": "cpu",
  "localmap-cpu": "cpu",
};

const qs = new URLSearchParams(location.search);
const KNN_MODE: KnnMode =
  qs.get("knn") === "gpu" ? "gpu"
  : qs.get("knn") === "nnd" || qs.get("knn") === "nndescent" ? "nndescent"
  : "cpu";
// Bare `pacmap` / `localmap` still mean the GPU implementations — that is what
// they meant before there was a second one, and links to them predate it.
const ALGO_PARAM = qs.get("algo") ?? "";
const ALGO_MODE: AlgoKey =
  ALGO_PARAM in VARIANT_OF ? (ALGO_PARAM as AlgoKey)
  : ALGO_PARAM === "localmap" ? "localmap-gpu"
  : "pacmap-gpu";
const KNN_CHECK = qs.get("knncheck") === "1";
const DIMS_MODE: Components = qs.get("dims") === "2" ? 2 : 3;

// Playback history. Every captured frame is a full N x d f32 snapshot kept in
// GPU memory, so the scrubber never reads positions back to the host — same
// constraint as the live render path. At 65k points a frame is 520KB in 2D and
// 780KB in 3D, so the budget, not the iteration count, is what decides how many
// we keep — and a 3D run banks correspondingly fewer, or strides coarser.
const HISTORY_BUDGET_BYTES = 128 << 20;

// Byte offset of `lerpT` in the view uniform — the last of the 24 floats. Its
// own tiny write, because it changes per drawn frame while everything else in
// that struct only changes when the camera or a pane control does.
const LERP_T_OFFSET = 92;

// The framing box: `lo.xyzw hi.xyzw`, one size at either dimensionality (see
// `struct Bounds` in shaders.ts). Every buffer, history slot and copy that
// carries a box is measured in this rather than in a literal.
const BOUNDS_BYTES = 32;

// Occlusion's depth attachment. Must match the format the pipeline's depth
// state declares, which check-shaders builds against the same constant.
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

// Straight source-over. Shared by the point renderer and the edge overlay, and
// they have to agree: the edges are drawn into the same pass immediately before
// the points, so a different blend would make an edge composite against the
// cloud differently than the cloud composites against itself.
const ALPHA_BLEND: GPUBlendState = {
  color: {
    srcFactor: "src-alpha",
    dstFactor: "one-minus-src-alpha",
    operation: "add",
  },
  // Not src-alpha, so the destination alpha accumulates rather than being
  // scaled by the source's — the canvas is composited against the page.
  alpha: {
    srcFactor: "one",
    dstFactor: "one-minus-src-alpha",
    operation: "add",
  },
};

// Occlusion's depth state, and the other half of the agreement above: both
// pipelines are set in one pass, and a pass carrying a depth attachment can
// only be drawn with pipelines that declare depth state. Both build their
// occluded variant from this, so neither can drift from the attachment.
const DEPTH_STATE: GPUDepthStencilState = {
  format: DEPTH_FORMAT,
  depthWriteEnabled: true,
  depthCompare: "less",
};

// With auto zoom off the whole trace is framed by one box, and during a live run
// the box that frames the *final* frame is not yet known. The previous run's is
// the best guess available, so it is carried across runs and page reloads. Same
// layout the bounds shader writes, so it drops straight into the render's
// bounds uniform.
const BOUNDS_KEY = "pacmap:lastBounds";
let seedBounds: Float32Array<ArrayBuffer> | null = readSeedBounds();

function readSeedBounds(): Float32Array<ArrayBuffer> | null {
  // A cosmetic cache; nothing here may take out startup.
  try {
    const raw = sessionStorage.getItem(BOUNDS_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw);
    // An entry written before the box widened is simply ignored; the run it
    // would have seeded falls back to auto zoom, which is already the
    // first-run-of-a-session behaviour.
    if (!Array.isArray(v) || v.length !== BOUNDS_BYTES / 4) return null;
    if (!v.every((x) => typeof x === "number" && Number.isFinite(x))) return null;
    return new Float32Array(v);
  } catch {
    return null;
  }
}

// Colors for digits 0-9. Distinct hues, roughly matched in luminance so no one
// class dominates visually.
const PALETTE: [number, number, number][] = [
  [0.90, 0.31, 0.31], [0.95, 0.60, 0.20], [0.85, 0.80, 0.22],
  [0.44, 0.78, 0.34], [0.22, 0.72, 0.62], [0.29, 0.62, 0.90],
  [0.45, 0.45, 0.92], [0.72, 0.40, 0.88], [0.92, 0.42, 0.70],
  [0.62, 0.62, 0.66],
];

// ---------------------------------------------------------------------------
// Digit thumbnails
// ---------------------------------------------------------------------------

/** u32 words per 28x28 tile, four 8-bit intensities to the word. */
const TILE_WORDS = IMAGE_SIZE / 4;

/** Which of `thumbColor`'s looks a digit is drawn in. */
type DigitStyle = 0 | 1 | 2;
/**
 * Annotated rather than inferred, for the reason the algorithm map is: Tweakpane
 * types `options` values as whatever they look like, so a fourth entry pointing
 * at a style the shader does not implement would type-check and then draw the
 * default.
 */
const DIGIT_STYLES: Record<string, DigitStyle> = {
  "coloured stroke": 0,
  "white on colour": 1,
  "black on colour": 2,
};

/**
 * Pack every digit into one atlas, and give every point a rank deciding how
 * early it becomes a thumbnail.
 *
 * The atlas holds *all* N bitmaps rather than a sampled subset, which is what
 * lets the `digit %` slider be a render-time control: it moves a threshold
 * against the rank, so no buffer is rebuilt and no run is restarted. That is
 * affordable only because the intensities are quantized back to the 8 bits they
 * arrived as — 784 bytes a digit, ~51MB at N=65k, against ~204MB as f32 and a
 * 128MB default limit on a storage binding. The tile index is then just the
 * point index, so nothing has to map one to the other.
 *
 * Rank is a random permutation drawn from the run's seed, so raising the slider
 * adds digits spread through the cloud rather than walking along the point
 * order, and two runs at one seed show the same digits. Which *look* they are
 * drawn in is a uniform, not stored here — see `thumbColor` in shaders.ts.
 *
 * Must be called *before* `pcaProject`, which runs with `inPlace: true` and
 * overwrites X. Reading it afterwards yields plausible noise rather than an
 * error.
 */
function buildDigitAtlas(X: Float32Array, N: number, seed: number) {
  const rand = mulberry32(seed ^ 0x7d1);

  // Full Fisher-Yates: rank[i] is where point i falls in a random order.
  const rank = new Uint32Array(N);
  for (let i = 0; i < N; i++) rank[i] = i;
  for (let i = N - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = rank[i];
    rank[i] = rank[j];
    rank[j] = t;
  }

  // At least one tile: a zero-length storage buffer is not a legal binding.
  const atlas = new Uint32Array(Math.max(1, N) * TILE_WORDS);
  for (let i = 0; i < N; i++) {
    const src = i * IMAGE_SIZE;
    const dst = i * TILE_WORDS;
    for (let w = 0; w < TILE_WORDS; w++) {
      const p = src + w * 4;
      atlas[dst + w] =
        ((X[p] * 255) & 0xff) |
        (((X[p + 1] * 255) & 0xff) << 8) |
        (((X[p + 2] * 255) & 0xff) << 16) |
        (((X[p + 3] * 255) & 0xff) << 24);
    }
  }

  return { thumbs: rank, atlas };
}

/** Duplicated from pca.ts / pacmap-webgpu.ts, which keep it module-private so
 *  each stays standalone. Same generator, so a seed means the same thing. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

let running = false;
// Bumped on every run so a previous run's render loop and resize handler stop
// touching the canvas (their buffers belong to a device that is now stale).
let runGen = 0;
/**
 * Aborts the run in flight.
 *
 * The CPU engine can occupy the machine for minutes, and before this the only
 * way out was closing the tab — which is what made "warn, never block" a safe
 * default rather than a trap. Held at module scope because the button that
 * triggers it lives here too, outside any one run.
 */
let runAbort: AbortController | null = null;

async function go() {
  if (running) return;
  running = true;
  startBtn.textContent = "Stop";
  const gen = ++runGen;
  const abort = new AbortController();
  runAbort = abort;
  resetTransport();
  // A new run starts from the designated view rather than inheriting wherever
  // the last one was left parked, on the same reasoning as the transport reset.
  resetCamera();
  pacmapFolder.disabled = true; // setup-time params; this run is committed

  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice();
    device.lost.then((info) => status(`Device lost: ${info.message}`));

    const N = parseInt(sampleSel.value, 10);
    const stepsPerFrame = parseInt(speedSel.value, 10);
    sampleSel.disabled = true;

    // --- Data --------------------------------------------------------------
    const { X, labels } = await loadMnist(N, status);

    // Before the PCA below, which runs in place and overwrites X.
    const thumbs = buildDigitAtlas(X, N, params.seed);

    status("Projecting 784d → 100d (randomized PCA)…");
    await frame();
    const t0 = performance.now();
    // inPlace is safe here: X is never read again after this call, and it saves
    // a second n*784 f32 array — 204MB at 65k.
    const Z = await pcaProject(X, N, IMAGE_SIZE, 100, {
      inPlace: true,
      onStatus: status,
    });
    const tPca = performance.now() - t0;

    // PCA is the one long stretch neither engine can interrupt — it runs on
    // this thread, yielding only to paint. A stop during it is honoured here,
    // before committing to a setup that is itself minutes long on the CPU.
    if (abort.signal.aborted) throw new Error("aborted");

    if (KNN_CHECK) await knnSelfCheck(device, Z, N);

    // Read once, here: the folder is disabled for the duration of the run, so
    // these are the values this run is committed to.
    const algorithm = params.algorithm;
    const variant = VARIANT_OF[algorithm];
    const engine = ENGINE_OF[algorithm];
    const algoLabel = ALGO_LABELS[algorithm];
    // Druid runs its own exact neighbour search, so the kNN backend is a
    // GPU-engine concept and naming one under the CPU engine would be a lie.
    const knnLabel = engine === "cpu" ? "druid exact" : KNN_LABELS[params.knnMethod];
    const nNeighbors = params.autoNeighbors ? undefined : params.nNeighbors;
    const nComponents = params.nComponents;
    // Rotation is on in 3D and off in 2D, and the mouse map differs with it —
    // set here rather than only on the dropdown so a `?dims=` load is right
    // before anything is drawn.
    setCameraMode(nComponents);

    status(
      `PCA ${tPca | 0}ms · building kNN graph (${knnLabel}) + sampling pairs…`
    );
    await frame();
    const t1 = performance.now();
    const pm: EmbeddingRun =
      engine === "cpu"
        ? await druidCPU(device, Z, N, 100, {
            variant,
            nComponents,
            signal: abort.signal,
            seed: params.seed,
            lowDistThres: params.lowDistThres,
            // Druid has no equivalent of the library's log10(N) rule, so "auto"
            // is resolved here rather than passed through as undefined.
            nNeighbors: nNeighbors ?? defaultNeighbors(N),
            mnRatio: params.mnRatio,
            fpRatio: params.fpRatio,
            onStatus: status,
          })
        : await pacmapWebGPU(device, Z, N, 100, {
            seed: params.seed,
            knn: params.knnMethod,
            variant,
            nComponents,
            lowDistThres: params.lowDistThres,
            nNeighbors,
            mnRatio: params.mnRatio,
            fpRatio: params.fpRatio,
            onStatus: status,
          });
    const tSetup = performance.now() - t1;

    // --- Render resources --------------------------------------------------
    const ctx = canvas.getContext("webgpu")!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: "premultiplied" });

    const labelBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Uint32Array(labelBuf.getMappedRange()).set(Uint32Array.from(labels));
    labelBuf.unmap();

    const thumbBuf = device.createBuffer({
      size: N * 4,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    new Uint32Array(thumbBuf.getMappedRange()).set(thumbs.thumbs);
    thumbBuf.unmap();

    const atlasBuf = device.createBuffer({
      size: thumbs.atlas.byteLength,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint32Array(atlasBuf.getMappedRange()).set(thumbs.atlas);
    atlasBuf.unmap();

    const boundsStorage = device.createBuffer({
      size: BOUNDS_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    // Two boxes, for the two keyframes an interpolated frame mixes. The live
    // path writes the same box into both, so `lerpT` cannot affect it whatever
    // it holds.
    const boundsUniform = device.createBuffer({
      size: BOUNDS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const boundsUniformB = device.createBuffer({
      size: BOUNDS_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // 96 bytes exactly: viewProj mat4, res vec2, radius, occlude, digitCount,
    // digitScale, digitStyle, lerpT. See `struct View`.
    const viewUniform = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const viewUniformData = new Float32Array(24);
    // Where the drawn frame sits between the two bound keyframes. Never
    // anything but 0 on the live path, which binds one keyframe to both slots.
    let lerpT = 0;
    const lerpScratch = new Float32Array(1);

    // --- Playback history --------------------------------------------------
    // One slot per captured frame, plus slot 0 for the pre-optimization init.
    // `stride` iterations are collapsed into one slot when the full trace would
    // not fit the budget, so the timeline stays complete (just coarser).
    const frameBytes = N * 4 * nComponents;
    const budget = Math.min(HISTORY_BUDGET_BYTES, device.limits.maxBufferSize);
    const maxFrames = Math.max(2, Math.floor(budget / frameBytes));
    const stride = Math.max(1, Math.ceil(pm.totalIters / (maxFrames - 1)));
    const frameCount = 1 + Math.ceil(pm.totalIters / stride);
    // Which iteration each slot holds, for the readout.
    const frameIters = new Int32Array(frameCount);

    const posHistory = device.createBuffer({
      size: frameCount * frameBytes,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    // The autoscale bounds are banked alongside the positions rather than
    // recomputed on scrub, so a replayed frame is framed exactly as it was live.
    const boundsHistory = device.createBuffer({
      size: frameCount * BOUNDS_BYTES,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });

    const boundsPipe = device.createComputePipeline({
      layout: "auto",
      compute: {
        module: device.createShaderModule({ code: boundsWGSL(N) }),
        entryPoint: "main",
      },
    });
    const boundsBG = device.createBindGroup({
      layout: boundsPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: pm.positions } },
        { binding: 1, resource: { buffer: boundsStorage } },
      ],
    });

    const renderModule = device.createShaderModule({
      code: renderWGSL(nComponents, DEFAULT_DIST),
    });
    // Explicit, not "auto". A pipeline created with "auto" gets its own
    // freshly-minted bind group layout, and layouts minted that way are never
    // compatible with another pipeline's — so a bind group built from
    // `renderPipe` cannot be set while `renderPipeDepth` is bound, and the
    // draw is dropped as a validation error. Symptom: turning occlusion on
    // makes every point vanish while the camera keeps working.
    const renderBGL = device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
        {
          // The fragment reads this one too, for the occlude flag.
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        {
          // The second keyframe's box, mixed against binding 0 on `lerpT`.
          binding: 3,
          visibility: GPUShaderStage.VERTEX,
          buffer: { type: "uniform" },
        },
      ],
    });
    /**
     * One keyframe of positions as a vertex buffer.
     *
     * The stride is the same `d` the shader was generated with and the same one
     * `frameBytes` is sized from; those three cannot be allowed to disagree, so
     * they are expressed once. Step mode is the caller's because it is the one
     * real difference between the two consumers: the point renderer draws one
     * instance per point, while the edge overlay walks the same memory per
     * vertex, two vertices to a line, fetched through an index buffer.
     */
    const posLayout = (
      shaderLocation: number,
      stepMode: GPUVertexStepMode
    ): GPUVertexBufferLayout => ({
      arrayStride: 4 * nComponents,
      stepMode,
      attributes: [
        {
          shaderLocation,
          offset: 0,
          format: `float32x${nComponents}` as GPUVertexFormat,
        },
      ],
    });

    const renderDesc: GPURenderPipelineDescriptor = {
      layout: device.createPipelineLayout({ bindGroupLayouts: [renderBGL] }),
      vertex: {
        module: renderModule,
        entryPoint: "vs",
        buffers: [
          posLayout(0, "instance"),
          {
            arrayStride: 4,
            stepMode: "instance",
            attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" }],
          },
          {
            // Tile index + style bit; NO_THUMB for a point drawn as a disc.
            arrayStride: 4,
            stepMode: "instance",
            attributes: [{ shaderLocation: 2, offset: 0, format: "uint32" }],
          },
          // The next keyframe. Same shape as slot 0 and, on the playback path,
          // the same buffer — bound one slot along. Binding one buffer to two
          // vertex slots at different offsets is legal; vertex buffers are
          // read-only.
          posLayout(3, "instance"),
        ],
      },
      fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [{ format, blend: ALPHA_BLEND }],
      },
      primitive: { topology: "triangle-list" },
    };
    const renderPipe = device.createRenderPipeline(renderDesc);
    // Depth state is baked into a pipeline, so the toggle is a second pipeline
    // rather than a flag — built only in 3D, where it is the only mode that
    // can use it.
    const renderPipeDepth =
      nComponents === 3
        ? device.createRenderPipeline({ ...renderDesc, depthStencil: DEPTH_STATE })
        : null;
    /**
     * Is this frame drawn occluded?
     *
     * One place, because three things have to agree: which pipeline is set,
     * whether the pass carries a depth attachment, and the `occlude` flag in
     * the view uniform that tells the fragment to paint solid. A pipeline with
     * depth state and a pass without an attachment is a validation error; the
     * flag disagreeing with either is a semi-transparent fragment writing
     * depth, which is the dark-streak failure this mode exists to avoid.
     */
    const occludingNow = () => renderPipeDepth !== null && view.occlusion;

    let depthTex: GPUTexture | null = null;
    let depthView: GPUTextureView | null = null;

    const renderBG = device.createBindGroup({
      layout: renderBGL,
      entries: [
        { binding: 0, resource: { buffer: boundsUniform } },
        { binding: 1, resource: { buffer: viewUniform } },
        { binding: 2, resource: { buffer: atlasBuf } },
        { binding: 3, resource: { buffer: boundsUniformB } },
      ],
    });

    // --- The pair-graph overlay --------------------------------------------
    // Null under the CPU engines: druid builds its own neighbour graph
    // internally and exposes nothing, so there is no overlay to offer and the
    // pane greys the control rather than drawing an empty one.
    const edges = (() => {
      if (!pm.graph) return null;
      // Seeded from the run's seed, so two runs at one seed show the same sample.
      const built = buildEdgeIndices(pm.graph, N, mulberry32(params.seed ^ 0x3dbe));
      if (built.indices.length === 0) return null;

      const indexBuf = device.createBuffer({
        size: built.indices.byteLength,
        usage: GPUBufferUsage.INDEX,
        mappedAtCreation: true,
      });
      new Uint32Array(indexBuf.getMappedRange()).set(built.indices);
      indexBuf.unmap();

      // One 256-byte-aligned slot per kind, addressed by dynamic offset. The
      // kind cannot ride in a vertex attribute — on an indexed draw
      // `vertex_index` is the fetched index value, not the position in the
      // index buffer — so it arrives per draw instead. Same mechanism the
      // optimizer uses for its per-iteration weights.
      const align = device.limits.minUniformBufferOffsetAlignment || 256;
      const styles = new Float32Array((align / 4) * EDGE_KINDS.length);
      EDGE_KINDS.forEach((kind, k) => {
        styles.set([...EDGE_COLORS[kind], view.edgeAlpha / 100], (k * align) / 4);
      });
      const styleBuf = device.createBuffer({
        size: styles.byteLength,
        // COPY_DST too: onViewChange rewrites the alpha float live via
        // writeBuffer, which silently no-ops against a buffer lacking it.
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
      });
      new Float32Array(styleBuf.getMappedRange()).set(styles);
      styleBuf.unmap();

      const bgl = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
          {
            // The fragment reads this one too, for the occlude flag.
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" },
          },
          { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
          {
            binding: 3,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 16 },
          },
        ],
      });
      const bindGroup = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: boundsUniform } },
          { binding: 1, resource: { buffer: viewUniform } },
          { binding: 2, resource: { buffer: boundsUniformB } },
          { binding: 3, resource: { buffer: styleBuf, size: 16 } },
        ],
      });

      const edgeModule = device.createShaderModule({ code: edgeWGSL(nComponents) });
      const desc: GPURenderPipelineDescriptor = {
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module: edgeModule,
          entryPoint: "vs",
          // The same two keyframes the point draw binds, at "vertex" step mode
          // rather than "instance" — same memory, different pipeline, different
          // layout. An edge is two of these, fetched through the index buffer.
          buffers: [posLayout(0, "vertex"), posLayout(1, "vertex")],
        },
        fragment: {
          module: edgeModule,
          entryPoint: "fs",
          targets: [{ format, blend: ALPHA_BLEND }],
        },
        primitive: { topology: "line-list" },
      };
      return {
        indexBuf,
        styleBuf,
        ranges: built.ranges,
        align,
        pipe: device.createRenderPipeline(desc),
        // Not optional: when occlusion is on the pass carries a depth
        // attachment, and a pipeline without depth state cannot be used in a
        // pass that has one. That is a validation error, not a difference in
        // how it looks.
        pipeDepth:
          nComponents === 3
            ? device.createRenderPipeline({ ...desc, depthStencil: DEPTH_STATE })
            : null,
        bindGroup,
      };
    })();

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      // The slider is in CSS px; 1.5 framebuffer px is the floor below which
      // points stop resolving at all. In 3D the radius reaching the shader is
      // no longer the drawn size — it is the size *at the reference distance*,
      // which depth then scales — so the flat clamp belongs there instead, and
      // the shader has a better one: it floors the per-point size at 1px and
      // spends the shortfall as opacity. Clamping here too would only make
      // every slider value below 1.5/dpr render identically.
      const radius =
        nComponents === 3 ? view.pointSize * dpr : Math.max(1.5, view.pointSize * dpr);

      camera.perspective({ aspect: w / h });
      // Refreshes viewMatrix and projectionViewMatrix. ogl does this inside
      // `render()`, which this pipeline never calls, so it has to be explicit.
      camera.updateMatrixWorld();

      const m = viewUniformData;
      m.set(camera.projectionViewMatrix, 0);
      // ogl emits WebGL clip space, where z runs [-1, 1]; WebGPU's runs [0, 1]
      // and clips anything below it. Remap with z' = (z + w)/2, which on a
      // column-major matrix is a rewrite of row 2 against row 3. Nothing renders
      // differently today — there is no depth attachment — but the 3D step turns
      // depth testing on, and this is where that would silently eat half the
      // points otherwise.
      for (let i = 0; i < 4; i++) {
        m[4 * i + 2] = 0.5 * (m[4 * i + 2] + m[4 * i + 3]);
      }
      m[16] = canvas.width;
      m[17] = canvas.height;
      m[18] = radius;
      m[19] = occludingNow() ? 1 : 0;
      // The slider is a percentage; the shader compares against a rank, so the
      // resolution against N happens here rather than needing N in the shader.
      m[20] = (view.digitPct / 100) * N;
      m[21] = view.digitScale;
      m[22] = view.digitStyle;
      // Owned by `drawFrame`, which rewrites just these four bytes per frame.
      // Carried here too so a resize mid-playback cannot stomp it back to 0.
      m[23] = lerpT;
      device.queue.writeBuffer(viewUniform, 0, m);

      // Sized with the canvas, and only in 3D. Recreated rather than resized —
      // a texture's dimensions are fixed at creation — so the old one is
      // destroyed rather than left to the GC holding framebuffer memory.
      if (renderPipeDepth && (depthTex?.width !== w || depthTex?.height !== h)) {
        depthTex?.destroy();
        depthTex = device.createTexture({
          size: [w, h],
          format: DEPTH_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
        depthView = depthTex.createView();
      }
    }
    // Point-size edits land immediately, even mid-run when nothing else is
    // rewriting the view uniform.
    let viewDirty = false;
    onViewChange = () => {
      if (gen !== runGen) return;
      viewDirty = true;
      // The alpha float lives in each kind's EdgeStyle slot, not the view
      // uniform, so it gets its own small writeBuffer here rather than
      // riding along with viewDirty's redraw of `boundsUniform`/`viewUniform`.
      if (edges) {
        EDGE_KINDS.forEach((kind, k) => {
          device.queue.writeBuffer(
            edges.styleBuf,
            k * edges.align + 12, // color is 3 floats, alpha is the 4th
            new Float32Array([view.edgeAlpha / 100])
          );
        });
      }
    };
    window.addEventListener("resize", () => {
      if (gen === runGen) viewDirty = true;
    });
    resize();

    // Orbit eases toward its target, so it needs a tick every frame, and the
    // camera has to stay responsive while the optimizer is running. During the
    // run the only other redraws come from `captureAndDraw`, which under the CPU
    // engine can be seconds apart — a gesture would freeze between frames. So
    // the live phase gets its own rAF, which hands over to the playback loop
    // below once the run is done rather than running alongside it.
    let livePhase = true;
    let camKey = "";
    /** Has the camera moved since the last tick? Also seeds `camKey`. */
    function cameraMoved() {
      const p = camera.position;
      const t = orbit.target;
      const key = `${p.x},${p.y},${p.z}|${t.x},${t.y},${t.z}`;
      if (key === camKey) return false;
      camKey = key;
      return true;
    }
    const camTick = () => {
      if (gen !== runGen || !livePhase) return;
      orbit.update();
      if (cameraMoved() || viewDirty) {
        viewDirty = false;
        resize();
        // `boundsUniform` already holds the box for the frame on screen, so
        // re-encoding the render alone is enough — no compute, no readback.
        const enc = device.createCommandEncoder();
        encodeRender(enc, pm.positions, 0);
        device.queue.submit([enc.finish()]);
      }
      requestAnimationFrame(camTick);
    };
    requestAnimationFrame(camTick);

    /**
     * Encode one draw.
     *
     * `offsetB` is the keyframe the vertex stage mixes toward on `lerpT`. The
     * live path passes the same offset twice, which makes the mix a no-op
     * whatever `lerpT` holds; only playback ever passes two.
     */
    function encodeRender(
      enc: GPUCommandEncoder,
      posBuf: GPUBuffer,
      offset: number,
      offsetB: number = offset
    ) {
      const occlude = occludingNow() && depthView !== null;
      const rp = enc.beginRenderPass({
        colorAttachments: [
          {
            view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0.055, g: 0.06, b: 0.075, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
        ...(occlude
          ? {
              depthStencilAttachment: {
                view: depthView!,
                depthClearValue: 1,
                depthLoadOp: "clear" as const,
                depthStoreOp: "store" as const,
              },
            }
          : {}),
      });
      // Edges first, into the same pass, so the points land on top of them. A
      // second pass with loadOp "load" would work and costs a pass per frame
      // for nothing.
      if (edges && view.edges) {
        rp.setPipeline(occlude ? edges.pipeDepth! : edges.pipe);
        rp.setVertexBuffer(0, posBuf, offset, frameBytes);
        rp.setVertexBuffer(1, posBuf, offsetB, frameBytes);
        rp.setIndexBuffer(edges.indexBuf, "uint32");
        EDGE_KINDS.forEach((kind, k) => {
          const { first, count } = edges.ranges[kind];
          // The percentage resolved against this range's own total. Because
          // the range was shuffled at setup, a prefix is a uniform sample of
          // that kind — so this is the whole of the control.
          const drawn = Math.round((view.edgePct[kind] / 100) * count);
          if (drawn === 0) return;
          rp.setBindGroup(0, edges.bindGroup, [k * edges.align]);
          rp.drawIndexed(drawn * 2, 1, first);
        });
      }

      rp.setPipeline(occlude ? renderPipeDepth! : renderPipe);
      rp.setBindGroup(0, renderBG);
      rp.setVertexBuffer(0, posBuf, offset, frameBytes);
      rp.setVertexBuffer(1, labelBuf);
      rp.setVertexBuffer(2, thumbBuf);
      rp.setVertexBuffer(3, posBuf, offsetB, frameBytes);
      rp.draw(6, N);
      rp.end();
    }

    /** Reduce bounds over the live positions, bank both into `slot`, and draw. */
    function captureAndDraw(slot: number) {
      const enc = device.createCommandEncoder();

      const cp = enc.beginComputePass();
      cp.setPipeline(boundsPipe);
      cp.setBindGroup(0, boundsBG);
      cp.dispatchWorkgroups(1);
      cp.end();
      // Banked unconditionally: 32 bytes, and toggling auto zoom back on mid-run
      // has to find a bound for every slot behind it.
      enc.copyBufferToBuffer(
        boundsStorage,
        0,
        boundsHistory,
        slot * BOUNDS_BYTES,
        BOUNDS_BYTES
      );
      // Both boxes, always the same one: a live frame has no next keyframe to
      // interpolate toward, so the mix in the shader has to come out at the box
      // it was given.
      if (view.autoZoom || !seedBounds) {
        enc.copyBufferToBuffer(boundsStorage, 0, boundsUniform, 0, BOUNDS_BYTES);
        enc.copyBufferToBuffer(boundsStorage, 0, boundsUniformB, 0, BOUNDS_BYTES);
      } else {
        // Held fixed, but the final frame's bound doesn't exist yet — the
        // previous run's is the only guess there is. Ordered against the submit
        // below on the queue timeline, so writing it here is safe.
        device.queue.writeBuffer(boundsUniform, 0, seedBounds);
        device.queue.writeBuffer(boundsUniformB, 0, seedBounds);
      }
      enc.copyBufferToBuffer(
        pm.positions,
        0,
        posHistory,
        slot * frameBytes,
        frameBytes
      );

      encodeRender(enc, pm.positions, 0);
      device.queue.submit([enc.finish()]);
    }

    /**
     * Draw a banked frame. No compute, no readback — copies and a pass.
     *
     * `f` is the fractional playhead, not a slot: the two keyframes it sits
     * between are bound together and mixed in the vertex stage, which is what
     * makes playback smooth rather than stepped. With `interpolation` off the
     * fraction is dropped and this is exactly the slot-at-a-time draw that
     * predates it.
     */
    function drawFrame(f: number) {
      const a = Math.max(0, Math.min(Math.floor(f), banked - 1));
      // Clamped, so the last frame mixes with itself rather than wrapping to
      // slot 0 — which matters most for a stopped run, where the trace ends
      // wherever it ended.
      const b = Math.min(a + 1, banked - 1);
      lerpT = view.interpolate ? f - a : 0;
      // Four bytes, and ordered against the submit below on the queue timeline
      // like the seedBounds write above. The rest of the struct is rewritten
      // only when the camera or a pane control moves.
      lerpScratch[0] = lerpT;
      device.queue.writeBuffer(viewUniform, LERP_T_OFFSET, lerpScratch);

      // Auto zoom frames each slot to its own extent, so the two keyframes'
      // boxes differ and get mixed alongside the positions; held, the whole
      // trace is framed by the last slot, so the scrubber shows travel rather
      // than a camera that cancels it out — and both bindings get that one box.
      const slotA = view.autoZoom ? a : banked - 1;
      const slotB = view.autoZoom ? b : banked - 1;
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(
        boundsHistory,
        slotA * BOUNDS_BYTES,
        boundsUniform,
        0,
        BOUNDS_BYTES
      );
      enc.copyBufferToBuffer(
        boundsHistory,
        slotB * BOUNDS_BYTES,
        boundsUniformB,
        0,
        BOUNDS_BYTES
      );
      encodeRender(enc, posHistory, a * frameBytes, b * frameBytes);
      device.queue.submit([enc.finish()]);
    }

    // --- Animate -----------------------------------------------------------
    status(
      `${N} points · ${algoLabel} · PCA ${tPca | 0}ms · ` +
        `kNN+pairs ${tSetup | 0}ms (${knnLabel}) · optimizing…`
    );

    // The transport stays put and inert during the run, tracking progress; the
    // playback block below is what hands the controls over.
    scrub.max = String(frameCount - 1);

    // Capture must happen on `stride` boundaries, so the loop steps in chunks
    // of `stride` and presents every `chunks` of them — that keeps the
    // iters/frame selector meaningful without dropping frames from the trace.
    const chunks = Math.max(1, Math.round(stepsPerFrame / stride));
    let it = 0;
    let slot = 0;
    // How many frames the trace actually holds. Equal to the planned
    // `frameCount` while the run is in flight and for any run that finishes;
    // narrowed below when a stop cuts the trace short. Declared here because
    // `setReadout` reads it from inside the loop.
    let banked = frameCount;
    frameIters[0] = 0;
    captureAndDraw(0); // the Gaussian init, before any optimizer step

    const tOpt = performance.now();
    while (it < pm.totalIters && !abort.signal.aborted) {
      for (
        let c = 0;
        c < chunks && it < pm.totalIters && !abort.signal.aborted;
        c++
      ) {
        const next = Math.min(it + stride, pm.totalIters);
        try {
          // Awaited for the CPU engine, where a step is a worker round-trip.
          // The GPU path returns void and resolves immediately, as before.
          await pm.runRange(it, next);
        } catch (e) {
          // Stopping mid-step terminates the worker, which rejects the request
          // this is awaiting. Everything banked before it is still a valid
          // trace, so break to the playback below rather than throwing past it
          // into the error path — the frames are the point of having stopped
          // rather than reloaded.
          if (abort.signal.aborted) break;
          throw e;
        }
        it = next;
        slot++;
        frameIters[slot] = it;
        captureAndDraw(slot); // reads pm.positions directly as a vertex buffer
      }
      scrub.value = String(slot);
      setReadout(slot);
      await frame();
    }
    const elapsed = performance.now() - tOpt;

    // A stopped run keeps everything it banked. The trace is shorter than the
    // schedule, so playback is bounded by what exists rather than by
    // `frameCount`, and the frames themselves are as valid as any other.
    const stopped = abort.signal.aborted;
    banked = slot + 1;

    // The bound the next run holds its live frames to, before it has one of its
    // own. Sixteen bytes, once, after the loop — not the per-frame `mapAsync` on
    // positions the render path forbids. Overwrites rather than unions, so a 2k
    // run after a 65k one isn't stuck with a view sized for the big one.
    {
      const staging = device.createBuffer({
        size: BOUNDS_BYTES,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(
        boundsHistory,
        (banked - 1) * BOUNDS_BYTES,
        staging,
        0,
        BOUNDS_BYTES
      );
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const last = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      if (last.every((x) => Number.isFinite(x))) {
        seedBounds = last;
        try {
          sessionStorage.setItem(BOUNDS_KEY, JSON.stringify(Array.from(last)));
        } catch {
          // Private mode or quota. The in-memory copy still serves this session.
        }
      }
    }

    // On a stop, this is what kills the druid worker — the whole point of the
    // button. Replay reads `posHistory`/`boundsHistory`, which are owned here
    // and outlive it, so the banked trace survives the teardown.
    //
    // Deliberately not done for a run that finished: that path is unchanged
    // from before this button existed, and it is the one someone is looking at
    // when the layout settles. Freeing there would be a real improvement — the
    // run's buffers currently live until the next run replaces them — but it is
    // a change to the main path that wants a browser to confirm, and this
    // change could not get one.
    if (stopped) pm.destroy();

    const mb = ((banked * frameBytes) / (1 << 20)).toFixed(0);
    status(
      `${N} points · ${algoLabel} · PCA ${tPca | 0}ms · ` +
        `kNN+pairs ${tSetup | 0}ms (${knnLabel}) · ` +
        `${stopped ? `stopped at ${it} of ${pm.totalIters}` : `${pm.totalIters}`} ` +
        `iters in ${elapsed | 0}ms ` +
        `(${(elapsed / Math.max(it, 1)).toFixed(2)}ms/iter, includes render + rAF) · ` +
        `${banked} frames banked on the GPU (${mb}MB` +
        `${stride > 1 ? `, every ${stride}th iter` : ""})`
    );

    // --- Playback ----------------------------------------------------------
    scrub.max = String(banked - 1);
    playhead = banked - 1;
    setPlaying(false);
    playBtn.disabled = false;
    scrub.disabled = false;
    onSeek = (f) => {
      playhead = Math.max(0, Math.min(f, banked - 1));
      scrub.value = String(playhead);
      setReadout(playhead);
    };
    onPlayToggle = () => {
      // Restarting from the top at the end is what a player is expected to do.
      if (!playing && playhead >= banked - 1) onSeek!(0);
      setPlaying(!playing);
    };
    setReadout(playhead);

    function setReadout(f: number) {
      f = Math.round(f);
      const iter = frameIters[f];
      iterEl.textContent = `${iter} / ${pm.totalIters}`;
      // LocalMAP differs from PaCMAP only in phase 3, and only strictly after
      // iteration 200 — same boundary the library applies to the near-pair
      // coefficient.
      phaseEl.textContent =
        iter <= 100 ? "1 · global (w_MN 1000→3)"
        : iter <= 200 ? "2 · local refine"
        : variant === "localmap" ? "3 · attract-repel + local graph"
        : "3 · attract-repel";
      readoutEl.textContent = `frame ${f + 1} / ${banked}`;
    }

    // Keeps drawing after convergence so resizing still works, and advances the
    // playhead when playing. Takes over the camera from `camTick`, which stops
    // itself on the next frame — this loop redraws unconditionally, so it needs
    // no dirty tracking.
    livePhase = false;
    let last = performance.now();
    const tick = (now: number) => {
      if (gen !== runGen) return; // a newer run owns the canvas
      const dt = (now - last) / 1000;
      last = now;
      orbit.update();
      if (playing) {
        const next = playhead + dt * view.playbackFps;
        if (next >= banked - 1) {
          onSeek!(banked - 1);
          setPlaying(false);
        } else {
          onSeek!(next);
        }
      }
      resize();
      // Fractional, not rounded: the fraction is what `drawFrame` mixes on.
      drawFrame(playhead);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch (e) {
    // A stop during setup surfaces here, as the rejection of whatever the run
    // was awaiting. It is a request that was granted, not a failure.
    if (abort.signal.aborted) {
      status("Stopped before the first iteration — nothing to replay.");
    } else {
      status(`Error: ${(e as Error).message}`);
      console.error(e);
    }
  } finally {
    running = false;
    runAbort = null;
    startBtn.textContent = "Run";
    sampleSel.disabled = false;
    pacmapFolder.disabled = false;
  }
}

function frame(): Promise<void> {
  return new Promise((r) => requestAnimationFrame(() => r()));
}

// ---------------------------------------------------------------------------
// Tweakpane
// ---------------------------------------------------------------------------

// The dense default (what the fixed radius used to be above 8k points) is the
// default at every N — a size that stays readable when crowded is a fine
// starting point when it isn't, and the slider is right there.
// `autoZoom` on is the behaviour that predates the checkbox: every frame framed
// to its own extent. Off holds one box for the whole trace, which is what makes
// the points' actual travel visible rather than cancelled out by the camera.
// `occlusion` is read only in 3D (see `occludingNow` in `go()`), where it
// decides between an opaque, depth-tested cloud and the blended haze that is
// the only sensible thing to draw when every point is coplanar.
const EDGE_TITLE = "pair graph";

const view = {
  pointSize: 1.8,
  autoZoom: false,
  occlusion: true,
  // What share of points draw as their own bitmap. Live, because the atlas
  // holds every digit and this only moves a threshold; 0 is no digits at all.
  digitPct: 100,
  // Digits get their own multiplier rather than dragging the whole cloud up
  // with them — at 1 a digit is exactly the point's own size.
  digitScale: 1,
  // Which of thumbColor's three looks. Live, like everything else here.
  digitStyle: 0 as DigitStyle,
  // Smoothed playback: adjacent keyframes mixed in the vertex stage rather than
  // one drawn at a time. Off is the stepped playback that predates it, exactly.
  interpolate: true,
  // How many keyframes a second playback advances, which is also what decides
  // whether there is anything to interpolate: at 60 on a 60Hz panel the cursor
  // lands on a keyframe every tick and the mix has nothing to fill. 30 gives
  // two drawn frames per pair there, more on a faster display. Capped at 60,
  // above which it only skips keyframes.
  playbackFps: 30,
  // The pair-graph overlay. Off by default: the point cloud is the thing being
  // demonstrated and three million lines over it is a wash of colour.
  edges: false,
  // Per kind, and all three low, because what is wanted from the overlay is the
  // *shape* of each relation rather than the whole of it — a sample big enough
  // to read the structure and small enough to see the cloud through. Every one
  // of these is a few percent of a set that also acts on the other endpoint, so
  // the drawn density is about double what the number suggests.
  //
  // Neighbours get the largest share: they are the local structure, they are
  // short, and they land inside clusters where there is room. Mid-near and
  // further both span the whole cloud, so a line of either crosses everything
  // between its endpoints — 1% of those is legible where 10% is a wash.
  //
  // Render-time: the index buffer holds every pair and this moves a draw count.
  edgePct: { near: 10, midNear: 1, further: 1 } as Record<EdgeKind, number>,
  // Percent, matching EDGE_ALPHA's 0.35 default. Live, like edgePct: this
  // rewrites the alpha float already sitting in each kind's EdgeStyle slot,
  // no buffer rebuilt and no shader touched.
  edgeAlpha: 35,
};

/** Installed by a run so an edit can rewrite that run's view uniform. */
let onViewChange: (() => void) | null = null;

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

// A real camera rather than a 2D pan/zoom transform, because 3D rendering is the
// next step: this stays as-is and gains rotation, where an affine {scale, offset}
// would have to be thrown away. Only ogl's camera and controller are used — its
// `Renderer` is never constructed and no WebGL context exists; the WGSL pipeline
// below does the drawing, and all this supplies is a view-projection matrix.
//
// `gl` is the first constructor argument and is genuinely unused by `Camera`
// (checked against ogl 1.0.11's source, which never assigns or reads it), which
// is what lets it be driven headless like this.
const FOV = 45;
const camera = new Camera(null as unknown as OGLRenderingContext, {
  fov: FOV,
  near: 0.01,
  far: 1000,
});

/**
 * The distance at which the projection reproduces the framing that predates the
 * camera, exactly. World space is `(p - ctr) / (span * 0.55)`, so the embedding's
 * half-extent is `1 / (2 * 0.55) = 0.909`; the old shader mapped that straight to
 * 0.909 of NDC height. A perspective camera sees a half-height of `d*tan(fov/2)`,
 * so wanting 0.909 world → 0.909 NDC means the visible half-height must be 1, i.e.
 * `d = 1/tan(fov/2)`. Same margin, same aspect handling, same pixels.
 */
const DEFAULT_DIST = 1 / Math.tan((FOV * Math.PI) / 360);
camera.position.set(0, 0, DEFAULT_DIST);

// The cast is because ogl's .d.ts omits `mouseButtons`, which its constructor
// does set (`Orbit.js`); the property is real, only the declaration is short.
type MouseButtons = { ORBIT: number; ZOOM: number; PAN: number };

/**
 * The controller for one dimensionality.
 *
 * Rebuilt rather than reconfigured when the mode changes, because ogl captures
 * `enableRotate` as a constructor closure variable rather than assigning it to
 * the instance — `orbit.enableRotate = true` looks right, type-checks against
 * `OrbitOptions`, and does nothing. (ogl 1.0.11; the .d.ts is accurate, listing
 * it under the options and not on the class.) `remove()` detaches every
 * listener, so this does not stack handlers the way constructing one per run
 * would, and the new controller seeds its spherical coordinates from the
 * camera's current position and the target it is handed — so the view carries
 * across a switch untouched.
 *
 * The mouse map differs with the mode. In 2D there is nothing to rotate, so
 * left-drag has to pan or it would do nothing at all, and that needs both
 * halves of the remap: the switch in `onMouseDown` tests ORBIT first and
 * returns early when rotation is disabled, so pointing PAN at button 0 is not
 * enough — ORBIT has to move off it too. In 3D left-drag orbits and right-drag
 * pans, ogl's own default and what a 3D viewer is expected to do (Orbit
 * suppresses the context menu itself, so the right button is free).
 *
 * One gesture is missing in 2D either way: one-finger touch drag, which Orbit
 * routes to rotate. Two-finger pinch and pan both work.
 */
function makeOrbit(d: Components, target: Vec3): Orbit {
  const orbit = new Orbit(camera, {
    element: canvas,
    target,
    enableRotate: d === 3,
    // These three go together, and none of them is right alone.
    //
    // `update()` adds the whole of panDelta *and* sphericalDelta to the target
    // every frame and only decays them by `inertia` afterwards — it never
    // clears either — so one drag delta is applied d + 0.85d + 0.85^2 d + ...
    // = d/(1 - inertia), i.e. 6.7x over at the default. Both of ogl's 0.1
    // speeds are tuned against that coasting rather than against their own
    // pixel conversions, so dropping inertia means restoring the 6.7x by hand
    // on both of them, not just one.
    //
    // inertia: 0 applies each delta once and drops the throw, which is what
    // makes a drag track the cursor exactly — the point under the pointer stays
    // under it, Google-Maps style. Pan is never eased (the target is moved
    // directly), so `ease` is left alone: it only smooths the zoom and rotate.
    //
    // panSpeed: Orbit's pixel-to-world conversion is exactly 1:1 already
    // (2*d*tan(fov/2)/clientHeight, and every point sits on the target plane),
    // so 1 is the value that tracks.
    //
    // rotateSpeed: 0.1/(1 - 0.85) = 2/3 reproduces ogl's own angular travel per
    // pixel, which is what the reference viewer this borrows from feels like.
    // At 0.1 a drag rotates 6.7x less than expected — the same factor, and the
    // reason this was missed at first: with rotation off there was nothing to
    // notice it on.
    panSpeed: 1,
    rotateSpeed: 0.1 / 0.15,
    inertia: 0,
    minDistance: DEFAULT_DIST / 40,
    maxDistance: DEFAULT_DIST * 20,
  });
  (orbit as Orbit & { mouseButtons: MouseButtons }).mouseButtons =
    d === 3 ? { ORBIT: 0, ZOOM: 1, PAN: 2 } : { ORBIT: 2, ZOOM: 1, PAN: 0 };
  return orbit;
}

// One controller at a time, at module scope rather than per run: `go()` already
// leaks a window resize listener per run, harmless only because of the
// `gen === runGen` guard, and a controller attaching its own handlers per run
// would stack them for real.
let cameraMode: Components = DIMS_MODE;
let orbit = makeOrbit(cameraMode, new Vec3(0, 0, 0));

/** Switch the controller's mode, carrying the current view across. */
function setCameraMode(d: Components) {
  if (d === cameraMode) return;
  cameraMode = d;
  const target = orbit.target.clone();
  orbit.remove();
  orbit = makeOrbit(d, target);
}

/** Back to the designated view: the framing that predates the camera. */
function resetCamera() {
  orbit.target.set(0, 0, 0);
  camera.position.set(0, 0, DEFAULT_DIST);
  camera.up.set(0, 1, 0);
  // Required, and easy to miss: Orbit keeps its own spherical coordinates and
  // eases toward them, so without this the next update() snaps straight back to
  // wherever the user left it.
  orbit.forcePosition();
}

canvas.addEventListener("dblclick", resetCamera);

/**
 * PaCMAP pair ratios. n_MN and n_FP are counts per point, derived as
 * round(n_neighbors * ratio) inside the library, so these are the knobs the
 * reference exposes rather than the counts themselves.
 *
 * Unlike point size these are setup-time: the pairs are sampled and the CSR is
 * built once, before the first iteration, so an edit cannot apply to a run
 * already in flight. The folder is disabled while one is running and the values
 * are read at the top of the next `go()`.
 */
const params = {
  // Off by default: upstream PaCMAP's signature default is a flat
  // n_neighbors=10, and its log10 rule only fires when the caller explicitly
  // passes None. This demo matches the signature default, not the rule.
  autoNeighbors: false,
  nNeighbors: 10,
  mnRatio: 0.5,
  fpRatio: 2.0,
  seed: 7,
  // Seeded from ?knn= / ?algo= so the URLs still work, but the dropdowns own
  // them after that — comparing backends and variants is the point, and that
  // shouldn't need a reload.
  knnMethod: KNN_MODE,
  algorithm: ALGO_MODE,
  nComponents: DIMS_MODE,
  // Upstream's default. Only read under localmap.
  lowDistThres: 10,
};

const KNN_LABELS: Record<KnnMode, string> = {
  gpu: "brute force (GPU)",
  nndescent: "NN-Descent (GPU)",
  cpu: "brute force (CPU)",
};

/**
 * Both halves of the choice, named where the reader is: which algorithm, and
 * whose implementation of it. "custom" is the WGSL in `pacmap-webgpu.ts`;
 * "druid" is DruidJS, an independent implementation of the same two papers,
 * which is the entire point of offering it — a layout you can compare against.
 */
const ALGO_LABELS: Record<AlgoKey, string> = {
  "pacmap-gpu": "PaCMAP (GPU - custom)",
  "localmap-gpu": "LocalMAP (GPU - custom)",
  "pacmap-cpu": "PaCMAP (CPU - druid)",
  "localmap-cpu": "LocalMAP (CPU - druid)",
};

const pane = new Pane({
  container: document.getElementById("pane") as HTMLElement,
  title: "controls",
});

// First, because it is where a run starts: nothing in `rendering` below matters
// until something has been embedded.
const pacmapFolder = pane.addFolder({ title: "dimensional reduction" });

// Every parameter below is shared by all four entries — LocalMAP inherits
// PaCMAP's whole setup path upstream, and druid takes the same knobs our
// library does — so this is one folder with a variant selector rather than a
// folder per algorithm.
//
// The map is annotated rather than inferred on purpose. Tweakpane types
// `options` as plain strings, so a typo in a *value* here type-checks, and
// `VARIANT_OF[key]` would then be undefined while the pane claimed otherwise.
// `Record<string, AlgoKey>` is what makes that a build error, and it matters
// more with four entries than it did with two. (The kNN dropdown below still
// has the same hole; separate change.)
const ALGO_OPTIONS: Record<string, AlgoKey> = {
  [ALGO_LABELS["pacmap-gpu"]]: "pacmap-gpu",
  [ALGO_LABELS["localmap-gpu"]]: "localmap-gpu",
  [ALGO_LABELS["pacmap-cpu"]]: "pacmap-cpu",
  [ALGO_LABELS["localmap-cpu"]]: "localmap-cpu",
};

pacmapFolder
  .addBinding(params, "algorithm", {
    label: "algorithm",
    options: ALGO_OPTIONS,
  })
  .on("change", (e) => syncAlgorithm(e.value as AlgoKey));

// Same annotation discipline as ALGO_OPTIONS above, and for the same reason:
// Tweakpane types option values as plain strings, so without it a typo here
// would type-check and reach the library as an unusable width.
const COMPONENT_OPTIONS: Record<string, Components> = { "2D": 2, "3D": 3 };

// Setup-time like everything else in this folder: the buffers, the generated
// shaders and the vertex layout are all sized from it before the first
// iteration, so it is read at the top of `go()` and the folder is disabled for
// the duration of the run.
pacmapFolder
  .addBinding(params, "nComponents", {
    label: "components",
    options: COMPONENT_OPTIONS,
  })
  .on("change", (e) => syncComponents(e.value as Components));

/**
 * Bring the camera and the view folder in line with the selected width.
 *
 * Occlusion is meaningless in 2D — every point sits on one plane, so a depth
 * buffer would resolve ties by whatever order the buffer happens to be in —
 * and the 2D pipeline is built without depth state at all, so the checkbox
 * would be inert as well as pointless. Greyed rather than hidden, so it does
 * not move the folder's other rows around.
 */
function syncComponents(d: Components) {
  setCameraMode(d);
  occlusionBinding.disabled = d === 2;
  edgeAlphaBinding.disabled = !occlusionBinding.disabled && view.occlusion;
}

/**
 * Grey out what the selected engine does not read, and refresh the cost hint.
 *
 * Both matter for honesty rather than tidiness: druid runs its own exact
 * neighbour search, so leaving the kNN dropdown live under the CPU engine would
 * imply a choice that has no effect, and the two engines' costs differ by orders
 * of magnitude, so the hint is wrong until it is told which one is selected.
 */
function syncAlgorithm(key: AlgoKey) {
  lowDistBinding.disabled = VARIANT_OF[key] === "pacmap";
  knnBinding.disabled = ENGINE_OF[key] === "cpu";
  // Druid builds its own neighbour graph internally and exposes nothing, so
  // `EmbeddingRun.graph` is undefined there and `go()` builds no overlay at all.
  // Greyed rather than hidden, like `occlusion`, so the folder keeps its size.
  edgeFolder.disabled = ENGINE_OF[key] === "cpu";
  // LocalMAP redraws the further pairs against the embedding 24 times during
  // phase 3, on the GPU, and those redraws never come back to the host — so the
  // red edges are the set drawn at setup for the whole run. Said here because
  // it is visible: they sit still while the points move.
  edgeFolder.title =
    VARIANT_OF[key] === "localmap"
      ? `${EDGE_TITLE} — further pairs are the initial draw`
      : EDGE_TITLE;
  updateSampleLabel();
}

// LocalMAP's only new knob. Read in two places — it scales the phase-3 near-pair
// gradient (as lowDistThres/2) and bounds how far a redrawn further pair may sit
// in the embedding — so it is one slider, not two. Inert under pacmap.
const lowDistBinding = pacmapFolder.addBinding(params, "lowDistThres", {
  label: "low_dist_thres",
  min: 2,
  max: 30,
  step: 0.5,
});
// Ordered by how much the backend takes on trust: the CPU oracle first as the
// default, then the exact GPU kernel, then the approximate one. No speed
// annotation on any of them — measured, the ordering doesn't match what the
// asymptotics suggest, so a label here would mislead.
//
// Inert under the CPU engine: druid builds its own neighbour graph and never
// consults this.
const knnBinding = pacmapFolder.addBinding(params, "knnMethod", {
  label: "kNN algo",
  options: {
    [KNN_LABELS.cpu]: "cpu",
    [KNN_LABELS.gpu]: "gpu",
    [KNN_LABELS.nndescent]: "nndescent",
  },
});

// Turning "auto neighbors" on hands n_neighbors to the library's log10-of-N
// rule; the slider then mirrors the value that rule picks rather than leaving
// it implicit, and goes read-only because it is no longer the input.
pacmapFolder
  .addBinding(params, "autoNeighbors", { label: "auto neighbors" })
  .on("change", (e) => {
    nnBinding.disabled = e.value;
    syncAutoNeighbors();
  });
const nnBinding = pacmapFolder.addBinding(params, "nNeighbors", {
  label: "n_neighbors",
  min: 3,
  max: 60,
  step: 1,
});

/** Keeps the displayed n_neighbors in step with the point-count slider. */
function syncAutoNeighbors() {
  if (!params.autoNeighbors) return;
  params.nNeighbors = defaultNeighbors(parseInt(sampleSel.value, 10));
  nnBinding.refresh();
}

// Ranges top out at 3x the reference defaults (0.5 / 2.0), enough travel to see
// the structure change: dropping MN toward 0 loses global layout, raising FP
// pushes clusters apart. 0 is a legal value for both — it just drops that pair
// kind entirely.
pacmapFolder.addBinding(params, "mnRatio", {
  label: "MN ratio",
  min: 0,
  max: 1.5,
  step: 0.05,
});
pacmapFolder.addBinding(params, "fpRatio", {
  label: "FP ratio",
  min: 0,
  max: 6,
  step: 0.05,
});
// The seed drives pair sampling and the Gaussian init, so scrubbing it is how
// you tell a stable structure from an artifact of one layout.
pacmapFolder.addBinding(params, "seed", { min: 0, max: 999, step: 1 });

const viewFolder = pane.addFolder({ title: "rendering" });
viewFolder
  .addBinding(view, "pointSize", {
    label: "point size",
    min: 0.05,
    max: 2,
    step: 0.05,
  })
  .on("change", () => onViewChange?.());
// Nothing to install for this one: the post-run rAF redraws every tick and
// `captureAndDraw` re-reads it per captured frame, so a toggle lands on the next
// frame either way. `onViewChange` is here only to keep the two bindings alike.
viewFolder
  .addBinding(view, "autoZoom", { label: "auto zoom" })
  .on("change", () => onViewChange?.());
// Nothing to install here either — the flag rides in the view uniform, which
// `resize()` rewrites on the next redraw, and `onViewChange` is what marks the
// live phase dirty enough to have one.
const occlusionBinding = viewFolder
  .addBinding(view, "occlusion", { label: "occlusion" })
  .on("change", () => {
    onViewChange?.();
    // Occlusion forces edge alpha to 1.0 in the shader (see edgeWGSL), so the
    // opacity slider is dead weight while it's on — grey it rather than leave
    // a live-looking control that does nothing.
    edgeAlphaBinding.disabled = !occlusionBinding.disabled && view.occlusion;
  });
// All three ride in the view uniform like `occlusion`, so there is nothing to
// install for them either.
viewFolder
  .addBinding(view, "digitPct", {
    label: "digit %",
    min: 0,
    max: 100,
    step: 0.5,
  })
  .on("change", () => onViewChange?.());
viewFolder
  .addBinding(view, "digitScale", {
    label: "digit scale",
    min: 1,
    max: 20,
    step: 0.5,
  })
  .on("change", () => onViewChange?.());
viewFolder
  .addBinding(view, "digitStyle", {
    label: "digit style",
    options: DIGIT_STYLES,
  })
  .on("change", () => onViewChange?.());
// Both read by the playback loop, which redraws unconditionally, so like
// `autoZoom` there is nothing to install — `onViewChange` is here to keep the
// bindings alike. Neither does anything during a live run: there is no next
// keyframe to interpolate toward, and the run's own cadence is `iters/frame`.
viewFolder
  .addBinding(view, "interpolate", { label: "interpolation" })
  .on("change", () => onViewChange?.());
viewFolder
  .addBinding(view, "playbackFps", {
    label: "playback fps",
    min: 1,
    max: 60,
    step: 1,
  })
  .on("change", () => onViewChange?.());
// Double-clicking the canvas does the same thing.
viewFolder
  .addButton({ title: "reset camera" })
  .on("click", () => resetCamera());

// ---------------------------------------------------------------------------
// The pair graph
//
// Its own folder rather than more rows under `rendering`, because it is about
// the algorithm's structure where the rest of that folder is about how the
// cloud is drawn. Live like `rendering` though, and for a stronger reason: the
// index buffer holds every pair, so a percentage moves a draw count and nothing
// else — no buffer rewritten, no run restarted.
//
// The labels carry what a colour cannot. Green pulls, red pushes, and yellow
// pulls only for the first 200 of 450 iterations — `weightsAt` sends w_MN to 0
// for the whole of phase 3, so a drawn mid-near edge is not necessarily a
// pulling one. That is the single place where "an edge means this pair exists"
// reads as a stronger claim than it is, and it is cheaper to say so than to
// encode weight into opacity.
// ---------------------------------------------------------------------------

const edgeFolder = pane.addFolder({ title: EDGE_TITLE });
edgeFolder
  .addBinding(view, "edges", { label: "show edges" })
  .on("change", () => onViewChange?.());
// Greyed whenever occlusion is actually in effect (3D and the checkbox on):
// the occluded fragment hardcodes alpha to 1.0 and ignores this entirely
// (see edgeWGSL), so a live slider there would silently do nothing.
const edgeAlphaBinding = edgeFolder
  .addBinding(view, "edgeAlpha", { label: "opacity", min: 0, max: 100, step: 1 })
  .on("change", () => onViewChange?.());
// (pos)/(neg) is the sign of the force, not of anything measured: near and
// mid-near pull, further pushes. Mid-near's is true only for the first 200 of
// 450 iterations — `weightsAt` sends w_MN to 0 for the whole of phase 3 — which
// the folder cannot fit and `CLAUDE.md` records instead.
const EDGE_LABELS: Record<EdgeKind, string> = {
  near: "% neighbors (pos)",
  midNear: "% mid-near (pos)",
  further: "% far (neg)",
};
for (const kind of EDGE_KINDS) {
  edgeFolder
    .addBinding(view.edgePct, kind, {
      label: EDGE_LABELS[kind],
      min: 0,
      max: 100,
      step: 0.5,
    })
    .on("change", () => onViewChange?.());
}

// Bindings all exist now, so the pane can be brought in line with whatever
// `?algo=` and `?dims=` selected. Also draws the first cost hint.
syncComponents(params.nComponents);
syncAlgorithm(params.algorithm);

// ---------------------------------------------------------------------------
// Transport (play/pause + scrubber)
//
// The controls live here, at module scope, but the frames they address are
// per-run GPU buffers, so the run installs `onSeek`/`onPlayToggle` once its
// history exists. Before that the buttons are disabled and these stay null.
// ---------------------------------------------------------------------------

let playing = false;
/** Fractional frame index; the render loop rounds it to a banked slot. */
let playhead = 0;
let onSeek: ((f: number) => void) | null = null;
let onPlayToggle: (() => void) | null = null;

function setPlaying(on: boolean) {
  playing = on;
  playBtn.textContent = on ? "❚❚" : "▶";
}

// The bar itself is never hidden — it holds its place in the layout and just
// goes inert, so starting a run doesn't reflow the canvas out from under you.
function resetTransport() {
  onSeek = null;
  onPlayToggle = null;
  setPlaying(false);
  playhead = 0;
  playBtn.disabled = true;
  scrub.disabled = true;
  scrub.value = "0";
  readoutEl.textContent = "—";
}

playBtn.addEventListener("click", () => onPlayToggle?.());
scrub.addEventListener("input", () => {
  setPlaying(false);
  onSeek?.(Number(scrub.value));
});

window.addEventListener("keydown", (e) => {
  const tag = (e.target as HTMLElement | null)?.tagName;
  // Sliders and buttons already handle these keys themselves — except the
  // scrubber, which since step="any" no longer steps by a keyframe under the
  // browser's own arrow handling but by 1% of the range. Whole keyframes are
  // what an arrow key should mean, so this handler takes it back once the
  // scrubber has focus (which one click on it is enough to do).
  if (e.target !== scrub && (tag === "SELECT" || tag === "INPUT" || tag === "BUTTON")) {
    return;
  }
  const step = e.shiftKey ? 10 : 1;
  if (e.code === "Space") {
    e.preventDefault();
    onPlayToggle?.();
  } else if (e.code === "ArrowRight" || e.code === "ArrowLeft") {
    e.preventDefault();
    setPlaying(false);
    onSeek?.(Math.round(playhead) + (e.code === "ArrowRight" ? step : -step));
  }
});

// ---------------------------------------------------------------------------
// kNN self-check (?knncheck=1)
// ---------------------------------------------------------------------------

interface KnnAgreement {
  recall: number;
  exactOrder: number;
  maxRel: number;
}

/**
 * Scores one backend's output against the CPU oracle's.
 *
 * Nothing here will agree bit-for-bit even among the exact backends — JS
 * accumulates distances in f64, WGSL in f32 — so near-ties legitimately swap
 * order and comparing index arrays for equality would cry wolf. Recall is the
 * metric that would actually catch a broken kernel.
 *
 * The other two need reading differently depending on the backend. For an exact
 * backend, a low exact-order score or a large max rel Δd² means a bug. For an
 * approximate one both are *expected* to move: NN-Descent legitimately returns
 * a different neighbor at some positions, and max rel Δd² then measures how
 * much worse that substitute is — which is the useful thing to know about it,
 * not a defect.
 */
function compareKnn(
  oracle: { idx: Uint32Array; d2: Float32Array },
  got: { idx: Uint32Array; d2: Float32Array },
  M: number,
  kCand: number
): KnnAgreement {
  let recallHits = 0;
  let exact = 0;
  let maxRel = 0;
  const seen = new Set<number>();
  for (let i = 0; i < M; i++) {
    const base = i * kCand;
    seen.clear();
    for (let k = 0; k < kCand; k++) seen.add(got.idx[base + k]);
    for (let k = 0; k < kCand; k++) {
      if (seen.has(oracle.idx[base + k])) recallHits++;
      if (oracle.idx[base + k] === got.idx[base + k]) exact++;
      // Both lists are sorted ascending, so comparing distances positionally
      // stays valid even where tied indices swapped.
      const a = oracle.d2[base + k];
      const rel = Math.abs(a - got.d2[base + k]) / Math.max(Math.abs(a), 1e-12);
      if (rel > maxRel) maxRel = rel;
    }
  }
  const tot = M * kCand;
  return { recall: recallHits / tot, exactOrder: exact / tot, maxRel };
}

/**
 * Runs every kNN backend over one identical input and reports how they compare.
 *
 * `bruteForceKnn` is the oracle — it defines what the right answer is, so it is
 * run once and everything else is scored against it. That makes this both the
 * regression guard on the exact GPU kernel and the recall measurement for the
 * approximate one, which are otherwise two different questions.
 */
async function knnSelfCheck(device: GPUDevice, Z: Float32Array, N: number) {
  // Capped: this runs every backend serially and the check is meant to stay
  // interactive. A prefix is a perfectly valid comparison — every backend sees
  // the identical input.
  const M = Math.min(N, 2000);
  const kCand = Math.min(60, M - 1);
  const Zm = Z.subarray(0, M * 100);

  status(`kNN self-check at ${M} points: CPU oracle…`);
  await frame();
  const tc = performance.now();
  const oracle = await bruteForceKnn(Zm, M, 100, kCand, status);
  const cpuMs = performance.now() - tc;

  const contenders: [string, () => Promise<{ idx: Uint32Array; d2: Float32Array }>][] =
    [
      ["brute force (GPU)", () => knnGPU(device, Zm, M, 100, kCand, status)],
      [
        "NN-Descent (GPU)",
        () =>
          nndescentGPU(device, Zm, M, 100, kCand, {
            seed: params.seed,
            onStatus: status,
          }),
      ],
    ];

  const lines = [
    `kNN check · N=${M} k=${kCand}`,
    `  brute force (CPU)   oracle` +
      `                                        ${(cpuMs | 0)
        .toString()
        .padStart(7)}ms`,
  ];
  const brief: string[] = [];

  for (const [name, run] of contenders) {
    status(`kNN self-check at ${M} points: ${name}…`);
    await frame();
    const t = performance.now();
    const got = await run();
    const ms = performance.now() - t;
    const a = compareKnn(oracle, got, M, kCand);

    lines.push(
      `  ${name.padEnd(20)}` +
        `recall ${(a.recall * 100).toFixed(3).padStart(7)}%  ` +
        `exact ${(a.exactOrder * 100).toFixed(1).padStart(5)}%  ` +
        `max rel Δd² ${a.maxRel.toExponential(1)}  ` +
        `${(ms | 0).toString().padStart(7)}ms  ` +
        `${(cpuMs / Math.max(ms, 1)).toFixed(1)}× vs CPU`
    );
    brief.push(
      `${name} ${(a.recall * 100).toFixed(2)}% / ${ms | 0}ms`
    );
  }

  console.log(lines.join("\n"));
  status(`kNN check · N=${M} k=${kCand} · CPU ${cpuMs | 0}ms · ${brief.join(" · ")} · full table in console`);
  await new Promise((r) => setTimeout(r, 4000));
}

// ---------------------------------------------------------------------------
// Sample-count slider
// ---------------------------------------------------------------------------

/**
 * Very rough cost estimate, so the price of the high end is visible before
 * committing to a run.
 *
 * The shared prefix is PCA, ~4*n*784*100 MACs of plain JS. After that the two
 * engines diverge by orders of magnitude, which is the reason this takes an
 * engine at all: on the GPU the optimizer is free and the quadratic kNN term
 * only shows past ~40k, while druid pays an exact O(N^2*D) neighbour search on
 * one CPU thread and then runs 450 f64 iterations.
 *
 * The druid coefficients are measured, not derived — 8.5e-8 s/N^2 for the
 * search and 2.7e-4 s/point for the whole optimizer, taken at D=100 over
 * N = 500…4000. Extrapolating a quadratic four-fold is exactly the kind of
 * estimate that should be distrusted, so treat the large end as an order of
 * magnitude ("this is minutes, not seconds") rather than a prediction.
 */
function estimateSetupSecs(n: number, engine: Engine): number {
  const pca = (n * 784 * 100 * 4) / 5e8;
  return engine === "cpu"
    ? pca + 8.5e-8 * n * n + 2.7e-4 * n
    : pca + (n * n * 100) / 3e11 + n * 4e-5;
}

sampleSel.max = String(NUM_AVAILABLE);

/** Seconds as something readable at both ends: 2.4s, 47s, 7min. */
function humanSecs(s: number): string {
  if (s < 10) return `${s.toFixed(1)}s`;
  if (s < 90) return `${Math.round(s)}s`;
  return `${Math.round(s / 60)}min`;
}

function updateSampleLabel() {
  const n = parseInt(sampleSel.value, 10);
  sampleOut.value = n.toLocaleString();
  const engine = ENGINE_OF[params.algorithm];
  const s = estimateSetupSecs(n, engine);
  // The CPU figure covers the whole run, not just setup, because under druid
  // the optimizer is no longer the free part.
  sampleHint.textContent =
    engine === "cpu" ? `~${humanSecs(s)} run (CPU)` : `~${humanSecs(s)} setup`;
  syncAutoNeighbors();
}
sampleSel.addEventListener("input", updateSampleLabel);
updateSampleLabel();

// Legend
const legend = document.getElementById("legend")!;
PALETTE.forEach((c, i) => {
  const s = document.createElement("span");
  s.className = "swatch";
  s.style.setProperty(
    "--c",
    `rgb(${c.map((v) => Math.round(v * 255)).join(",")})`
  );
  s.textContent = String(i);
  legend.appendChild(s);
});

// One button, two jobs. A run that can take minutes needs a way out that is
// where you already are — and the only alternative was closing the tab.
startBtn.addEventListener("click", () => {
  if (running) runAbort?.abort();
  else void go();
});
