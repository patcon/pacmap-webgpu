import {
  pacmapWebGPU,
  bruteForceKnn,
  knnGPU,
  nndescentGPU,
  defaultNeighbors,
  type PacmapRun,
} from "./pacmap-webgpu";
import { loadMnist, IMAGE_SIZE, NUM_AVAILABLE } from "./mnist";
import { pcaProject } from "./pca";
import { Pane } from "tweakpane";

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
//   ?knn=gpu|nnd  pick the kNN backend (default cpu brute force)
//   ?knncheck=1   run every backend over one input and report how they compare
// ---------------------------------------------------------------------------

type KnnMode = "gpu" | "cpu" | "nndescent";

const qs = new URLSearchParams(location.search);
const KNN_MODE: KnnMode =
  qs.get("knn") === "gpu" ? "gpu"
  : qs.get("knn") === "nnd" || qs.get("knn") === "nndescent" ? "nndescent"
  : "cpu";
const KNN_CHECK = qs.get("knncheck") === "1";

// Playback history. Every captured frame is a full N x 2 f32 snapshot kept in
// GPU memory, so the scrubber never reads positions back to the host — same
// constraint as the live render path. At 65k points a frame is 520KB, so the
// budget, not the iteration count, is what decides how many we keep.
const HISTORY_BUDGET_BYTES = 128 << 20;
const PLAYBACK_FPS = 60;

// Colors for digits 0-9. Distinct hues, roughly matched in luminance so no one
// class dominates visually.
const PALETTE: [number, number, number][] = [
  [0.90, 0.31, 0.31], [0.95, 0.60, 0.20], [0.85, 0.80, 0.22],
  [0.44, 0.78, 0.34], [0.22, 0.72, 0.62], [0.29, 0.62, 0.90],
  [0.45, 0.45, 0.92], [0.72, 0.40, 0.88], [0.92, 0.42, 0.70],
  [0.62, 0.62, 0.66],
];

// ---------------------------------------------------------------------------
// Shaders for the demo's own render + bounds passes
// ---------------------------------------------------------------------------

const boundsWGSL = (N: number) => /* wgsl */ `
@group(0) @binding(0) var<storage, read>       Y : array<f32>;
@group(0) @binding(1) var<storage, read_write> B : array<f32>;  // loX loY hiX hiY

const N : u32 = ${N}u;
var<workgroup> sLo : array<vec2<f32>, 256>;
var<workgroup> sHi : array<vec2<f32>, 256>;

// Single workgroup, grid-stride over all points. Keeps autoscaling on the GPU
// so the render loop never has to read positions back to the host.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  var lo = vec2<f32>( 1e30,  1e30);
  var hi = vec2<f32>(-1e30, -1e30);
  for (var i : u32 = t; i < N; i = i + 256u) {
    let p = vec2<f32>(Y[2u * i], Y[2u * i + 1u]);
    lo = min(lo, p);
    hi = max(hi, p);
  }
  sLo[t] = lo;
  sHi[t] = hi;
  workgroupBarrier();
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    if (t < s) {
      sLo[t] = min(sLo[t], sLo[t + s]);
      sHi[t] = max(sHi[t], sHi[t + s]);
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    B[0] = sLo[0].x; B[1] = sLo[0].y;
    B[2] = sHi[0].x; B[3] = sHi[0].y;
  }
}
`;

const renderWGSL = /* wgsl */ `
struct Bounds { lo : vec2<f32>, hi : vec2<f32> };
struct View   { res : vec2<f32>, radius : f32, _pad : f32 };

@group(0) @binding(0) var<uniform> B : Bounds;
@group(0) @binding(1) var<uniform> V : View;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       col  : vec3<f32>,
  @location(1)       uv   : vec2<f32>,
};

const PALETTE = array<vec3<f32>, 10>(
  vec3<f32>(0.90, 0.31, 0.31), vec3<f32>(0.95, 0.60, 0.20),
  vec3<f32>(0.85, 0.80, 0.22), vec3<f32>(0.44, 0.78, 0.34),
  vec3<f32>(0.22, 0.72, 0.62), vec3<f32>(0.29, 0.62, 0.90),
  vec3<f32>(0.45, 0.45, 0.92), vec3<f32>(0.72, 0.40, 0.88),
  vec3<f32>(0.92, 0.42, 0.70), vec3<f32>(0.62, 0.62, 0.66),
);

// Two triangles per point, instanced. @location(0) is bound directly to the
// PaCMAP position buffer — no copy, no readback.
@vertex
fn vs(
  @builtin(vertex_index) vi  : u32,
  @location(0)           p   : vec2<f32>,
  @location(1)           lab : u32,
) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vi];

  // Uniform scale on both axes so the embedding isn't stretched.
  let ctr  = (B.lo + B.hi) * 0.5;
  let span = max(max(B.hi.x - B.lo.x, B.hi.y - B.lo.y), 1e-6);
  var q = (p - ctr) / (span * 0.55);
  let aspect = V.res.x / V.res.y;
  q.x = q.x / aspect;

  let r = vec2<f32>(2.0 * V.radius / V.res.x, 2.0 * V.radius / V.res.y);

  var out : VSOut;
  out.clip = vec4<f32>(q + c * r, 0.0, 1.0);
  out.col  = PALETTE[min(lab, 9u)];
  out.uv   = c;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  return vec4<f32>(in.col, (1.0 - smoothstep(0.7, 1.0, d)) * 0.85);
}
`;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

let running = false;
// Bumped on every run so a previous run's render loop and resize handler stop
// touching the canvas (their buffers belong to a device that is now stale).
let runGen = 0;

async function go() {
  if (running) return;
  running = true;
  startBtn.disabled = true;
  const gen = ++runGen;
  resetTransport();
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

    if (KNN_CHECK) await knnSelfCheck(device, Z, N);

    const knnLabel = KNN_LABELS[params.knnMethod];
    status(
      `PCA ${tPca | 0}ms · building kNN graph (${knnLabel}) + sampling pairs…`
    );
    await frame();
    const t1 = performance.now();
    const pm: PacmapRun = await pacmapWebGPU(device, Z, N, 100, {
      seed: params.seed,
      knn: params.knnMethod,
      nNeighbors: params.autoNeighbors ? undefined : params.nNeighbors,
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

    const boundsStorage = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const boundsUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const viewUniform = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // --- Playback history --------------------------------------------------
    // One slot per captured frame, plus slot 0 for the pre-optimization init.
    // `stride` iterations are collapsed into one slot when the full trace would
    // not fit the budget, so the timeline stays complete (just coarser).
    const frameBytes = N * 8;
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
      size: frameCount * 16,
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

    const renderModule = device.createShaderModule({ code: renderWGSL });
    const renderPipe = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: renderModule,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 8,
            stepMode: "instance",
            attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }],
          },
          {
            arrayStride: 4,
            stepMode: "instance",
            attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" }],
          },
        ],
      },
      fragment: {
        module: renderModule,
        entryPoint: "fs",
        targets: [
          {
            format,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
    });
    const renderBG = device.createBindGroup({
      layout: renderPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: boundsUniform } },
        { binding: 1, resource: { buffer: viewUniform } },
      ],
    });

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
      const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
      }
      // The slider is in CSS px; 1.5 framebuffer px is the floor below which
      // points stop resolving at all.
      const radius = Math.max(1.5, view.pointSize * dpr);
      device.queue.writeBuffer(
        viewUniform,
        0,
        new Float32Array([canvas.width, canvas.height, radius, 0])
      );
    }
    // Point-size edits land immediately, even mid-run when nothing else is
    // rewriting the view uniform.
    onViewChange = () => {
      if (gen === runGen) resize();
    };
    window.addEventListener("resize", () => {
      if (gen === runGen) resize();
    });
    resize();

    function encodeRender(
      enc: GPUCommandEncoder,
      posBuf: GPUBuffer,
      offset: number
    ) {
      const rp = enc.beginRenderPass({
        colorAttachments: [
          {
            view: ctx.getCurrentTexture().createView(),
            clearValue: { r: 0.055, g: 0.06, b: 0.075, a: 1 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      });
      rp.setPipeline(renderPipe);
      rp.setBindGroup(0, renderBG);
      rp.setVertexBuffer(0, posBuf, offset, frameBytes);
      rp.setVertexBuffer(1, labelBuf);
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
      enc.copyBufferToBuffer(boundsStorage, 0, boundsUniform, 0, 16);
      enc.copyBufferToBuffer(boundsStorage, 0, boundsHistory, slot * 16, 16);
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

    /** Draw a banked frame. No compute, no readback — two copies and a pass. */
    function drawFrame(slot: number) {
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(boundsHistory, slot * 16, boundsUniform, 0, 16);
      encodeRender(enc, posHistory, slot * frameBytes);
      device.queue.submit([enc.finish()]);
    }

    // --- Animate -----------------------------------------------------------
    status(
      `${N} points · PCA ${tPca | 0}ms · ` +
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
    frameIters[0] = 0;
    captureAndDraw(0); // the Gaussian init, before any optimizer step

    const tOpt = performance.now();
    while (it < pm.totalIters) {
      for (let c = 0; c < chunks && it < pm.totalIters; c++) {
        const next = Math.min(it + stride, pm.totalIters);
        pm.runRange(it, next); // optimizer steps stay entirely on the GPU
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

    const mb = ((frameCount * frameBytes) / (1 << 20)).toFixed(0);
    status(
      `${N} points · PCA ${tPca | 0}ms · ` +
        `kNN+pairs ${tSetup | 0}ms (${knnLabel}) · ` +
        `${pm.totalIters} iters in ${elapsed | 0}ms ` +
        `(${(elapsed / pm.totalIters).toFixed(2)}ms/iter, includes render + rAF) · ` +
        `${frameCount} frames banked on the GPU (${mb}MB` +
        `${stride > 1 ? `, every ${stride}th iter` : ""})`
    );

    // --- Playback ----------------------------------------------------------
    playhead = frameCount - 1;
    setPlaying(false);
    playBtn.disabled = false;
    scrub.disabled = false;
    onSeek = (f) => {
      playhead = Math.max(0, Math.min(f, frameCount - 1));
      scrub.value = String(playhead);
      setReadout(playhead);
    };
    onPlayToggle = () => {
      // Restarting from the top at the end is what a player is expected to do.
      if (!playing && playhead >= frameCount - 1) onSeek!(0);
      setPlaying(!playing);
    };
    setReadout(playhead);

    function setReadout(f: number) {
      f = Math.round(f);
      const iter = frameIters[f];
      iterEl.textContent = `${iter} / ${pm.totalIters}`;
      phaseEl.textContent =
        iter <= 100 ? "1 · global (w_MN 1000→3)"
        : iter <= 200 ? "2 · local refine"
        : "3 · attract-repel";
      readoutEl.textContent = `frame ${f + 1} / ${frameCount}`;
    }

    // Keeps drawing after convergence so resizing still works, and advances the
    // playhead when playing.
    let last = performance.now();
    const tick = (now: number) => {
      if (gen !== runGen) return; // a newer run owns the canvas
      const dt = (now - last) / 1000;
      last = now;
      if (playing) {
        const next = playhead + dt * PLAYBACK_FPS;
        if (next >= frameCount - 1) {
          onSeek!(frameCount - 1);
          setPlaying(false);
        } else {
          onSeek!(next);
        }
      }
      resize();
      drawFrame(Math.round(playhead));
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  } catch (e) {
    status(`Error: ${(e as Error).message}`);
    console.error(e);
  } finally {
    running = false;
    startBtn.disabled = false;
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
const view = { pointSize: 1.8 };

/** Installed by a run so an edit can rewrite that run's view uniform. */
let onViewChange: (() => void) | null = null;

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
  // Seeded from ?knn= so the URL still works, but the dropdown owns it after
  // that — comparing backends is the point, and that shouldn't need a reload.
  knnMethod: KNN_MODE,
};

const KNN_LABELS: Record<KnnMode, string> = {
  gpu: "brute force (GPU)",
  nndescent: "NN-Descent (GPU)",
  cpu: "brute force (CPU)",
};

const pane = new Pane({
  container: document.getElementById("pane") as HTMLElement,
  title: "controls",
});

const viewFolder = pane.addFolder({ title: "view" });
viewFolder
  .addBinding(view, "pointSize", {
    label: "point size",
    min: 0.2,
    max: 8,
    step: 0.1,
  })
  .on("change", () => onViewChange?.());

const pacmapFolder = pane.addFolder({ title: "pacmap · next run" });

// Ordered by how much the backend takes on trust: the CPU oracle first as the
// default, then the exact GPU kernel, then the approximate one. It is O(N^2*D)
// in plain JS — ~40 minutes at 65k — so the entry stays labelled as slow even
// though it leads.
pacmapFolder.addBinding(params, "knnMethod", {
  label: "kNN algo",
  options: {
    [`${KNN_LABELS.cpu} · slow`]: "cpu",
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
  // Sliders and buttons already handle these keys themselves.
  if (tag === "SELECT" || tag === "INPUT" || tag === "BUTTON") return;
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
  // Capped: the CPU reference is O(N^2*D), so checking at full 65k would take
  // ~40 minutes. A prefix is a perfectly valid comparison — every backend sees
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
 * Very rough setup-time estimate, so the cost of the high end is visible before
 * committing to a run. PCA dominates (~4*n*784*100 MACs of plain JS); the GPU
 * kNN term is small but quadratic, so it only starts to show past ~40k.
 */
function estimateSetupSecs(n: number): number {
  return (n * 784 * 100 * 4) / 5e8 + (n * n * 100) / 3e11 + n * 4e-5;
}

sampleSel.max = String(NUM_AVAILABLE);

function updateSampleLabel() {
  const n = parseInt(sampleSel.value, 10);
  sampleOut.value = n.toLocaleString();
  const s = estimateSetupSecs(n);
  sampleHint.textContent = `~${s < 10 ? s.toFixed(1) : Math.round(s)}s setup`;
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

startBtn.addEventListener("click", go);
