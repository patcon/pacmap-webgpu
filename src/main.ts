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
import { boundsWGSL, renderWGSL } from "./shaders";
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
//   ?algo=localmap          run LocalMAP rather than PaCMAP (default pacmap)
//   ?algo=pacmap-cpu        run DruidJS on the CPU rather than our WGSL
//   ?knn=gpu|nnd            pick the kNN backend (default cpu brute force)
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

// Playback history. Every captured frame is a full N x 2 f32 snapshot kept in
// GPU memory, so the scrubber never reads positions back to the host — same
// constraint as the live render path. At 65k points a frame is 520KB, so the
// budget, not the iteration count, is what decides how many we keep.
const HISTORY_BUDGET_BYTES = 128 << 20;
const PLAYBACK_FPS = 60;

// With auto zoom off the whole trace is framed by one box, and during a live run
// the box that frames the *final* frame is not yet known. The previous run's is
// the best guess available, so it is carried across runs and page reloads. Same
// `loX loY hiX hiY` layout the bounds shader writes, so it drops straight into
// the render's bounds uniform.
const BOUNDS_KEY = "pacmap:lastBounds";
let seedBounds: Float32Array<ArrayBuffer> | null = readSeedBounds();

function readSeedBounds(): Float32Array<ArrayBuffer> | null {
  // A cosmetic cache; nothing here may take out startup.
  try {
    const raw = sessionStorage.getItem(BOUNDS_KEY);
    if (!raw) return null;
    const v = JSON.parse(raw);
    if (!Array.isArray(v) || v.length !== 4) return null;
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

    status(
      `PCA ${tPca | 0}ms · building kNN graph (${knnLabel}) + sampling pairs…`
    );
    await frame();
    const t1 = performance.now();
    const pm: EmbeddingRun =
      engine === "cpu"
        ? await druidCPU(device, Z, N, 100, {
            variant,
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
      // Banked unconditionally: 16 bytes, and toggling auto zoom back on mid-run
      // has to find a bound for every slot behind it.
      enc.copyBufferToBuffer(boundsStorage, 0, boundsHistory, slot * 16, 16);
      if (view.autoZoom || !seedBounds) {
        enc.copyBufferToBuffer(boundsStorage, 0, boundsUniform, 0, 16);
      } else {
        // Held fixed, but the final frame's bound doesn't exist yet — the
        // previous run's is the only guess there is. Ordered against the submit
        // below on the queue timeline, so writing it here is safe.
        device.queue.writeBuffer(boundsUniform, 0, seedBounds);
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

    /** Draw a banked frame. No compute, no readback — two copies and a pass. */
    function drawFrame(slot: number) {
      // Auto zoom frames each slot to its own extent; held, the whole trace is
      // framed by the last slot, so the scrubber shows travel rather than a
      // camera that cancels it out.
      const boundsSlot = view.autoZoom ? slot : banked - 1;
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(boundsHistory, boundsSlot * 16, boundsUniform, 0, 16);
      encodeRender(enc, posHistory, slot * frameBytes);
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
        size: 16,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(boundsHistory, (banked - 1) * 16, staging, 0, 16);
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
    // playhead when playing.
    let last = performance.now();
    const tick = (now: number) => {
      if (gen !== runGen) return; // a newer run owns the canvas
      const dt = (now - last) / 1000;
      last = now;
      if (playing) {
        const next = playhead + dt * PLAYBACK_FPS;
        if (next >= banked - 1) {
          onSeek!(banked - 1);
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
const view = { pointSize: 1.8, autoZoom: true };

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
  // Seeded from ?knn= / ?algo= so the URLs still work, but the dropdowns own
  // them after that — comparing backends and variants is the point, and that
  // shouldn't need a reload.
  knnMethod: KNN_MODE,
  algorithm: ALGO_MODE,
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

const viewFolder = pane.addFolder({ title: "view" });
viewFolder
  .addBinding(view, "pointSize", {
    label: "point size",
    min: 0.2,
    max: 2,
    step: 0.1,
  })
  .on("change", () => onViewChange?.());
// Nothing to install for this one: the post-run rAF redraws every tick and
// `captureAndDraw` re-reads it per captured frame, so a toggle lands on the next
// frame either way. `onViewChange` is here only to keep the two bindings alike.
viewFolder
  .addBinding(view, "autoZoom", { label: "auto zoom" })
  .on("change", () => onViewChange?.());

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

// Bindings all exist now, so the pane can be brought in line with whatever
// `?algo=` selected. Also draws the first cost hint.
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
