import { pacmapWebGPU, type PacmapRun } from "./pacmap-webgpu";
import { loadMnist, IMAGE_SIZE } from "./mnist";
import { pcaProject } from "./pca";

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

const canvas = document.getElementById("view") as HTMLCanvasElement;
const statusEl = document.getElementById("status") as HTMLElement;
const iterEl = document.getElementById("iter") as HTMLElement;
const phaseEl = document.getElementById("phase") as HTMLElement;
const startBtn = document.getElementById("start") as HTMLButtonElement;
const sampleSel = document.getElementById("samples") as HTMLSelectElement;
const speedSel = document.getElementById("speed") as HTMLSelectElement;

const status = (m: string) => (statusEl.textContent = m);

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

async function go() {
  if (running) return;
  running = true;
  startBtn.disabled = true;

  try {
    if (!navigator.gpu) throw new Error("WebGPU unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no GPU adapter");
    const device = await adapter.requestDevice();
    device.lost.then((info) => status(`Device lost: ${info.message}`));

    const N = parseInt(sampleSel.value, 10);
    const stepsPerFrame = parseInt(speedSel.value, 10);

    // --- Data --------------------------------------------------------------
    const { X, labels } = await loadMnist(N, status);

    status("Projecting 784d → 100d (randomized PCA)…");
    await frame();
    const t0 = performance.now();
    const Z = pcaProject(X, N, IMAGE_SIZE, 100);
    const tPca = performance.now() - t0;

    status(`PCA ${tPca | 0}ms · building kNN graph + sampling pairs…`);
    await frame();
    const t1 = performance.now();
    const pm: PacmapRun = await pacmapWebGPU(device, Z, N, 100, { seed: 7 });
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
      const radius = Math.max(1.5, 2.6 * dpr) * (N > 8000 ? 0.7 : 1);
      device.queue.writeBuffer(
        viewUniform,
        0,
        new Float32Array([canvas.width, canvas.height, radius, 0])
      );
    }
    window.addEventListener("resize", resize);
    resize();

    function draw() {
      const enc = device.createCommandEncoder();

      const cp = enc.beginComputePass();
      cp.setPipeline(boundsPipe);
      cp.setBindGroup(0, boundsBG);
      cp.dispatchWorkgroups(1);
      cp.end();
      enc.copyBufferToBuffer(boundsStorage, 0, boundsUniform, 0, 16);

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
      rp.setVertexBuffer(0, pm.positions);
      rp.setVertexBuffer(1, labelBuf);
      rp.draw(6, N);
      rp.end();

      device.queue.submit([enc.finish()]);
    }

    // --- Animate -----------------------------------------------------------
    status(
      `${N} points · PCA ${tPca | 0}ms · kNN+pairs ${tSetup | 0}ms · optimizing…`
    );

    let it = 0;
    const tOpt = performance.now();
    while (it < pm.totalIters) {
      const next = Math.min(it + stepsPerFrame, pm.totalIters);
      pm.runRange(it, next); // optimizer steps stay entirely on the GPU
      it = next;
      draw(); // reads pm.positions directly as a vertex buffer
      iterEl.textContent = `${it} / ${pm.totalIters}`;
      phaseEl.textContent =
        it <= 100 ? "1 · global (w_MN 1000→3)"
        : it <= 200 ? "2 · local refine"
        : "3 · attract-repel";
      await frame();
    }
    const elapsed = performance.now() - tOpt;

    status(
      `${N} points · PCA ${tPca | 0}ms · kNN+pairs ${tSetup | 0}ms · ` +
        `${pm.totalIters} iters in ${elapsed | 0}ms ` +
        `(${(elapsed / pm.totalIters).toFixed(2)}ms/iter, includes render + rAF)`
    );

    // Keep redrawing so resizing still works after convergence.
    const idle = () => {
      resize();
      draw();
      requestAnimationFrame(idle);
    };
    idle();
  } catch (e) {
    status(`Error: ${(e as Error).message}`);
    console.error(e);
  } finally {
    running = false;
    startBtn.disabled = false;
  }
}

function frame(): Promise<void> {
  return new Promise((r) => requestAnimationFrame(() => r()));
}

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
