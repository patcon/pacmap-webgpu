/**
 * Minimal PaCMAP on WebGPU.
 *
 * Split of responsibilities:
 *   CPU (once, at setup)  - kNN, sigma scaling, pair sampling, CSR build
 *   GPU (450 iterations)  - gradient accumulation + Adam, fully resident
 *
 * The whole optimization loop is encoded into a single command buffer with no
 * host round-trip. Y never leaves the GPU unless you ask for it, so the position
 * buffer can be bound directly as a vertex attribute for per-iteration rendering.
 *
 * [Omitted for minimality: PCA-to-100d preprocessing, PCA init (using scaled
 *  gaussian instead), GPU brute-force kNN. All three are noted at their sites.]
 */

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface PacmapOptions {
  /** Nearest neighbors per point. Default follows the reference heuristic. */
  nNeighbors?: number;
  /** n_MN = round(nNeighbors * mnRatio). Default 0.5 */
  mnRatio?: number;
  /** n_FP = round(nNeighbors * fpRatio). Default 2.0 */
  fpRatio?: number;
  /** Iterations per phase. Default [100, 100, 250] */
  phases?: [number, number, number];
  /** Adam learning rate. Default 1.0 */
  lr?: number;
  seed?: number;
}

const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-7;

// Pair type tags, packed into the top 2 bits of each adjacency entry.
const T_NB = 0;
const T_MN = 1;
const T_FP = 2;

// ---------------------------------------------------------------------------
// RNG — deterministic so animation captures are reproducible
// ---------------------------------------------------------------------------

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
// Brute-force kNN (CPU)
// ---------------------------------------------------------------------------

/**
 * Returns the kCand nearest neighbors of every point by squared euclidean
 * distance. O(N^2 * D) with a bounded insertion sort per row.
 *
 * [This is the piece to replace first if you outgrow it: either a tiled WGSL
 *  distance kernel with a per-thread bounded heap, or NN-Descent above ~20k.]
 */
function bruteForceKnn(
  X: Float32Array,
  N: number,
  D: number,
  kCand: number
): { idx: Uint32Array; d2: Float32Array } {
  const idx = new Uint32Array(N * kCand);
  const d2 = new Float32Array(N * kCand);

  const heapI = new Uint32Array(kCand);
  const heapD = new Float32Array(kCand);

  for (let i = 0; i < N; i++) {
    let count = 0;
    let worst = Infinity;

    for (let j = 0; j < N; j++) {
      if (j === i) continue;

      // Squared distance with early exit once we're full.
      let s = 0;
      const oi = i * D;
      const oj = j * D;
      for (let k = 0; k < D; k++) {
        const diff = X[oi + k] - X[oj + k];
        s += diff * diff;
        if (count === kCand && s >= worst) break;
      }
      if (count === kCand && s >= worst) continue;

      // Insert into the sorted bounded list.
      let pos = count < kCand ? count : kCand - 1;
      while (pos > 0 && heapD[pos - 1] > s) {
        heapD[pos] = heapD[pos - 1];
        heapI[pos] = heapI[pos - 1];
        pos--;
      }
      heapD[pos] = s;
      heapI[pos] = j;
      if (count < kCand) count++;
      worst = heapD[count - 1];
    }

    idx.set(heapI.subarray(0, count), i * kCand);
    d2.set(heapD.subarray(0, count), i * kCand);
  }

  return { idx, d2 };
}

function dist2(X: Float32Array, D: number, a: number, b: number): number {
  let s = 0;
  const oa = a * D;
  const ob = b * D;
  for (let k = 0; k < D; k++) {
    const diff = X[oa + k] - X[ob + k];
    s += diff * diff;
  }
  return s;
}

// ---------------------------------------------------------------------------
// Pair sampling
// ---------------------------------------------------------------------------

interface Pairs {
  i: Uint32Array;
  j: Uint32Array;
  t: Uint8Array;
}

function samplePairs(
  X: Float32Array,
  N: number,
  D: number,
  nNB: number,
  nMN: number,
  nFP: number,
  rand: () => number
): Pairs {
  const kCand = Math.min(nNB + 50, N - 1);
  const { idx, d2 } = bruteForceKnn(X, N, D, kCand);

  // sigma_i = mean euclidean distance to the 4th-6th nearest neighbor.
  const sig = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const lo = Math.min(3, kCand - 1);
    const hi = Math.min(6, kCand);
    let s = 0;
    let n = 0;
    for (let k = lo; k < hi; k++) {
      s += Math.sqrt(d2[i * kCand + k]);
      n++;
    }
    sig[i] = Math.max(n > 0 ? s / n : 1e-10, 1e-10);
  }

  const total = N * (nNB + nMN + nFP);
  const pi = new Uint32Array(total);
  const pj = new Uint32Array(total);
  const pt = new Uint8Array(total);
  let w = 0;

  // Reusable scratch for the rescale-and-reselect step.
  const order = new Uint32Array(kCand);
  const scaled = new Float32Array(kCand);
  const nbSet = new Set<number>();

  for (let i = 0; i < N; i++) {
    // --- Near pairs: re-rank candidates by sigma-scaled distance. ---
    for (let k = 0; k < kCand; k++) {
      const j = idx[i * kCand + k];
      order[k] = k;
      scaled[k] = d2[i * kCand + k] / (sig[i] * sig[j]);
    }
    const ord = Array.from(order.subarray(0, kCand)).sort(
      (a, b) => scaled[a] - scaled[b]
    );

    nbSet.clear();
    const take = Math.min(nNB, kCand);
    for (let k = 0; k < take; k++) {
      const j = idx[i * kCand + ord[k]];
      nbSet.add(j);
      pi[w] = i;
      pj[w] = j;
      pt[w] = T_NB;
      w++;
    }

    // --- Mid-near pairs: draw 6 at random, keep the 2nd closest. ---
    for (let m = 0; m < nMN; m++) {
      let best = -1;
      let bestD = Infinity;
      let second = -1;
      let secondD = Infinity;
      for (let s = 0; s < 6; s++) {
        let c = (rand() * N) | 0;
        if (c === i) c = (c + 1) % N;
        const dd = dist2(X, D, i, c);
        if (dd < bestD) {
          secondD = bestD;
          second = best;
          bestD = dd;
          best = c;
        } else if (dd < secondD) {
          secondD = dd;
          second = c;
        }
      }
      pi[w] = i;
      pj[w] = second >= 0 ? second : best;
      pt[w] = T_MN;
      w++;
    }

    // --- Further pairs: uniform random, rejecting true neighbors. ---
    for (let f = 0; f < nFP; f++) {
      let c = 0;
      for (let tries = 0; tries < 64; tries++) {
        c = (rand() * N) | 0;
        if (c !== i && !nbSet.has(c)) break;
      }
      pi[w] = i;
      pj[w] = c;
      pt[w] = T_FP;
      w++;
    }
  }

  return { i: pi.subarray(0, w), j: pj.subarray(0, w), t: pt.subarray(0, w) };
}

// ---------------------------------------------------------------------------
// CSR build — each pair is duplicated into BOTH endpoints' lists so the
// gradient kernel can gather instead of scatter. This is what buys us a
// float-atomic-free shader.
// ---------------------------------------------------------------------------

function buildCSR(
  N: number,
  pairs: Pairs
): { offsets: Uint32Array; data: Uint32Array } {
  const P = pairs.i.length;
  const deg = new Uint32Array(N);
  for (let p = 0; p < P; p++) {
    deg[pairs.i[p]]++;
    deg[pairs.j[p]]++;
  }

  const offsets = new Uint32Array(N + 1);
  for (let i = 0; i < N; i++) offsets[i + 1] = offsets[i] + deg[i];

  const data = new Uint32Array(offsets[N]);
  const cursor = offsets.slice(0, N);
  for (let p = 0; p < P; p++) {
    const a = pairs.i[p];
    const b = pairs.j[p];
    const tag = pairs.t[p] << 30;
    data[cursor[a]++] = tag | b;
    data[cursor[b]++] = tag | a;
  }

  return { offsets, data };
}

// ---------------------------------------------------------------------------
// Weight schedule
// ---------------------------------------------------------------------------

function weightsAt(
  iter: number,
  n1: number,
  n2: number
): [number, number, number] {
  if (iter < n1) {
    const t = iter / n1;
    return [2.0, 1000.0 * (1.0 - t) + 3.0 * t, 1.0]; // NB, MN, FP
  }
  if (iter < n1 + n2) return [3.0, 3.0, 1.0];
  return [1.0, 0.0, 1.0];
}

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------

function shaderSource(N: number): string {
  return /* wgsl */ `
struct Params {
  w  : vec4<f32>,   // wNB, wMN, wFP, lr
  bc : vec4<f32>,   // 1/(1-b1^t), 1/(1-b2^t), _, _
};

@group(0) @binding(0) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(1) var<storage, read>       Off  : array<u32>;
@group(0) @binding(2) var<storage, read>       Adj  : array<u32>;
@group(0) @binding(3) var<storage, read_write> Grad : array<f32>;
@group(0) @binding(4) var<storage, read_write> M    : array<f32>;
@group(0) @binding(5) var<storage, read_write> V    : array<f32>;
@group(0) @binding(6) var<uniform>             P    : Params;

const N : u32 = ${N}u;

// One thread per point. Gathers every pair this point participates in and sums
// the force. The sign works out automatically: for the reverse entry the
// displacement flips while the scalar coefficient (a function of |d|^2 only)
// does not, which is exactly the symmetric counterpart of the forward pair.
@compute @workgroup_size(64)
fn grad_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }

  let yi = vec2<f32>(Y[2u * i], Y[2u * i + 1u]);
  var g = vec2<f32>(0.0, 0.0);

  let lo = Off[i];
  let hi = Off[i + 1u];

  for (var p : u32 = lo; p < hi; p = p + 1u) {
    let packed = Adj[p];
    let kind = packed >> 30u;
    let j = packed & 0x3FFFFFFFu;

    let d = yi - vec2<f32>(Y[2u * j], Y[2u * j + 1u]);
    let dd = 1.0 + dot(d, d);

    var c : f32 = 0.0;
    if (kind == 0u) {          // near:      w * dd / (10 + dd)
      let r = 10.0 + dd;
      c = P.w.x * 20.0 / (r * r);
    } else if (kind == 1u) {   // mid-near:  w * dd / (10000 + dd)
      let r = 10000.0 + dd;
      c = P.w.y * 20000.0 / (r * r);
    } else {                   // further:   w * 1 / (1 + dd)
      let r = 1.0 + dd;
      c = -P.w.z * 2.0 / (r * r);
    }

    g = g + c * d;
  }

  Grad[2u * i]      = g.x;
  Grad[2u * i + 1u] = g.y;
}

@compute @workgroup_size(64)
fn adam_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }

  for (var c : u32 = 0u; c < 2u; c = c + 1u) {
    let k = 2u * i + c;
    let g = Grad[k];

    let m = ${ADAM_B1} * M[k] + ${1 - ADAM_B1} * g;
    let v = ${ADAM_B2} * V[k] + ${1 - ADAM_B2} * g * g;
    M[k] = m;
    V[k] = v;

    let mh = m * P.bc.x;
    let vh = v * P.bc.y;
    Y[k] = Y[k] - P.w.w * mh / (sqrt(vh) + ${ADAM_EPS});
  }
}
`;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

export interface PacmapRun {
  /** N x 2 f32 positions. Bindable as a vertex buffer for zero-copy rendering. */
  positions: GPUBuffer;
  /** Encode and submit the full optimization. No host round-trip inside. */
  run(): void;
  /** Encode iterations [from, to) only — for stepping an animation. */
  runRange(from: number, to: number): void;
  /** Copy positions back to the host. */
  read(): Promise<Float32Array>;
  totalIters: number;
  destroy(): void;
}

export async function pacmapWebGPU(
  device: GPUDevice,
  X: Float32Array,
  N: number,
  D: number,
  opts: PacmapOptions = {}
): Promise<PacmapRun> {
  const nNB =
    opts.nNeighbors ??
    (N <= 10000 ? 10 : Math.round(10 + 15 * (Math.log10(N) - 4)));
  const nMN = Math.round(nNB * (opts.mnRatio ?? 0.5));
  const nFP = Math.round(nNB * (opts.fpRatio ?? 2.0));
  const [n1, n2, n3] = opts.phases ?? [100, 100, 250];
  const lr = opts.lr ?? 1.0;
  const totalIters = n1 + n2 + n3;
  const rand = mulberry32(opts.seed ?? 42);

  // --- CPU setup ---------------------------------------------------------
  // [X is used as given. The reference applies PCA to 100 dims first when
  //  D > 100; add that here if your input is high-dimensional.]
  const pairs = samplePairs(X, N, D, nNB, nMN, nFP, rand);
  const { offsets, data } = buildCSR(N, pairs);

  // [Reference default init is scaled PCA. Gaussian is used here for brevity;
  //  it converges but embeddings will differ run-to-run across seeds.]
  const Y0 = new Float32Array(N * 2);
  for (let i = 0; i < N * 2; i++) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    Y0[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 0.0001;
  }

  // --- Buffers -----------------------------------------------------------
  const mk = (arr: ArrayBufferView, usage: GPUBufferUsageFlags) => {
    const b = device.createBuffer({
      size: Math.max(arr.byteLength, 4),
      usage,
      mappedAtCreation: true,
    });
    new Uint8Array(b.getMappedRange()).set(
      new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength)
    );
    b.unmap();
    return b;
  };

  const S = GPUBufferUsage.STORAGE;
  const yBuf = mk(Y0, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX);
  const offBuf = mk(offsets, S);
  const adjBuf = mk(data, S);
  const gradBuf = device.createBuffer({ size: N * 2 * 4, usage: S });
  const mBuf = mk(new Float32Array(N * 2), S);
  const vBuf = mk(new Float32Array(N * 2), S);

  // One 256-byte-aligned slot per iteration, addressed by dynamic offset.
  // This is what lets the whole loop live in a single command buffer.
  const align = device.limits.minUniformBufferOffsetAlignment || 256;
  const params = new Float32Array((totalIters * align) / 4);
  for (let it = 0; it < totalIters; it++) {
    const o = (it * align) / 4;
    const [wNB, wMN, wFP] = weightsAt(it, n1, n2);
    params[o + 0] = wNB;
    params[o + 1] = wMN;
    params[o + 2] = wFP;
    params[o + 3] = lr;
    params[o + 4] = 1 / (1 - Math.pow(ADAM_B1, it + 1));
    params[o + 5] = 1 / (1 - Math.pow(ADAM_B2, it + 1));
  }
  const paramBuf = mk(params, GPUBufferUsage.UNIFORM);

  // --- Pipelines ---------------------------------------------------------
  const module = device.createShaderModule({ code: shaderSource(N) });

  const storageEntry = (
    binding: number,
    type: GPUBufferBindingType
  ): GPUBindGroupLayoutEntry => ({
    binding,
    visibility: GPUShaderStage.COMPUTE,
    buffer: { type },
  });

  const layout = device.createBindGroupLayout({
    entries: [
      storageEntry(0, "storage"),
      storageEntry(1, "read-only-storage"),
      storageEntry(2, "read-only-storage"),
      storageEntry(3, "storage"),
      storageEntry(4, "storage"),
      storageEntry(5, "storage"),
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 32 },
      },
    ],
  });

  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout],
  });

  const gradPipe = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: "grad_main" },
  });
  const adamPipe = device.createComputePipeline({
    layout: pipelineLayout,
    compute: { module, entryPoint: "adam_main" },
  });

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: yBuf } },
      { binding: 1, resource: { buffer: offBuf } },
      { binding: 2, resource: { buffer: adjBuf } },
      { binding: 3, resource: { buffer: gradBuf } },
      { binding: 4, resource: { buffer: mBuf } },
      { binding: 5, resource: { buffer: vBuf } },
      { binding: 6, resource: { buffer: paramBuf, size: 32 } },
    ],
  });

  const groups = Math.ceil(N / 64);

  // --- Encode ------------------------------------------------------------
  // WebGPU synchronizes automatically between dispatches in a pass, so the
  // grad -> adam dependency needs no explicit barrier.
  function runRange(from: number, to: number) {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    for (let it = from; it < to; it++) {
      pass.setBindGroup(0, bindGroup, [it * align]);
      pass.setPipeline(gradPipe);
      pass.dispatchWorkgroups(groups);
      pass.setPipeline(adamPipe);
      pass.dispatchWorkgroups(groups);
    }
    pass.end();
    device.queue.submit([enc.finish()]);
  }

  async function read(): Promise<Float32Array> {
    const staging = device.createBuffer({
      size: N * 2 * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(yBuf, 0, staging, 0, N * 2 * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  return {
    positions: yBuf,
    totalIters,
    run: () => runRange(0, totalIters),
    runRange,
    read,
    destroy() {
      for (const b of [yBuf, offBuf, adjBuf, gradBuf, mBuf, vBuf, paramBuf]) {
        b.destroy();
      }
    },
  };
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------
//
//   const adapter = await navigator.gpu.requestAdapter();
//   const device = await adapter!.requestDevice();
//
//   const pm = await pacmapWebGPU(device, X, N, D, { seed: 7 });
//
//   // One shot:
//   pm.run();
//   const Y = await pm.read();
//
//   // Or step it for animation, rendering pm.positions directly as a vertex
//   // buffer between calls — no readback, no stall:
//   for (let it = 0; it < pm.totalIters; it++) {
//     pm.runRange(it, it + 1);
//     drawFrom(pm.positions);
//     await new Promise(requestAnimationFrame);
//   }
