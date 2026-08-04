/**
 * Minimal PaCMAP — and LocalMAP — on WebGPU.
 *
 * Split of responsibilities:
 *   CPU (once, at setup)  - kNN, sigma scaling, pair sampling, CSR build
 *   GPU (450 iterations)  - gradient accumulation + Adam, fully resident
 *
 * The whole optimization loop is encoded into a single command buffer with no
 * host round-trip. Y never leaves the GPU unless you ask for it, so the position
 * buffer can be bound directly as a vertex attribute for per-iteration rendering.
 *
 * Three kNN backends, all producing the identical N x kCand output contract:
 * `bruteForceKnn` (CPU, exact, the oracle), `knnGPU` (WGSL, exact) and
 * `nndescentGPU` (WGSL, approximate). Select with the `knn` option.
 *
 * [Omitted for minimality: PCA-to-100d preprocessing, PCA init (using scaled
 *  gaussian instead). Both are noted at their sites.]
 */

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface PacmapOptions {
  /**
   * Which algorithm to run. Default "pacmap".
   *
   * "localmap" is LocalMAP (Wang et al., AAAI 2025), which upstream ships as a
   * subclass of PaCMAP reusing its whole setup path — so it is a variant here
   * rather than a separate entry point. It differs only inside phase 3, in two
   * ways: the near-pair gradient picks up a `(lowDistThres/2)/|d|` factor (see
   * `shaderSource`), and the further pairs are periodically redrawn against the
   * *embedding* rather than staying fixed for the whole run.
   */
  variant?: "pacmap" | "localmap";
  /**
   * LocalMAP's `low_dist_thres`. Default 10, as upstream. Ignored under
   * "pacmap". Sets both the near-pair coefficient (as `lowDistThres/2`) and the
   * radius the redrawn further pairs are restricted to.
   */
  lowDistThres?: number;
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
  /**
   * Where and how to build the kNN graph. Default "gpu" (brute force).
   *
   * "cpu" keeps the O(N^2*D) reference path, which is only tolerable below a
   * few thousand points but is useful as a correctness oracle. "nndescent" is
   * approximate — see `nndescentGPU` for what that costs and buys.
   */
  knn?: "gpu" | "cpu" | "nndescent";
  /** Tuning for `knn: "nndescent"`. Ignored otherwise. */
  nndescent?: Omit<NndOptions, "seed" | "onStatus">;
  /** Progress callback for the setup phase. */
  onStatus?: (msg: string) => void;
}

/**
 * The reference heuristic for `nNeighbors`: flat below 10k, then growing with
 * log10(N). Exported so a caller offering an "auto" toggle can show the value
 * it would get rather than duplicating the formula.
 */
export function defaultNeighbors(N: number): number {
  return N <= 10000 ? 10 : Math.round(10 + 15 * (Math.log10(N) - 4));
}

const ADAM_B1 = 0.9;
const ADAM_B2 = 0.999;
const ADAM_EPS = 1e-7;

// Pair type tags, packed into the top 2 bits of each adjacency entry. Only the
// two kinds the CSR still carries: further pairs moved to their own arrays when
// LocalMAP made them mutable (see `Pairs`), so they never reach this packing.
const T_NB = 0;
const T_MN = 1;

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
// Buffer helper
// ---------------------------------------------------------------------------

function mkBuf(
  device: GPUDevice,
  arr: ArrayBufferView,
  usage: GPUBufferUsageFlags
): GPUBuffer {
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
}

// ---------------------------------------------------------------------------
// Cooperative yielding
//
// Same mechanism as `pca.ts`, duplicated rather than imported so this file
// stays standalone (`mulberry32` is duplicated for the same reason). Not
// setTimeout(0): that clamps to ~4ms once nested. A MessageChannel task has no
// clamp and still lets the browser paint between slices. Guarded so the library
// still runs headlessly under Node.
// ---------------------------------------------------------------------------

/** Target time between yields. Long enough to amortize, short enough to feel live. */
const SLICE_MS = 25;

const chan = typeof MessageChannel !== "undefined" ? new MessageChannel() : null;

function tick(): Promise<void> {
  if (!chan) return Promise.resolve();
  return new Promise<void>((resolve) => {
    chan.port1.onmessage = () => resolve();
    chan.port2.postMessage(0);
  });
}

// ---------------------------------------------------------------------------
// Brute-force kNN (CPU) — reference implementation
// ---------------------------------------------------------------------------

/**
 * Returns the kCand nearest neighbors of every point by squared euclidean
 * distance. O(N^2 * D) with a bounded insertion sort per row.
 *
 * Superseded by `knnGPU` for anything but small N — this is ~40 minutes at 65k,
 * on whatever thread you call it from. Kept as the correctness oracle for both
 * GPU paths (see the `?knncheck=1` mode in the demo) and as an escape hatch.
 *
 * Above ~100k, where O(N^2) stops being viable even at GPU throughput, neither
 * brute-force path is usable and `nndescentGPU` is the one that still scales.
 *
 * Async purely so it can yield: at 65k this runs for tens of minutes, and a
 * synchronous version pins the event loop for the whole of it, so the caller's
 * `onStatus` fires into a page that can never repaint. The arithmetic is
 * unchanged and the yields cost well under 1% — one `tick` per ${SLICE_MS}ms
 * slice — but note that a timing harness measuring this will see that overhead.
 *
 * Yielding keeps the page painting; it does not keep it *responsive*, since the
 * main thread is still ~100% busy between slices. A worker would fix that, at
 * the cost of making this file no longer standalone. Not worth it for a debug
 * oracle — see the README.
 */
export async function bruteForceKnn(
  X: Float32Array,
  N: number,
  D: number,
  kCand: number,
  onStatus: (msg: string) => void = () => {}
): Promise<{ idx: Uint32Array; d2: Float32Array }> {
  const idx = new Uint32Array(N * kCand);
  const d2 = new Float32Array(N * kCand);

  const heapI = new Uint32Array(kCand);
  const heapD = new Float32Array(kCand);

  const t0 = performance.now();
  let last = t0;

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

    // The clock check is per-row rather than every Nth row (as pca.ts does)
    // because one row here is N*D work — milliseconds at any interesting N — so
    // the check is already lost in the noise, and striding it would make the
    // yield rate depend on N.
    const now = performance.now();
    if (now - last >= SLICE_MS) {
      const done = i + 1;
      const elapsed = (now - t0) / 1000;
      // Every row costs the same, so elapsed/done extrapolates honestly. Worth
      // showing: this path runs for tens of minutes at the top of the slider.
      const left = (elapsed / done) * (N - done);
      onStatus(
        `Building kNN graph on CPU… ${((done / N) * 100) | 0}% ` +
          `(${done}/${N}, ${elapsed.toFixed(1)}s, ~${fmtDuration(left)} left)`
      );
      await tick();
      last = performance.now();
    }
  }

  return { idx, d2 };
}

/** Compact "left" figure — seconds under a minute, then minutes, then hours. */
function fmtDuration(secs: number): string {
  if (secs < 60) return `${Math.ceil(secs)}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}min`;
  return `${(secs / 3600).toFixed(1)}h`;
}

// ---------------------------------------------------------------------------
// Brute-force kNN (GPU)
// ---------------------------------------------------------------------------

const KNN_WG = 64;

function knnShaderSource(N: number, D: number, K: number): string {
  return /* wgsl */ `
struct KParams { range : vec4<u32> };   // qStart, qEnd, _, _

@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read_write> OutI : array<u32>;
@group(0) @binding(2) var<storage, read_write> OutD : array<f32>;
@group(0) @binding(3) var<uniform>             P    : KParams;

const N : u32 = ${N}u;
const D : u32 = ${D}u;
const K : u32 = ${K}u;

// One thread per query point, scanning every candidate and keeping a bounded
// sorted list. Output matches bruteForceKnn exactly in shape and ordering.
//
// There is deliberately no cooperative shared-memory tile. A tile of ${KNN_WG}
// candidates x ${D} dims is ${KNN_WG * D * 4} bytes, over the 16KB
// maxComputeWorkgroupStorageSize limit, and tiling by dimension instead would
// force every thread to hold one partial sum per tile candidate — moving the
// pressure rather than relieving it. So the query row goes in a private array
// and candidate rows are read straight from global memory: all ${KNN_WG}
// threads in the workgroup hit the same candidate address at the same time,
// which caches and broadcasts well.
//
// Also unlike the CPU reference, there is no early exit inside the distance
// loop. It would be divergent, and every thread waits for the slowest anyway,
// so the full distance is always computed and only the insert is branched on.
// Inserts are rare (~K + K*ln(N/K) per query, well under 1% of the work), which
// is what makes the insertion sort free in practice.
@compute @workgroup_size(${KNN_WG})
fn knn_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = P.range.x + gid.x;
  if (i >= P.range.y) { return; }

  var q : array<f32, ${D}>;
  let oi = i * D;
  for (var k : u32 = 0u; k < D; k = k + 1u) { q[k] = X[oi + k]; }

  var hD : array<f32, ${K}>;
  var hI : array<u32, ${K}>;
  var count : u32 = 0u;
  var worst : f32 = 0.0;   // only read once count == K

  for (var j : u32 = 0u; j < N; j = j + 1u) {
    if (j == i) { continue; }

    var s : f32 = 0.0;
    let oj = j * D;
    for (var k : u32 = 0u; k < D; k = k + 1u) {
      let diff = q[k] - X[oj + k];
      s = s + diff * diff;
    }

    if (count == K && s >= worst) { continue; }

    // Shift the sorted list down one slot and drop s into place.
    var pos : u32 = select(count, K - 1u, count == K);
    loop {
      if (pos == 0u) { break; }
      if (hD[pos - 1u] <= s) { break; }
      hD[pos] = hD[pos - 1u];
      hI[pos] = hI[pos - 1u];
      pos = pos - 1u;
    }
    hD[pos] = s;
    hI[pos] = j;
    if (count < K) { count = count + 1u; }
    worst = hD[count - 1u];
  }

  let base = i * K;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    OutD[base + k] = hD[k];
    OutI[base + k] = hI[k];
  }
}
`;
}

/**
 * GPU equivalent of `bruteForceKnn`. Same O(N^2 * D) work, same output shape and
 * ordering (N x kCand row-major, ascending by squared distance), several orders
 * of magnitude more throughput.
 *
 * Results are not bit-identical to the CPU path: JS accumulates distances in
 * f64 and WGSL in f32, so near-ties can swap order. Recall is what matters and
 * it should be ~1.0 — see the demo's `?knncheck=1` mode.
 */
export async function knnGPU(
  device: GPUDevice,
  X: Float32Array,
  N: number,
  D: number,
  kCand: number,
  onStatus: (msg: string) => void = () => {}
): Promise<{ idx: Uint32Array; d2: Float32Array }> {
  if (kCand < 1) throw new Error("kCand must be >= 1");
  if (kCand > N - 1) throw new Error("kCand must be <= N-1");

  const S = GPUBufferUsage.STORAGE;
  const outBytes = N * kCand * 4;
  const xBuf = mkBuf(device, X, S);
  const idxBuf = device.createBuffer({
    size: outBytes,
    usage: S | GPUBufferUsage.COPY_SRC,
  });
  const d2Buf = device.createBuffer({
    size: outBytes,
    usage: S | GPUBufferUsage.COPY_SRC,
  });
  const pBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const pipe = device.createComputePipeline({
    layout: "auto",
    compute: {
      module: device.createShaderModule({
        code: knnShaderSource(N, D, kCand),
      }),
      entryPoint: "knn_main",
    },
  });
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: idxBuf } },
      { binding: 2, resource: { buffer: d2Buf } },
      { binding: 3, resource: { buffer: pBuf } },
    ],
  });

  // Dispatch in query-range chunks. This is required, not an optimization: one
  // dispatch covering all N*N*D work runs for seconds at scale and trips the OS
  // GPU watchdog (~2s on Windows), which loses the device. Each query is fully
  // resolved inside a single dispatch — it scans every candidate — so chunks
  // carry no state between them.
  //
  // The first chunk is deliberately tiny, then size is steered toward TARGET_MS:
  // fast GPUs converge on large well-utilized dispatches, slow ones stay under
  // the watchdog. Awaiting onSubmittedWorkDone is also the yield point that
  // keeps the host responsive (this library stays DOM-free, so no rAF here).
  const TARGET_MS = 40;
  const range = new Uint32Array(4);
  let chunk = Math.max(KNN_WG, Math.min(N, Math.ceil(2e8 / (N * D))));
  let q = 0;
  const t0 = performance.now();

  while (q < N) {
    const end = Math.min(q + chunk, N);
    range[0] = q;
    range[1] = end;
    device.queue.writeBuffer(pBuf, 0, range);

    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil((end - q) / KNN_WG));
    pass.end();
    device.queue.submit([enc.finish()]);

    const t = performance.now();
    await device.queue.onSubmittedWorkDone();
    const dt = Math.max(performance.now() - t, 0.1);

    q = end;
    const scale = Math.min(4, Math.max(0.25, TARGET_MS / dt));
    chunk = Math.max(KNN_WG, Math.min(N, Math.round(chunk * scale)));

    const pct = ((q / N) * 100) | 0;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    onStatus(`Building kNN graph on GPU… ${pct}% (${q}/${N}, ${secs}s)`);
  }

  // --- Readback ----------------------------------------------------------
  const stageI = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stageD = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(idxBuf, 0, stageI, 0, outBytes);
  enc.copyBufferToBuffer(d2Buf, 0, stageD, 0, outBytes);
  device.queue.submit([enc.finish()]);

  await stageI.mapAsync(GPUMapMode.READ);
  await stageD.mapAsync(GPUMapMode.READ);
  const idx = new Uint32Array(stageI.getMappedRange().slice(0));
  const d2 = new Float32Array(stageD.getMappedRange().slice(0));
  stageI.unmap();
  stageD.unmap();

  for (const b of [xBuf, idxBuf, d2Buf, pBuf, stageI, stageD]) b.destroy();
  return { idx, d2 };
}

// ---------------------------------------------------------------------------
// NN-Descent (GPU)
// ---------------------------------------------------------------------------

const NND_WG = 64;

export interface NndOptions {
  seed?: number;
  /** Hard cap on descent iterations. Default 12. */
  maxIters?: number;
  /** Stop once updates in an iteration fall below delta*N*K. Default 0.001 */
  delta?: number;
  /** Reverse-neighbor list cap per point. Default 16. */
  revCap?: number;
  onStatus?: (msg: string) => void;
}

function nndShaderSource(N: number, D: number, K: number, R: number): string {
  return /* wgsl */ `
struct NParams { range : vec4<u32> };   // qStart, qEnd, seed, _

@group(0) @binding(0) var<storage, read>       X    : array<f32>;
@group(0) @binding(1) var<storage, read_write> NbrI : array<u32>;
@group(0) @binding(2) var<storage, read_write> NbrD : array<f32>;
@group(0) @binding(3) var<storage, read_write> RevI : array<u32>;
@group(0) @binding(4) var<storage, read_write> RevC : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> Upd  : array<atomic<u32>>;
@group(0) @binding(6) var<uniform>             P    : NParams;

const N : u32 = ${N}u;
const D : u32 = ${D}u;
const K : u32 = ${K}u;
const R : u32 = ${R}u;

// Counter-based hash RNG. Stateless, so a thread can derive its whole draw
// sequence from (seed, i, k) without carrying state between dispatches.
fn hash3(a : u32, b : u32, c : u32) -> u32 {
  // WGSL defines no relative precedence between bitwise and arithmetic
  // operators, so the parens are required, not style.
  var h = (a * 0x9E3779B1u) ^ (b * 0x85EBCA6Bu) ^ (c * 0xC2B2AE35u);
  h = h ^ (h >> 16u); h = h * 0x7FEB352Du;
  h = h ^ (h >> 15u); h = h * 0x846CA68Bu;
  return h ^ (h >> 16u);
}

fn d2(i : u32, j : u32) -> f32 {
  var s : f32 = 0.0;
  let oi = i * D;
  let oj = j * D;
  for (var k : u32 = 0u; k < D; k = k + 1u) {
    let diff = X[oi + k] - X[oj + k];
    s = s + diff * diff;
  }
  return s;
}

// Random K distinct neighbors per point, distance-sorted. Fills all K slots, so
// every later kernel can assume a full row and skip the count < K branch that
// knn_main needs.
@compute @workgroup_size(${NND_WG})
fn nnd_init(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }

  var hD : array<f32, ${K}>;
  var hI : array<u32, ${K}>;
  var count : u32 = 0u;

  var draw : u32 = 0u;
  loop {
    if (count == K) { break; }
    // N-1 >= K is guaranteed by the caller, so this terminates; the bound is
    // only insurance against a pathological hash.
    if (draw > 8u * K + 64u) { break; }
    let c = hash3(P.range.z, i, draw) % N;
    draw = draw + 1u;
    if (c == i) { continue; }

    var dup = false;
    for (var k : u32 = 0u; k < count; k = k + 1u) {
      if (hI[k] == c) { dup = true; break; }
    }
    if (dup) { continue; }

    let s = d2(i, c);
    var pos = count;
    loop {
      if (pos == 0u) { break; }
      if (hD[pos - 1u] <= s) { break; }
      hD[pos] = hD[pos - 1u];
      hI[pos] = hI[pos - 1u];
      pos = pos - 1u;
    }
    hD[pos] = s;
    hI[pos] = c;
    count = count + 1u;
  }

  // A short row would break the N x K contract. Pad by repeating the last entry
  // rather than leaving garbage; the join will displace it on the first pass.
  let base = i * K;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    let src = select(count - 1u, k, k < count);
    NbrD[base + k] = hD[src];
    NbrI[base + k] = hI[src];
  }
}

@compute @workgroup_size(${NND_WG})
fn nnd_clear(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  atomicStore(&RevC[i], 0u);
  if (i == 0u) { atomicStore(&Upd[0], 0u); }
}

// Reverse adjacency, capped at R entries per point. Integer atomics only.
//
// The cap is what bounds worst-case join work: MNIST has hub points that
// thousands of rows point at, and an uncapped reverse list would hand one
// thread all of them. The cost is that WHICH R writers win is a race, so this
// path is not bit-reproducible across runs even at a fixed seed. See the
// nndescentGPU docstring.
@compute @workgroup_size(${NND_WG})
fn nnd_reverse(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let base = i * K;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    let j = NbrI[base + k];
    let s = atomicAdd(&RevC[j], 1u);
    if (s < R) { RevI[j * R + s] = i; }
  }
}

// One thread per point, owning row i and writing nothing else. Candidates are
// the neighbors-of-neighbors set: for each b in fwd(i) + rev(i), scan
// fwd(b) + rev(b). Best K survive in registers, same bounded insertion sort as
// knn_main.
//
// Other threads are rewriting the rows this one reads. That is safe rather than
// merely tolerated: only INDICES are read from foreign rows, never distances —
// every distance is recomputed here from X. So a torn read yields a stale or
// duplicated candidate index, which the dedup scan below already handles, and
// never a mismatched index/distance pair. No double buffering needed.
@compute @workgroup_size(${NND_WG})
fn nnd_join(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = P.range.x + gid.x;
  if (i >= P.range.y) { return; }

  let base = i * K;
  var hD : array<f32, ${K}>;
  var hI : array<u32, ${K}>;
  for (var k : u32 = 0u; k < K; k = k + 1u) {
    hD[k] = NbrD[base + k];
    hI[k] = NbrI[base + k];
  }
  var worst = hD[K - 1u];
  var changed : u32 = 0u;

  let revN = min(atomicLoad(&RevC[i]), R);
  let outer = K + revN;

  for (var o : u32 = 0u; o < outer; o = o + 1u) {
    var b : u32;
    if (o < K) { b = hI[o]; } else { b = RevI[i * R + (o - K)]; }
    if (b >= N) { continue; }

    let bRevN = min(atomicLoad(&RevC[b]), R);
    let inner = K + bRevN;

    for (var q : u32 = 0u; q < inner; q = q + 1u) {
      var c : u32;
      if (q < K) { c = NbrI[b * K + q]; } else { c = RevI[b * R + (q - K)]; }
      if (c == i || c >= N) { continue; }

      let s = d2(i, c);
      if (s >= worst) { continue; }

      var dup = false;
      for (var k : u32 = 0u; k < K; k = k + 1u) {
        if (hI[k] == c) { dup = true; break; }
      }
      if (dup) { continue; }

      var pos = K - 1u;
      loop {
        if (pos == 0u) { break; }
        if (hD[pos - 1u] <= s) { break; }
        hD[pos] = hD[pos - 1u];
        hI[pos] = hI[pos - 1u];
        pos = pos - 1u;
      }
      hD[pos] = s;
      hI[pos] = c;
      worst = hD[K - 1u];
      changed = changed + 1u;
    }
  }

  if (changed > 0u) {
    for (var k : u32 = 0u; k < K; k = k + 1u) {
      NbrD[base + k] = hD[k];
      NbrI[base + k] = hI[k];
    }
    atomicAdd(&Upd[0], changed);
  }
}
`;
}

/**
 * Approximate kNN by NN-Descent, as a drop-in alternative to `knnGPU`. Same
 * return shape and ordering (N x kCand row-major, ascending by squared
 * distance), so `samplePairs` cannot tell them apart.
 *
 * Approximate is the operative word: expect recall around 0.95 rather than the
 * 1.0 the brute-force paths give. That is the same trade the reference PaCMAP
 * makes — it uses ANNOY — so this is arguably closer to upstream than the exact
 * paths are. Measure it with the demo's `?knncheck=1`.
 *
 * Two properties differ from `knnGPU` beyond recall:
 *
 * - **Not bit-reproducible.** The reverse-list cap is first-R-writers-win, and
 *   that race is not seeded. Same seed, slightly different graph, slightly
 *   different layout. `knnGPU` and `bruteForceKnn` remain deterministic. This is
 *   observable: repeated runs at a fixed seed agree to about 1e-5 of recall.
 * - **It is slower than brute force at every size this demo reaches.** Measured
 *   several times slower than `knnGPU` at N=2000-5000, narrowing as N grows but
 *   nowhere near crossing over by 65k. Cost is `~N*K^2*D` per iteration against
 *   brute force's `N^2*D`, which should win — but `candidateCount` asks for
 *   `nNB + 50` candidates, so K is large (60 by default), and more importantly
 *   each thread walks its own scattered candidate set while `knn_main` has every
 *   thread in a workgroup hitting the same address at once. That memory pattern,
 *   not the arithmetic, is what dominates.
 *
 * So this is here for the asymptotics and for comparison, not as an
 * optimization: past ~100k, where `N^2` stops being viable at any throughput,
 * it is the only one of the three still standing. Below that, use `knnGPU`.
 */
export async function nndescentGPU(
  device: GPUDevice,
  X: Float32Array,
  N: number,
  D: number,
  kCand: number,
  opts: NndOptions = {}
): Promise<{ idx: Uint32Array; d2: Float32Array }> {
  if (kCand < 1) throw new Error("kCand must be >= 1");
  if (kCand > N - 1) throw new Error("kCand must be <= N-1");

  const K = kCand;
  const R = opts.revCap ?? 16;
  const maxIters = opts.maxIters ?? 12;
  const delta = opts.delta ?? 0.001;
  const seed = opts.seed ?? 42;
  const onStatus = opts.onStatus ?? (() => {});

  const S = GPUBufferUsage.STORAGE;
  const outBytes = N * K * 4;
  const xBuf = mkBuf(device, X, S);
  const nbrIBuf = device.createBuffer({
    size: outBytes,
    usage: S | GPUBufferUsage.COPY_SRC,
  });
  const nbrDBuf = device.createBuffer({
    size: outBytes,
    usage: S | GPUBufferUsage.COPY_SRC,
  });
  const revIBuf = device.createBuffer({ size: N * R * 4, usage: S });
  const revCBuf = device.createBuffer({ size: N * 4, usage: S });
  const updBuf = device.createBuffer({
    size: 4,
    usage: S | GPUBufferUsage.COPY_SRC,
  });
  const pBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const updStage = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const module = device.createShaderModule({
    code: nndShaderSource(N, D, K, R),
  });

  // The layout is declared explicitly rather than left to `layout: "auto"`.
  // Auto derives a *separate* layout per entry point containing only the
  // bindings that entry point actually reads — nnd_init never touches the
  // reverse lists or the update counter, so its auto layout would have four
  // bindings, not seven, and one shared bind group would fail validation
  // against it. Declaring the full set once lets all four pipelines share a
  // single layout and a single bind group.
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
      storageEntry(0, "read-only-storage"),
      storageEntry(1, "storage"),
      storageEntry(2, "storage"),
      storageEntry(3, "storage"),
      storageEntry(4, "storage"),
      storageEntry(5, "storage"),
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
  const pipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [layout],
  });

  const mkPipe = (entryPoint: string) =>
    device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint },
    });
  const initPipe = mkPipe("nnd_init");
  const clearPipe = mkPipe("nnd_clear");
  const revPipe = mkPipe("nnd_reverse");
  const joinPipe = mkPipe("nnd_join");

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: xBuf } },
      { binding: 1, resource: { buffer: nbrIBuf } },
      { binding: 2, resource: { buffer: nbrDBuf } },
      { binding: 3, resource: { buffer: revIBuf } },
      { binding: 4, resource: { buffer: revCBuf } },
      { binding: 5, resource: { buffer: updBuf } },
      { binding: 6, resource: { buffer: pBuf } },
    ],
  });

  const allGroups = Math.ceil(N / NND_WG);
  const range = new Uint32Array(4);
  const t0 = performance.now();

  // --- Init --------------------------------------------------------------
  range[0] = 0;
  range[1] = N;
  range[2] = seed >>> 0;
  device.queue.writeBuffer(pBuf, 0, range);
  {
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(initPipe);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(allGroups);
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }
  onStatus(`NN-Descent: random init done (${N} points, k=${K})…`);

  // --- Descent -----------------------------------------------------------
  // Same adaptive chunking as knnGPU, for the same reason: one dispatch over
  // all N runs for seconds at scale and trips the ~2s OS GPU watchdog.
  //
  // Unlike knnGPU's chunks these are NOT independent — a later chunk reads rows
  // an earlier one already improved. For NN-Descent that is benign and if
  // anything accelerates convergence (pynndescent updates in place too); the
  // graph is a fixed point being approached, not a partitioned output.
  const TARGET_MS = 40;
  const stopBelow = delta * N * K;
  let chunk = Math.max(NND_WG, Math.min(N, Math.ceil(2e8 / (K * K * D))));
  let iters = 0;
  let lastUpd = 0;

  for (let iter = 0; iter < maxIters; iter++) {
    // Reverse lists are rebuilt from the current graph each iteration.
    {
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(clearPipe);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(allGroups);
      pass.setPipeline(revPipe);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(allGroups);
      pass.end();
      device.queue.submit([enc.finish()]);
    }

    let q = 0;
    while (q < N) {
      const end = Math.min(q + chunk, N);
      range[0] = q;
      range[1] = end;
      device.queue.writeBuffer(pBuf, 0, range);

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(joinPipe);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil((end - q) / NND_WG));
      pass.end();
      device.queue.submit([enc.finish()]);

      const t = performance.now();
      await device.queue.onSubmittedWorkDone();
      const dt = Math.max(performance.now() - t, 0.1);

      q = end;
      const scale = Math.min(4, Math.max(0.25, TARGET_MS / dt));
      chunk = Math.max(NND_WG, Math.min(N, Math.round(chunk * scale)));

      const pct = ((q / N) * 100) | 0;
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      onStatus(
        `NN-Descent iter ${iter + 1}/${maxIters}… ${pct}% (${secs}s)`
      );
    }

    // A 4-byte readback per iteration. Cheap, and it doubles as the host yield
    // point — this library stays DOM-free, so there is no rAF to hand back to.
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(updBuf, 0, updStage, 0, 4);
    device.queue.submit([enc.finish()]);
    await updStage.mapAsync(GPUMapMode.READ);
    lastUpd = new Uint32Array(updStage.getMappedRange())[0];
    updStage.unmap();

    iters = iter + 1;
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    onStatus(
      `NN-Descent iter ${iters}/${maxIters}: ${lastUpd} updates (${secs}s)`
    );
    if (lastUpd < stopBelow) break;
  }

  onStatus(
    `NN-Descent converged in ${iters} iters ` +
      `(${lastUpd} final updates, ${((performance.now() - t0) / 1000).toFixed(1)}s)`
  );

  // --- Readback ----------------------------------------------------------
  const stageI = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const stageD = device.createBuffer({
    size: outBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(nbrIBuf, 0, stageI, 0, outBytes);
  enc.copyBufferToBuffer(nbrDBuf, 0, stageD, 0, outBytes);
  device.queue.submit([enc.finish()]);

  await stageI.mapAsync(GPUMapMode.READ);
  await stageD.mapAsync(GPUMapMode.READ);
  const idx = new Uint32Array(stageI.getMappedRange().slice(0));
  const dd = new Float32Array(stageD.getMappedRange().slice(0));
  stageI.unmap();
  stageD.unmap();

  for (const b of [
    xBuf, nbrIBuf, nbrDBuf, revIBuf, revCBuf, updBuf, pBuf,
    updStage, stageI, stageD,
  ]) {
    b.destroy();
  }
  return { idx, d2: dd };
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
  /** Near and mid-near only. Further pairs are `fpFwd` — see `buildCSR`. */
  i: Uint32Array;
  j: Uint32Array;
  t: Uint8Array;
  /**
   * N x nNB, row-major: i's own near partners, densely indexable from a shader.
   * This is the reject list the LocalMAP resample needs, and it is the same set
   * the T_NB entries of the CSR carry — kept separately because the CSR
   * interleaves i's own pairs with the ones pointing back at it, and the
   * resample must reject only the former. Short rows (possible only when
   * kCand < nNB, i.e. at tiny N) are padded with N, which matches no candidate.
   */
  nbFwd: Uint32Array;
  /**
   * N x nFP, row-major: i's own further partners. Held outside the CSR because
   * LocalMAP redraws it mid-run; under PaCMAP it simply never changes.
   */
  fpFwd: Uint32Array;
}

/** Candidate pool size the kNN stage must produce for `samplePairs`. */
function candidateCount(N: number, nNB: number): number {
  return Math.min(nNB + 50, N - 1);
}

/**
 * Takes the kNN result rather than computing it, so this stays synchronous
 * while the graph itself is built on the GPU (an async step).
 */
function samplePairs(
  X: Float32Array,
  N: number,
  D: number,
  nNB: number,
  nMN: number,
  nFP: number,
  rand: () => number,
  knn: { idx: Uint32Array; d2: Float32Array },
  kCand: number
): Pairs {
  const { idx, d2 } = knn;

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

  // Near and mid-near go to the CSR; further pairs go to their own dense array.
  const total = N * (nNB + nMN);
  const pi = new Uint32Array(total);
  const pj = new Uint32Array(total);
  const pt = new Uint8Array(total);
  const nbFwd = new Uint32Array(N * nNB).fill(N);
  const fpFwd = new Uint32Array(N * nFP);
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
      nbFwd[i * nNB + k] = j;
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
      fpFwd[i * nFP + f] = c;
    }
  }

  return {
    i: pi.subarray(0, w),
    j: pj.subarray(0, w),
    t: pt.subarray(0, w),
    nbFwd,
    fpFwd,
  };
}

/**
 * Reverse adjacency for the further pairs: for each point, every point that
 * drew it. One buffer, two regions — `[0, N+1)` is the offset table and
 * everything after it is the index list, so the gradient shader spends one
 * binding on this rather than two (it is already at the 8-storage-buffer limit).
 *
 * The GPU rebuilds this every 10 iterations under LocalMAP; this is the CPU
 * build that seeds it, and the oracle its kernels were checked against.
 *
 * Sources land in ascending order because `i` ascends. `fp_scatter` on the GPU
 * has no such guarantee — its cursor is atomic — which is why it is followed by
 * a sort: the summation order of an f32 gradient has to be reproducible.
 */
function buildFpReverse(
  N: number,
  nFP: number,
  fpFwd: Uint32Array
): Uint32Array {
  const out = new Uint32Array(N + 1 + N * nFP);

  for (let p = 0; p < fpFwd.length; p++) out[fpFwd[p] + 1]++;
  for (let i = 0; i < N; i++) out[i + 1] += out[i];

  const cursor = out.slice(0, N);
  for (let i = 0; i < N; i++) {
    for (let f = 0; f < nFP; f++) out[N + 1 + cursor[fpFwd[i * nFP + f]]++] = i;
  }
  return out;
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

/**
 * Every WGSL source in this file, exposed so `scripts/check-shaders.ts` can
 * compile them headlessly. Nothing at runtime reads this — the pipelines call
 * the functions directly.
 */
export const shaderSources = {
  knnShaderSource,
  nndShaderSource,
  fpShaderSource,
  shaderSource,
};

const FP_WG = 64;

/**
 * Rebuilds the further-pair reverse CSR that `grad_main` gathers from, entirely
 * on the GPU. LocalMAP redraws the forward pairs every 10 iterations of phase 3
 * against the *embedding*, so this has to run where Y already lives — reading
 * positions back two dozen times a run to rebuild the structure on the host is
 * exactly what this library is built not to do. (At the default phases the
 * chain fires 24 times: iterations 210, 220 … 440.)
 *
 * The chain is clear -> count -> scan -> scatter -> sort, and it is the GPU
 * equivalent of `buildFpReverse`, which stays the oracle it was checked against.
 *
 * Two notes on the shape it takes:
 *
 * - **The scan is one serial invocation.** N is at most a few hundred thousand
 *   and this runs two dozen times a run, so a single thread walking N counters
 *   costs a fraction of a millisecond. A parallel two-level scan would buy µs
 *   and cost a page of WGSL with real correctness subtleties — the same trade
 *   the demo's single-workgroup bounds reduce makes.
 * - **The sort is not decoration.** `fp_scatter` claims slots with an atomic
 *   cursor, so which writer lands where is a race; sorting each reverse list
 *   afterwards makes the gradient's f32 summation order reproducible again.
 *   Without it a fixed seed would stop determining the layout, which is a
 *   property only `nndescentGPU` is allowed to give up.
 */
function fpShaderSource(
  N: number,
  nFP: number,
  nNB: number,
  lowDistThres: number,
  seed: number
): string {
  return /* wgsl */ `
struct Params {
  w  : vec4<f32>,
  bc : vec4<f32>,
  it : vec4<f32>,   // iteration index, _, _, _
};

@group(0) @binding(0) var<storage, read_write> FpF : array<u32>;
@group(0) @binding(1) var<storage, read_write> FpR : array<u32>;
@group(0) @binding(2) var<storage, read_write> Cnt : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>       Y   : array<f32>;
@group(0) @binding(4) var<storage, read>       NbF : array<u32>;
@group(0) @binding(5) var<uniform>             P   : Params;

const N   : u32 = ${N}u;
const NFP : u32 = ${nFP}u;
const NNB : u32 = ${nNB}u;
const SEED : u32 = ${seed >>> 0}u;
// Compared squared, so the reference's euclid_dist(...) > low_dist_thres needs
// no root here. Both sides are non-negative, so the comparison is equivalent.
const THRES2 : f32 = ${lowDistThres * lowDistThres};
// Same two-region layout grad_main reads: [0, N+1) offsets, then the indices.
const FPR_BASE : u32 = ${N + 1}u;

// Counter-based hash RNG, same one nnd_init uses: stateless, so a thread can
// derive its whole draw sequence from (seed, slot, try) with nothing carried
// between dispatches. That is what makes the resample reproducible where
// upstream's np.random.randint is not.
fn hash3(a : u32, b : u32, c : u32) -> u32 {
  var h = (a * 0x9E3779B1u) ^ (b * 0x85EBCA6Bu) ^ (c * 0xC2B2AE35u);
  h = h ^ (h >> 16u); h = h * 0x7FEB352Du;
  h = h ^ (h >> 15u); h = h * 0x846CA68Bu;
  return h ^ (h >> 16u);
}

// LocalMAP's local graph adjustment: redraw this point's further partners,
// keeping only candidates that are already CLOSE in the embedding. That is what
// turns the further-pair term from a global scatter into a local one, and it is
// why the pair set cannot be fixed for the whole run.
//
// Rejects, in the reference's order: self, a partner already drawn this round,
// one of this point's own near partners, and anything beyond THRES2.
//
// MAX_TRIES is a hard bound, which is a deliberate departure from the
// reference's control flow. There, both the self-hit and the distance-failure
// paths 'continue' past the 'if count > 100' escape, so a point with no
// eligible partner inside low_dist_thres loops forever. On a CPU that is a hang
// nobody hits often enough to notice; on a GPU it takes out the device. The
// escape is plainly the intent, so it is applied to every rejection path.
const MAX_TRIES : u32 = 100u;

@compute @workgroup_size(${FP_WG})
fn fp_resample(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }

  let yi = vec2<f32>(Y[2u * i], Y[2u * i + 1u]);
  let round = u32(P.it.x);
  let base = i * NFP;

  // Drawn into registers rather than straight into FpF, so the duplicate scan
  // sees only this round's picks — matching the reference, whose result array
  // holds -1 for slots not yet filled. N is the sentinel: no candidate equals it.
  var picks : array<u32, ${nFP}>;
  for (var s : u32 = 0u; s < NFP; s = s + 1u) { picks[s] = N; }

  for (var s : u32 = 0u; s < NFP; s = s + 1u) {
    for (var t : u32 = 0u; t < MAX_TRIES; t = t + 1u) {
      let c = hash3(SEED, i * NFP + s, round * 128u + t) % N;
      if (c == i) { continue; }

      var reject = false;
      for (var k : u32 = 0u; k < s; k = k + 1u) {
        if (picks[k] == c) { reject = true; break; }
      }
      if (reject) { continue; }

      // Short rows are padded with N, which matches no candidate.
      for (var k : u32 = 0u; k < NNB; k = k + 1u) {
        if (NbF[i * NNB + k] == c) { reject = true; break; }
      }
      if (reject) { continue; }

      let d = yi - vec2<f32>(Y[2u * c], Y[2u * c + 1u]);
      if (dot(d, d) > THRES2) { continue; }

      picks[s] = c;
      break;
    }
  }

  // A slot that exhausted its budget keeps whatever partner it already had,
  // which is what the reference does when it falls back to old_pair_FP.
  for (var s : u32 = 0u; s < NFP; s = s + 1u) {
    if (picks[s] != N) { FpF[base + s] = picks[s]; }
  }
}

@compute @workgroup_size(${FP_WG})
fn fp_clear(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  atomicStore(&Cnt[i], 0u);
}

// Reverse degree: how many points drew i. Unlike nnd_reverse there is no cap —
// the total is exactly N*NFP however it distributes, so the allocation is known
// in advance and no contribution has to be dropped.
@compute @workgroup_size(${FP_WG})
fn fp_count(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let base = i * NFP;
  for (var s : u32 = 0u; s < NFP; s = s + 1u) {
    atomicAdd(&Cnt[FpF[base + s]], 1u);
  }
}

// Exclusive prefix sum of the degrees into the offset region, reseating each
// counter as that point's write cursor on the way past. One invocation, one
// pass; see the docstring for why this is deliberate rather than lazy.
@compute @workgroup_size(1)
fn fp_scan() {
  var acc : u32 = 0u;
  for (var i : u32 = 0u; i < N; i = i + 1u) {
    let c = atomicLoad(&Cnt[i]);
    FpR[i] = acc;
    atomicStore(&Cnt[i], acc);
    acc = acc + c;
  }
  FpR[N] = acc;
}

@compute @workgroup_size(${FP_WG})
fn fp_scatter(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let base = i * NFP;
  for (var s : u32 = 0u; s < NFP; s = s + 1u) {
    let slot = atomicAdd(&Cnt[FpF[base + s]], 1u);
    FpR[FPR_BASE + slot] = i;
  }
}

// Sorts each reverse list ascending, which is the order the CPU build produces
// and the order the atomic scatter above destroys. Insertion sort: these lists
// average NFP entries, and the pass is over a few tens of elements per thread.
@compute @workgroup_size(${FP_WG})
fn fp_sort(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= N) { return; }
  let lo = FpR[i];
  let hi = FpR[i + 1u];

  for (var a : u32 = lo + 1u; a < hi; a = a + 1u) {
    let v = FpR[FPR_BASE + a];
    var b = a;
    loop {
      if (b == lo) { break; }
      if (FpR[FPR_BASE + b - 1u] <= v) { break; }
      FpR[FPR_BASE + b] = FpR[FPR_BASE + b - 1u];
      b = b - 1u;
    }
    FpR[FPR_BASE + b] = v;
  }
}
`;
}

function shaderSource(N: number, nFP: number): string {
  return /* wgsl */ `
struct Params {
  w  : vec4<f32>,   // wNB, wMN, wFP, lr
  bc : vec4<f32>,   // 1/(1-b1^t), 1/(1-b2^t), nnA, nnB
  it : vec4<f32>,   // iteration index, _, _, _
};

@group(0) @binding(0) var<storage, read_write> Y    : array<f32>;
@group(0) @binding(1) var<storage, read>       Off  : array<u32>;
@group(0) @binding(2) var<storage, read>       Adj  : array<u32>;
@group(0) @binding(3) var<storage, read_write> Grad : array<f32>;
@group(0) @binding(4) var<storage, read_write> M    : array<f32>;
@group(0) @binding(5) var<storage, read_write> V    : array<f32>;
@group(0) @binding(6) var<uniform>             P    : Params;
@group(0) @binding(7) var<storage, read>       FpF  : array<u32>;
@group(0) @binding(8) var<storage, read>       FpR  : array<u32>;

const N   : u32 = ${N}u;
const NFP : u32 = ${nFP}u;
// FpR is one buffer holding two regions: [0, N+1) offsets, then the indices.
const FPR_BASE : u32 = ${N + 1}u;

// The further-pair force, shared by the forward and reverse loops below. The
// sign works out for both from the one expression, exactly as it does for the
// CSR's duplicated entries: the displacement flips between the two directions
// while the scalar, a function of |d|^2 only, does not.
fn fpForce(yi : vec2<f32>, j : u32) -> vec2<f32> {
  let d = yi - vec2<f32>(Y[2u * j], Y[2u * j + 1u]);
  let dd = 1.0 + dot(d, d);
  let r = 1.0 + dd;
  return (-P.w.z * 2.0 / (r * r)) * d;
}

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
      // The trailing factor is what separates LocalMAP from PaCMAP. Upstream
      // writes it as a branch — phase 3 calls a whole second gradient kernel
      // that multiplies this coefficient by (low_dist_thres/2)/sqrt(d_ij) —
      // but as an affine (nnA, nnB) pair it needs neither a branch nor a
      // second entry point, and costs one FMA on the path that already runs.
      //   PaCMAP, and LocalMAP up to n1+n2:  (1, 0)              -> unchanged
      //   LocalMAP after n1+n2:              (0, lowDistThres/2)
      c = P.w.x * 20.0 / (r * r) * (P.bc.z + P.bc.w * inverseSqrt(dd));
    } else {                   // mid-near:  w * dd / (10000 + dd)
      let r = 10000.0 + dd;
      c = P.w.y * 20000.0 / (r * r);
    }

    g = g + c * d;
  }

  // Further pairs, gathered from their own arrays rather than the CSR: this
  // point's own NFP draws, then every point that drew this one. Together these
  // are the same both-endpoints duplication buildCSR does for the other two
  // kinds, split across two arrays instead of interleaved into one.
  let fpBase = i * NFP;
  for (var s : u32 = 0u; s < NFP; s = s + 1u) {
    g = g + fpForce(yi, FpF[fpBase + s]);
  }
  let rlo = FpR[i];
  let rhi = FpR[i + 1u];
  for (var p : u32 = rlo; p < rhi; p = p + 1u) {
    g = g + fpForce(yi, FpR[FPR_BASE + p]);
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
  const nNB = opts.nNeighbors ?? defaultNeighbors(N);
  const nMN = Math.round(nNB * (opts.mnRatio ?? 0.5));
  const nFP = Math.round(nNB * (opts.fpRatio ?? 2.0));
  const [n1, n2, n3] = opts.phases ?? [100, 100, 250];
  const lr = opts.lr ?? 1.0;
  const variant = opts.variant ?? "pacmap";
  const lowDistThres = opts.lowDistThres ?? 10;
  const totalIters = n1 + n2 + n3;
  const rand = mulberry32(opts.seed ?? 42);
  const onStatus = opts.onStatus ?? (() => {});

  // --- Setup -------------------------------------------------------------
  // [X is used as given. The reference applies PCA to 100 dims first when
  //  D > 100; add that here if your input is high-dimensional.]
  const kCand = candidateCount(N, nNB);
  const knn =
    opts.knn === "cpu"
      ? await bruteForceKnn(X, N, D, kCand, onStatus)
      : opts.knn === "nndescent"
        ? await nndescentGPU(device, X, N, D, kCand, {
            ...opts.nndescent,
            seed: opts.seed ?? 42,
            onStatus,
          })
        : await knnGPU(device, X, N, D, kCand, onStatus);

  onStatus("Sampling pairs…");
  const pairs = samplePairs(X, N, D, nNB, nMN, nFP, rand, knn, kCand);
  const { offsets, data } = buildCSR(N, pairs);
  const fpRev = buildFpReverse(N, nFP, pairs.fpFwd);

  // [Reference default init is scaled PCA. Gaussian is used here for brevity;
  //  it converges but embeddings will differ run-to-run across seeds.]
  const Y0 = new Float32Array(N * 2);
  for (let i = 0; i < N * 2; i++) {
    const u = Math.max(rand(), 1e-12);
    const v = rand();
    Y0[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * 0.0001;
  }

  // --- Buffers -----------------------------------------------------------
  const mk = (arr: ArrayBufferView, usage: GPUBufferUsageFlags) =>
    mkBuf(device, arr, usage);

  const S = GPUBufferUsage.STORAGE;
  const yBuf = mk(Y0, S | GPUBufferUsage.COPY_SRC | GPUBufferUsage.VERTEX);
  const offBuf = mk(offsets, S);
  const adjBuf = mk(data, S);
  const gradBuf = device.createBuffer({ size: N * 2 * 4, usage: S });
  const mBuf = mk(new Float32Array(N * 2), S);
  const vBuf = mk(new Float32Array(N * 2), S);
  // These two put the gradient shader at 8 storage buffers, which is the
  // default maxStorageBuffersPerShaderStage — no headroom left. If a ninth is
  // ever needed, M and V are the ones to interleave: same length, same access
  // pattern, and adam_main already walks both in lockstep.
  const fpFwdBuf = mk(pairs.fpFwd, S);
  const fpRevBuf = mk(fpRev, S);

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
    // (nnA, nnB) — see the near-pair branch in `shaderSource`. Note the strict
    // `>`: upstream guards this with `itr > num_iters[0] + num_iters[1]`, so at
    // the default phases the modified gradient starts at iteration 201, one
    // step after phase 3 itself does.
    const localNB = variant === "localmap" && it > n1 + n2;
    params[o + 6] = localNB ? 0 : 1;
    params[o + 7] = localNB ? lowDistThres / 2 : 0;
    // Read only by the further-pair resample, which needs a per-iteration draw
    // counter. It has to travel in this slot rather than a buffer written per
    // resample: queue writes are ordered against the single submit(), so every
    // resample encoded into one command buffer would see the last value.
    params[o + 8] = it;
  }
  const paramBuf = mk(params, GPUBufferUsage.UNIFORM);

  // --- Pipelines ---------------------------------------------------------
  const module = device.createShaderModule({ code: shaderSource(N, nFP) });

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
        buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 48 },
      },
      storageEntry(7, "read-only-storage"),
      storageEntry(8, "read-only-storage"),
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
      { binding: 6, resource: { buffer: paramBuf, size: 48 } },
      { binding: 7, resource: { buffer: fpFwdBuf } },
      { binding: 8, resource: { buffer: fpRevBuf } },
    ],
  });

  const groups = Math.ceil(N / 64);

  // --- LocalMAP's local graph adjustment ---------------------------------
  // Built only for the variant that uses it: under "pacmap" nothing below is
  // allocated and nothing is encoded, so that path is exactly what it was.
  //
  // Its own bind group, because the shaders disagree about Y — the optimizer
  // writes it, the resample only reads it — and because `layout: "auto"` would
  // derive a different narrow layout per entry point.
  const fp = (() => {
    if (variant !== "localmap") return null;

    const cntBuf = device.createBuffer({ size: N * 4, usage: S });
    const nbBuf = mk(pairs.nbFwd, S);
    const fpModule = device.createShaderModule({
      code: fpShaderSource(N, nFP, nNB, lowDistThres, opts.seed ?? 42),
    });
    const fpLayout = device.createBindGroupLayout({
      entries: [
        storageEntry(0, "storage"),
        storageEntry(1, "storage"),
        storageEntry(2, "storage"),
        storageEntry(3, "read-only-storage"),
        storageEntry(4, "read-only-storage"),
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: "uniform",
            hasDynamicOffset: true,
            minBindingSize: 48,
          },
        },
      ],
    });
    const fpPipelineLayout = device.createPipelineLayout({
      bindGroupLayouts: [fpLayout],
    });

    // Order is the chain: redraw, then rebuild the reverse CSR around it.
    // fp_scan is the one serial stage — see its docstring.
    const stages = (
      [
        ["fp_resample", groups],
        ["fp_clear", groups],
        ["fp_count", groups],
        ["fp_scan", 1],
        ["fp_scatter", groups],
        ["fp_sort", groups],
      ] as const
    ).map(([entryPoint, n]) => ({
      pipeline: device.createComputePipeline({
        layout: fpPipelineLayout,
        compute: { module: fpModule, entryPoint },
      }),
      groups: n,
    }));

    return {
      stages,
      buffers: [cntBuf, nbBuf],
      bindGroup: device.createBindGroup({
        layout: fpLayout,
        entries: [
          { binding: 0, resource: { buffer: fpFwdBuf } },
          { binding: 1, resource: { buffer: fpRevBuf } },
          { binding: 2, resource: { buffer: cntBuf } },
          { binding: 3, resource: { buffer: yBuf } },
          { binding: 4, resource: { buffer: nbBuf } },
          { binding: 5, resource: { buffer: paramBuf, size: 48 } },
        ],
      }),
    };
  })();

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

      // The graph adjustment runs after the step, on the embedding the step
      // just produced — the order the reference's loop uses. The guard is its
      // guard: strictly after n1+n2, every tenth iteration, so at the default
      // phases this is 210, 220 … 440 and never 200.
      //
      // It chains into the same pass, so no host round-trip appears here and
      // runRange still submits exactly one command buffer. Dispatches inside a
      // pass are synchronized by WebGPU, so the chain needs no explicit barrier
      // between its stages any more than grad -> adam does.
      if (fp && it > n1 + n2 && it % 10 === 0) {
        pass.setBindGroup(0, fp.bindGroup, [it * align]);
        for (const stage of fp.stages) {
          pass.setPipeline(stage.pipeline);
          pass.dispatchWorkgroups(stage.groups);
        }
      }
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
      for (const b of [
        yBuf, offBuf, adjBuf, gradBuf, mBuf, vBuf, paramBuf,
        fpFwdBuf, fpRevBuf,
        ...(fp?.buffers ?? []),
      ]) {
        b.destroy();
      }
    },
  };
}

/**
 * LocalMAP, for callers who would rather name the algorithm than pass a flag.
 * Exactly `pacmapWebGPU` with `variant: "localmap"` — upstream's `LocalMAP` is
 * likewise a subclass that reuses `PaCMAP.fit` wholesale.
 */
export function localmapWebGPU(
  device: GPUDevice,
  X: Float32Array,
  N: number,
  D: number,
  opts: Omit<PacmapOptions, "variant"> = {}
): Promise<PacmapRun> {
  return pacmapWebGPU(device, X, N, D, { ...opts, variant: "localmap" });
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
