/**
 * Randomized PCA projection, D -> k.
 *
 * PaCMAP's reference implementation reduces to 100 dims before building the
 * kNN graph, and on MNIST it's the difference between a demo that runs and one
 * that doesn't: the kNN step is O(N^2 * D), so 784 -> 100 cuts it by ~8x on its
 * own.
 *
 * This uses a randomized range finder rather than a full eigendecomposition.
 * The output spans (an approximation of) the top-k principal subspace but is
 * NOT rotated to the principal axes. That's fine here: pairwise distances
 * inside the subspace are identical either way, and distances are all the kNN
 * step consumes.
 *
 * [If you want true axis-ordered components — e.g. to use PC1/PC2 as a PaCMAP
 *  init — you'd need an SVD of B below.]
 *
 * Everything here is on the CPU and the total is ~4*n*d*k MACs, which at 65k
 * points is ~20 GMAC and tens of seconds. So every loop long enough to matter
 * yields to the event loop as it goes; the work is unchanged, it just stops
 * freezing the page. [Moving these matmuls to WGSL is the next real win — it
 * would take PCA from tens of seconds to under one.]
 */

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
// Cooperative yielding
// ---------------------------------------------------------------------------

/** Target time between yields. Long enough to amortize, short enough to feel live. */
const SLICE_MS = 25;

/** Rows between clock checks. ~64 rows of a 784x100 matmul is well under a slice. */
const ROW_STRIDE = 64;

const chan = typeof MessageChannel !== "undefined" ? new MessageChannel() : null;

/**
 * Hand control back to the event loop.
 *
 * Not setTimeout(0): that clamps to ~4ms once nested, which at this yield rate
 * would tack ~15% onto the total. A MessageChannel task has no clamp and still
 * lets the browser paint between slices.
 */
function tick(): Promise<void> {
  if (!chan) return Promise.resolve();
  return new Promise<void>((resolve) => {
    chan.port1.onmessage = () => resolve();
    chan.port2.postMessage(0);
  });
}

class Pacer {
  private last = performance.now();
  private readonly onStatus: (msg: string) => void;

  constructor(onStatus: (msg: string) => void) {
    this.onStatus = onStatus;
  }

  async maybeYield(label: string, done: number, total: number): Promise<void> {
    if (performance.now() - this.last < SLICE_MS) return;
    this.onStatus(`PCA · ${label} ${((done / total) * 100) | 0}%`);
    await tick();
    this.last = performance.now();
  }
}

// ---------------------------------------------------------------------------
// Kernels. Each loops over rows outermost, which is what makes chunking cheap.
// ---------------------------------------------------------------------------

/** A (n x m) @ B (m x p) -> n x p */
async function matmul(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number,
  pace: Pacer,
  label: string
): Promise<Float32Array> {
  const C = new Float32Array(n * p);
  for (let i = 0; i < n; i++) {
    const ai = i * m;
    const ci = i * p;
    for (let k = 0; k < m; k++) {
      const a = A[ai + k];
      if (a === 0) continue;
      const bk = k * p;
      for (let j = 0; j < p; j++) C[ci + j] += a * B[bk + j];
    }
    if ((i & (ROW_STRIDE - 1)) === 0) await pace.maybeYield(label, i, n);
  }
  return C;
}

/** A^T (m x n) @ B (n x p) -> m x p, where A is stored n x m */
async function matmulAtB(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number,
  pace: Pacer,
  label: string
): Promise<Float32Array> {
  const C = new Float32Array(m * p);
  for (let i = 0; i < n; i++) {
    const ai = i * m;
    const bi = i * p;
    for (let k = 0; k < m; k++) {
      const a = A[ai + k];
      if (a === 0) continue;
      const ck = k * p;
      for (let j = 0; j < p; j++) C[ck + j] += a * B[bi + j];
    }
    if ((i & (ROW_STRIDE - 1)) === 0) await pace.maybeYield(label, i, n);
  }
  return C;
}

/** A (n x m) @ B^T (m x p), where B is stored p x m -> n x p */
async function matmulABt(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number,
  pace: Pacer,
  label: string
): Promise<Float32Array> {
  const C = new Float32Array(n * p);
  for (let i = 0; i < n; i++) {
    const ai = i * m;
    for (let j = 0; j < p; j++) {
      const bj = j * m;
      let s = 0;
      for (let k = 0; k < m; k++) s += A[ai + k] * B[bj + k];
      C[i * p + j] = s;
    }
    if ((i & (ROW_STRIDE - 1)) === 0) await pace.maybeYield(label, i, n);
  }
  return C;
}

/** Modified Gram-Schmidt over the columns of M (n x p), in place. */
async function orthonormalizeCols(
  M: Float32Array,
  n: number,
  p: number,
  pace: Pacer
): Promise<void> {
  for (let j = 0; j < p; j++) {
    for (let q = 0; q < j; q++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += M[i * p + j] * M[i * p + q];
      for (let i = 0; i < n; i++) M[i * p + j] -= dot * M[i * p + q];
    }
    let nrm = 0;
    for (let i = 0; i < n; i++) nrm += M[i * p + j] * M[i * p + j];
    nrm = Math.sqrt(nrm);
    const inv = nrm > 1e-12 ? 1 / nrm : 0;
    for (let i = 0; i < n; i++) M[i * p + j] *= inv;
    await pace.maybeYield("orthonormalizing", j, p);
  }
}

/** Modified Gram-Schmidt over the rows of M (p x d), in place. */
function orthonormalizeRows(M: Float32Array, p: number, d: number): void {
  for (let r = 0; r < p; r++) {
    const rr = r * d;
    for (let q = 0; q < r; q++) {
      const qq = q * d;
      let dot = 0;
      for (let i = 0; i < d; i++) dot += M[rr + i] * M[qq + i];
      for (let i = 0; i < d; i++) M[rr + i] -= dot * M[qq + i];
    }
    let nrm = 0;
    for (let i = 0; i < d; i++) nrm += M[rr + i] * M[rr + i];
    nrm = Math.sqrt(nrm);
    const inv = nrm > 1e-12 ? 1 / nrm : 0;
    for (let i = 0; i < d; i++) M[rr + i] *= inv;
  }
}

// ---------------------------------------------------------------------------

export interface PcaOptions {
  seed?: number;
  powerIters?: number;
  /**
   * Center X in place instead of allocating a second n*d array. Saves 204MB at
   * 65k x 784 — worth it when the caller has no further use for X, which is the
   * common case. Destroys the input.
   */
  inPlace?: boolean;
  onStatus?: (msg: string) => void;
}

export async function pcaProject(
  X: Float32Array,
  n: number,
  d: number,
  k: number,
  opts: PcaOptions = {}
): Promise<Float32Array> {
  const seed = opts.seed ?? 1;
  const powerIters = opts.powerIters ?? 1;
  const p = Math.min(k, d, n);
  const pace = new Pacer(opts.onStatus ?? (() => {}));

  // Center. Note this kills the sparsity that the `a === 0` skips above would
  // otherwise exploit: MNIST is ~80% zero pixels, but zero minus the mean is
  // not zero.
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) mean[j] += X[i * d + j];
    if ((i & (ROW_STRIDE - 1)) === 0) await pace.maybeYield("centering", i, 2 * n);
  }
  for (let j = 0; j < d; j++) mean[j] /= n;

  const Xc = opts.inPlace ? X : new Float32Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) Xc[i * d + j] = X[i * d + j] - mean[j];
    if ((i & (ROW_STRIDE - 1)) === 0)
      await pace.maybeYield("centering", n + i, 2 * n);
  }

  // Random test matrix, D x p.
  const rand = mulberry32(seed);
  const Om = new Float32Array(d * p);
  for (let i = 0; i < d * p; i++) {
    const u = Math.max(rand(), 1e-12);
    Om[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // Sketch the column space, then push it toward the dominant subspace.
  let Y = await matmul(Xc, Om, n, d, p, pace, "sketching"); // n x p
  for (let t = 0; t < powerIters; t++) {
    const Z = await matmulAtB(Xc, Y, n, d, p, pace, "power iteration"); // d x p
    Y = await matmul(Xc, Z, n, d, p, pace, "power iteration"); // n x p
  }
  await orthonormalizeCols(Y, n, p, pace); // Q, n x p

  // B = Q^T Xc  (p x d); its rows span the estimated principal subspace.
  const B = await matmulAtB(Y, Xc, n, p, d, pace, "extracting components");
  orthonormalizeRows(B, p, d);

  // Z = Xc @ B^T  (n x p)
  return matmulABt(Xc, B, n, d, p, pace, "projecting");
}
