/**
 * Randomized PCA projection, D -> k.
 *
 * PaCMAP's reference implementation reduces to 100 dims before building the
 * kNN graph, and on MNIST it's the difference between a demo that runs and one
 * that doesn't: the CPU brute-force kNN is O(N^2 * D), so 784 -> 100 cuts it by
 * ~8x on its own.
 *
 * This uses a randomized range finder rather than a full eigendecomposition.
 * The output spans (an approximation of) the top-k principal subspace but is
 * NOT rotated to the principal axes. That's fine here: pairwise distances
 * inside the subspace are identical either way, and distances are all the kNN
 * step consumes.
 *
 * [If you want true axis-ordered components — e.g. to use PC1/PC2 as a PaCMAP
 *  init — you'd need an SVD of B below.]
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

/** A (n x m) @ B (m x p) -> n x p */
function matmul(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number
): Float32Array {
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
  }
  return C;
}

/** A^T (m x n) @ B (n x p) -> m x p, where A is stored n x m */
function matmulAtB(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number
): Float32Array {
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
  }
  return C;
}

/** A (n x m) @ B^T (m x p), where B is stored p x m -> n x p */
function matmulABt(
  A: Float32Array,
  B: Float32Array,
  n: number,
  m: number,
  p: number
): Float32Array {
  const C = new Float32Array(n * p);
  for (let i = 0; i < n; i++) {
    const ai = i * m;
    for (let j = 0; j < p; j++) {
      const bj = j * m;
      let s = 0;
      for (let k = 0; k < m; k++) s += A[ai + k] * B[bj + k];
      C[i * p + j] = s;
    }
  }
  return C;
}

/** Modified Gram-Schmidt over the columns of M (n x p), in place. */
function orthonormalizeCols(M: Float32Array, n: number, p: number): void {
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

export function pcaProject(
  X: Float32Array,
  n: number,
  d: number,
  k: number,
  opts: { seed?: number; powerIters?: number } = {}
): Float32Array {
  const seed = opts.seed ?? 1;
  const powerIters = opts.powerIters ?? 1;
  const p = Math.min(k, d, n);

  // Center.
  const mean = new Float64Array(d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) mean[j] += X[i * d + j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  const Xc = new Float32Array(n * d);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < d; j++) Xc[i * d + j] = X[i * d + j] - mean[j];

  // Random test matrix, D x p.
  const rand = mulberry32(seed);
  const Om = new Float32Array(d * p);
  for (let i = 0; i < d * p; i++) {
    const u = Math.max(rand(), 1e-12);
    Om[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
  }

  // Sketch the column space, then push it toward the dominant subspace.
  let Y = matmul(Xc, Om, n, d, p); // n x p
  for (let t = 0; t < powerIters; t++) {
    const Z = matmulAtB(Xc, Y, n, d, p); // d x p
    Y = matmul(Xc, Z, n, d, p); // n x p
  }
  orthonormalizeCols(Y, n, p); // Q, n x p

  // B = Q^T Xc  (p x d); its rows span the estimated principal subspace.
  const B = matmulAtB(Y, Xc, n, p, d);
  orthonormalizeRows(B, p, d);

  // Z = Xc @ B^T  (n x p)
  return matmulABt(Xc, B, n, d, p);
}
