/**
 * Headless behaviour checks for the kernels `check:shaders` can only compile.
 *
 * A WGSL source that compiles and pipelines can still be wrong, and the wrong
 * answer here looks like a plausible embedding rather than an error — the same
 * failure mode that motivated `check:shaders`, one level up. These run the real
 * kernels under Dawn and check what they produce.
 *
 * Every case is self-contained: it either compares against an oracle computed
 * here on the CPU, or asserts an invariant the output must hold, or runs the
 * same thing twice and demands the same answer. Nothing depends on a stored
 * baseline, which is what lets this run in CI. For "did this refactor change
 * the embedding", which does need a baseline, see `scripts/ab-embedding.ts`.
 *
 * **Known gap, measured rather than assumed.** These cover the kernels, not the
 * *wiring* of the resample into `runRange`. Deleting the `it > n1 + n2 && it %
 * 10 === 0` block entirely leaves every check below green, because LocalMAP's
 * other change — the near-pair coefficient — is enough on its own to make
 * `e2e/localmap-differs-from-pacmap` pass. Closing that here would need the
 * further-pair state exposed on `PacmapRun` purely for tests. Until something
 * else wants it, the schedule is covered by `npm run check:ab -- <ref>
 * --variant=localmap`, which sees it as a ~0.08 relative move. If you touch
 * that block, run check:ab.
 *
 * Run with `npm run check:kernels`. Pass --strict to fail rather than skip when
 * no adapter is available (CI does).
 */

import { pacmapWebGPU, shaderSources } from "../src/pacmap-webgpu";
import { openDawn, upload, readback, rng } from "./dawn";

const STRICT = process.argv.includes("--strict");

let failures = 0;
let checks = 0;

function check(name: string, ok: boolean, detail = ""): void {
  checks++;
  if (ok) {
    console.log(`ok   ${name}`);
  } else {
    failures++;
    console.error(`FAIL ${name}${detail ? `: ${detail}` : ""}`);
  }
}

// ---------------------------------------------------------------------------
// The further-pair reverse CSR
// ---------------------------------------------------------------------------

/**
 * Independent CPU oracle: bucket by target, then flatten.
 *
 * Deliberately NOT `buildFpReverse` — that is the implementation this is meant
 * to corroborate, and an oracle sharing its code cannot catch a shared
 * misunderstanding. `buildFpReverse` stays unexported for the same reason.
 */
function oracleReverse(N: number, nFP: number, fpFwd: Uint32Array): Uint32Array {
  const buckets: number[][] = Array.from({ length: N }, () => []);
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < nFP; s++) buckets[fpFwd[i * nFP + s]].push(i);
  }
  const out = new Uint32Array(N + 1 + N * nFP);
  let w = 0;
  for (let j = 0; j < N; j++) {
    out[j] = w;
    for (const src of buckets[j].slice().sort((a, b) => a - b)) {
      out[N + 1 + w] = src;
      w++;
    }
  }
  out[N] = w;
  return out;
}

/** Layout shared by every entry point of `fpShaderSource`. */
function fpLayout(device: GPUDevice): GPUBindGroupLayout {
  return device.createBindGroupLayout({
    entries: [
      ...([0, 1, 2] as const).map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" as const },
      })),
      ...([3, 4] as const).map((binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" as const },
      })),
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform" as const,
          hasDynamicOffset: true,
          minBindingSize: 48,
        },
      },
    ],
  });
}

const FP_ALIGN = 256;

async function reverseCase(
  device: GPUDevice,
  name: string,
  N: number,
  nFP: number,
  fpFwd: Uint32Array
): Promise<void> {
  const module = device.createShaderModule({
    // nNB / lowDistThres / seed reach only fp_resample, which this case does
    // not dispatch — but the module declares their bindings either way.
    code: shaderSources.fpShaderSource(N, nFP, 4, 10, 7),
  });
  const bgl = fpLayout(device);
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bgl] });

  const S = GPUBufferUsage.STORAGE;
  const revLen = N + 1 + N * nFP;
  // Poison-filled, so anything the chain fails to write shows up as a
  // mismatch rather than as a plausible zero.
  const revBuf = upload(
    device,
    new Uint32Array(revLen).fill(0xdeadbeef),
    S | GPUBufferUsage.COPY_SRC
  );
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: upload(device, fpFwd, S) } },
      { binding: 1, resource: { buffer: revBuf } },
      { binding: 2, resource: { buffer: upload(device, new Uint32Array(N), S) } },
      { binding: 3, resource: { buffer: upload(device, new Uint32Array(N * 2), S) } },
      { binding: 4, resource: { buffer: upload(device, new Uint32Array(N * 4), S) } },
      {
        binding: 5,
        resource: {
          buffer: upload(
            device,
            new Float32Array(FP_ALIGN / 4),
            GPUBufferUsage.UNIFORM
          ),
          size: 48,
        },
      },
    ],
  });

  const groups = Math.ceil(N / 64);
  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setBindGroup(0, bindGroup, [0]);
  for (const [entryPoint, n] of [
    ["fp_clear", groups],
    ["fp_count", groups],
    ["fp_scan", 1],
    ["fp_scatter", groups],
    ["fp_sort", groups],
  ] as const) {
    pass.setPipeline(
      device.createComputePipeline({ layout, compute: { module, entryPoint } })
    );
    pass.dispatchWorkgroups(n);
  }
  pass.end();
  device.queue.submit([enc.finish()]);

  const got = await readback(device, revBuf, revLen * 4);
  const want = oracleReverse(N, nFP, fpFwd);

  let firstBad = -1;
  let nBad = 0;
  for (let k = 0; k < revLen; k++) {
    if (got[k] !== want[k]) {
      nBad++;
      if (firstBad < 0) firstBad = k;
    }
  }
  check(
    `fp-reverse/${name}`,
    nBad === 0,
    nBad === 0
      ? ""
      : `${nBad}/${revLen} words differ; first at ${firstBad} ` +
        `(${firstBad <= N ? "offsets" : "indices"}) ` +
        `want ${want[firstBad]} got ${got[firstBad]}`
  );
}

async function checkReverseCsr(device: GPUDevice): Promise<void> {
  {
    // nFP deliberately not 20, to catch anything hardcoded to the demo default.
    const N = 500;
    const nFP = 7;
    const r = rng(1);
    await reverseCase(
      device,
      "uniform",
      N,
      nFP,
      new Uint32Array(N * nFP).map(() => (r() * N) | 0)
    );
  }
  {
    // A hub taking most of the draws: worst case for the sort, and for any
    // per-point capacity assumption. nnd_reverse caps for exactly this reason;
    // this chain must not need to.
    const N = 500;
    const nFP = 8;
    const r = rng(2);
    await reverseCase(
      device,
      "hub",
      N,
      nFP,
      new Uint32Array(N * nFP).map(() => (r() < 0.6 ? 0 : (r() * N) | 0))
    );
  }
  {
    // Most points end with an empty reverse list — the case fp_sort's lo/hi
    // bounds can trip on.
    const N = 300;
    const nFP = 4;
    const f = new Uint32Array(N * nFP);
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < nFP; s++) f[i * nFP + s] = (i + 1) % N;
    }
    await reverseCase(device, "empty-lists", N, nFP, f);
  }
  {
    // N not a multiple of the workgroup size.
    const N = 517;
    const nFP = 3;
    const r = rng(4);
    await reverseCase(
      device,
      "ragged-N",
      N,
      nFP,
      new Uint32Array(N * nFP).map(() => (r() * N) | 0)
    );
  }
}

// ---------------------------------------------------------------------------
// The further-pair resample
// ---------------------------------------------------------------------------

const RES_N = 400;
const RES_NFP = 5;
const RES_NNB = 3;
const RES_THRES = 1.0;
const RES_SEED = 7;
/** Parked with nothing inside the threshold: the case that must not hang. */
const ISOLATED = RES_N - 1;

/**
 * Points on a line at spacing 0.1, so |y_i - y_j| = 0.1*|i-j| and "within
 * low_dist_thres" is exactly "within ten indices". The constraint is then
 * checkable rather than merely plausible.
 */
function resampleFixture() {
  const Y = new Float32Array(RES_N * 2);
  for (let i = 0; i < RES_N; i++) Y[2 * i] = 0.1 * i;
  Y[2 * ISOLATED] = 1e6;

  const nbFwd = new Uint32Array(RES_N * RES_NNB);
  for (let i = 0; i < RES_N; i++) {
    for (let k = 0; k < RES_NNB; k++) {
      nbFwd[i * RES_NNB + k] = (i + 1 + k) % RES_N;
    }
  }

  // Every slot starts somewhere ineligible, so anything eligible afterwards
  // must have been redrawn.
  const fp0 = new Uint32Array(RES_N * RES_NFP);
  for (let i = 0; i < RES_N; i++) {
    for (let s = 0; s < RES_NFP; s++) fp0[i * RES_NFP + s] = (i + 200) % RES_N;
  }
  return { Y, nbFwd, fp0 };
}

async function resample(
  device: GPUDevice,
  round: number
): Promise<Uint32Array> {
  const { Y, nbFwd, fp0 } = resampleFixture();
  const module = device.createShaderModule({
    code: shaderSources.fpShaderSource(
      RES_N,
      RES_NFP,
      RES_NNB,
      RES_THRES,
      RES_SEED
    ),
  });
  const bgl = fpLayout(device);
  const pipe = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "fp_resample" },
  });

  // Two aligned slots so the dynamic offset is genuinely exercised rather than
  // always landing on zero.
  const params = new Float32Array((2 * FP_ALIGN) / 4);
  params[FP_ALIGN / 4 + 8] = round;

  const S = GPUBufferUsage.STORAGE;
  const fpBuf = upload(device, fp0, S | GPUBufferUsage.COPY_SRC);
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: fpBuf } },
      {
        binding: 1,
        resource: {
          buffer: upload(
            device,
            new Uint32Array(RES_N + 1 + RES_N * RES_NFP),
            S
          ),
        },
      },
      { binding: 2, resource: { buffer: upload(device, new Uint32Array(RES_N), S) } },
      { binding: 3, resource: { buffer: upload(device, Y, S) } },
      { binding: 4, resource: { buffer: upload(device, nbFwd, S) } },
      {
        binding: 5,
        resource: {
          buffer: upload(device, params, GPUBufferUsage.UNIFORM),
          size: 48,
        },
      },
    ],
  });

  const enc = device.createCommandEncoder();
  const pass = enc.beginComputePass();
  pass.setPipeline(pipe);
  pass.setBindGroup(0, bindGroup, [FP_ALIGN]);
  pass.dispatchWorkgroups(Math.ceil(RES_N / 64));
  pass.end();
  device.queue.submit([enc.finish()]);

  return readback(device, fpBuf, fp0.byteLength);
}

async function checkResample(device: GPUDevice): Promise<void> {
  const { Y, nbFwd, fp0 } = resampleFixture();
  const got = await resample(device, 3);
  const dist = (a: number, b: number) =>
    Math.hypot(Y[2 * a] - Y[2 * b], Y[2 * a + 1] - Y[2 * b + 1]);

  let redrawn = 0;
  const bad = { self: 0, dup: 0, nbr: 0, far: 0 };
  for (let i = 0; i < RES_N; i++) {
    const nbrs = new Set(
      Array.from({ length: RES_NNB }, (_, k) => nbFwd[i * RES_NNB + k])
    );
    const seen = new Set<number>();
    for (let s = 0; s < RES_NFP; s++) {
      const j = got[i * RES_NFP + s];
      if (j !== fp0[i * RES_NFP + s]) {
        redrawn++;
        if (j === i) bad.self++;
        if (seen.has(j)) bad.dup++;
        if (nbrs.has(j)) bad.nbr++;
        if (dist(i, j) > RES_THRES + 1e-6) bad.far++;
      }
      seen.add(j);
    }
  }

  check("fp-resample/rejects-self", bad.self === 0, `${bad.self} violations`);
  check("fp-resample/rejects-duplicates", bad.dup === 0, `${bad.dup} violations`);
  check("fp-resample/rejects-near-partners", bad.nbr === 0, `${bad.nbr} violations`);
  check(
    "fp-resample/respects-low-dist-thres",
    bad.far === 0,
    `${bad.far} violations`
  );
  // Guards the checks above from passing vacuously on a kernel that wrote
  // nothing at all.
  check(
    "fp-resample/actually-redraws",
    redrawn > RES_N * RES_NFP * 0.5,
    `only ${redrawn} of ${RES_N * RES_NFP} slots redrawn`
  );

  // The reference cannot terminate here: its self-hit and distance-failure
  // branches both `continue` past the try-budget check.
  const isolatedKept = Array.from({ length: RES_NFP }, (_, s) => s).every(
    (s) => got[ISOLATED * RES_NFP + s] === fp0[ISOLATED * RES_NFP + s]
  );
  check("fp-resample/exhausted-slot-keeps-partner", isolatedKept);

  const same = await resample(device, 3);
  check(
    "fp-resample/same-round-reproduces",
    same.every((v, k) => v === got[k])
  );
  const other = await resample(device, 4);
  check(
    "fp-resample/different-round-differs",
    other.some((v, k) => v !== got[k])
  );
}

// ---------------------------------------------------------------------------
// End to end
// ---------------------------------------------------------------------------

const E2E_N = 400;
const E2E_D = 16;

/** Ten gaussian blobs, so the layout has structure rather than being a smear. */
function blobs(): Float32Array {
  const r = rng(12345);
  const gauss = () =>
    Math.sqrt(-2 * Math.log(Math.max(r(), 1e-12))) * Math.cos(2 * Math.PI * r());
  const centers = Array.from(
    { length: 10 },
    () => new Float32Array(E2E_D).map(() => gauss() * 4)
  );
  const X = new Float32Array(E2E_N * E2E_D);
  for (let i = 0; i < E2E_N; i++) {
    const c = centers[i % 10];
    for (let k = 0; k < E2E_D; k++) X[i * E2E_D + k] = c[k] + gauss();
  }
  return X;
}

async function embed(
  device: GPUDevice,
  variant: "pacmap" | "localmap"
): Promise<Float32Array> {
  const pm = await pacmapWebGPU(device, blobs(), E2E_N, E2E_D, {
    seed: 7,
    knn: "cpu",
    variant,
  });
  pm.run();
  const Y = await pm.read();
  pm.destroy();
  return Y;
}

async function checkEndToEnd(device: GPUDevice): Promise<void> {
  const pac = await embed(device, "pacmap");
  const pac2 = await embed(device, "pacmap");
  const lm = await embed(device, "localmap");
  const lm2 = await embed(device, "localmap");

  const identical = (a: Float32Array, b: Float32Array) =>
    a.every((v, k) => v === b[k]);
  const spread = (a: Float32Array) => Math.max(...a) - Math.min(...a);
  const maxDelta = (a: Float32Array, b: Float32Array) =>
    a.reduce((m, v, k) => Math.max(m, Math.abs(v - b[k])), 0);

  check("e2e/pacmap-reproducible", identical(pac, pac2));
  // This one is the whole reason fp_sort exists: the resample's scatter claims
  // slots with an atomic cursor, and without the sort the gradient's f32
  // summation order — and so the layout — would drift between runs.
  check("e2e/localmap-reproducible", identical(lm, lm2));

  const rel = maxDelta(pac, lm) / spread(pac);
  check(
    "e2e/localmap-differs-from-pacmap",
    rel > 1e-3,
    `relative delta only ${rel.toExponential(2)}`
  );

  // A finite, non-degenerate layout. Catches a kernel that quietly produced
  // NaN or collapsed every point onto one spot.
  check(
    "e2e/localmap-layout-is-sane",
    lm.every(Number.isFinite) && spread(lm) > 1e-3,
    `spread ${spread(lm)}`
  );
}

// ---------------------------------------------------------------------------

async function main(): Promise<number> {
  const session = await openDawn();
  if (!session) {
    console.error("no WebGPU adapter available");
    if (STRICT) return 1;
    console.error("SKIP (pass --strict to make this a failure)");
    return 0;
  }

  try {
    await checkReverseCsr(session.device);
    await checkResample(session.device);
    await checkEndToEnd(session.device);
  } finally {
    session.close();
  }

  console.log(
    failures === 0
      ? `\n${checks} kernel checks passed.`
      : `\n${failures} of ${checks} kernel checks failed.`
  );
  return failures === 0 ? 0 : 1;
}

// The bindings hold an interval per instance, so the process would otherwise
// sit here after main() resolves. Same note as scripts/check-shaders.ts.
process.exit(await main());
