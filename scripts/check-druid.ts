/**
 * Headless behaviour checks for the DruidJS CPU backends.
 *
 * The other three checks all point at our own code: `check:shaders` compiles our
 * WGSL, `check:kernels` runs our kernels against a CPU oracle, `check:ab`
 * measures whether a refactor moved our embedding. This one covers the level
 * none of them reach — a *second, independent* implementation of the same two
 * algorithms, which is the whole reason the CPU backends exist.
 *
 * Be clear about what it does and does not establish. It is **not** an oracle
 * for druid's mathematics; druid is a third-party library and its correctness is
 * its own business. What this guards is *our use of it*: that the class we asked
 * for is the class that ran, that the parameters we pass are parameters druid
 * actually reads, and that the output satisfies the contract `druid-cpu.ts` will
 * hand to a vertex buffer.
 *
 * That middle one carries the most weight. Druid takes its parameters as a plain
 * object and silently ignores any key it does not recognise, so a typo'd
 * `n_neigbors` would not throw — it would quietly run the default and produce a
 * perfectly plausible layout. That is the same failure mode as a WGSL shader
 * that compiles and computes the wrong thing, and `paramsAreRead` below is what
 * turns it into a test failure.
 *
 * Unlike the other checks this needs no GPU and no Dawn: the druid path is plain
 * JS, so it runs anywhere Node does. Run with `npm run check:druid`.
 */

import { LocalMAP, Matrix, PaCMAP } from "@saehrimnir/druidjs";
import { buildDruidParams } from "../src/druid-protocol";

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
// Input
// ---------------------------------------------------------------------------

const N = 400;
const D = 20;
const BLOBS = 4;
const SEED = 7;

/** Deterministic, and the same generator the rest of the repo uses. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Four well-separated Gaussian blobs, and which blob each row came from.
 *
 * Separated enough that any working DR method keeps them apart, which is what
 * makes `structurePreserved` a statement about the embedding rather than about
 * how hard the input was.
 */
function blobs(): { rows: Float64Array[]; blobOf: Int32Array } {
  const rand = rng(11);
  const gauss = () =>
    Math.sqrt(-2 * Math.log(1 - rand())) * Math.cos(2 * Math.PI * rand());

  const centers = Array.from({ length: BLOBS }, () =>
    Float64Array.from({ length: D }, () => rand() * 20 - 10)
  );

  const rows: Float64Array[] = [];
  const blobOf = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const b = i % BLOBS;
    blobOf[i] = b;
    const c = centers[b];
    rows.push(Float64Array.from({ length: D }, (_, k) => c[k] + gauss()));
  }
  return { rows, blobOf };
}

const { rows, blobOf } = blobs();

// ---------------------------------------------------------------------------
// Running druid
//
// These are exactly the parameters `druid-worker.ts` will pass, so a mapping
// that breaks there breaks here. `num_iters` matches our own phase split so both
// engines run 450 iterations and the demo's history/stride math needs no case
// for the CPU path.
// ---------------------------------------------------------------------------

interface RunOpts {
  variant: "pacmap" | "localmap";
  n_neighbors?: number;
  seed?: number;
}

/** One embedding, flattened to the N x 2 f32 the GPU buffer would receive. */
function embed(o: RunOpts): { Y: Float32Array; ms: number } {
  // The real mapping, the one the worker ships — so a typo there fails the
  // behavioural checks below, not just the key-set check.
  const params = buildDruidParams({
    variant: o.variant,
    nNeighbors: o.n_neighbors ?? 10,
    mnRatio: 0.5,
    fpRatio: 2.0,
    lowDistThres: 10,
    seed: o.seed ?? SEED,
    phases: [100, 100, 250],
  });

  const t0 = performance.now();
  const dr =
    o.variant === "localmap"
      ? new LocalMAP(rows, params)
      : new PaCMAP(rows, params);
  const out = dr.transform() as Float64Array[];
  const ms = performance.now() - t0;

  const Y = new Float32Array(N * 2);
  for (let i = 0; i < out.length; i++) {
    Y[i * 2] = out[i][0];
    Y[i * 2 + 1] = out[i][1];
  }
  return { Y, ms };
}

const identical = (a: Float32Array, b: Float32Array) =>
  a.length === b.length && a.every((v, k) => v === b[k]);

const spread = (a: Float32Array) => Math.max(...a) - Math.min(...a);

/** Relative separation between two layouts, scaled so it is comparable across runs. */
function relDelta(a: Float32Array, b: Float32Array): number {
  const s = spread(a);
  return a.reduce((m, v, k) => Math.max(m, Math.abs(v - b[k])), 0) / (s || 1);
}

// ---------------------------------------------------------------------------
// Checks
// ---------------------------------------------------------------------------

function contractHolds(name: string, Y: Float32Array, ms: number): void {
  check(
    `${name}/shape-and-finite`,
    Y.length === N * 2 && Y.every(Number.isFinite),
    `length ${Y.length}, ${Y.filter((v) => !Number.isFinite(v)).length} non-finite`
  );
  // A layout that collapsed to a point is finite and the right shape, and would
  // render as a single dot. Catch it here rather than in the browser.
  check(`${name}/non-degenerate`, spread(Y) > 1e-3, `spread ${spread(Y)}`);
  console.log(`     ${name}: ${ms | 0}ms for 450 iterations at N=${N}`);
}

/**
 * Mean intra-blob distance must beat mean inter-blob distance in 2-d.
 *
 * The weakest possible statement that the embedding means anything: a run that
 * silently degenerated into noise passes every other check here.
 */
function structurePreserved(name: string, Y: Float32Array): void {
  let intra = 0;
  let intraN = 0;
  let inter = 0;
  let interN = 0;
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = Y[i * 2] - Y[j * 2];
      const dy = Y[i * 2 + 1] - Y[j * 2 + 1];
      const d = Math.hypot(dx, dy);
      if (blobOf[i] === blobOf[j]) {
        intra += d;
        intraN++;
      } else {
        inter += d;
        interN++;
      }
    }
  }
  const ratio = intra / intraN / (inter / interN);
  check(
    `${name}/blobs-stay-separated`,
    ratio < 0.5,
    `intra/inter distance ratio ${ratio.toFixed(3)}, expected well below 1`
  );
}

/**
 * Every key `buildDruidParams` may emit, transcribed by hand from druid's
 * shipped `ParametersPaCMAP` / `ParametersLocalMAP` declarations.
 *
 * Written out here rather than derived from `DruidParams` on purpose. If this
 * list were computed from the same source the builder uses, a misspelling would
 * appear identically on both sides and cancel out — the exact way an oracle
 * that shares code with its subject fails to catch anything. Transcribing it
 * independently is what gives the comparison teeth.
 */
const EXPECTED_KEYS = [
  "MN_ratio",
  "FP_ratio",
  "apply_pca",
  "d",
  "knn",
  "low_dist_thres",
  "n_neighbors",
  "num_iters",
  "seed",
];

/**
 * The keys the worker sends are keys druid declares.
 *
 * This is the check that covers the *worker's* mapping. The behavioural checks
 * further down can only see parameters that change the layout; a mis-typed
 * `apply_pca` or `num_iters` would sail past them, because druid would fall
 * back to a default that still produces a sane embedding.
 */
function paramKeysAreReal(): void {
  const pacKeys = Object.keys(
    buildDruidParams({
      variant: "pacmap",
      nNeighbors: 10,
      mnRatio: 0.5,
      fpRatio: 2,
      lowDistThres: 10,
      seed: SEED,
      phases: [100, 100, 250],
    })
  ).sort();
  const lmKeys = Object.keys(
    buildDruidParams({
      variant: "localmap",
      nNeighbors: 10,
      mnRatio: 0.5,
      fpRatio: 2,
      lowDistThres: 10,
      seed: SEED,
      phases: [100, 100, 250],
    })
  ).sort();

  const unknown = lmKeys.filter((k) => !EXPECTED_KEYS.includes(k));
  check(
    "params/keys-are-declared-by-druid",
    unknown.length === 0,
    `druid declares no such parameter: ${unknown.join(", ")}`
  );

  // Everything except low_dist_thres, which druid's PaCMAP does not declare.
  const wanted = EXPECTED_KEYS.filter((k) => k !== "low_dist_thres");
  const missing = wanted.filter((k) => !pacKeys.includes(k));
  check(
    "params/pacmap-sends-every-key",
    missing.length === 0 && !pacKeys.includes("low_dist_thres"),
    missing.length
      ? `never sent: ${missing.join(", ")}`
      : "low_dist_thres sent under pacmap, which druid does not declare there"
  );

  check(
    "params/localmap-adds-low-dist-thres",
    lmKeys.includes("low_dist_thres"),
    "LocalMAP's only extra parameter is never sent"
  );

  // The two engines must agree on iteration count or the demo's history stride
  // arithmetic diverges between them.
  const p = buildDruidParams({
    variant: "pacmap",
    nNeighbors: 10,
    mnRatio: 0.5,
    fpRatio: 2,
    lowDistThres: 10,
    seed: SEED,
    phases: [100, 100, 250],
  });
  check(
    "params/iterations-match-the-gpu-path",
    p.num_iters.reduce((a, b) => a + b, 0) === 450 && p.apply_pca === false,
    `num_iters ${JSON.stringify(p.num_iters)}, apply_pca ${p.apply_pca}`
  );
}

/**
 * The two things `druid-worker.ts` does differently from `embed` above, checked
 * where a browser is not needed to see them.
 *
 * The worker cannot call `transform()`: the demo animates, so it drives
 * `generator()` and pulls it forward a chunk at a time, reading the embedding
 * only on capture boundaries. And it hands druid a `Matrix` rather than
 * `Float64Array[]`, because druid's `projection` getter copies via
 * `to2dArray()` for the array forms and would do so on every one of the 450
 * iterations.
 *
 * Both are performance decisions, and both would be silent if wrong — a chunked
 * run that diverged from a straight one, or a Matrix input that took a different
 * path through druid, would still produce a plausible layout. So: same seed,
 * same params, all three routes, bit-for-bit.
 */
function workerDrivingIsEquivalent(): void {
  const params = buildDruidParams({
    variant: "pacmap",
    nNeighbors: 10,
    mnRatio: 0.5,
    fpRatio: 2.0,
    lowDistThres: 10,
    seed: SEED,
    phases: [100, 100, 250],
  });

  const asMatrix = () => new Matrix(N, D, (i, j) => rows[i][j]);
  const flatten = (v: Float64Array) => {
    const Y = new Float32Array(N * 2);
    for (let k = 0; k < Y.length; k++) Y[k] = v[k];
    return Y;
  };

  // Straight through, Matrix in.
  const whole = new PaCMAP(asMatrix(), params);
  whole.transform();
  const straight = flatten(whole.Y.values);

  // Chunked exactly as the worker drives it — deliberately an uneven stride, so
  // a boundary assumption baked into druid would show up.
  const stepped = new PaCMAP(asMatrix(), params);
  stepped.check_init();
  const steps = stepped.generator(450);
  let it = 0;
  for (const to of [37, 100, 101, 250, 449, 450]) {
    while (it < to) {
      if (steps.next().done) break;
      it++;
    }
  }
  check(
    "worker/chunked-generator-matches-transform",
    it === 450 && identical(straight, flatten(stepped.Y.values)),
    `stopped at iteration ${it}`
  );

  // Matrix input against the row input every other check here uses, so those
  // checks keep speaking about the same computation the worker runs.
  const viaRows = new PaCMAP(rows, params);
  viaRows.transform();
  check(
    "worker/matrix-input-matches-rows",
    identical(straight, flatten(viaRows.Y.values))
  );
}

function main(): number {
  paramKeysAreReal();
  workerDrivingIsEquivalent();

  const pac = embed({ variant: "pacmap" });
  const lm = embed({ variant: "localmap" });

  contractHolds("pacmap", pac.Y, pac.ms);
  contractHolds("localmap", lm.Y, lm.ms);
  structurePreserved("pacmap", pac.Y);
  structurePreserved("localmap", lm.Y);

  // Determinism at a fixed seed. The demo banks a trace and replays it, so a run
  // that is not reproducible cannot be compared against anything.
  check("pacmap/reproducible", identical(pac.Y, embed({ variant: "pacmap" }).Y));
  check(
    "localmap/reproducible",
    identical(lm.Y, embed({ variant: "localmap" }).Y)
  );

  // The variant we asked for is the one that ran. Without this, a dropdown that
  // never reached druid would look like it worked.
  const variantDelta = relDelta(pac.Y, lm.Y);
  check(
    "variant-selected",
    variantDelta > 1e-3,
    `pacmap and localmap agree to ${variantDelta.toExponential(2)} — same class ran twice?`
  );

  // The load-bearing one. Druid ignores unrecognised parameter keys silently, so
  // a typo'd name runs the default and returns a plausible layout. Two runs
  // differing only in n_neighbors must differ; if they don't, the key isn't
  // being read.
  const paramDelta = relDelta(
    pac.Y,
    embed({ variant: "pacmap", n_neighbors: 40 }).Y
  );
  check(
    "params-are-read",
    paramDelta > 1e-3,
    `n_neighbors 10 vs 40 agree to ${paramDelta.toExponential(2)} — is the key spelled right?`
  );

  // Same, for the one parameter only LocalMAP reads.
  const seedDelta = relDelta(pac.Y, embed({ variant: "pacmap", seed: 99 }).Y);
  check(
    "seed-is-read",
    seedDelta > 1e-3,
    `seeds 7 and 99 agree to ${seedDelta.toExponential(2)}`
  );

  console.log(
    failures === 0
      ? `\n${checks} druid checks passed.`
      : `\n${failures} of ${checks} druid checks failed.`
  );
  return failures === 0 ? 0 : 1;
}

process.exit(main());
