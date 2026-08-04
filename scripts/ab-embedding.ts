/**
 * Does this change move the embedding, and by how much?
 *
 * `check:kernels` asks whether the kernels are correct. This asks a different
 * question that no self-contained check can: whether a change that is *supposed*
 * to preserve behaviour actually did. It bundles the working tree's library and
 * a git ref's library, runs both over identical input, and reports the largest
 * coordinate difference relative to the layout's own extent.
 *
 * Not in CI, because it needs a baseline to compare against and the answer is a
 * number to interpret rather than a pass/fail. It is the tool for refactors:
 * moving the further pairs out of the CSR was landed on the strength of a
 * 2.9e-5 here, against 0.50 for a version with the reverse-FP loop deleted.
 *
 *   npm run check:ab                      # working tree vs HEAD, pacmap
 *   npm run check:ab -- HEAD~3
 *   npm run check:ab -- HEAD --variant=localmap
 *
 * Interpreting the number: exact zero means bit-identical. Around 1e-5 means
 * the arithmetic is unchanged and only the f32 summation ORDER moved — expected
 * from any reordering of gradient accumulation, and amplified by 450 iterations
 * of a chaotic optimizer. Anything at 1e-2 or above is a behaviour change.
 *
 * Always sanity-check the harness with `npm run check:ab -- HEAD` on a clean
 * tree first: that must print exactly 0. A confounded comparison otherwise
 * reads as a pass.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { pacmapWebGPU, type PacmapOptions } from "../src/pacmap-webgpu";
import { openDawn, rng } from "./dawn";

const CACHE = "node_modules/.cache";

const args = process.argv.slice(2);
const flag = (name: string, fallback: string) =>
  args.find((a) => a.startsWith(`--${name}=`))?.split("=")[1] ?? fallback;

const REF = args.find((a) => !a.startsWith("--")) ?? "HEAD";
const VARIANT = flag("variant", "pacmap") as "pacmap" | "localmap";
const ITERS = Number(flag("iters", "450"));
const N = Number(flag("n", "800"));
const D = Number(flag("d", "20"));
const SEED = Number(flag("seed", "7"));

/** Ten gaussian blobs. Independent of the library's RNG so both builds agree. */
function blobs(): Float32Array {
  const r = rng(12345);
  const gauss = () =>
    Math.sqrt(-2 * Math.log(Math.max(r(), 1e-12))) * Math.cos(2 * Math.PI * r());
  const centers = Array.from(
    { length: 10 },
    () => new Float32Array(D).map(() => gauss() * 4)
  );
  const X = new Float32Array(N * D);
  for (let i = 0; i < N; i++) {
    const c = centers[i % 10];
    for (let k = 0; k < D; k++) X[i * D + k] = c[k] + gauss();
  }
  return X;
}

type Pacmap = typeof pacmapWebGPU;

async function embed(device: GPUDevice, impl: Pacmap): Promise<Float32Array> {
  const opts: PacmapOptions = { seed: SEED, knn: "cpu", variant: VARIANT };
  const pm = await impl(device, blobs(), N, D, opts);
  pm.runRange(0, ITERS);
  const Y = await pm.read();
  pm.destroy();
  return Y;
}

/** Bundle `<ref>:src/pacmap-webgpu.ts` on its own and load it. */
async function loadRef(ref: string): Promise<Pacmap> {
  mkdirSync(CACHE, { recursive: true });
  const src = `${CACHE}/ab-ref-src.ts`;
  const out = `${CACHE}/ab-ref.mjs`;
  writeFileSync(
    src,
    execFileSync("git", ["show", `${ref}:src/pacmap-webgpu.ts`], {
      encoding: "utf8",
      maxBuffer: 1 << 26,
    })
  );
  execFileSync(
    "esbuild",
    [
      src,
      "--bundle",
      "--format=esm",
      "--platform=node",
      "--external:@kmamal/gpu",
      `--outfile=${out}`,
    ],
    { stdio: "pipe" }
  );
  // Cache-busted: without the query a second run in the same process would get
  // the first ref's module back.
  const mod = await import(
    /* @vite-ignore */ `${process.cwd()}/${out}?t=${Date.now()}`
  );
  return mod.pacmapWebGPU as Pacmap;
}

async function main(): Promise<number> {
  const session = await openDawn();
  if (!session) {
    console.error("no WebGPU adapter available");
    return 1;
  }

  try {
    const refImpl = await loadRef(REF);
    console.log(
      `${VARIANT}  N=${N} D=${D} seed=${SEED} iters=${ITERS}  ` +
        `working tree vs ${REF}`
    );

    const a = await embed(session.device, refImpl);
    const b = await embed(session.device, pacmapWebGPU);
    if (a.length !== b.length) {
      console.error(`length ${a.length} vs ${b.length}`);
      return 1;
    }

    let maxAbs = 0;
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = 0; k < a.length; k++) {
      maxAbs = Math.max(maxAbs, Math.abs(a[k] - b[k]));
      lo = Math.min(lo, a[k]);
      hi = Math.max(hi, a[k]);
    }
    const rel = maxAbs / (hi - lo);

    console.log(
      `  extent ${(hi - lo).toFixed(4)}  ` +
        `max|delta| ${maxAbs.toExponential(3)}  ` +
        `relative ${rel.toExponential(3)}`
    );
    console.log(
      `  ${
        maxAbs === 0
          ? "bit-identical"
          : rel < 1e-3
            ? "same arithmetic, different f32 summation order"
            : "BEHAVIOUR CHANGED"
      }`
    );
  } finally {
    session.close();
  }
  return 0;
}

// See scripts/check-shaders.ts: the bindings hold an interval per instance.
process.exit(await main());
