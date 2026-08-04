/**
 * Headless WGSL check: compiles every shader in the project under Dawn and
 * builds the real pipeline for each entry point.
 *
 * Nothing else in this repo looks at a shader. `tsc --noEmit` sees template
 * literals, and Vite ships them unread — so a WGSL error survives the build and
 * only surfaces at runtime, where it takes out every pipeline in the module at
 * once and looks like a plausible-but-wrong result rather than a failure. Both
 * kNN shaders shipped broken that way once (eccc675).
 *
 * Module compilation alone is not enough. `createShaderModule` may succeed and
 * defer its diagnostics; entry-point names, bind-group layout compatibility and
 * the render pipeline's vertex/blend state are validated only when the pipeline
 * is created. So each case does both, and uses the same layout the app uses —
 * notably NN-Descent's explicit 7-entry layout, whose whole point is that
 * `layout: "auto"` would derive a narrower one per entry point.
 *
 * Run with `npm run check:shaders`. Pass --strict to fail rather than skip when
 * no adapter is available (CI does).
 */

import { shaderSources } from "../src/pacmap-webgpu";
import { boundsWGSL, renderWGSL } from "../src/shaders";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const gpu = require("@kmamal/gpu");

const { GPUShaderStage } = gpu;

// Sizes are baked into the sources as WGSL constants, and several become the
// length of a private array (`array<f32, D>`, `array<f32, K>`). Toy values
// would compile where the real ones might not, so these mirror what the demo
// actually runs: PCA output width, the auto-neighbor count at 65k, and the
// NN-Descent reverse-list cap.
const N = 2000;
const D = 100;
const K = 60;
const R = 32;
// round(n_neighbors * FP_ratio) at the demo's defaults (10 and 2.0). It sets
// the length of a per-thread loop bound, not an array, but keeping it realistic
// costs nothing.
const NFP = 20;

const STRICT = process.argv.includes("--strict");

type Case = {
  name: string;
  code: string;
  build: (device: GPUDevice, module: GPUShaderModule) => void;
};

const computeLayout = (
  device: GPUDevice,
  types: (GPUBufferBindingType | "uniform-dynamic")[]
): GPUPipelineLayout => {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: types.map((type, binding) => ({
      binding,
      visibility: GPUShaderStage.COMPUTE,
      buffer:
        type === "uniform-dynamic"
          ? { type: "uniform" as const, hasDynamicOffset: true, minBindingSize: 48 }
          : { type },
    })),
  });
  return device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
};

const cases: Case[] = [
  {
    name: "knn",
    code: shaderSources.knnShaderSource(N, D, K),
    build: (device, module) => {
      device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "knn_main" },
      });
    },
  },
  {
    name: "nndescent",
    code: shaderSources.nndShaderSource(N, D, K, R),
    build: (device, module) => {
      // The explicit layout is load-bearing: all four entry points share one
      // bind group, which `layout: "auto"` could not express.
      const layout = computeLayout(device, [
        "read-only-storage",
        "storage",
        "storage",
        "storage",
        "storage",
        "storage",
        "uniform",
      ]);
      for (const entryPoint of [
        "nnd_init",
        "nnd_clear",
        "nnd_reverse",
        "nnd_join",
      ]) {
        device.createComputePipeline({ layout, compute: { module, entryPoint } });
      }
    },
  },
  {
    name: "pacmap",
    code: shaderSources.shaderSource(N, NFP),
    build: (device, module) => {
      // Eight storage buffers is the default per-stage limit, so this layout is
      // also the check that the shader still fits inside it.
      const layout = computeLayout(device, [
        "storage",
        "read-only-storage",
        "read-only-storage",
        "storage",
        "storage",
        "storage",
        "uniform-dynamic",
        "read-only-storage",
        "read-only-storage",
      ]);
      for (const entryPoint of ["grad_main", "adam_main"]) {
        device.createComputePipeline({ layout, compute: { module, entryPoint } });
      }
    },
  },
  {
    name: "bounds",
    code: boundsWGSL(N),
    build: (device, module) => {
      device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });
    },
  },
  {
    name: "render",
    code: renderWGSL,
    build: (device, module) => {
      device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module,
          entryPoint: "vs",
          buffers: [
            {
              arrayStride: 8,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x2" },
              ],
            },
            {
              arrayStride: 4,
              stepMode: "instance",
              attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" }],
            },
          ],
        },
        fragment: {
          module,
          entryPoint: "fs",
          // There is no canvas here to ask for a preferred format. The choice
          // does not affect WGSL validation, only the target's own validity.
          targets: [
            {
              format: "bgra8unorm",
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
    },
  },
];

async function main(): Promise<number> {
  const instance = gpu.create([]);
  const adapter = await instance.requestAdapter();
  if (!adapter) {
    console.error("no WebGPU adapter available");
    if (STRICT) return 1;
    console.error("SKIP (pass --strict to make this a failure)");
    return 0;
  }
  const device: GPUDevice = await adapter.requestDevice();

  let failed = 0;
  for (const c of cases) {
    const problems: string[] = [];
    let warnings = 0;

    // Uncaptured errors are asynchronous and would otherwise land after the
    // summary, attributed to nothing.
    device.onuncapturederror = (e: GPUUncapturedErrorEvent) => {
      problems.push(`uncaptured: ${e.error.message}`);
    };
    device.pushErrorScope("validation");

    const module = device.createShaderModule({ code: c.code });
    const info = await module.getCompilationInfo();
    for (const m of info.messages) {
      const where = `${c.name}.wgsl:${m.lineNum}:${m.linePos}`;
      if (m.type === "error") problems.push(`${where}: ${m.message}`);
      else warnings++;
    }

    // Only worth building pipelines if the module itself is sound; otherwise
    // every entry point fails for the same reason and buries the real message.
    if (problems.length === 0) c.build(device, module);

    const scoped = await device.popErrorScope();
    if (scoped) problems.push(scoped.message);

    if (problems.length > 0) {
      failed++;
      console.error(`FAIL ${c.name}`);
      for (const p of problems) console.error(`  ${p}`);
    } else {
      const note = warnings > 0 ? ` (${warnings} warning(s))` : "";
      console.log(`ok   ${c.name}${note}`);
    }
  }

  console.log(
    failed === 0
      ? `\n${cases.length} shaders compiled and pipelined.`
      : `\n${failed} of ${cases.length} shaders failed.`
  );

  // The bindings hold an interval per live instance, which would keep the
  // process alive after main() resolves.
  device.destroy();
  gpu.destroy(instance);
  return failed === 0 ? 0 : 1;
}

process.exitCode = await main();
