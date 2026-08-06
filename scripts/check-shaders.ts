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

const { GPUShaderStage, GPUBufferUsage } = gpu;

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

// Must match main.ts: the pipeline's depth state and the attachment's texture
// format are validated against each other at draw time.
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";

// One MNIST tile. The atlas in the draw case below is a single tile — the count
// is irrelevant there, only that the binding is accepted under the pipeline.
const TILE = 28;

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

// The embedding dimensionalities the library generates code for. Both are
// built here because the sources are templated on it: a `vec3` form that never
// compiled would look exactly like a working one until the dropdown was moved.
const DIMS = [2, 3] as const;

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
  ...DIMS.map((d) => ({
    name: `fp-rebuild-${d}d`,
    code: shaderSources.fpShaderSource(N, NFP, K, 10, 7, d),
    build: (device: GPUDevice, module: GPUShaderModule) => {
      // One layout across all six entry points, for the same reason NN-Descent
      // declares its own: `layout: "auto"` derives a narrower one per entry
      // point (fp_clear never touches Y or the near-partner list), and a single
      // shared bind group would then fail validation against it.
      const layout = computeLayout(device, [
        "storage",
        "storage",
        "storage",
        "read-only-storage",
        "read-only-storage",
        "uniform-dynamic",
      ]);
      for (const entryPoint of [
        "fp_resample",
        "fp_clear",
        "fp_count",
        "fp_scan",
        "fp_scatter",
        "fp_sort",
      ]) {
        device.createComputePipeline({ layout, compute: { module, entryPoint } });
      }
    },
  })),
  ...DIMS.map((d) => ({
    name: `pacmap-${d}d`,
    code: shaderSources.shaderSource(N, NFP, d),
    build: (device: GPUDevice, module: GPUShaderModule) => {
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
  })),
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
  // The 3D renderer builds a second pipeline for the occlusion toggle, and
  // depth state is baked into a pipeline rather than set at draw time — so
  // both have to be built here. A depth format the device rejects, or a
  // fragment that stopped writing the target, fails only at this call.
  ...DIMS.flatMap((d) =>
    (d === 3 ? [false, true] : [false]).map((depth) => ({
    name: `render-${d}d${depth ? "-depth" : ""}`,
    code: renderWGSL(d),
    build: (device: GPUDevice, module: GPUShaderModule) => {
      device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module,
          entryPoint: "vs",
          buffers: [
            {
              // Must agree with the attribute's own type in the WGSL, and with
              // the per-frame copies main.ts sizes from the same `d`.
              arrayStride: 4 * d,
              stepMode: "instance",
              attributes: [
                {
                  shaderLocation: 0,
                  offset: 0,
                  format: `float32x${d}` as GPUVertexFormat,
                },
              ],
            },
            {
              arrayStride: 4,
              stepMode: "instance",
              attributes: [{ shaderLocation: 1, offset: 0, format: "uint32" as const }],
            },
            {
              // The digit-thumbnail tile + style. A vertex input no buffer
              // supplies fails right here.
              arrayStride: 4,
              stepMode: "instance",
              attributes: [{ shaderLocation: 2, offset: 0, format: "uint32" as const }],
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
              format: "bgra8unorm" as const,
              blend: {
                color: {
                  srcFactor: "src-alpha" as const,
                  dstFactor: "one-minus-src-alpha" as const,
                  operation: "add" as const,
                },
                alpha: {
                  srcFactor: "one" as const,
                  dstFactor: "one-minus-src-alpha" as const,
                  operation: "add" as const,
                },
              },
            },
          ],
        },
        primitive: { topology: "triangle-list" as const },
        ...(depth
          ? {
              depthStencil: {
                format: DEPTH_FORMAT,
                depthWriteEnabled: true,
                depthCompare: "less" as const,
              },
            }
          : {}),
      });
    },
    }))
  ),
  {
    // The one case that encodes a draw.
    //
    // Creating both 3D pipelines proves each is individually valid. It says
    // nothing about whether one bind group can serve both, or whether what
    // they are drawn into carries the attachments their state demands — and
    // those are the mistakes that produce a blank canvas rather than an error
    // anyone sees. A pipeline built with `layout: "auto"` mints its own bind
    // group layout, incompatible with any other pipeline's, so the draw is
    // dropped at validation. Occlusion shipped that way once: every point
    // vanished the moment it was ticked, while the camera kept working.
    //
    // A render *bundle* encoder rather than a pass, because it takes
    // attachment formats instead of views. That validates exactly what is
    // wanted here — setBindGroup against the pipeline's layout, and the
    // pipeline's depth state against the declared depth format — and it is
    // also the only way to encode a draw under these bindings at all:
    // @kmamal/gpu's `createView()` unconditionally sends a component swizzle
    // that this adapter does not support, so no texture view can be made and
    // no render pass begun.
    name: "render-3d-occlusion-draw",
    code: renderWGSL(3),
    build: (device, module) => {
      const M = 8; // instances; the count is irrelevant, the encoding is not
      const bgl = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.VERTEX,
            buffer: { type: "uniform" as const },
          },
          {
            // The fragment reads this one too, for the occlude flag.
            binding: 1,
            visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
            buffer: { type: "uniform" as const },
          },
          // The digit-thumbnail atlas. A storage buffer read from the fragment
          // stage, which is a binding kind nothing else here uses.
          {
            binding: 2,
            visibility: GPUShaderStage.FRAGMENT,
            buffer: { type: "read-only-storage" as const },
          },
        ],
      });
      const desc: GPURenderPipelineDescriptor = {
        layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
        vertex: {
          module,
          entryPoint: "vs",
          buffers: [
            {
              arrayStride: 12,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x3" as const },
              ],
            },
            {
              arrayStride: 4,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 1, offset: 0, format: "uint32" as const },
              ],
            },
            {
              arrayStride: 4,
              stepMode: "instance",
              attributes: [
                { shaderLocation: 2, offset: 0, format: "uint32" as const },
              ],
            },
          ],
        },
        fragment: {
          module,
          entryPoint: "fs",
          targets: [{ format: "rgba8unorm" as const }],
        },
        primitive: { topology: "triangle-list" as const },
      };
      const blended = device.createRenderPipeline(desc);
      const occluded = device.createRenderPipeline({
        ...desc,
        depthStencil: {
          format: DEPTH_FORMAT,
          depthWriteEnabled: true,
          depthCompare: "less" as const,
        },
      });

      const uniform = (size: number) =>
        device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM });
      const atlas = device.createBuffer({
        size: TILE * TILE * 4,
        usage: GPUBufferUsage.STORAGE,
      });
      const bindGroup = device.createBindGroup({
        layout: bgl,
        entries: [
          { binding: 0, resource: { buffer: uniform(32) } }, // Bounds
          { binding: 1, resource: { buffer: uniform(96) } }, // View
          { binding: 2, resource: { buffer: atlas } }, // one f32 per texel
        ],
      });
      const vbuf = (size: number) =>
        device.createBuffer({ size, usage: GPUBufferUsage.VERTEX });
      const positions = vbuf(M * 12);
      const labels = vbuf(M * 4);
      const thumbs = vbuf(M * 4);

      for (const occlude of [false, true]) {
        const bundle = device.createRenderBundleEncoder({
          colorFormats: ["rgba8unorm"],
          ...(occlude ? { depthStencilFormat: DEPTH_FORMAT } : {}),
        });
        bundle.setPipeline(occlude ? occluded : blended);
        bundle.setBindGroup(0, bindGroup);
        bundle.setVertexBuffer(0, positions);
        bundle.setVertexBuffer(1, labels);
        bundle.setVertexBuffer(2, thumbs);
        bundle.draw(6, M);
        bundle.finish();
      }
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
