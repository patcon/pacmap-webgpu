/**
 * Opening a real WebGPU device from Node, for the checks under `scripts/`.
 *
 * The library is DOM-free, so an esbuild bundle of it runs under `@kmamal/gpu`
 * (Dawn bindings for Node) against real WGSL and real kernels. That is the only
 * way anything here is verifiable without a browser — see CLAUDE.md.
 *
 * Two things about the bindings are easy to lose an hour to, so they live here
 * rather than in each caller:
 *
 * - The WebGPU constants are module properties, not globals. Anything written
 *   for a browser — the library included — reads `GPUBufferUsage.STORAGE` off
 *   the global object and gets `undefined` without the install below.
 * - Each live instance holds an interval, so a script that finishes its work
 *   still will not exit. `close()` disposes the instance, and callers should
 *   still `process.exit` rather than trust the event loop to drain.
 */

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const gpu = require("@kmamal/gpu");

for (const k of ["GPUBufferUsage", "GPUShaderStage", "GPUMapMode"] as const) {
  (globalThis as Record<string, unknown>)[k] = gpu[k];
}

export interface DawnSession {
  device: GPUDevice;
  close(): void;
}

/** Null when the platform has no adapter — callers decide if that's fatal. */
export async function openDawn(): Promise<DawnSession | null> {
  const instance = gpu.create([]);
  const adapter = await instance.requestAdapter();
  if (!adapter) {
    gpu.destroy(instance);
    return null;
  }
  const device: GPUDevice = await adapter.requestDevice();
  return {
    device,
    close() {
      device.destroy();
      gpu.destroy(instance);
    },
  };
}

/** Upload a typed array as a storage/uniform buffer. */
export function upload(
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

/** Copy a buffer back to the host. Only for checks; never on a render path. */
export async function readback(
  device: GPUDevice,
  src: GPUBuffer,
  bytes: number
): Promise<Uint32Array> {
  const stage = device.createBuffer({
    size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, 0, stage, 0, bytes);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(stage.getMappedRange().slice(0));
  stage.unmap();
  stage.destroy();
  return out;
}

/** Deterministic RNG for check inputs, independent of the library's own. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
