/**
 * Runs DruidJS's PaCMAP / LocalMAP off the main thread.
 *
 * These are the CPU comparison backends: a second, independent implementation
 * of the two algorithms `pacmap-webgpu.ts` implements in WGSL, so the demo can
 * show them side by side over the same input at the same seed. Druid's
 * optimizer is plain JS in f64 and its neighbour search is exact O(N^2*D), so
 * this is slow by construction — which is exactly why it must not run on the
 * thread that paints.
 *
 * No DOM and no WebGPU here. The worker owns the algorithm and nothing else;
 * `druid-cpu.ts` owns the buffer the frames land in.
 *
 * Two details are load-bearing and easy to undo by accident:
 *
 * **The input is a `Matrix`, not `Float64Array[]`.** Druid's `projection`
 * getter returns `this.Y` directly when the input was a Matrix, but calls
 * `to2dArray()` for either array form — and `generator()` reads `projection`
 * once per iteration. Handing it rows would therefore allocate N fresh
 * Float64Arrays on every one of the 450 steps, every one of them discarded,
 * because this worker reads the embedding only on capture boundaries. At 65k
 * points that is 29 million throwaway arrays.
 *
 * **The optimizer is driven through `generator()`, not by calling `next()` in a
 * loop.** A generator that runs out or is closed releases the WASM buffers druid
 * holds between iterations (`DR.release`); a hand-driven `next()` loop leaves
 * them allocated until the worker dies.
 */

import { LocalMAP, Matrix, PaCMAP } from "@saehrimnir/druidjs";
import {
  buildDruidParams,
  type DruidCommand,
  type DruidEvent,
  type DruidInitCommand,
} from "./druid-protocol";

// `self` is typed as a Window under the DOM lib, whose postMessage has a
// different signature. Narrowing it here beats reshuffling tsconfig's libs for
// one file.
const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<DruidCommand>) => void) | null;
  postMessage(msg: DruidEvent, transfer?: Transferable[]): void;
};

/** The druid instance, its iteration stream, and where that stream has got to. */
let dr: PaCMAP<Matrix> | LocalMAP<Matrix> | null = null;
let steps: Generator<Matrix, Matrix, void> | null = null;
let it = 0;
let N = 0;
let d = 2;

function init(cmd: DruidInitCommand): void {
  N = cmd.N;
  d = cmd.nComponents;
  const { X, D } = cmd;

  // f32 in, f64 out: druid computes in f64 throughout, so this widening happens
  // once here rather than implicitly on every access.
  const M = new Matrix(N, D, (i, j) => X[i * D + j]);
  const params = buildDruidParams(cmd);

  const t0 = performance.now();
  dr =
    cmd.variant === "localmap"
      ? new LocalMAP<Matrix>(M, params)
      : new PaCMAP<Matrix>(M, params);

  // Explicit, so the expensive part — neighbour search and pair sampling — is
  // attributed to setup and reported. `generator()` would otherwise do it
  // lazily inside the first step, where it would look like a stall.
  dr.check_init();
  steps = dr.generator(cmd.phases.reduce((a, b) => a + b, 0));
  it = 0;

  ctx.postMessage({ type: "ready", setupMs: performance.now() - t0 });
}

/** Advance to iteration `to`, then send the embedding as it stands. */
function step(to: number): void {
  if (!dr || !steps) throw new Error("step before init");

  while (it < to) {
    if (steps.next().done) break; // the schedule ran out; send what we have
    it++;
  }

  // Narrow to f32 for the vertex buffer. `values` is exactly N*d, packed the
  // same way the renderer's vertex layout expects.
  const values = dr.Y.values;
  const Y = new Float32Array(N * d);
  for (let k = 0; k < Y.length; k++) Y[k] = values[k];

  ctx.postMessage({ type: "frame", it, Y }, [Y.buffer]);
}

ctx.onmessage = (e: MessageEvent<DruidCommand>) => {
  try {
    const cmd = e.data;
    if (cmd.type === "init") init(cmd);
    else step(cmd.to);
  } catch (err) {
    // Including a throw from druid itself. The main thread turns this into a
    // status line; without it a failure here is silent and the run just hangs.
    ctx.postMessage({
      type: "error",
      message: err instanceof Error ? err.message : String(err),
    });
  }
};
