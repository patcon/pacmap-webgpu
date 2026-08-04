/**
 * The DruidJS CPU backends, dressed as an `EmbeddingRun`.
 *
 * Everything downstream of the optimizer in `main.ts` — the bounds reduce, the
 * banked position history, the scrubber, the renderer — is written against
 * `{ positions, runRange, totalIters, destroy }`. Satisfying that shape is what
 * lets a worker-driven CPU run reuse all of it rather than growing a second
 * render path, and it is why this file exists at all: the worker could have
 * posted straight to `main.ts`, but then every consumer would need to know which
 * engine it was talking to.
 *
 * **The no-readback invariant survives, and the direction is the whole reason.**
 * `positions` is a real GPU buffer that the worker's snapshot is uploaded *into*
 * with `writeBuffer`. Nothing is ever mapped back to the host on the render
 * path, so banking, scrubbing and replay behave exactly as they do for the GPU
 * engine. An upload per captured frame is not a violation of that constraint;
 * a `mapAsync` would be.
 */

import {
  type DruidCommand,
  type DruidEvent,
  type DruidVariant,
} from "./druid-protocol";
import type { EmbeddingRun } from "./pacmap-webgpu";

export interface DruidCpuOptions {
  variant?: DruidVariant;
  nNeighbors?: number;
  mnRatio?: number;
  fpRatio?: number;
  /** LocalMAP's only extra knob. Ignored under "pacmap", as upstream. */
  lowDistThres?: number;
  seed?: number;
  /** Iterations per phase. Default [100, 100, 250], matching the GPU path. */
  phases?: [number, number, number];
  /**
   * Kills the worker, including part way through setup.
   *
   * That last part is the reason this exists rather than the caller simply
   * declining to step: druid's neighbour search is one synchronous call that
   * owns the worker for the entire time — minutes, at the top of the range —
   * so there is no point in it where a cooperative check could run.
   * `terminate()` is the only thing that stops it, and without this the demo's
   * stop button would be inert during the longest phase of the run.
   */
  signal?: AbortSignal;
  onStatus?: (msg: string) => void;
}

/**
 * Starts a druid run in a worker and returns once its setup has finished.
 *
 * Signature mirrors `pacmapWebGPU` so `main.ts` can pick between them on one
 * flag and change nothing else.
 */
export async function druidCPU(
  device: GPUDevice,
  X: Float32Array,
  N: number,
  D: number,
  opts: DruidCpuOptions = {}
): Promise<EmbeddingRun> {
  const phases = opts.phases ?? [100, 100, 250];
  const totalIters = phases[0] + phases[1] + phases[2];
  const onStatus = opts.onStatus ?? (() => {});

  const positions = device.createBuffer({
    size: N * 8,
    // STORAGE for the demo's bounds reduce, VERTEX so the renderer can bind it
    // directly, COPY_DST for the upload, COPY_SRC for banking into the history.
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.VERTEX |
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC,
  });

  const worker = new Worker(new URL("./druid-worker.ts", import.meta.url), {
    type: "module",
  });

  // One request outstanding at a time — `runRange` is awaited before the next
  // is issued, so a single slot is the whole protocol.
  let pending: {
    resolve: (e: DruidEvent) => void;
    reject: (err: Error) => void;
  } | null = null;
  let dead: Error | null = null;

  function fail(err: Error) {
    dead = err;
    pending?.reject(err);
    pending = null;
  }

  worker.onmessage = (e: MessageEvent<DruidEvent>) => {
    if (e.data.type === "error") {
      fail(new Error(`druid worker: ${e.data.message}`));
      return;
    }
    const p = pending;
    pending = null;
    p?.resolve(e.data);
  };
  // A module worker that fails to load reports here and never sends anything,
  // so without this the first await would hang forever rather than report.
  worker.onerror = (e) =>
    fail(new Error(`druid worker failed to load: ${e.message}`));
  worker.onmessageerror = () => fail(new Error("druid worker: uncloneable message"));

  // Wired before the first message so an abort during setup is caught. `fail`
  // also rejects whatever request is outstanding, which is what unblocks the
  // caller's await rather than leaving it pending forever.
  const onAbort = () => {
    worker.terminate();
    fail(new Error("aborted"));
  };
  if (opts.signal) {
    if (opts.signal.aborted) onAbort();
    opts.signal.addEventListener("abort", onAbort, { once: true });
  }

  function request(cmd: DruidCommand, transfer: Transferable[] = []) {
    return new Promise<DruidEvent>((resolve, reject) => {
      if (dead) return reject(dead);
      pending = { resolve, reject };
      worker.postMessage(cmd, transfer);
    });
  }

  // Copied, then the copy is transferred. `pcaProject` returns a plain
  // `Float32Array`, whose type admits a SharedArrayBuffer that could not be
  // transferred at all; taking a copy settles that statically and spares the
  // caller a "must not touch X again" hazard for the sake of one 26MB
  // allocation at 65k, on the path that is about to spend minutes inside druid.
  const payload = new Float32Array(X);
  const ready = await request(
    {
      type: "init",
      X: payload,
      N,
      D,
      variant: opts.variant ?? "pacmap",
      nNeighbors: opts.nNeighbors ?? 10,
      mnRatio: opts.mnRatio ?? 0.5,
      fpRatio: opts.fpRatio ?? 2.0,
      lowDistThres: opts.lowDistThres ?? 10,
      seed: opts.seed ?? 42,
      phases,
    },
    [payload.buffer]
  );
  if (ready.type !== "ready") throw new Error(`unexpected ${ready.type}`);
  onStatus(`druid setup (kNN + pairs) ${ready.setupMs | 0}ms`);

  /** The last frame uploaded, so `read()` needs no readback either. */
  let last = new Float32Array(N * 2);
  /** Where the worker's stream has got to, to catch an out-of-order caller. */
  let at = 0;

  async function runRange(from: number, to: number): Promise<void> {
    // The worker is a stream, not a random-access encoder: it can only carry on
    // from where it stopped. Silently ignoring `from` would let a caller that
    // rewound get frames from the wrong iteration, which would look like a
    // rendering bug rather than a protocol one.
    if (from !== at) {
      throw new Error(
        `druid runs forward only: asked for [${from}, ${to}) at iteration ${at}`
      );
    }
    const evt = await request({ type: "step", to });
    if (evt.type !== "frame") throw new Error(`unexpected ${evt.type}`);
    at = evt.it;
    last = evt.Y;
    device.queue.writeBuffer(positions, 0, evt.Y);
  }

  return {
    positions,
    totalIters,
    run: () => runRange(0, totalIters),
    runRange,
    read: async () => last,
    destroy() {
      opts.signal?.removeEventListener("abort", onAbort);
      worker.terminate();
      positions.destroy();
    },
  };
}
