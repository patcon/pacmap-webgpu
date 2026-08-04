/**
 * The contract between `druid-cpu.ts` (main thread) and `druid-worker.ts`, plus
 * the one piece of logic they disagree about at their peril: the mapping from
 * this demo's option names to DruidJS's.
 *
 * No imports. Not DOM, not WebGPU, not druid itself — so the worker, the main
 * thread and `scripts/check-druid.ts` can all reach it, which is the point.
 * `buildDruidParams` living here rather than inside the worker is what lets the
 * check assert the key spelling without a browser; see the note on its
 * `EXPECTED_KEYS` counterpart there.
 */

export type DruidVariant = "pacmap" | "localmap";

/**
 * Exactly the parameter object druid receives.
 *
 * Deliberately not `Partial<ParametersPaCMAP>` from the library: the whole
 * hazard being defended against is a key druid does not recognise, and typing
 * this against druid's own interface would make the compiler complicit only for
 * the keys that *are* spelled right. Naming them here, once, is what
 * `check-druid.ts` compares against.
 */
export interface DruidParams {
  n_neighbors: number;
  MN_ratio: number;
  FP_ratio: number;
  d: number;
  num_iters: number[];
  apply_pca: boolean;
  knn: null;
  seed: number;
  low_dist_thres?: number;
}

export interface DruidRunOptions {
  variant: DruidVariant;
  nNeighbors: number;
  mnRatio: number;
  fpRatio: number;
  lowDistThres: number;
  seed: number;
  phases: [number, number, number];
}

/**
 * Our option names → druid's.
 *
 * Two of these carry more than a rename:
 *
 * `apply_pca: false` because the demo has already run its own PCA to 100d
 * (`pca.ts`) and hands the result to both engines. Letting druid run a second
 * one would mean the two backends were no longer looking at the same input,
 * which is the only reason to have both.
 *
 * `num_iters` mirrors our `phases`, so both engines run the same 450 iterations
 * and the demo's history-stride arithmetic needs no special case for the CPU
 * path.
 *
 * `low_dist_thres` is present only under localmap. Druid's PaCMAP does not
 * declare it, and passing it there would be indistinguishable from a typo.
 */
export function buildDruidParams(o: DruidRunOptions): DruidParams {
  const params: DruidParams = {
    n_neighbors: o.nNeighbors,
    MN_ratio: o.mnRatio,
    FP_ratio: o.fpRatio,
    d: 2,
    num_iters: [...o.phases],
    apply_pca: false,
    knn: null,
    seed: o.seed,
  };
  if (o.variant === "localmap") params.low_dist_thres = o.lowDistThres;
  return params;
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

/**
 * Hands the worker the data and starts it. `X` is transferred, not copied — the
 * caller must not touch it afterwards.
 */
export interface DruidInitCommand extends DruidRunOptions {
  type: "init";
  /** N x D, row-major. The demo's PCA output, so D is 100. */
  X: Float32Array<ArrayBuffer>;
  N: number;
  D: number;
}

/** Advance the optimizer to iteration `to` and send back the embedding. */
export interface DruidStepCommand {
  type: "step";
  to: number;
}

export type DruidCommand = DruidInitCommand | DruidStepCommand;

export type DruidEvent =
  /** Setup (kNN + pair sampling) is done; the optimizer is ready to step. */
  | { type: "ready"; setupMs: number }
  /**
   * N x 2 f32, transferred. `it` is the iteration it was taken after.
   *
   * `ArrayBuffer` rather than the default `ArrayBufferLike`: these are
   * transferred, so they are never shared, and `queue.writeBuffer` will not
   * accept a possibly-shared view.
   */
  | { type: "frame"; it: number; Y: Float32Array<ArrayBuffer> }
  | { type: "error"; message: string };
