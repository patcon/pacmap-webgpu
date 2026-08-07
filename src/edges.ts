// The pair-graph overlay's index buffer.
//
// Its own module, with no DOM and no device, for the same reason `shaders.ts`
// has one: `main.ts` touches the DOM at module scope and cannot be imported
// outside a browser. This is the piece of the overlay where a mistake is an
// off-by-one rather than a validation error — an edge drawn between the wrong
// two points looks like a plausible mesh — and nothing headless renders a
// pixel, so being importable is the only way it gets checked at all.

import type { PairGraph } from "./pacmap-webgpu";

/** The three pair kinds, in the order their ranges sit in the index buffer. */
export const EDGE_KINDS = ["near", "midNear", "further"] as const;
export type EdgeKind = (typeof EDGE_KINDS)[number];

/**
 * Near green, mid-near yellow, further red — attract, attract-weakly, repel.
 *
 * Held a little away from the digit palette's own green/yellow/red so an edge
 * reads as an edge over a cluster of the same hue, and dimmer than the points
 * so the cloud stays the foreground.
 */
export const EDGE_COLORS: Record<EdgeKind, [number, number, number]> = {
  near: [0.25, 0.85, 0.42],
  midNear: [0.95, 0.83, 0.22],
  further: [0.92, 0.28, 0.28],
};

/** Flat, deliberately not scaled by the pair's weight or the schedule's phase. */
export const EDGE_ALPHA = 0.35;

export interface EdgeRange {
  /** Offset into the index buffer, in indices — i.e. `drawIndexed`'s firstIndex. */
  first: number;
  /** Pairs, not indices. Two indices are drawn per pair. */
  count: number;
}

/**
 * The pair graph as one index buffer: three contiguous ranges of `line-list`
 * endpoint pairs, drawn against the position buffer the points already use.
 *
 * **Each range is shuffled**, so any prefix of it is a uniform random sample of
 * that kind. That is what lets the percentage be a render-time control — it
 * moves an index count and nothing else, no buffer rewritten and no run
 * restarted — and it means edges appear spread through the cloud rather than
 * walking along the sample order. Exactly the argument `digit %` rests on, and
 * the reason every pair is held rather than a sampled subset.
 *
 * The cost of holding all of them is 8 bytes a pair, and a point draws
 * `nNB + nMN + nFP` of them — 35 at the demo's defaults (10 neighbours, then
 * `round(nNB * 0.5)` and `round(nNB * 2)`). So 65k points is 2.3M pairs and
 * 18MB, or 5.0M and 40MB with "auto neighbors" ticked, where `defaultNeighbors`
 * puts nNB at 22. Against the digit atlas's 51MB, either is the small one.
 *
 * `rand` is supplied rather than seeded here, so the caller owns the policy —
 * the demo derives it from the run's seed, which is what makes two runs at one
 * seed draw the same sample rather than making a comparison also a comparison
 * of two different subsets.
 */
export function buildEdgeIndices(
  graph: PairGraph,
  N: number,
  rand: () => number
): { indices: Uint32Array; ranges: Record<EdgeKind, EdgeRange> } {
  // Each kind is a forward array of `width` partners per point, laid out row
  // by row — the only two things this needs to know about any of them.
  const rows: Record<EdgeKind, { fwd: Uint32Array; width: number }> = {
    near: { fwd: graph.nbFwd, width: graph.nNB },
    midNear: { fwd: graph.mnFwd, width: graph.nMN },
    further: { fwd: graph.fpFwd, width: graph.nFP },
  };

  let total = 0;
  for (const kind of EDGE_KINDS) total += rows[kind].fwd.length;
  const idx = new Uint32Array(total * 2);

  const ranges = {} as Record<EdgeKind, EdgeRange>;
  let w = 0;
  for (const kind of EDGE_KINDS) {
    const { fwd, width } = rows[kind];
    const first = w;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < width; k++) {
        const j = fwd[i * width + k];
        // `nbFwd` pads short rows with N — possible only when the candidate
        // pool is smaller than nNB, i.e. at a handful of points. Not an edge.
        if (j >= N) continue;
        idx[w] = i;
        idx[w + 1] = j;
        w += 2;
      }
    }
    // Fisher-Yates over whole pairs, not over the u32s: swapping individual
    // indices would rewire the graph rather than reorder it. Hence both
    // endpoints moving together below.
    for (let e = (w - first) / 2 - 1; e > 0; e--) {
      const f = Math.floor(rand() * (e + 1));
      const a = first + e * 2;
      const b = first + f * 2;
      const a0 = idx[a];
      const a1 = idx[a + 1];
      idx[a] = idx[b];
      idx[a + 1] = idx[b + 1];
      idx[b] = a0;
      idx[b + 1] = a1;
    }
    ranges[kind] = { first, count: (w - first) / 2 };
  }

  // subarray, not the whole allocation: skipping pad entries above can leave a
  // tail unused, and a buffer sized from `total` would end in zeros that draw
  // as edges from point 0 to point 0.
  return { indices: idx.subarray(0, w), ranges };
}
