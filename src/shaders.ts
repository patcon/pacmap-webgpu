// Shaders for the demo's own render + bounds passes.
//
// These live apart from `main.ts` only because `main.ts` reaches for the DOM at
// module scope, which makes it unimportable outside a browser. Keeping the WGSL
// in a module with no imports is what lets `scripts/check-shaders.ts` compile it
// headlessly.

export const boundsWGSL = (N: number) => /* wgsl */ `
@group(0) @binding(0) var<storage, read>       Y : array<f32>;
@group(0) @binding(1) var<storage, read_write> B : array<f32>;  // lo.xyzw hi.xyzw

const N : u32 = ${N}u;
// vec4 rather than vec2 because the box is eight floats at either
// dimensionality — see struct Bounds below for why that costs nothing. The
// unused components carry 0 through the reduce rather than the sentinel, so
// they land as 0 in the output instead of +-1e30.
var<workgroup> sLo : array<vec4<f32>, 256>;
var<workgroup> sHi : array<vec4<f32>, 256>;

// Single workgroup, grid-stride over all points. Keeps autoscaling on the GPU
// so the render loop never has to read positions back to the host.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  var lo = vec4<f32>( 1e30);
  var hi = vec4<f32>(-1e30);
  for (var i : u32 = t; i < N; i = i + 256u) {
    let p = vec4<f32>(Y[2u * i], Y[2u * i + 1u], 0.0, 0.0);
    lo = min(lo, p);
    hi = max(hi, p);
  }
  sLo[t] = lo;
  sHi[t] = hi;
  workgroupBarrier();
  for (var s : u32 = 128u; s > 0u; s = s >> 1u) {
    if (t < s) {
      sLo[t] = min(sLo[t], sLo[t + s]);
      sHi[t] = max(sHi[t], sHi[t + s]);
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    B[0] = sLo[0].x; B[1] = sLo[0].y; B[2] = sLo[0].z; B[3] = sLo[0].w;
    B[4] = sHi[0].x; B[5] = sHi[0].y; B[6] = sHi[0].z; B[7] = sHi[0].w;
  }
}
`;

/**
 * The point renderer, templated on the embedding's width.
 *
 * `d` decides two things and nothing else: the vertex attribute's type, and
 * whether the world position takes its z from the data or from the constant 0
 * the 2D path has always passed. Everything downstream — the projection, the
 * screen-space quad, the fragment — is already dimension-agnostic.
 */
export const renderWGSL = (d = 2, refDist = 2.414213562373095) => /* wgsl */ `
// 32 bytes. vec4 rather than vec3 because a vec3 in a uniform pads to 16 bytes
// anyway, so the padded form is free — and it keeps one buffer size, one
// history slot stride and one sessionStorage schema across the 2D/3D switch,
// instead of a layout that changes with a dropdown.
struct Bounds { lo : vec4<f32>, hi : vec4<f32> };
// 80 bytes: mat4 64 + vec2 8 + f32 4 + f32 4. viewProj already carries the
// WebGL-to-WebGPU depth remap (see resize() in main.ts), so it is used as-is.
// occlude took the slot that used to be padding — see the fragment.
struct View   { viewProj : mat4x4<f32>, res : vec2<f32>, radius : f32, occlude : f32 };

@group(0) @binding(0) var<uniform> B : Bounds;
@group(0) @binding(1) var<uniform> V : View;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       col  : vec3<f32>,
  @location(1)       uv   : vec2<f32>,
  ${d === 3 ? "@location(2)       fade : f32," : ""}
};

const PALETTE = array<vec3<f32>, 10>(
  vec3<f32>(0.90, 0.31, 0.31), vec3<f32>(0.95, 0.60, 0.20),
  vec3<f32>(0.85, 0.80, 0.22), vec3<f32>(0.44, 0.78, 0.34),
  vec3<f32>(0.22, 0.72, 0.62), vec3<f32>(0.29, 0.62, 0.90),
  vec3<f32>(0.45, 0.45, 0.92), vec3<f32>(0.72, 0.40, 0.88),
  vec3<f32>(0.92, 0.42, 0.70), vec3<f32>(0.62, 0.62, 0.66),
);

// Two triangles per point, instanced. @location(0) is bound directly to the
// PaCMAP position buffer — no copy, no readback.
@vertex
fn vs(
  @builtin(vertex_index) vi  : u32,
  @location(0)           p   : vec${d}<f32>,
  @location(1)           lab : u32,
) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vi];

  // Data → world. One scale for every axis so the embedding isn't stretched,
  // and centred on the origin, which is why the camera never has to re-fit: the
  // bounds reduce keeps doing the framing and the camera only moves when the
  // user moves it. In 2D the box's z extent is exactly 0, so one span
  // expression serves both — measured, a 3D layout comes out near-isotropic
  // (per-axis spreads within 3% of each other), so one span frames it well.
  let ctr  = (B.lo.xyz + B.hi.xyz) * 0.5;
  let ext  = B.hi.xyz - B.lo.xyz;
  let span = max(max(max(ext.x, ext.y), ext.z), 1e-6);
  let w = ${d === 3
    ? "(p - ctr) / (span * 0.55)"
    : "vec3<f32>((p - ctr.xy) / (span * 0.55), 0.0)"};
  // The projection owns the aspect divide now, so there is none here.
  var clip = V.viewProj * vec4<f32>(w, 1.0);

  ${d === 3 ? `
  // Perspective size, normalized against the default framing distance so
  // REF_DIST keeps V.radius meaning "pixels at the default framing" rather
  // than raw view-space depth — the reference viewer this borrows from
  // (marimo-pacmap-animation/app/app.js) does the same against uRefDist. A
  // point closer than the reference distance reads bigger, farther reads
  // smaller: the depth cue 2D deliberately does not want (see below).
  //
  // Depth is clip.w — the depth this point is *drawn* at, from the same
  // transform as its position. It has to be the same one: size against any
  // other depth and a point can read huge while being drawn in a distant
  // clump. That is not hypothetical; it shipped. Sizing ran a second
  // transform against the frame's own bound so that the auto-zoom-off
  // framing could not affect it, which is precisely backwards — with auto
  // zoom off the two transforms differ by the ratio of the held span to the
  // frame's own, so at the dense start the drawn cloud was a small far
  // clump while the sizing transform had spread the same points across the
  // whole world box, some of them onto the camera. Those points' 1/w blew
  // up and the canvas filled with colour. A point clumped on screen is
  // genuinely clumped; drawing it small is right.
  //
  // Clamping the depth rather than the size handles both ends of that in one
  // expression: it caps growth at MAX_GROW×, and it disposes of w <= 0 for
  // points behind the camera (Orbit's minDistance lets the camera inside the
  // cloud), which would otherwise invert the quad. The natural in-cloud
  // variation at the default framing is only about 0.73x–1.6x, so a cap of 4
  // is well clear of it and bites only once the camera has dollied in among
  // the points — which is exactly when the runaway used to start.
  const REF_DIST = ${refDist};
  const MAX_GROW = 4.0;
  let pxAtDepth = V.radius * REF_DIST / max(clip.w, REF_DIST / MAX_GROW);
  let drawPx = max(pxAtDepth, 1.0);
  // Below one device pixel there is nothing left to shrink, so the shortfall
  // is spent as opacity instead of a floor that would make near and far
  // points read as the same size — same trick as the reference's vFade.
  let fade = clamp(pxAtDepth, 0.0, 1.0);
  let r = vec2<f32>(2.0 * drawPx / V.res.x, 2.0 * drawPx / V.res.y);
  ` : `
  let r = vec2<f32>(2.0 * V.radius / V.res.x, 2.0 * V.radius / V.res.y);
  `}
  // Times clip.w so the perspective divide cancels: in 2D that keeps a point
  // a constant pixel size at any camera distance, so zooming resolves a
  // cluster rather than magnifying it. In 3D, drawPx above already carries
  // the desired depth falloff, and this cancellation is still needed for the
  // same reason — the GPU always divides clip.xy by clip.w on rasterization,
  // so any offset meant to land at a specific *post-divide* size has to be
  // pre-multiplied by clip.w regardless of how that size was computed.
  clip = vec4<f32>(clip.xy + c * r * clip.w, clip.zw);

  var out : VSOut;
  out.clip = clip;
  out.col  = PALETTE[min(lab, 9u)];
  out.uv   = c;
  ${d === 3 ? "out.fade = fade;" : ""}
  return out;
}

// Two looks, because in 3D they answer different questions.
//
// Blended (occlude = 0): every point contributes wherever it lands, so density
// reads directly as opacity. Correct for 2D, where the points are coplanar and
// there is nothing to hide anything else. In 3D it is a haze — depth is
// conveyed only by parallax as the camera moves.
//
// Occluded (occlude = 1): an opaque disc, so the depth buffer resolves which
// point is in front and near clusters genuinely hide far ones. The opacity is
// the load-bearing part, not a side effect: a semi-transparent fragment that
// writes depth culls everything behind it while compositing against the
// background, which is what produces dark streaks cutting through clusters
// (observed in the sibling project's fastplotlib backend). Nothing here writes
// depth unless it painted the pixel solid.
//
// The cost is a hard edge — the smoothstep feather cannot survive, since a
// feathered edge is exactly a semi-transparent fragment — and a dense cloud
// that stops accumulating opacity. Which is why this is a toggle.
//
// In 3D, in.fade (see vs above) also scales alpha here, including in the
// occluded branch — so a sub-1px-at-this-depth point still writes depth as
// the "solid disc" comment above describes, just faintly. That is the same
// tradeoff occlusion already accepts, now varying with size instead of
// constant; worth eyeballing the two together, not a reason to gate this on.
@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  ${d === 3 ? `
  if (V.occlude > 0.5) { return vec4<f32>(in.col, in.fade); }
  return vec4<f32>(in.col, (1.0 - smoothstep(0.7, 1.0, d)) * 0.85 * in.fade);
  ` : `
  if (V.occlude > 0.5) { return vec4<f32>(in.col, 1.0); }
  return vec4<f32>(in.col, (1.0 - smoothstep(0.7, 1.0, d)) * 0.85);
  `}
}
`;
