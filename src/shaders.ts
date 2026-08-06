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

// The two uniform structs and the data→world mapping, as text, because the
// point renderer and the edge overlay below have to agree about all three
// exactly. An edge whose endpoints are placed by even slightly different
// arithmetic than the points it connects would detach from them, and the
// framing is mixed per frame so the disagreement would come and go — which is
// the kind of bug that is hunted for hours rather than noticed.
//
// Spliced in rather than factored into a WGSL function, so `renderWGSL`'s
// output stays character-for-character what it was before the overlay existed.

// 32 bytes. vec4 rather than vec3 because a vec3 in a uniform pads to 16 bytes
// anyway, so the padded form is free — and it keeps one buffer size, one
// history slot stride and one sessionStorage schema across the 2D/3D switch,
// instead of a layout that changes with a dropdown.
const BOUNDS_STRUCT = `struct Bounds { lo : vec4<f32>, hi : vec4<f32> };`;

// Exactly 96 bytes: mat4 64 + vec2 8 + f32 x 6. It used to be 92 padded to 96
// by the mat4's 16-byte alignment; lerpT took the padding, as occlude took the
// slot before it. viewProj already carries the WebGL-to-WebGPU depth remap (see
// resize() in main.ts), so it is used as-is.
const VIEW_STRUCT = `struct View {
  viewProj   : mat4x4<f32>,
  res        : vec2<f32>,
  radius     : f32,
  occlude    : f32,
  // How many points draw as digits: the slider's percentage resolved against N
  // on the CPU, compared against each point's rank. A count and not a fraction
  // so the shader needs no N.
  digitCount : f32,
  digitScale : f32,
  // Which of the three thumbnail looks — see thumbColor.
  digitStyle : f32,
  // Where between the two bound keyframes this frame sits. Playback binds
  // adjacent history slots to @location(0) and @location(3) and the vertex
  // stage mixes them; 0 draws keyframe A exactly, which is what the live path
  // and an unticked "interpolation" both leave it at.
  lerpT      : f32,
};`;

/**
 * Data → world, as statements. Reads `p`, `pB`, `B`, `B2` and `V` from the
 * surrounding scope and defines `w`; every consumer declares those bindings
 * under those names, whatever binding numbers it gives them.
 */
const worldStatements = (d: number) => `
  // One scale for every axis so the embedding isn't stretched, and centred on
  // the origin, which is why the camera never has to re-fit: the bounds reduce
  // keeps doing the framing and the camera only moves when the user moves it.
  // In 2D the box's z extent is exactly 0, so one span expression serves both —
  // measured, a 3D layout comes out near-isotropic (per-axis spreads within 3%
  // of each other), so one span frames it well.
  //
  // The box is mixed on the same t as the position. Under "auto zoom" each
  // keyframe is framed by its own banked bound, so a snapping box under gliding
  // points would pulse once per keyframe; held, both bindings carry the same
  // box and the mix is a no-op.
  let lo   = mix(B.lo, B2.lo, V.lerpT);
  let hi   = mix(B.hi, B2.hi, V.lerpT);
  let ctr  = (lo.xyz + hi.xyz) * 0.5;
  let ext  = hi.xyz - lo.xyz;
  let span = max(max(max(ext.x, ext.y), ext.z), 1e-6);
  // Straight lerp between adjacent keyframes — no easing, no spline. At lerpT
  // 0 this is p exactly, so an unticked "interpolation" draws what predates it
  // bit for bit.
  let pos = mix(p, pB, V.lerpT);
  let w = ${d === 3
    ? "(pos - ctr) / (span * 0.55)"
    : "vec3<f32>((pos - ctr.xy) / (span * 0.55), 0.0)"};`;

/**
 * The point renderer, templated on the embedding's width.
 *
 * `d` decides two things and nothing else: the vertex attribute's type, and
 * whether the world position takes its z from the data or from the constant 0
 * the 2D path has always passed. Everything downstream — the projection, the
 * screen-space quad, the fragment — is already dimension-agnostic.
 */
export const renderWGSL = (d = 2, refDist = 2.414213562373095) => /* wgsl */ `
${BOUNDS_STRUCT}
${VIEW_STRUCT}

// A and B: the two keyframes being mixed. Two 32-byte bindings rather than one
// 64-byte struct, so BOUNDS_BYTES keeps meaning both the history slot stride
// and a uniform's size — the staging buffer and the sessionStorage schema are
// written against that one constant. Both hold the same box on the live path.
@group(0) @binding(0) var<uniform> B  : Bounds;
@group(0) @binding(3) var<uniform> B2 : Bounds;
@group(0) @binding(1) var<uniform> V : View;
// The digit atlas: every point's 28x28 bitmap, tile-major, four 8-bit
// intensities to the u32. Quantized because holding *all* of them is what makes
// the digit percentage a render-time slider rather than a setup parameter — at
// f32 the buffer would be ~204MB at N=65k against a 128MB default limit on a
// storage binding, and at 8 bits it is ~51MB. The tile index is the point
// index, so there is no mapping to keep.
//
// A storage buffer and not an r8unorm texture with a filtering sampler, which
// is the natural tool for this. @kmamal/gpu's createView() unconditionally
// sends a component swizzle Dawn rejects, so under the headless checks no
// texture view can be created and therefore no bind group containing one — the
// same quirk that already forces check-shaders to encode its draw into a render
// bundle. A texture here would have meant deleting the one case that encodes a
// draw, at the exact moment of adding a binding that could break it. Filtering
// by hand below is the cheaper half of that trade.
@group(0) @binding(2) var<storage, read> atlas : array<u32>;

const TILE_PX : u32 = 28u;
const TILE_WORDS : u32 = TILE_PX * TILE_PX / 4u;
// What the vertex stage forwards for a point drawn as a disc, so the fragment
// has one thing to test.
const NO_THUMB : u32 = 0xFFFFFFFFu;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       col  : vec3<f32>,
  @location(1)       uv   : vec2<f32>,
  // The tile to draw, or NO_THUMB for an ordinary disc.
  @location(3) @interpolate(flat) thumb : u32,
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
  @builtin(vertex_index)   vi  : u32,
  @builtin(instance_index) ii  : u32,
  @location(0)             p   : vec${d}<f32>,
  @location(1)             lab : u32,
  @location(2)             thm : u32,
  // The next keyframe, bound from the same history buffer one slot along. A
  // vertex input location, which is a separate namespace from VSOut's — the 3
  // it shares with the thumb interpolant is not a collision.
  @location(3)             pB  : vec${d}<f32>,
) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vi];

  // A thumbnail is drawn at digitScale times the point radius. Matching the
  // point size exactly would cap a digit at a handful of pixels, which is a
  // smudge rather than a glyph; points keep their own range and digits get
  // theirs. Resolved here so sizeMul is the only thing the size arithmetic
  // below has to know about thumbnails, at either dimensionality.
  //
  // Which points are digits is a threshold on their rank — a random
  // permutation fixed at setup — so the slider adds and removes them spread
  // through the cloud, and moving it costs one number in this uniform rather
  // than a rebuilt buffer.
  let isThumb = f32(thm) < V.digitCount;
  let sizeMul = select(1.0, V.digitScale, isThumb);

  // Data → world. Shared verbatim with the edge overlay — see worldStatements.
${worldStatements(d)}
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
  // Two separate guards, because they answer two separate questions.
  //
  // The floor on w is only about w <= 0 — points behind the camera, which
  // Orbit's minDistance makes reachable, and whose negative w would invert
  // the quad. It is deliberately tiny: a quad whose vertices all sit behind
  // the near plane is clipped whole, so this only has to keep the arithmetic
  // sane on the way there, not bound the size.
  //
  // Bounding the size is maxPx's job, and it is expressed as a fraction of
  // the viewport rather than as a multiple of V.radius. A multiple caps the
  // wrong quantity: what should never happen is one point covering the
  // canvas, and how many radii that takes depends on the canvas. This lets a
  // point at the lens grow to half the short side and no further. The near
  // plane (0.01, the camera in main.ts) is the other limit and sits at about
  // 240x, so between them a close point gets genuinely large before it is
  // clipped — which is the point of a perspective size cue.
  const REF_DIST = ${refDist};
  let maxPx = 0.25 * min(V.res.x, V.res.y);
  // sizeMul lands before the clamp, so a thumbnail still falls off with depth
  // and is still bounded by the viewport fraction below.
  let pxAtDepth = V.radius * sizeMul * REF_DIST / max(clip.w, 1e-4);
  let drawPx = clamp(pxAtDepth, 1.0, maxPx);
  // Below one device pixel there is nothing left to shrink, so the shortfall
  // is spent as opacity instead of a floor that would make near and far
  // points read as the same size — same trick as the reference's vFade.
  let fade = clamp(pxAtDepth, 0.0, 1.0);
  let r = vec2<f32>(2.0 * drawPx / V.res.x, 2.0 * drawPx / V.res.y);
  ` : `
  let px = V.radius * sizeMul;
  let r = vec2<f32>(2.0 * px / V.res.x, 2.0 * px / V.res.y);
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
  // The tile is this point's own bitmap, so the attribute carries only rank.
  out.thumb = select(NO_THUMB, ii, isThumb);
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

// A share of points draw their actual 28x28 MNIST bitmap instead of a disc, so
// a cluster boundary can be read as digits rather than as colours. Three styles,
// picked live from the pane, because which reads best in a dense cloud depends
// on how dense it is:
//
//   0 coloured stroke — the strokes take the label colour and the black
//     background is transparent, so the tile sits in the cloud the way a point
//     does. Loses definition where clusters overlap.
//   1 white on colour — a solid label-coloured tile with the digit painted
//     white through it. Reads at smaller sizes, but occludes what is behind.
//   2 black on colour — the same tile with the strokes knocked out to black,
//     which keeps the label colour dominant at a distance.
//
// A uniform and not a per-point bit: the style costs nothing to store and
// nothing to change, and picking it per point (as a coinflip did while both
// were being tried) means never seeing either one alone.
fn texel(base : u32, x : i32, y : i32) -> f32 {
  let cx = u32(clamp(x, 0, i32(TILE_PX) - 1));
  let cy = u32(clamp(y, 0, i32(TILE_PX) - 1));
  let i = cy * TILE_PX + cx;
  let w = atlas[base + (i >> 2u)];
  return f32((w >> ((i & 3u) * 8u)) & 0xFFu) / 255.0;
}

fn thumbColor(thumb : u32, uv : vec2<f32>, col : vec3<f32>) -> vec4<f32> {
  let base = thumb * TILE_WORDS;
  // Quad space is [-1, 1] with y up; the sprite's row 0 is the top of the
  // digit, so v is flipped here and the digits come out upright.
  let g = vec2<f32>(uv.x * 0.5 + 0.5, 0.5 - uv.y * 0.5);
  // Bilinear by hand — what a linear sampler with no mips would have given.
  // Half-texel offset so the filter is centred on texel centres rather than
  // on their corners, which would shift the digit by half a texel.
  let t = g * f32(TILE_PX) - 0.5;
  let i = floor(t);
  let f = t - i;
  let x = i32(i.x);
  let y = i32(i.y);
  let v = mix(
    mix(texel(base, x, y),     texel(base, x + 1, y),     f.x),
    mix(texel(base, x, y + 1), texel(base, x + 1, y + 1), f.x),
    f.y);
  if (V.digitStyle > 1.5) { return vec4<f32>(col * (1.0 - v), 1.0); }
  if (V.digitStyle > 0.5) { return vec4<f32>(mix(col, vec3<f32>(1.0), v), 1.0); }
  return vec4<f32>(col, v);
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  if (in.thumb != NO_THUMB) {
    let t = thumbColor(in.thumb, in.uv, in.col);
    // A digit fills its square, so there is no disc discard on this path.
    ${d === 3 ? `
    let a = t.a * in.fade;
    // Occluded, a fragment that writes depth hides everything behind it, and
    // most of a digit's square is background. Dropping the transparent part
    // keeps it from punching a square hole through the cloud — a disc never
    // had this much empty area to give away.
    if (V.occlude > 0.5 && a < 0.02) { discard; }
    return vec4<f32>(t.rgb, a);` : `
    return t;`}
  }
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

/**
 * The pair-graph overlay: near, mid-near and further pairs as coloured lines.
 *
 * An **indexed** `line-list` over the same position buffer the point renderer
 * binds, which is what makes this cheap. The alternatives were both worse:
 *
 *   - Positions as a storage buffer, indexed per edge. That needs a dynamic
 *     offset to reach a playback keyframe, and a storage binding's dynamic
 *     offset must be 256-byte aligned — `frameBytes = N*4*d` generally is not
 *     (520,000 at N=65k, d=2). Binding all of `posHistory` instead runs into
 *     its 128MB budget being exactly the default maxStorageBufferBindingSize.
 *   - A second render pass. One more pass per frame to no end; the points are
 *     drawn straight after these into the same one.
 *
 * Indexing also makes playback interpolation free: slot `a` goes to vertex
 * buffer 0 and slot `b` to vertex buffer 1, and the mix below is the same one
 * the point shader runs, from the same shared text.
 *
 * What it costs is that the pair *kind* cannot be a vertex attribute — on an
 * indexed draw `@builtin(vertex_index)` is the fetched index value, not the
 * position in the index buffer, so it cannot be compared against a range
 * boundary. So the kind is not in the shader at all: one index buffer holds the
 * three kinds in three contiguous ranges, and the colour arrives per draw
 * through a dynamic-offset uniform. Same mechanism the optimizer already uses
 * for its per-iteration weights.
 *
 * Line width is always one device pixel in WebGPU — there is no lineWidth — so
 * there is nothing to expose for it short of drawing quads.
 */
export const edgeWGSL = (d = 2) => /* wgsl */ `
${BOUNDS_STRUCT}
${VIEW_STRUCT}
// One vec4 per pair kind, selected by dynamic offset at draw time. Only .rgb
// and .a are read; the alpha is a flat constant, deliberately not scaled by the
// pair's weight or the schedule's phase (see the pane).
struct EdgeStyle { color : vec4<f32> };

// Same names as the point shader, because worldStatements below is the same
// text. The binding *numbers* differ — there is no atlas here — which is fine:
// the shared statements refer to B, B2 and V and never to a binding index.
@group(0) @binding(0) var<uniform> B  : Bounds;
@group(0) @binding(1) var<uniform> V  : View;
@group(0) @binding(2) var<uniform> B2 : Bounds;
@group(0) @binding(3) var<uniform> E  : EdgeStyle;

@vertex
fn vs(
  @location(0) p  : vec${d}<f32>,
  // The next keyframe. On playback this is the same buffer as location 0, one
  // history slot along.
  @location(1) pB : vec${d}<f32>,
) -> @builtin(position) vec4<f32> {
${worldStatements(d)}
  return V.viewProj * vec4<f32>(w, 1.0);
}

@fragment
fn fs() -> @location(0) vec4<f32> {
  // Opaque when occluding. A semi-transparent fragment that writes depth culls
  // everything behind it while compositing against the background — the same
  // dark-streak failure the point shader's occluded branch avoids, and a line
  // laid across a cluster would draw a very visible one.
  if (V.occlude > 0.5) { return vec4<f32>(E.color.rgb, 1.0); }
  return E.color;
}
`;
