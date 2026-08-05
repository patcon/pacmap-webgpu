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
export const renderWGSL = (d = 2) => /* wgsl */ `
// 32 bytes. vec4 rather than vec3 because a vec3 in a uniform pads to 16 bytes
// anyway, so the padded form is free — and it keeps one buffer size, one
// history slot stride and one sessionStorage schema across the 2D/3D switch,
// instead of a layout that changes with a dropdown.
struct Bounds { lo : vec4<f32>, hi : vec4<f32> };
// 80 bytes: mat4 64 + vec2 8 + f32 4 + pad 4. viewProj already carries the
// WebGL-to-WebGPU depth remap (see resize() in main.ts), so it is used as-is.
struct View   { viewProj : mat4x4<f32>, res : vec2<f32>, radius : f32, _pad : f32 };

@group(0) @binding(0) var<uniform> B : Bounds;
@group(0) @binding(1) var<uniform> V : View;

struct VSOut {
  @builtin(position) clip : vec4<f32>,
  @location(0)       col  : vec3<f32>,
  @location(1)       uv   : vec2<f32>,
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

  let r = vec2<f32>(2.0 * V.radius / V.res.x, 2.0 * V.radius / V.res.y);
  // Times clip.w so the perspective divide cancels: a point keeps a constant
  // pixel size at any camera distance, so zooming resolves a cluster rather
  // than magnifying it.
  clip = vec4<f32>(clip.xy + c * r * clip.w, clip.zw);

  var out : VSOut;
  out.clip = clip;
  out.col  = PALETTE[min(lab, 9u)];
  out.uv   = c;
  return out;
}

@fragment
fn fs(in : VSOut) -> @location(0) vec4<f32> {
  let d = length(in.uv);
  if (d > 1.0) { discard; }
  return vec4<f32>(in.col, (1.0 - smoothstep(0.7, 1.0, d)) * 0.85);
}
`;
