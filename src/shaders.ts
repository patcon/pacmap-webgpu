// Shaders for the demo's own render + bounds passes.
//
// These live apart from `main.ts` only because `main.ts` reaches for the DOM at
// module scope, which makes it unimportable outside a browser. Keeping the WGSL
// in a module with no imports is what lets `scripts/check-shaders.ts` compile it
// headlessly.

export const boundsWGSL = (N: number) => /* wgsl */ `
@group(0) @binding(0) var<storage, read>       Y : array<f32>;
@group(0) @binding(1) var<storage, read_write> B : array<f32>;  // loX loY hiX hiY

const N : u32 = ${N}u;
var<workgroup> sLo : array<vec2<f32>, 256>;
var<workgroup> sHi : array<vec2<f32>, 256>;

// Single workgroup, grid-stride over all points. Keeps autoscaling on the GPU
// so the render loop never has to read positions back to the host.
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid : vec3<u32>) {
  let t = lid.x;
  var lo = vec2<f32>( 1e30,  1e30);
  var hi = vec2<f32>(-1e30, -1e30);
  for (var i : u32 = t; i < N; i = i + 256u) {
    let p = vec2<f32>(Y[2u * i], Y[2u * i + 1u]);
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
    B[0] = sLo[0].x; B[1] = sLo[0].y;
    B[2] = sHi[0].x; B[3] = sHi[0].y;
  }
}
`;

export const renderWGSL = /* wgsl */ `
struct Bounds { lo : vec2<f32>, hi : vec2<f32> };
struct View   { res : vec2<f32>, radius : f32, _pad : f32 };

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
  @location(0)           p   : vec2<f32>,
  @location(1)           lab : u32,
) -> VSOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0),
  );
  let c = corners[vi];

  // Uniform scale on both axes so the embedding isn't stretched.
  let ctr  = (B.lo + B.hi) * 0.5;
  let span = max(max(B.hi.x - B.lo.x, B.hi.y - B.lo.y), 1e-6);
  var q = (p - ctr) / (span * 0.55);
  let aspect = V.res.x / V.res.y;
  q.x = q.x / aspect;

  let r = vec2<f32>(2.0 * V.radius / V.res.x, 2.0 * V.radius / V.res.y);

  var out : VSOut;
  out.clip = vec4<f32>(q + c * r, 0.0, 1.0);
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
