# TODO

Things worth doing, not scheduled. Each entry should say enough that picking it up
does not mean re-deriving why it is here.

## Scale point radius with distance from the camera (3D only)

Today the quad offset is added *after* projection and multiplied by `clip.w`
(`src/shaders.ts`, `renderWGSL`), which cancels the perspective divide and makes a
point's radius **constant in screen pixels** at any camera distance.

In 2D that is the right call and deliberate: zooming in resolves a dense cluster
into separate points rather than magnifying blobs. The 3D analogue of that
behaviour is **not** the same rule, though — it is per-point distance scaling. With
every point the same size regardless of depth, the near face and the far face of a
cloud are drawn identically, which throws away the strongest size cue the eye has
and leaves depth to occlusion and parallax alone.

**Shape of the change.** Dropping the `* clip.w` is the one-line version and gives
true perspective size, but it ties the radius to the raw view-space depth, so the
apparent size of everything changes as the camera dollies. The reference viewer
(`~/scratch/marimo-pacmap-animation/app/app.js:88-121`) normalizes against the
default framing distance instead:

```glsl
float size = uSize * uRefDist / max(-mv.z, 0.001);
```

so `uSize` keeps meaning "pixels at the default framing", and only the *relative*
near/far difference comes from depth. `DEFAULT_DIST` in `src/main.ts` is already
exactly that reference distance.

**Watch out for:**

- The existing floor (`Math.max(1.5, view.pointSize * dpr)` in `resize()`) is a
  per-frame CPU clamp on a now per-point quantity; the floor has to move into the
  shader. The reference expresses the shortfall below one pixel as opacity instead
  (`vFade`), which is worth copying — but keep it a separate varying rather than
  conflating it with a depth cue, which is a mistake that file documents in itself.
- Interaction with `occlusion`: solid discs of varying size change how much a near
  cluster hides, so the two want to be judged together.
- Whether it should apply in 2D as well (it should not — see above) means this is a
  third thing keyed off the components mode, alongside `enableRotate` and the mouse
  map.

Raised 2026-08-05 while reviewing the 3D renderer.
