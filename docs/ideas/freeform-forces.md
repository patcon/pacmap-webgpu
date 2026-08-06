# Freeform forces

## Problem Statement

How might we let someone turn the PaCMAP force weights into live knobs — playing the
embedding as an instrument rather than watching a recording of a fixed schedule?

## Recommended Direction

**A `freeform` mode: a new pane folder with three live sliders (NN, MN, FP) driving a
perpetual optimizer, with no phase schedule at all.**

The architecture already supports this almost entirely. `weightsAt`
(`pacmap-webgpu.ts:1169`) is the only place forces enter the system, and its output
already lives in a per-iteration uniform slot (`params[o+0..2]`). Nothing caches them.
A pane edit is a JS rewrite plus one `queue.writeBuffer`, and the next encoded chunk
picks it up — queue writes are ordered against `submit()`, which CLAUDE.md already
relies on for the further-pair resample counter.

Freeform is *simpler* than the scheduled path, not an addition to it: with weights
constant-until-edited there is no per-iteration variation to encode, so the whole
`params` array (`:1662-1684`) collapses to **a single 256-byte slot at dynamic offset
0**, rewritten on edit. Adam's two bias corrections both converge to 1.0 at large
iteration counts and can be pinned there.

Dropping the 3-phase structure is what makes every knob always live. In a scheduled
run, two-thirds of a 3x3 table is inactive at any moment — you turn a phase-1 knob at
iteration 250 and correctly nothing happens, which reads as a dead control. With one
perpetual phase that failure mode cannot occur by construction. The cost is that the
schedule stops being the subject: you can no longer feel *why* MN ramps 1000 -> 3.
That is the right trade for an instrument and the wrong one for a teaching tool.

## Key Assumptions to Validate

- [ ] **A converged embedding still responds to the knobs.** Adam's second moment `v`
      accumulates, so by iteration ~2000 the effective step may be small enough that
      turning MN produces almost nothing. The dramatic motion lives in the first ~100
      iterations while `v` is still small. **Test before building any UI:** run past
      `totalIters`, turn a weight by hand in the console, watch. If it's numb, the
      outs are (a) partially decay `M`/`V` on edit — a "kick", (b) a higher fixed `lr`
      in freeform, (c) short auto-cycles that keep you in the fluid regime.
- [ ] **One chunk of edit latency is imperceptible.** Weights can't change for
      iterations already encoded into a submitted command buffer. The demo steps in
      `stride`-sized chunks so this should be a frame or two — but `run()`
      (`:1872`) encodes the entire run in one submit, so freeform must never use it.
- [ ] **Useful slider ranges exist.** MN spans 1000 -> 3 in the real schedule, so a
      linear 0-1000 slider puts everything interesting in the bottom 0.3% of travel.
      Assume MN needs a log/exponential scale; NN (1-3) and FP (1) probably don't.

## MVP Scope

**In:**
- A `freeform` entry in the `algorithm` dropdown (or a checkbox that swaps modes),
  GPU engines only.
- A pane folder with three sliders: NN, MN, FP. Each declares an explicit
  `{min, max, curve}` in one table — see "MIDI" below for why now rather than later.
- Single-slot params buffer; `paramBuf` gains `COPY_DST` (`:1685` currently lacks it).
- `reset()` on the run surface: restore `yBuf` from a retained `Y0`, zero `M` and `V`.
  Needed for *either* restart mechanism.
- Restart: a hotkey, and/or auto-cycle after N iterations (N itself a slider — see the
  assumption above, this may be load-bearing rather than a convenience).
- Transport goes inert; no `posHistory` banking. The scrubber is meaningless in a
  perpetual run, and skipping the history copies avoids ~350MB/cycle at 65k.

**Out of MVP but designed for:** MIDI. Declaring each slider's range and curve in one
table now makes MIDI a mapping layer later instead of a redesign — MIDI CC is 7-bit
normalized and needs exactly that metadata.

## Not Doing (and Why)

- **Boomerang / reverse playback** — deferred by decision, and separately: the
  optimizer cannot run backward. Adam is a lossy stateful integrator and LocalMAP's
  further-pair resample is a random redraw. Any boomerang is necessarily a *playback*
  feature over banked frames, sharing no code with this. Different conversation.
- **The 3x3 phase table (rows NN/MN/FP x cols phase1/2/3)** — the original idea. Two
  thirds of it is inactive at any playhead position, so most knobs are dead most of
  the time. Freeform is the version where that can't happen.
- **`linear(1000, 3)` / expression cells and a parser** — with no phases there is no
  ramp to express. A cell is one number. If the schedule ever comes back, note that
  `check:ab` demands the default table emit `1000*(1-t) + 3*t` character-for-
  character: a generic `a + (b-a)*t` is not bit-identical in f32.
- **Editing phase lengths** — `n1`/`n2` are baked into `totalIters` (buffer and
  history sizing) *and* into `runRange`'s encode-time guards (`:1842`). Weights-only
  stays a uniform rewrite; phase-length editing is a full teardown.
- **Re-baking banked traces on edit** — the architecture this was first framed around.
  Live injection into a running sim needs no re-bake, no invalidation, and no 350MB
  re-bank. It was the wrong model.
- **CPU/druid support** — druid takes `num_iters` and exposes no weight-schedule hook.
  Grey the folder out under CPU engines, same pattern as `kNN algo`.

## Open Questions

- **What does LocalMAP mean in freeform?** The further-pair resample is gated on
  `it > n1+n2 && it % 10 === 0` (`:1842`), and the modified near-pair coefficient on
  `it > n1+n2` (`:1676`). With no phases: does resampling run every 10 iterations
  forever, and is the LocalMAP gradient always on? Both are plausible and both are
  choices. A resample-cadence slider is a candidate fourth knob.
- **Hotkey restart, auto-cycle, or both?** They're different products — auto-cycle is
  ambient and suits a MIDI performance; a hotkey is deliberate. Both are cheap. If the
  numbness assumption bites, auto-cycle stops being optional.
- **Is a mid-flight weight change acceptable?** Adam's `M`/`V` carry across the
  discontinuity, so the trajectory on screen is a splice, not a run of any schedule
  you could write down. For play that's the point. It does mean the sliders describe
  the *current force field*, not the thing you're watching — and a future "bake this
  exact schedule" button would be a second mode, not a tweak.
- **How is freeform entered?** A fourth `AlgoKey`-style entry, or an orthogonal
  toggle? It crosses the existing algorithm/engine axes rather than extending them —
  freeform-PaCMAP and freeform-LocalMAP are both meaningful, freeform-CPU is not.
- **Which pane lifetime does the folder have?** `dimensional reduction` is
  setup-time-and-locked; `rendering` is live. This is live-but-affects-the-simulation,
  a third category the pane has no vocabulary for.

## Coverage Note

Per CLAUDE.md's three known gaps, this adds a fourth of the same shape: nothing
headless can see a slider move. `check:shaders` would cover a changed params layout;
`check:kernels` could assert that a single-slot freeform params buffer at the default
weights reproduces the scheduled path's iteration-0 step. The wiring — folder, hotkey,
cycle, MIDI — needs a browser.
