# TODO

Things worth doing, not scheduled. Each entry should say enough that picking it up
does not mean re-deriving why it is here.

## `samplePairs` doesn't reject duplicates the way upstream does

Two undocumented deviations in `samplePairs` (`src/pacmap-webgpu.ts`), found while reading
`pacmap/pacmap.py` for the pair-graph overlay. Both make a point's pair set slightly more
concentrated than the reference's — the same partner can be counted twice, so it pulls or
pushes twice as hard as it should.

Upstream's `sample_FP(n_samples, maximum, reject_ind, self_ind)` rejects three things: the
point itself, **anything already drawn in this same call** (`for k in range(i): if j ==
result[k]`), and the caller's `reject_ind` list. Ours rejects the first and the third only.

- **Mid-near.** The 6 candidates drawn per slot are not deduplicated against each other,
  and — unlike upstream, which passes `reject_ind=pair_MN[i*n_MN : i*n_MN+j, 1]` — not
  against the partners already picked for this same `i`. So one point can end up as two
  of `i`'s mid-near partners.
- **Further.** The 64-try loop rejects self and `nbSet`, but not partners already drawn
  for this `i`, so `fpFwd`'s row can repeat an index.

Not obviously worth fixing on its own: the effect is small at any realistic `nMN`/`nFP`
and it is not what anyone is looking at. Worth knowing about if a layout ever disagrees
with the reference's in a way the Gaussian-vs-PCA init doesn't explain. Fixing it changes
the RNG stream, so `check:ab` will move and the diff has to be justified rather than
zeroed — do it deliberately, not as a drive-by.
