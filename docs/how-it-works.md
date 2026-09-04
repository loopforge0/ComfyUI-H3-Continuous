# How it works, and how it was measured

Everything in this file is background. You do not need any of it to use the pack --
start at the [README](../README.md). It is here because most of the design is not
obvious, and because the numbers behind the defaults should be checkable rather than
asserted.

Every measurement was taken on a 12 GB RTX 3060 at 480x864. `tools/check_joins.py`,
`tools/face_drift.py`, `tools/latent_drift.py` and `tools/bench_chain.py` reproduce
all of it.

---

## Why this isn't just "feed the last frame back in"

Two things make a chained H3 take hold together, and both are easy to get wrong.

### 1. Anchor the tail, don't reference it

`MiniMaxH3ReferenceToVideo` has a `ref_videos` input, and it is the obvious place to
put the previous shot. It does not work well. `ref_videos` is *reference* conditioning:
the model is trained to continue from it, but it reconstructs a plausible continuation
rather than copying pixels. The framing and the light carry over; the head pose pops.

A **keyframe** instead pins the clip into `minimax_keyframes` at a frame index. It
adds no sequence length — it writes into frames the latent already has — so it is
*cheaper* than `ref_videos`, and the anchor can stay at native resolution.

That is what this pack does, per segment, at `frame_idx=0`.

**And it never goes near the VAE.** `MiniMaxH3AddGuide` builds that keyframe by
VAE-encoding pixels, which means the default route to it is decode the previous
segment, write an mp4, read it back, encode again. The pack skips all of that: the
handoff is **sliced straight out of the previous segment's own sampled latent** and
pinned as-is. The anchor is then exact by construction rather than 30 dB of a round
trip, and each segment saves a 39-frame VAE encode.

That slice is always token-aligned, which is why it is safe. H3's frame grid is
17k+5 frames → 5k+2 latent steps, and a valid handoff is 17m+5 → 5m+2, so the tail
always starts at token 5(k−m) — a multiple of 5, which is cycle position 0 of the
1/4/4/4/4 frames-per-token pattern. Start anywhere else and the tail's first token
would claim to cover 4 frames where a fresh encode covers 1, putting the whole
anchor 3 frames out. `handoff_mode=pixel` keeps the old route for the one case the
latent slice cannot serve: changing resolution partway through a session.

**Latent against pixel, measured.** The same six-shot take rendered twice — identical
prompts, reference, seed, resolution and sampler, only the handoff route different
(6 × 4 s at 480×864, 4-step turbo):

| | pixel handoff | latent handoff |
| --- | --- | --- |
| anchor fidelity, per join | 22.0 / 19.4 / 17.1 / 17.3 / 17.6 dB | **22.7 / 21.9 / 18.4 / 19.0 / 19.0 dB** |
| mean | 18.7 dB | **20.2 dB** (+1.5, better at all five) |
| seam vs local motion, mean | −1.82 dB | **−0.40 dB** |
| latent drift from segment 1 | 2.049 | **1.366** (−33 %) |
| per segment | 5 min 06 s | **4 min 38 s** (−8.5 %) |

Better at every join, not on average-with-exceptions. The drift gap is 4.3× the
sampler's own noise (see [the noise floor](#drift-the-thing-you-cannot-see-at-any-one-join) below),
so it is a result rather than luck.

What it is *not*: on this pair, ArcFace identity came out 0.813 pixel against 0.791
latent and face dE 2.14 against 2.03 — both differences smaller than the ±0.039 and
±0.38 those metrics wobble by between two identical renders. The handoff demonstrably
carries the model's own state across more faithfully; it does not follow, and is not
claimed here, that the *face* survives measurably better for it.

**Earlier measurements, pixel handoff only.** Comparing segment N's last 39 frames
against segment N+1's first 39:

| | anchor fidelity | seam vs the motion around it |
| --- | --- | --- |
| 8 s segments, 480×864, 20 steps | **30.3 / 26.3 / 27.1 dB** | **+3.1 / +6.3 / −1.4 dB** |
| 8 s segments, 864×480, 20 steps (7 segments) | 23.7 – 26.2 dB | −0.6 to −5.6 dB |
| 5 s segments, 640×352, 20 steps | 21.1 / 22.8 dB | −5.5 / −7.5 dB |
| 5 s segments, 640×352, 4-step turbo | 19.9 / 20.9 dB | −6.6 / −7.5 dB |

Two unrelated spans of the same footage sit at 11–15 dB, and a genuine hard cut at
13 dB, so every row above is a join rather than a cut. **30 dB is the signature of the
same picture after a VAE round trip** — that top row is the anchor doing exactly what
it claims.

The second column is the one that predicts whether you *see* a join: the transition at
the splice, measured against the median frame-to-frame delta in the second either side
of it. Positive means the join moves *less* than the footage around it, i.e. invisible.
Do not use a whole-clip percentile for this — it is confounded by how much the footage
moves, and rates identical joins anywhere from the 1st to the 31st percentile depending
on whether the shot is a busy street or a slow misty walk.

**Segment length matters more than you would guess.** The bottom two rows are not worse
because of resolution — they are worse because a 39-frame handoff is **31 % of a 5-second
segment but only 20 % of an 8-second one**. Past roughly a quarter, the model has too
little free runway after the pinned region and the seam degrades. Keep the handoff near
a fifth of the segment. 8 s segments also put the audio exactly on grid: measured drift
is 0.000 s at 192 frames against 0.017 s at 124.

`tools/check_joins.py <session>` runs all of this against a rendered session, and
`tools/bench_chain.py --arms pixel latent arrest` renders the same take under
different settings and measures the difference, which is the only way to answer
"did that help" on a model where one segment costs minutes.

### Handoff length, exactly

**Why 1.625 s (39 frames) is the default:** guide fidelity peaks exactly at frames 0, 17
and 34 and sags between — the latent grouping showing through. A longer handoff joins
more securely but costs you that much new footage: at 8 s segments with a 1.625 s
handoff, every segment after the first contributes 6.375 s to the finished cut.

39 is also the only handoff that lands *exactly* on the packed sequence's shared time
axis. Picture advances 5/3 t-units per frame and audio 1 per latent step, so a handoff
is only phase-exact when 5/3 × frames is a whole number:

| handoff | frames | latent steps | audio steps | 5/3 × frames |
| --- | --- | --- | --- | --- |
| 0.21 s | 5 | 2 | 8 | 8.33 |
| 0.92 s | 22 | 7 | 37 | 36.67 |
| **1.63 s** | **39** | **12** | **65** | **65.00** |
| 2.33 s | 56 | 17 | 93 | 93.33 |

**But match it to the segment, not to this table.** The default is wrong for short
segments, and badly so: 39 frames is 20 % of an 8 s segment and 36 % of a 4 s one, well
past the quarter where seams measurably degrade. For 4-second segments use **0.917 s
(22 frames)**, which is the same 21 % the 8 s default gets. Give up the phase-exactness
rather than the runway — 0.33 of an audio step is 8 ms, and too little free footage
after the pinned region is a visible seam.

### 2. The anchored clip has no tag

A keyframe never reaches the tokenizer — by either route — so the carried-in clip gets
**no** `<Video>` / `<Audio>` label. Describe it in prose:

> The target video opens on an exact replay of the closing 1.6 seconds of the preceding
> segment of the same unbroken take and carries straight on from it…

Writing `<Video 1>` for it is an unresolved reference and the segment falls apart.

### And one thing that will catch you out

**Nothing new can happen in the first 1.6 s** of any segment after the first — those
frames are a fixed replay. A beat written for "the one-second mark" lands inside frames
that are already pinned, so it never happens.

---

## Drift: the thing you cannot see at any one join

Every join can be invisible and the take can still fail. Measure segment 6 against
segment 1 rather than against segment 5 and something else shows up:

| chain | segments | ArcFace seg1→last | face L* | per hop |
| --- | --- | --- | --- | --- |
| `kate_rant6` | 6 × 8 s | 0.761 | 42.1 → 33.6 | −0.048 |
| `kate_rant` | 6 | 0.641 | 43.0 → 35.3 | −0.072 |
| `mia_influencer` | 4 | 0.792 | 42.7 → 37.5 | −0.069 |

Three chains, two subjects, two reference images, and all three do the same thing: the
face gets steadily darker and steadily less like itself, about 0.06 of ArcFace cosine
per join. Nothing about it is visible at a cut — adjacent segments are always close.
Extrapolate the per-hop rate and a ten-shot chain lands near 0.35, which is below the
threshold where a face-verification model will still call it the same person.

`tools/face_drift.py <session>` is the other half of `check_joins.py`: it measures
identity, colour, detail and framing against segment 1 instead of against the
neighbour.

**Know the noise floor before believing any of it.** Rendering the *same* segment
twice — same seed, same settings, same everything — does not give the same file: sage
attention, fp16 accumulation and dynamic VRAM loading are all non-deterministic, and
four sampling steps are enough to diverge. Measured, two such renders sit at:

| | between two identical renders |
| --- | --- |
| ArcFace cosine | 0.961 (so identity noise is ±0.039) |
| face dE | 0.38 |

That is the resolution of every number on this page. The drift above — 0.19 of identity
over five hops — is roughly five times it, so the drift is real. But a difference of
0.03 between two settings on one chain each is not a result, it is the same number
twice. Comparing settings needs either a gap bigger than ~0.06, or more seeds.

### It is not a random walk, and that is the useful part

A random walk would wander in either direction and disagree in sign between chains.
This does not. Measure the reference image's own face and the direction is obvious:

| | face L* |
| --- | --- |
| `kate_closeup.png` (the reference) | 45.6 |
| `kate_rant6` segment 1 | 40.1 |
| `kate_rant6` segment 5 | 34.2 |
| `mia-default.png` (the reference) | 58.6 |
| `mia_influencer` segment 1 | 42.9 |
| `mia_influencer` segment 4 | 37.3 |

Segment 1 already renders well below its reference image, and every later segment
slides further the same way. What is happening is **relaxation, not diffusion**:
segment 1 is the only shot rendered from the prompt and the reference alone, so it sits
closest to them; every segment after it is told nothing except "continue from this one",
forgets a little more of the reference, and slides toward whatever the model would have
produced unprompted. Each segment also darkens *within itself* as it free-runs away
from its pinned opening — measurably about 2 L* across segments 2-4 of `kate_rant6` —
and the next segment inherits that decayed ending as its new truth.

That is a ratchet, and a ratchet can be closed.

### `drift_arrest`

The problem is that a chain has no absolute reference. Segment 1 is the only candidate,
so each handoff is shifted back toward segment 1's per-channel latent means before it is
pinned:

```
drift_arrest = 0.5     on H3 Chain Settings (advanced)
```

Two limits are deliberate. It is a **fraction, not a reset** — the previous segment
really did end where it ended and the finished cut keeps those frames, so correcting
the whole error would put a step at every join; a fraction leaves a residual small
enough to hide under the join while still bounding the walk. And each channel's
correction is **clamped** to a fraction of that channel's own spread, for the one case
the measurement cannot distinguish from drift: a shot that is legitimately darker
because the light changed. Real lighting changes are large and get clamped; accumulated
drift is small and passes through.

The signature is taken over each segment's **whole** latent, not its tail, so a
particular pose or expression at the moment of the handoff averages out and only the
shot's standing appearance is compared. Audio is never touched — it does not drift this
way, and shifting its latent detunes the voice.

`drift_arrest` needs `handoff_mode=latent`; there is no latent to correct on the pixel
path.

**What it is actually worth, and what that does not show.** Same six-shot take, three
settings, everything else identical:

| | latent drift from segment 1 | mean face dE | ArcFace identity |
| --- | --- | --- | --- |
| noise floor (two identical renders) | 0.16 | 0.38 | 0.039 |
| `pixel` handoff | 2.049 | 2.21 | 0.813 |
| `latent` handoff | 1.366 | 2.41 | 0.791 |
| `latent` + `drift_arrest 0.5` | **1.078** | 2.60 | 0.794 |

In the model's own representation the ordering is clean and far outside the noise:
each change cuts the drift, and the loop visibly works — the arrest arm's trajectory
goes 0.372 · 0.726 · **0.514** · 0.799 · 1.078, falling back at segment 4 as the
correction catches up, which a chain without it never does.

The face columns show nothing. That is **not** a null result, it is an underpowered
one, and the arithmetic says so in advance: identity falls 0.187 over these five hops,
so a 21 % improvement is 0.039 — exactly the noise floor. One chain per arm cannot
resolve an effect the size of its own error bar. Separating these would need roughly
four to eight chains per arm, which is eight to sixteen GPU-hours on a 12 GB card.

There is also a cost, and it is visible. The correction is a proportional loop with a
one-segment delay and no damping, so it overshoots — and the overshoot scales with the
gain, which is exactly the signature of an under-damped controller:

| face L* at segment 6 (segment 1 ≈ 42.3) | |
| --- | --- |
| no arrest | 43.0 |
| `drift_arrest 0.5` | 45.7 |
| `drift_arrest 0.8` | 46.3 |

Turning the gain up does not help either: 0.5 and 0.8 land at 1.078 and 1.044 of latent
drift, a gap of 0.034 against a noise floor of 0.16. **The loop saturates** — the model
only partly honours a shifted anchor, and pushing harder does not change that.

The overshoot is entirely removable, because `stabilize` does not care what caused a
brightness trend:

| | mean face dE | segment 6 L* |
| --- | --- | --- |
| `pixel` | 2.21 | 42.8 |
| `pixel` + `stabilize` | 1.41 | 43.0 |
| `drift_arrest 0.5` | 2.60 | 45.7 |
| `drift_arrest 0.5` + `stabilize` | **1.14** | 41.3 |

So the two compose cleanly. But 1.14 against 1.41 is a gap of 0.27 on a metric that
wobbles by 0.38, so **`drift_arrest` cannot be shown to add anything on top of
`stabilize`** on this evidence.

The honest summary: it demonstrably steers the latent, it overshoots the picture unless
`stabilize` cleans up after it, and it has never been shown to improve a frame anyone
will watch. It is left **defaulted to 0** and documented rather than removed, because
the latent-space effect is real and may matter on chains far longer than six shots —
but do not turn it on expecting a visible win. `stabilize` is the drift fix that *is*
resolvable, at three to thirteen times its own noise floor.

### `stabilize`, on the join

`drift_arrest` acts on the generation. The other half of the problem is the footage
already on disk, and that needs no GPU at all:

```
stabilize = 1.0        on H3 Chain to Video (advanced)
```

The drift is slow by definition, which is what makes it separable. Each channel's
per-frame mean is smoothed over two seconds; what is left is the drift, with gestures,
blinks and real flicker averaged out. The correction is a **gain** aimed at segment 1's
level — gains preserve black, where an offset would lift the shadows — and because the
curve it comes from is smooth, it is continuous across every join by construction.

Re-joining the three chains above with it on, changing nothing else:

| chain | dE as rendered | dE stabilised | | identity before → after |
| --- | --- | --- | --- | --- |
| `kate_rant6` | 6.39 | **1.34** | −79 % | 0.761 → 0.739 |
| `kate_rant` | 8.11 | **3.97** | −51 % | 0.641 → 0.650 |
| `mia_influencer` | 5.94 | **3.03** | −49 % | 0.792 → 0.762 |

Half to four-fifths of the colour drift, gone for no GPU time. On `kate_rant6` the face
L* goes from 42.1 · 40.0 · 37.6 · 36.2 · 33.6 · 36.1 to 41.9 · 41.6 · 42.6 · 41.8 ·
41.1 · 42.3 — flat — and the peak dE lands at 1.90, under the 2.3 just-noticeable
difference. The two weaker rows are chains whose segments also move a lot *within*
themselves, which the two-second smoothing deliberately does not chase.

It is seam-neutral, measured: the frame-to-frame delta at each cut relative to the
motion around it is unchanged (×1.48 / 2.24 / 0.99 / 1.77 / 1.59 before,
×1.48 / 2.23 / 1.00 / 1.78 / 1.66 after).

**It does not fix identity** — that column barely moves, in either direction, on any of
the three. That is the point, not a shortfall: colour is a property of the pixels and
can be graded afterwards; identity was decided when the frames were sampled. Grading
cannot put it back, so use both — `stabilize` for the look, `drift_arrest` for the face.

(One measurement artefact worth knowing: `face_drift.py`'s `detail` column reads high
on stabilised footage — ×1.70 above — because scaling a darkened image back up scales
its noise too. It is not sharpening.)

### Give every segment a still of itself

The one thing that moved identity, and it needs no feature in this pack — only a
second `LoadImage`.

A photo reference is *out of distribution*. It was shot on a real camera, and every
segment has to translate it into H3's own rendition before it can match it. Segment 1's
own output is not: same lens, same light, same skin, already rendered. So pull a frontal
frame out of `seg_01.mp4`, wire it into `image_1` on every **later** segment, and give
it a line in the prompt:

> `<Picture 2>` is a frame taken from earlier in this same continuous take. It shows
> exactly how `<Subject 1>` looks in this footage — her face, hair, colouring, clothing,
> the lighting on her and the framing of the shot — and the target video must match it
> exactly in all of those respects.

Segment 1 gets neither the image nor the clause; it cannot reference itself, and it is
the shot everything else is being held to.

Across the five-arm comparison, measured against its own control (the same chain with
the latent handoff and nothing else):

| arm | ArcFace seg1→6 | identity lost per hop |
| --- | --- | --- |
| `latent` handoff (control) | 0.791 | 0.042 |
| `+ drift_arrest 0.5` | 0.794 | 0.041 |
| `+ drift_arrest 0.8` | 0.790 | 0.042 |
| **`+ self-reference`** | **0.834** | **0.033** |

It is the only one of the three that moves — +0.043, or 1.1× the noise floor. Read that
honestly: it is the best result in the table and it is *one chain*, sitting barely above
the level where a single chain can tell anything apart at all. It costs about 5 % more
time per segment for the extra reference tokens. Treat it as the most promising lever
found rather than a settled one, and if identity over a long chain is what you care
about, this is the thing to try first.

### What to do about the reference image

The same measurement says something about assets. `mia-default.png` sits at L* 58.6 and
the chain it drives runs at 37-43 — the model is being asked to match a reference 16-21
L* brighter than anything it wants to produce, and loses. **Grade the reference toward
what the model actually renders** and segment 1 starts closer to the prior, which leaves
less distance for the chain to travel. This costs nothing at render time.

---
