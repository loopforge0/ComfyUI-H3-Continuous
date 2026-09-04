"""Self-checks for the latent handoff, runnable without a GPU or any model.

The handoff is frame arithmetic on a latent, and frame arithmetic is exactly the
kind of thing that is silently three frames wrong for a month. These build synthetic
AV latents of every legal shape and assert the slice lands where it should.

    python tools/test_latent_handoff.py --comfy /path/to/ComfyUI
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=os.environ.get("H3_COMFY_ROOT"),
                    help="ComfyUI checkout, for comfy.nested_tensor and the H3 constants")
    args = ap.parse_args()
    if args.comfy:
        sys.path.insert(0, args.comfy)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    import torch
    import comfy.nested_tensor
    from h3_continuous.common import (
        AUDIO_LATENT_FPS, FPS, audio_latent_frames, generation_length, guide_length,
        video_latent_t,
    )
    from h3_continuous.engine import (
        anchor_frames, arrest_drift, latent_signature, latent_tail, picture_only,
    )

    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)
            print("  FAIL %s" % message)

    def make(frames, h=30, w=54, seed=0):
        """A synthetic H3 AV latent for `frames` pixel frames."""
        g = torch.Generator().manual_seed(seed)
        t = video_latent_t(frames)
        at = int(round(frames / float(FPS) * AUDIO_LATENT_FPS))
        video = torch.randn(1, 24, t, h, w, generator=g)
        audio = torch.randn(1, 32, 2, at, generator=g)
        return comfy.nested_tensor.NestedTensor((video, audio))

    print("grid arithmetic")
    for k in range(0, 22):
        frames = 17 * k + 5
        check(generation_length(frames) == frames, "generation_length(%d)" % frames)
        check(guide_length(frames) == frames, "guide_length(%d)" % frames)
        t = video_latent_t(frames)
        check(t == 5 * k + 2, "video_latent_t(%d) == %d, got %d" % (frames, 5 * k + 2, t))
    check(guide_length(38) == 22, "guide_length rounds 38 down to 22")
    check(generation_length(96) == 107, "generation_length rounds 96 up to 107")
    check(guide_length(4) == 0, "a handoff under 5 frames is no handoff")

    print("tail slicing, every legal segment/handoff pair")
    for k in range(1, 14):
        segment = 17 * k + 5
        samples = make(segment)
        for m in range(0, k):
            handoff = 17 * m + 5
            anchor = latent_tail(samples, handoff)
            got = anchor_frames(anchor)
            check(got == handoff,
                  "seg %d handoff %d -> anchor_frames %d" % (segment, handoff, got))
            # The slice must be the literal tail of the source, not a copy of
            # anything else: compare against the source's own last tokens.
            source = samples.unbind()[0]
            tokens = video_latent_t(handoff)
            check(torch.equal(anchor["video_latent"], source[:1, :, source.shape[2] - tokens:]),
                  "seg %d handoff %d slice is the tail" % (segment, handoff))
            rt = audio_latent_frames(handoff)
            check(anchor["audio_latent"].shape[-1] == rt,
                  "seg %d handoff %d audio steps %d != %d"
                  % (segment, handoff, anchor["audio_latent"].shape[-1], rt))

    print("no handoff, no anchor")
    check(latent_tail(make(107), 0) is None, "handoff 0 yields no anchor")
    check(anchor_frames(None) == 0, "anchor_frames(None)")

    print("a handoff longer than the segment is refused")
    try:
        latent_tail(make(22), 39)
        failures.append("39-frame handoff out of a 22-frame segment should raise")
        print("  FAIL oversized handoff was accepted")
    except ValueError:
        pass

    print("picture_only drops the soundtrack and keeps the picture")
    anchor = latent_tail(make(107), 22)
    bare = picture_only(anchor)
    check(bare["audio_latent"] is None, "picture_only clears audio_latent")
    check(torch.equal(bare["video_latent"], anchor["video_latent"]),
          "picture_only keeps the video latent untouched")
    pixels = {"images": torch.zeros(22, 480, 864, 3), "audio": {"waveform": None}}
    check(picture_only(pixels)["audio"] is None, "picture_only works on a pixel anchor")
    check(anchor_frames(pixels) == 22, "anchor_frames counts pixel frames")

    print("drift arrest")
    reference = latent_signature(make(107, seed=1))
    current = latent_signature(make(107, seed=2))
    check(tuple(reference["mean"].shape) == (24,), "signature has one mean per channel")
    check(tuple(reference["std"].shape) == (24,), "signature has one std per channel")

    anchor = latent_tail(make(107, seed=2), 22)
    same = arrest_drift(anchor, reference, reference, 0.5)
    check(float((same["video_latent"] - anchor["video_latent"]).abs().max()) == 0.0,
          "no error means no correction")
    check(arrest_drift(anchor, reference, current, 0.0) is anchor,
          "strength 0 is a no-op")
    check(arrest_drift(anchor, None, current, 0.5) is anchor,
          "no reference is a no-op")

    half = arrest_drift(anchor, reference, current, 0.5)
    full = arrest_drift(anchor, reference, current, 1.0)
    d_half = (half["video_latent"] - anchor["video_latent"]).mean(dim=(0, 2, 3, 4))
    d_full = (full["video_latent"] - anchor["video_latent"]).mean(dim=(0, 2, 3, 4))
    check(torch.allclose(d_half * 2, d_full, atol=1e-5),
          "the correction is linear in strength")
    error = reference["mean"] - current["mean"]
    check(bool(((d_full.sign() == error.sign()) | (error.abs() < 1e-6)).all()),
          "the correction moves toward the reference, not away from it")

    # The clamp has to bite on a large error, or a real lighting change gets fought.
    wild = {"mean": reference["mean"] + 99.0, "std": reference["std"]}
    clamped = arrest_drift(anchor, wild, current, 1.0)
    shift = (clamped["video_latent"] - anchor["video_latent"]).mean(dim=(0, 2, 3, 4))
    check(bool((shift.abs() <= wild["std"] * 0.35 + 1e-4).all()),
          "a large error is clamped to 0.35 sigma")

    check(arrest_drift(pixels, reference, current, 0.5) is pixels,
          "a pixel anchor has no latent to correct and is passed through")

    print()
    if failures:
        print("%d FAILURE(S)" % len(failures))
        return 1
    print("all latent-handoff self-checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
