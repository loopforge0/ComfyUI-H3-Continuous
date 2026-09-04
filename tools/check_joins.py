"""Check a rendered session: are the joins actually joins?

    python tools/check_joins.py my_chain

Three things get measured, in increasing order of how much they tell you:

1. **Frame and audio bookkeeping.** Segment lengths against the manifest, each
   segment's soundtrack against its own picture length, handoff clips at exactly the
   frame count they claim. Cheap, and catches the failures that silently corrupt a
   chain rather than making it look wrong.

2. **Anchor fidelity.** Segment N's last frames against segment N+1's first frames.
   They should be near-identical; the floor to compare against is two unrelated spans
   of the same footage, printed beside it.

3. **Seam visibility.** The number that actually predicts a pop. Every adjacent frame
   pair in the joined cut is measured, and each join is placed in that distribution.
   A join in the normal range is a join you cannot see. A join in the bottom few
   percent is a bump, however good its raw PSNR looked.

Run it from the pack directory with ComfyUI's Python, or pass --comfy.
"""

import argparse
import json
import os
import sys


def psnr(a, b):
    import torch
    mse = float(((a - b) ** 2).mean())
    return float("inf") if mse == 0 else 10.0 * float(torch.log10(torch.tensor(1.0 / mse)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--comfy", default=None,
                    help="Path to the ComfyUI directory (default: three levels up)")
    ap.add_argument("--output", default=None,
                    help="ComfyUI output directory (default: <comfy>/output)")
    a = ap.parse_args()

    comfy = a.comfy or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    sys.path.insert(0, comfy)
    try:
        from comfy_api.input_impl import VideoFromFile
    except ImportError:
        raise SystemExit("could not import ComfyUI from %s -- pass --comfy" % comfy)

    out_dir = a.output or os.path.join(comfy, "output")
    directory = os.path.join(out_dir, "h3_continuous", a.session)
    manifest_path = os.path.join(directory, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit("no session at %s" % directory)

    def load(path):
        return VideoFromFile(path).get_components()

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    records = manifest["segments"]
    fps = manifest.get("fps", 24)
    print("session %r: %d segments at %dx%d\n"
          % (a.session, len(records), manifest.get("width", 0), manifest.get("height", 0)))

    ok = True
    frames, audio = {}, {}
    print("--- segments ---")
    for r in records:
        c = load(os.path.join(directory, r["file"]))
        frames[r["index"]], audio[r["index"]] = c.images, c.audio
        n = frames[r["index"]].shape[0]
        note = "" if n == r["length"] else "  <-- disagrees with the manifest (%d)" % r["length"]
        ok &= n == r["length"]
        print("  %2d  %3d frames (%.2fs)  handoff %2d  seed %d%s"
              % (r["index"] + 1, n, n / float(fps), r["handoff"], r["seed"], note))

    print("\n--- audio against picture ---")
    for r in records:
        a_ = audio[r["index"]]
        if a_ is None:
            print("  %2d  NO AUDIO TRACK" % (r["index"] + 1)); ok = False; continue
        seconds = a_["waveform"].shape[-1] / float(a_["sample_rate"])
        drift = seconds - r["length"] / float(fps)
        flag = "" if abs(drift) <= 0.15 else "  <-- more than a few frames out"
        ok &= abs(drift) <= 0.15
        print("  %2d  %.3fs audio, %+.3fs against picture%s"
              % (r["index"] + 1, seconds, drift, flag))

    print("\n--- anchor fidelity (segment end vs next segment's opening) ---")
    for i in range(len(records) - 1):
        h = records[i]["handoff"]
        if not h:
            continue
        d = psnr(frames[i][-h:], frames[i + 1][:h])
        floor = psnr(frames[i][:h], frames[i + 1][:h])
        print("  %d -> %d : %5.1f dB   (unrelated-span floor %.1f dB)" % (i + 1, i + 2, d, floor))

    joined = os.path.join(directory, "%s.mp4" % a.session)
    if os.path.exists(joined):
        print("\n--- seam visibility in the joined cut ---")
        j = load(joined).images
        expected = (sum(r["length"] for r in records)
                    - sum(r["handoff"] for r in records[:-1]))
        flag = "" if j.shape[0] == expected else "  <-- expected %d" % expected
        ok &= j.shape[0] == expected
        print("  %d frames (%.2fs)%s" % (j.shape[0], j.shape[0] / float(fps), flag))
        deltas = [psnr(j[k], j[k + 1]) for k in range(j.shape[0] - 1)]

        # Compare each join against the frames *around* it, not against the whole cut.
        # Whole-cut percentiles are confounded by how much the footage moves: a calm shot
        # has small frame-to-frame deltas, so any join looks terrible against its own
        # median, while a busy shot hides the same join. Measured on two chains of
        # identical join quality, whole-cut percentiles read 1-3% and 13-31% purely
        # because one was a slow misty walk and the other a busy street. The local
        # window has the same motion as the join itself, so the gap is comparable
        # across content.
        window = 24  # a second either side
        cut = 0
        for i, r in enumerate(records[:-1]):
            cut += r["length"] - (records[i - 1]["handoff"] if i else 0)
            if not 0 < cut < len(deltas):
                continue
            d = deltas[cut - 1]
            near = ([deltas[k] for k in range(max(0, cut - 1 - window), cut - 1)] +
                    [deltas[k] for k in range(cut, min(len(deltas), cut + window))])
            local = sorted(near)[len(near) // 2]
            gap = d - local
            # Calibrated on a seven-segment 864x480 chain that reads as one unbroken
            # take on screen. Its six joins spanned -0.6 to -5.6 dB against local
            # motion, mean -2.9. So a join being a few dB below its neighbourhood is
            # normal and invisible; past about -6 dB you are outside anything that
            # reference build ever did.
            if gap >= -3.0:
                verdict = "sits in its own neighbourhood"
            elif gap >= -6.0:
                verdict = "normal for a chain (reference: -0.6 to -5.6 dB)"
            else:
                verdict = "bigger than any join in the reference build -- look at it"
            print("  join %d->%d at frame %4d: %5.1f dB vs %5.1f dB locally "
                  "(%+.1f dB)  [%s]" % (i + 1, i + 2, cut, d, local, gap, verdict))
            ok &= gap >= -6.0
        print("  (whole cut: median %.1f dB, worst %.1f dB -- context only, the local "
              "comparison above is the one that means anything)"
              % (sorted(deltas)[len(deltas) // 2], min(deltas)))
    else:
        print("\n  no joined cut yet -- run H3 Chain to Video")

    print("\n%s" % ("looks good" if ok else "something above needs a look"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
