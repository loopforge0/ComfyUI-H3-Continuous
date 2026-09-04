"""Drift measured in the model's own representation, straight from the manifest.

``face_drift.py`` answers the question a viewer would ask -- is it still the same
person -- but it pays for a face detector and an ArcFace embedding to do it, and
those are noisy: two renders of the *same* segment with the *same* seed land 0.961
apart in cosine, because sage attention, fp16 accumulation and dynamic VRAM loading
are all non-deterministic.

This measures the same drift with no detector and no model at all. Every segment
records the per-channel mean of its own sampled latent, so the distance between
segment N's signature and segment 1's *is* the drift, in the representation the
model actually works in. On real chains it comes out around twelve times its own
noise floor, against about five for the face metric.

It is also exactly the quantity ``drift_arrest`` acts on -- which cuts both ways.
For comparing *handoff modes* it is the cleanest instrument available. For judging
``drift_arrest`` itself it is partly circular: the correction is applied to the
anchor, the anchor is about a fifth of the next segment's tokens, and those tokens
are inside the signature being measured. So a fifth of any improvement it shows is
mechanical rather than earned. Judge ``drift_arrest`` on ``face_drift.py`` instead,
which samples from a fifth of the way into each clip and therefore never looks at
the pinned region at all.

    python latent_drift.py <session> [<session> ...]

Needs nothing but the session's manifest.json -- no GPU, no ONNX, no video decode.
Sessions rendered before signatures existed are reported as such.
"""

import argparse
import json
import os


def comfy_root():
    """ComfyUI's root directory.

    Installed normally this pack sits in ComfyUI/custom_nodes/<pack>, so the root is two
    levels up. Set COMFYUI_DIR when the checkout lives somewhere else.
    """
    return os.environ.get("COMFYUI_DIR") or os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))



def signatures(session_dir):
    path = os.path.join(session_dir, "manifest.json")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    rows = [(r["index"], r["signature"]["mean"], r.get("drift_correction"))
            for r in manifest["segments"] if r.get("signature")]
    return rows


def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def report(name, rows):
    if not rows:
        print("  %-22s no signatures (rendered before they existed)" % name)
        return None
    base = rows[0][1]
    walk = [distance(mean, base) for _, mean, _ in rows]
    corrections = [c for _, _, c in rows if c]
    print("  %-22s %s" % (name, "  ".join("%5.3f" % v for v in walk)))
    if corrections:
        print("  %-22s corrections applied: %s" % (
            "", "  ".join("%.3f" % c for c in corrections)))
    return walk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--comfy", default=comfy_root(),
                    help="ComfyUI root, if the sessions are given as bare names "
                         "(default: derived, or $COMFYUI_DIR)")
    args = ap.parse_args()

    resolved = []
    for s in args.sessions:
        if os.path.isdir(s):
            resolved.append(s)
        else:
            resolved.append(os.path.join(args.comfy, "output", "h3_continuous", s))

    print("distance of each segment's latent signature from segment 1's")
    print("  %-22s %s" % ("session", "  ".join("seg%-2d" % (i + 1) for i in range(8))))
    walks = {}
    for path in resolved:
        name = os.path.basename(os.path.normpath(path))
        try:
            walks[name] = report(name, signatures(path))
        except (OSError, ValueError, KeyError) as exc:
            print("  %-22s %s" % (name, exc))

    good = {k: v for k, v in walks.items() if v}
    if len(good) >= 2:
        print("\n  (segment 1 is 0.000 by definition in every column above)")
        print("\nfinal drift, least to most")
        for name, walk in sorted(good.items(), key=lambda kv: kv[1][-1]):
            print("  %-22s %6.3f   over %d hops -> %.3f per hop"
                  % (name, walk[-1], max(1, len(walk) - 1),
                     walk[-1] / max(1, len(walk) - 1)))
        print("\nA difference here only means something if it clears the sampler's own")
        print("noise. Render one segment twice with the same seed and measure that gap")
        print("to calibrate; on a 3060 at 4 turbo steps it was 0.16.")


if __name__ == "__main__":
    main()
