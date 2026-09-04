"""Measure how far a chain drifts from its own opening.

``check_joins.py`` answers "is the seam visible". This answers the other half:
after N segments, is it still the same face, the same exposure, the same lens?

Drift is a random walk. Every segment is conditioned on the one before it, so a
small error in segment 2 is the starting point for segment 3. What that looks
like on screen is the face slowly becoming someone else's, the picture warming or
washing out, and detail dissolving -- none of it visible at any single join.

Four measurements, all against **segment 1** rather than against the neighbour,
because a chain can drift arbitrarily far while every adjacent pair stays close:

  identity   ArcFace (buffalo_l w600k_r50) cosine on SCRFD-aligned crops. The
             standard face-verification metric: >0.6 is "same person" with a
             clear margin, 0.4-0.6 is "probably", <0.3 is a different face.
  colour     mean CIE-Lab of the face region. dE is plain Euclidean distance in
             Lab, so ~2.3 is the classic just-noticeable difference.
  detail     variance of the Laplacian on the aligned 112px crop, as a ratio to
             segment 1. Below 1.0 is softening; the usual failure.
  framing    face box width as a fraction of frame width, and centre offset.
             Catches the shot slowly pushing in or the head walking out of frame.

Usage:
    python face_drift.py <session-dir> [--frames 5] [--json out.json]
    python face_drift.py <dir-a> <dir-b> ...     # compare arms side by side
"""

import argparse
import glob
import json
import os
import sys

import av
import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
# Set H3_FACE_MODELS to a folder holding det_10g.onnx and w600k_r50.onnx
# (insightface's buffalo_l pair). They are not vendored: 190 MB of weights do not
# belong in a node pack that does not otherwise need them.
MODEL_DIR = os.environ.get("H3_FACE_MODELS", os.path.join(HERE, "face_models"))

# insightface's 112x112 alignment template. Every ArcFace weight ever released
# was trained on crops warped onto exactly these five points.
ARCFACE_DST = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def _distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


class SCRFD:
    """det_10g: three strides, two anchors each, outputs ordered score/bbox/kps."""

    def __init__(self, path, providers):
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self._centers = {}

    def _anchor_centers(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._centers:
            ys, xs = np.mgrid[:h, :w]
            centers = np.stack([xs, ys], axis=-1).astype(np.float32) * stride
            centers = centers.reshape(-1, 2)
            if self.num_anchors > 1:
                centers = np.repeat(centers, self.num_anchors, axis=0)
            self._centers[key] = centers
        return self._centers[key]

    def detect(self, bgr, size=640, thresh=0.5):
        h0, w0 = bgr.shape[:2]
        scale = min(size / w0, size / h0)
        resized = cv2.resize(bgr, (int(round(w0 * scale)), int(round(h0 * scale))))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (size, size), (127.5,) * 3,
                                     swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})

        boxes, kpss, scores = [], [], []
        for i, stride in enumerate(self.strides):
            score = outs[i].reshape(-1)
            bbox = outs[i + len(self.strides)].reshape(-1, 4) * stride
            kps = outs[i + len(self.strides) * 2].reshape(-1, 10) * stride
            grid = size // stride
            centers = self._anchor_centers(grid, grid, stride)
            keep = score >= thresh
            if not keep.any():
                continue
            boxes.append(_distance2bbox(centers[keep], bbox[keep]))
            kpss.append(_distance2kps(centers[keep], kps[keep]).reshape(-1, 5, 2))
            scores.append(score[keep])
        if not boxes:
            return None, None
        boxes = np.concatenate(boxes) / scale
        kpss = np.concatenate(kpss) / scale
        scores = np.concatenate(scores)

        idx = cv2.dnn.NMSBoxes(
            [[float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
             for b in boxes], scores.tolist(), thresh, 0.4)
        if len(idx) == 0:
            return None, None
        idx = np.array(idx).reshape(-1)
        # The talking head is the biggest face in frame, never a background one.
        areas = (boxes[idx, 2] - boxes[idx, 0]) * (boxes[idx, 3] - boxes[idx, 1])
        best = idx[int(np.argmax(areas))]
        return boxes[best], kpss[best]


def align(bgr, kps):
    """Warp to the 112x112 ArcFace template using the 5 landmarks."""
    matrix, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_DST,
                                            method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0)


class ArcFace:
    def __init__(self, path, providers):
        self.session = ort.InferenceSession(path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, crop112):
        blob = cv2.dnn.blobFromImage(crop112, 1.0 / 127.5, (112, 112), (127.5,) * 3,
                                     swapRB=True)
        vec = self.session.run(None, {self.input_name: blob})[0][0]
        return vec / (np.linalg.norm(vec) + 1e-9)


def sample_frames(path, count, skip_head=True):
    """``count`` frames spread across the clip, skipping the replayed opening.

    The first frames of every segment after the first are a regeneration of the
    previous segment's ending, so they measure the *previous* segment. Starting a
    fifth of the way in keeps each sample inside footage that segment authored.
    """
    frames = []
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            frames.append(frame)
    if not frames:
        return []
    n = len(frames)
    start = n // 5 if skip_head else 0
    picks = np.linspace(start, n - 1, count).round().astype(int)
    return [frames[i].to_ndarray(format="bgr24") for i in sorted(set(picks.tolist()))]


def measure_clip(path, det, rec, frames_per_clip):
    rows = []
    for bgr in sample_frames(path, frames_per_clip):
        box, kps = det.detect(bgr)
        if box is None:
            continue
        crop = align(bgr, kps)
        if crop is None:
            continue
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        # OpenCV packs 8-bit Lab as L*255/100, a+128, b+128.
        lab_mean = np.array([lab[..., 0].mean() * 100.0 / 255.0,
                             lab[..., 1].mean() - 128.0,
                             lab[..., 2].mean() - 128.0])
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rows.append({
            "embedding": rec.embed(crop),
            "lab": lab_mean,
            "detail": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            "face_w": float(box[2] - box[0]) / bgr.shape[1],
            "face_cx": float(box[0] + box[2]) / 2.0 / bgr.shape[1],
            "face_cy": float(box[1] + box[3]) / 2.0 / bgr.shape[0],
        })
    if not rows:
        return None
    embeddings = np.stack([r["embedding"] for r in rows])
    mean_embedding = embeddings.mean(axis=0)
    return {
        "n": len(rows),
        "embedding": mean_embedding / (np.linalg.norm(mean_embedding) + 1e-9),
        "lab": np.stack([r["lab"] for r in rows]).mean(axis=0),
        "detail": float(np.mean([r["detail"] for r in rows])),
        "face_w": float(np.mean([r["face_w"] for r in rows])),
        "face_cx": float(np.mean([r["face_cx"] for r in rows])),
        "face_cy": float(np.mean([r["face_cy"] for r in rows])),
    }


def analyse(session_dir, frames_per_clip=5, providers=None):
    providers = providers or ["CPUExecutionProvider"]
    det = SCRFD(os.path.join(MODEL_DIR, "det_10g.onnx"), providers)
    rec = ArcFace(os.path.join(MODEL_DIR, "w600k_r50.onnx"), providers)

    paths = sorted(glob.glob(os.path.join(session_dir, "seg_*.mp4")))
    paths = [p for p in paths if ".tail." not in p and ".replaced." not in p]
    if not paths:
        raise FileNotFoundError("no seg_NN.mp4 in %s" % session_dir)

    clips = []
    for path in paths:
        stats = measure_clip(path, det, rec, frames_per_clip)
        if stats is None:
            print("  !! no face found in %s" % os.path.basename(path), file=sys.stderr)
            continue
        stats["file"] = os.path.basename(path)
        clips.append(stats)
    if not clips:
        raise RuntimeError("no faces found anywhere in %s" % session_dir)

    first = clips[0]
    for clip in clips:
        clip["identity"] = float(np.dot(first["embedding"], clip["embedding"]))
        clip["dE"] = float(np.linalg.norm(clip["lab"] - first["lab"]))
        clip["detail_ratio"] = clip["detail"] / max(first["detail"], 1e-6)
        clip["framing"] = clip["face_w"] / max(first["face_w"], 1e-6)
        clip["shift"] = float(np.hypot(clip["face_cx"] - first["face_cx"],
                                       clip["face_cy"] - first["face_cy"]))
    return clips


def report(name, clips):
    print("\n%s" % name)
    print("  seg   identity   dE(Lab)   detail   framing   shift    L*    a*    b*")
    for i, c in enumerate(clips):
        print("  %3d   %8.4f   %7.2f   %6.2f   %7.3f   %5.3f  %5.1f %5.1f %5.1f"
              % (i + 1, c["identity"], c["dE"], c["detail_ratio"], c["framing"],
                 c["shift"], c["lab"][0], c["lab"][1], c["lab"][2]))
    last = clips[-1]
    print("  first->last:  identity %.4f   dE %.2f   detail x%.2f   framing x%.3f"
          % (last["identity"], last["dE"], last["detail_ratio"], last["framing"]))
    # Per-hop rate is the number that predicts a longer chain: a chain twice as
    # long drifts about twice as far, so quoting only the endpoint hides how a
    # 6-segment result scales to 20.
    hops = max(1, len(clips) - 1)
    print("  per hop:      identity -%.4f   dE +%.2f"
          % ((1.0 - last["identity"]) / hops, last["dE"] / hops))
    return {"identity": last["identity"], "dE": last["dE"],
            "detail": last["detail_ratio"], "framing": last["framing"],
            "hops": hops, "segments": len(clips)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--frames", type=int, default=5,
                    help="frames sampled per segment (default 5)")
    ap.add_argument("--json", help="write the full per-segment table here")
    ap.add_argument("--gpu", action="store_true",
                    help="run the ONNX models on CUDA if a provider is available")
    args = ap.parse_args()

    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu
                 else ["CPUExecutionProvider"])

    summary, payload = {}, {}
    for session in args.sessions:
        clips = analyse(session, args.frames, providers)
        name = os.path.basename(os.path.normpath(session))
        summary[name] = report(name, clips)
        payload[name] = [
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in c.items() if k != "embedding"} for c in clips]

    if len(summary) > 1:
        print("\nside by side (first -> last)")
        print("  %-28s %9s %8s %8s %9s" % ("arm", "identity", "dE", "detail", "per-hop"))
        for name, s in summary.items():
            print("  %-28s %9.4f %8.2f %8.2f %9.4f"
                  % (name, s["identity"], s["dE"], s["detail"],
                     (1.0 - s["identity"]) / s["hops"]))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print("\nwrote %s" % args.json)


if __name__ == "__main__":
    main()
