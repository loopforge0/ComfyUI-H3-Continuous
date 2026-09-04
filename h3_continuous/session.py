"""On-disk state for one chain.

A chain is not a single render, it is N renders that each take minutes. Keeping
them in a session folder is what makes the pack usable:

* **Resume.** Re-queueing after a crash, an OOM or a Ctrl-C reuses every segment
  whose settings did not change and only renders the rest.
* **Edit one segment.** Change segment 4's prompt and segments 1-3 are reused;
  4 onward re-render, because each cache key folds in the key of the segment
  before it.
* **Repair.** The repair node needs the previous segment's tail and the tail the
  *next* segment already continued from. Both are on disk.

Everything lives under ``output/h3_continuous/<session>/``.
"""

import hashlib
import json
import os
import re

import folder_paths

ROOT = "h3_continuous"
MANIFEST_VERSION = 2


def sanitize(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    name = re.sub(r"\.{2,}", "_", name)  # never let a session name walk up a directory
    name = name.strip("._") or "session"
    return name[:64]


def digest(value):
    """Stable short digest of anything a segment input can be.

    Tensors are sampled rather than hashed whole: a 2048px reference image is
    50 MB and this runs for every segment on every queue. Shape, dtype and a
    strided slice change whenever the picture does, which is all a cache key needs.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        t = value
        step = max(1, t.shape[0] // 8) if t.ndim > 0 and t.shape[0] > 8 else 1
        sample = t[::step].contiguous().cpu().numpy().tobytes()
        h = hashlib.sha1()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(sample)
        return h.hexdigest()[:16]
    if isinstance(value, dict) and "waveform" in value:
        return "audio:%s:%s" % (digest(value["waveform"]), value.get("sample_rate"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(digest(v) for v in value) + "]"
    return hashlib.sha1(repr(value).encode("utf-8", "replace")).hexdigest()[:16]


def sampler_digest(sampler):
    """Stable fingerprint of a SAMPLER object.

    ``repr()`` will not do: it carries a memory address, so it changes on every
    queue and would invalidate the whole cache every run.
    """
    fn = getattr(sampler, "sampler_function", None)
    name = getattr(fn, "__name__", None) or type(sampler).__name__
    extra = getattr(sampler, "extra_options", None) or {}
    return hashlib.sha1((name + repr(sorted(extra.keys()))).encode()).hexdigest()[:16]


def model_digest(model):
    """Fingerprint the model patcher well enough to notice a LoRA change.

    Weights themselves are never hashed -- far too slow. This sees the model class
    and every patch key with its strengths, which covers adding, removing or
    re-weighting a LoRA. It does *not* see a swapped checkpoint file of the same
    class: rename the session or turn ``resume`` off when you change one.
    """
    try:
        parts = [type(getattr(model, "model", model)).__name__]
        for key, patches in sorted(getattr(model, "patches", {}).items()):
            strengths = []
            for patch in patches:
                strengths += [round(float(v), 6) for v in patch[:1] + patch[2:3]
                              if isinstance(v, (int, float))]
            parts.append("%s:%s" % (key, strengths))
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    except Exception:  # never let a cache key break a render
        return "unknown"


def segment_key(settings, segment, handoff, previous_key):
    """Fingerprint of everything that changes what this segment renders.

    ``previous_key`` is in the hash on purpose: a segment is a continuation of the
    one before it, so editing segment 2 must invalidate 3..N even though their own
    prompts are untouched.
    """
    payload = {
        "prev": previous_key,
        "prompt": segment["prompt"].strip(),
        "length": segment["length"],
        "handoff": handoff,
        "seed": segment["resolved_seed"],
        "width": settings["width"],
        "height": settings["height"],
        "ref_image_size": settings["ref_image_size"],
        "sigmas": digest(settings["sigmas"]),
        "sampler": sampler_digest(settings["sampler"]),
        "model": model_digest(settings["model"]),
        "images": [digest(t) for t in segment["images"]],
        "videos": [digest(t) for _, t in segment["videos"]],
        "video_audios": [digest(a) for _, a in segment["video_audios"]],
        "audios": [digest(a) for a in segment["audios"]],
    }
    # Only a segment that actually consumes an anchor cares how that anchor was
    # built, and the key is left byte-identical when it does not. That keeps the
    # first segment of every pre-existing session cached across this upgrade -- the
    # segment renders the same either way, and its own outgoing tail is now written
    # in both forms regardless. Segments 2..N do change, because their handoff
    # genuinely did.
    if previous_key and settings.get("handoff_mode"):
        payload["handoff_mode"] = settings["handoff_mode"]
    if previous_key and settings.get("drift_arrest"):
        payload["drift_arrest"] = round(float(settings["drift_arrest"]), 4)
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def signature_to_record(signature):
    """A latent signature as plain JSON -- 24 means and 24 spreads."""
    if not signature:
        return None
    return {"mean": [round(float(v), 6) for v in signature["mean"]],
            "std": [round(float(v), 6) for v in signature["std"]]}


def signature_from_record(record):
    """The inverse, for a segment being reused from cache.

    Returns None for a session rendered before signatures existed, which simply
    means drift arrest has no reference to work from until something re-renders.
    """
    import torch

    payload = (record or {}).get("signature")
    if not payload:
        return None
    return {"mean": torch.tensor(payload["mean"]), "std": torch.tensor(payload["std"])}


class Session:
    def __init__(self, name):
        self.name = sanitize(name)
        self.dir = os.path.join(folder_paths.get_output_directory(), ROOT, self.name)
        os.makedirs(self.dir, exist_ok=True)

    # --- paths -----------------------------------------------------------------
    def segment_path(self, index):
        return os.path.join(self.dir, "seg_%02d.mp4" % (index + 1))

    def tail_path(self, index):
        """The clip segment index+1 opens on. Kept even when a segment is repaired:
        it is the thing downstream already continues from."""
        return os.path.join(self.dir, "seg_%02d.tail.mp4" % (index + 1))

    def joined_path(self):
        return os.path.join(self.dir, "%s.mp4" % self.name)

    @property
    def manifest_path(self):
        return os.path.join(self.dir, "manifest.json")

    # --- manifest --------------------------------------------------------------
    def load(self):
        try:
            with open(self.manifest_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        if data.get("version") != MANIFEST_VERSION:
            return None
        return data

    def save(self, records, extra=None):
        data = {"version": MANIFEST_VERSION, "session": self.name, "segments": records}
        data.update(extra or {})
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.manifest_path)
        return data

    def cached(self, manifest, index, key, needs_tail):
        """True when segment ``index`` on disk is still exactly what we would render."""
        if not manifest:
            return False
        segments = manifest.get("segments", [])
        if index >= len(segments):
            return False
        record = segments[index]
        if record.get("key") != key:
            return False
        if not os.path.exists(self.segment_path(index)):
            return False
        if needs_tail and not os.path.exists(self.tail_path(index)):
            return False
        return True
