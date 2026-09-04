"""Frame maths, audio slicing, and the core MiniMax H3 nodes this pack drives.

Nothing here reimplements H3 conditioning. The chain calls the core nodes
(``MiniMaxH3ReferenceToVideo``, ``MiniMaxH3AddGuide``) and the core sampler, so a
ComfyUI update carries straight through instead of silently diverging.
"""

import logging

import torch

FPS = 24
AUDIO_LATENT_FPS = 40

try:
    from comfy_extras.nodes_minimax_h3 import (  # noqa: F401
        MiniMaxH3AddGuide,
        MiniMaxH3ReferenceToVideo,
        video_latent_t,
    )
except ImportError as exc:  # pragma: no cover - install-time guard
    raise ImportError(
        "ComfyUI-H3-Continuous needs a ComfyUI with MiniMax H3 support "
        "(comfy_extras/nodes_minimax_h3.py, ComfyUI 0.34.0 or newer). "
        "Update ComfyUI and restart."
    ) from exc

# One t-unit of the packed sequence per audio latent frame, 5/3 per pixel frame.
# The guide's audio and its picture have to be placed on that shared axis, which is
# the only reason this constant leaves the model file.
from comfy.ldm.minimax.model import FRAME_RESCALE  # noqa: E402


def log(msg, *args):
    logging.info("[H3 Continuous] " + msg, *args)


def generation_length(frames):
    """Snap a frame count UP onto H3's 17k+5 grid.

    H3 only generates 5, 22, 39, ... 124, 141, ... frames. The trained range is
    124-362 (about 5-15 s at 24 fps); shorter and longer both work but degrade.
    """
    frames = max(5, int(round(frames)))
    while frames % 17 != 5:
        frames += 1
    return frames


def guide_length(frames):
    """Snap a guide-clip frame count DOWN onto the same grid.

    ``MiniMaxH3AddGuide`` truncates guide clips to 17k+5 internally. Doing it here
    too means the handoff length we record in the manifest is the length actually
    anchored, so the join arithmetic and the anchor agree to the frame.
    """
    frames = int(round(frames))
    if frames < 5:
        return 0
    while frames % 17 != 5:
        frames -= 1
    return frames


def seconds_to_frames(seconds):
    return int(round(float(seconds) * FPS))


def audio_latent_frames(frames):
    """Audio latent steps covering ``frames`` pixel frames.

    H3 runs its audio latent at 40 Hz against 24 fps video. This is the same
    rounding ``_empty_av_latent`` uses to size a segment's audio stream, so a tail
    cut with it lines up with the tail of the stream it came from.
    """
    return int(round(float(frames) / FPS * AUDIO_LATENT_FPS))


def ordered_autogrow(values):
    """Autogrow inputs arrive as ``{"image_0": t, "image_3": t}``; return them in
    ordinal order, dropping unconnected slots but keeping the ordinal for pairing."""
    out = []
    for name, value in (values or {}).items():
        if value is None:
            continue
        tail = name.rsplit("_", 1)[-1]
        try:
            ordinal = int(tail)
        except ValueError:
            ordinal = 0
        out.append((ordinal, value))
    out.sort(key=lambda kv: kv[0])
    return out


def slice_audio(audio, start_seconds, duration_seconds):
    """Cut an AUDIO dict to a window, padding with silence if it falls short.

    The reference build did this with ffmpeg and hit the bug this function exists to
    make impossible: an anchor clip whose audio outran its video by six seconds.
    ``AddGuide`` crops audio against the *target's* remaining duration, never against
    the image clip beside it, so an over-long soundtrack is anchored in full and a
    segment with no dialogue of its own replays the previous segment's lines.
    """
    if audio is None:
        return None
    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    start = max(0, int(round(start_seconds * rate)))
    want = max(1, int(round(duration_seconds * rate)))
    cut = waveform[..., start:start + want].clone()
    if cut.shape[-1] < want:
        pad = torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype)
        cut = torch.cat([cut, pad], dim=-1)
    return {"waveform": cut, "sample_rate": rate}
