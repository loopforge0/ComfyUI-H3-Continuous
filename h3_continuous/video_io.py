"""Reading and writing the session's clips.

Segments are held on disk as ordinary mp4s so they can be scrubbed, dragged into
an editor, or handed back to the repair node. They are written at CRF 10 -- near
enough to lossless that a re-anchored tail measures the same ~30 dB as a plain VAE
round trip, which is the accuracy the seamless join depends on.
"""

import logging
import math
import os
from fractions import Fraction

import av
import comfy.utils
import numpy as np
import torch

from comfy_api.input_impl import VideoFromComponents, VideoFromFile
from comfy_api.util import VideoCodec, VideoComponents, VideoContainer

from .common import FPS

SEGMENT_CRF = 10.0


def latent_path(path):
    """Sidecar holding a handoff's latent, next to the mp4 of the same clip."""
    return os.path.splitext(path)[0] + ".latent.safetensors"


def save_latent_tail(path, anchor):
    """Persist a latent handoff so a resumed run can pin it without a VAE.

    Written as fp16: this is a 39-frame slice of a diffusion latent, and half
    precision costs about 1e-4 of its own standard deviation -- three orders of
    magnitude below what a VAE round trip costs, which is the thing the latent
    handoff exists to avoid. At 480x864 it is 0.9 MB a segment.
    """
    if anchor is None or anchor.get("video_latent") is None:
        return None
    target = latent_path(path)
    payload = {"video": anchor["video_latent"].contiguous().cpu().half()}
    audio = anchor.get("audio_latent")
    if audio is not None:
        payload["audio"] = audio.contiguous().cpu().half()
    tmp = target + ".tmp"
    comfy.utils.save_torch_file(payload, tmp)
    os.replace(tmp, target)
    return target


def load_latent_tail(path):
    """The latent handoff for a clip, or None if this session predates them."""
    target = latent_path(path)
    if not os.path.exists(target):
        return None
    try:
        payload = comfy.utils.load_torch_file(target, safe_load=True)
    except Exception as exc:  # a truncated sidecar must not kill a whole chain
        logging.warning("[H3 Continuous] could not read %s (%s); falling back to the "
                        "pixel handoff", os.path.basename(target), exc)
        return None
    video = payload.get("video")
    if video is None:
        return None
    return {"video_latent": video.float(),
            "audio_latent": payload["audio"].float() if "audio" in payload else None}


def save_clip(path, images, audio, fps=FPS, crf=SEGMENT_CRF, exact_audio=False):
    """Write a clip. ``exact_audio`` adds a sidecar WAV -- see ``_wav_path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    components = VideoComponents(images=images, audio=audio, frame_rate=Fraction(fps))
    tmp = path + ".tmp.mp4"
    VideoFromComponents(components).save_to(
        tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, path)
    if exact_audio and audio is not None:
        save_wav(_wav_path(path), audio)
    return path


def _wav_path(path):
    """Sidecar for a handoff clip's soundtrack.

    AAC is not sample-exact on the way back in: the decoder hands you the encoder's
    priming samples, about 20-30 ms of them. That is a fifth of an audio latent frame
    of slip on every anchor, for no reason -- the tail is small, so keep it as
    float PCM and read that instead.
    """
    return os.path.splitext(path)[0] + ".wav"


def save_wav(path, audio):
    waveform = audio["waveform"][0].contiguous().cpu().float().numpy()
    rate = int(audio["sample_rate"])
    layout = {1: "mono", 2: "stereo", 6: "5.1"}.get(waveform.shape[0], "stereo")
    with av.open(path, "w", format="wav") as container:
        stream = container.add_stream("pcm_f32le", rate=rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(waveform, format="fltp", layout=layout)
        frame.sample_rate = rate
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def load_clip(path, prefer_wav=False):
    """Whole clip back as (IMAGE, AUDIO)."""
    components = VideoFromFile(path).get_components()
    audio = components.audio
    if audio is not None and not isinstance(audio, dict):
        audio = dict(audio)
    if prefer_wav and os.path.exists(_wav_path(path)):
        audio = read_audio(_wav_path(path)) or audio
    return components.images, audio


def read_audio(path):
    """Audio only, without paying to decode the video stream.

    The join needs every segment's soundtrack up front but only one segment's
    frames at a time, so the two are read separately.
    """
    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            return None
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        chunks = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray())
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())
    if not chunks:
        return None
    data = np.concatenate(chunks, axis=1)  # (channels, samples)
    return {"waveform": torch.from_numpy(data).unsqueeze(0),
            "sample_rate": int(stream.sample_rate or 1)}


def drift_profile(parts, strength, window_seconds=2.0, fps=FPS, clamp=0.25):
    """Per-frame gains that flatten a chain's slow exposure/colour drift.

    A chained take slides away from its own opening -- steadily darker, steadily
    less saturated -- without any single join showing it. That trend is slow by
    definition, which is exactly what makes it separable: smooth each channel's
    per-frame mean over a couple of seconds and what is left is the drift, with
    gestures, blinks and real lighting flicker averaged out of it.

    The correction is a **gain** per frame per channel, aimed at the smoothed level
    of the first segment. Gains preserve black, so nothing is lifted into a milky
    shadow the way an offset would. Because the curve it is built from is smooth,
    the correction is continuous across the joins by construction -- there is no
    step at a cut, which is the whole reason to do this on the finished timeline
    rather than per segment.

    ``clamp`` bounds it. What this cannot tell apart from drift is a shot that is
    legitimately darker because the light changed, and a real lighting change is
    large. Bounding the gain means such a change survives, mostly, while
    accumulated drift -- which is small -- is removed.

    Returns None when there is nothing to do, so the caller can skip pass two.
    """
    if strength <= 0 or not parts:
        return None

    means = []
    for path, skip in parts:
        images, _ = load_clip(path)
        if images.shape[0] > skip:
            means.append(images[skip:].mean(dim=(1, 2)).float().cpu())
        del images
    if not means:
        return None
    series = torch.cat(means, dim=0)  # [frames, 3]
    if series.shape[0] < 3:
        return None

    window = max(3, int(round(window_seconds * fps)) | 1)
    padded = torch.cat([series[:1].repeat(window // 2, 1), series,
                        series[-1:].repeat(window // 2, 1)], dim=0)
    kernel = torch.ones(window) / window
    smooth = torch.stack([
        torch.nn.functional.conv1d(padded[:, c].view(1, 1, -1),
                                   kernel.view(1, 1, -1)).view(-1)
        for c in range(series.shape[1])], dim=1)

    # The target is the opening's own smoothed level: segment 1 is the only shot
    # rendered from the prompt and the reference alone, so it is the one place in a
    # chain that has not yet drifted.
    target = smooth[:max(3, window // 2)].mean(dim=0)
    gains = target.view(1, -1) / smooth.clamp_min(1e-4)
    gains = 1.0 + (gains - 1.0) * float(strength)
    gains = gains.clamp(1.0 - clamp, 1.0 + clamp)
    logging.info("[H3 Continuous] drift stabiliser: gain %.4f..%.4f across %d frames",
                 float(gains.min()), float(gains.max()), gains.shape[0])
    return gains


class _StreamedFrames:
    """A stand-in for the joined IMAGE batch that never exists all at once.

    ``VideoFromComponents.save_to`` only ever asks for ``.shape`` and then iterates,
    so the join can hold one segment's frames at a time instead of the whole cut.
    Eight 8-second segments at 768x1344 would otherwise be about 12 GB of float32.
    """

    def __init__(self, parts, height, width, gains=None):
        self.parts = parts  # [(path, skip_frames)]
        self.gains = gains
        self.total = 0
        self._counts = []
        for path, skip in parts:
            n = max(0, frame_count(path) - skip)
            self._counts.append(n)
            self.total += n
        self.shape = (self.total, height, width, 3)

    def __len__(self):
        return self.total

    def _apply(self, frame, index):
        if self.gains is None or index >= self.gains.shape[0]:
            return frame
        return (frame * self.gains[index].to(frame.dtype)).clamp(0.0, 1.0)

    def __iter__(self):
        index = 0
        for path, skip in self.parts:
            images, _ = load_clip(path)
            for i in range(skip, images.shape[0]):
                yield self._apply(images[i], index)
                index += 1
            del images

    def materialize(self):
        return torch.cat([f.unsqueeze(0) for f in self], dim=0)


def frame_count(path):
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def join(parts, out_path, fps=FPS, crf=14.0, stabilize=0.0):
    """Concatenate segments, dropping each one's replayed opening.

    ``parts`` is ``[(path, skip_frames), ...]``. Segment N opens on an exact replay
    of segment N-1's last ``skip_frames`` frames -- that replay is the join -- so
    those frames come off the front of every segment after the first or the motion
    stutters once per cut.

    ``stabilize`` additionally flattens the chain's slow drift away from its own
    opening; see ``drift_profile``. It costs one extra decoding pass and no GPU.
    """
    if not parts:
        raise ValueError("nothing to join")

    probe, _ = load_clip(parts[0][0])
    height, width = int(probe.shape[1]), int(probe.shape[2])
    del probe

    audio = _join_audio(parts, fps)
    frames = _StreamedFrames(parts, height, width,
                             gains=drift_profile(parts, stabilize, fps=fps))
    components = VideoComponents(images=frames, audio=audio, frame_rate=Fraction(fps))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".tmp.mp4"
    try:
        VideoFromComponents(components).save_to(
            tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    except (AttributeError, TypeError) as exc:
        # Streaming leans on save_to only touching .shape and iteration. If a future
        # ComfyUI indexes or moves the batch instead, fall back to the honest way.
        logging.warning("[H3 Continuous] streamed join failed (%s); "
                        "joining in memory instead", exc)
        components = VideoComponents(images=frames.materialize(), audio=audio,
                                     frame_rate=Fraction(fps))
        VideoFromComponents(components).save_to(
            tmp, format=VideoContainer.MP4, codec=VideoCodec.H264, crf=crf)
    os.replace(tmp, out_path)
    return out_path, frames.total


def _join_audio(parts, fps):
    """Concatenate the soundtracks, each cut to exactly its own kept frames.

    Every part is trimmed or padded to its video length before it is appended.
    Letting audio run even a fraction long is how a chain drifts: the error is not
    corrected at the next join, it accumulates across all of them.
    """
    rate = None
    pieces = []
    for path, skip in parts:
        clip_audio = read_audio(path)
        kept = max(0, frame_count(path) - skip)
        want_seconds = kept / float(fps)
        if clip_audio is None:
            pieces.append((None, want_seconds))
            continue
        if rate is None:
            rate = int(clip_audio["sample_rate"])
        waveform = clip_audio["waveform"]
        if int(clip_audio["sample_rate"]) != rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, int(clip_audio["sample_rate"]), rate)
        start = int(round(skip / float(fps) * rate))
        want = int(round(want_seconds * rate))
        cut = waveform[..., start:start + want]
        if cut.shape[-1] < want:
            cut = torch.cat(
                [cut, torch.zeros(cut.shape[:-1] + (want - cut.shape[-1],), dtype=cut.dtype)],
                dim=-1)
        pieces.append((cut, want_seconds))

    if rate is None:
        return None

    channels = next((p.shape[1] for p, _ in pieces if p is not None), 1)
    filled = []
    for piece, seconds in pieces:
        if piece is None:
            # A segment with no audio track still occupies time in the cut.
            filled.append(torch.zeros(1, channels, int(round(seconds * rate))))
        else:
            filled.append(piece)
    return {"waveform": torch.cat(filled, dim=-1), "sample_rate": rate}


def probe_seconds(path):
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        if stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
    return float(frame_count(path)) / FPS


def ceil_div(a, b):
    return int(math.ceil(a / float(b)))


def last_frame(path):
    """Decode just the final frame of a clip, without reading the whole thing.

    Used for the per-segment preview of a segment that was reused from cache: the
    footage is already on disk and decoding 192 frames of 480x864 to show one
    thumbnail would cost a gigabyte for nothing.
    """
    with av.open(path) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        duration = float(stream.duration * stream.time_base) if stream.duration else 0.0
        if duration > 0.5:
            try:
                container.seek(int((duration - 0.5) / stream.time_base), stream=stream)
            except av.error.PyAVError:
                pass
        frame = None
        for frame in container.decode(stream):
            pass
        if frame is None:  # seek overshot a very short clip; start over
            container.seek(0, stream=stream)
            for frame in container.decode(stream):
                pass
        if frame is None:
            return None
        # gbrpf32le, not rgb24: it is what ComfyUI's own decoder uses, and it carries the
        # stream's BT.709 tagging through the conversion. Going via rgb24 picks up
        # swscale's default matrix instead and shifts saturated colours by up to 0.09.
        return torch.from_numpy(frame.to_ndarray(format="gbrpf32le").copy())
