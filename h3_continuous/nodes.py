"""The five nodes.

    H3 Chain Settings   ->  models, sampler, canvas and seed, bundled once
    H3 Render Segment   ->  one shot: prompt, references and length IN, its rendered
                            video and a chain_state OUT that the next segment's node
                            wires into -- the chain is the graph, not a hidden loop
    H3 Chain to Video   ->  joins a chain_state's segments into one cut
    H3 Repair Segment   ->  re-renders one segment without disturbing its neighbours
    H3 Load Session     ->  re-join an earlier session, or extend it with more
                            H3 Render Segment nodes, without re-rendering what is done

Every H3 Render Segment is its own node execution, so its ``video`` output populates
-- and can feed a Preview Video node, or anything else -- the moment that one segment
finishes, independent of how many more are still to come. That is also what makes
resume robust to a hard kill: the manifest is rewritten after every segment's own
execute() returns, not once at the end of a run that might not reach one.
"""

import os
import shutil

import comfy.utils
from comfy_api.input_impl import VideoFromFile
from comfy_api.latest import ComfyExtension, io

from . import session as session_mod
from . import video_io
from .common import (
    FPS,
    generation_length,
    guide_length,
    log,
    ordered_autogrow,
    seconds_to_frames,
)
from .engine import (
    anchor_frames,
    arrest_drift,
    describe,
    free_between_segments,
    latent_signature,
    latent_tail,
    picture_only,
    push_preview,
    render_segment,
    take_tail,
)

CATEGORY = "MiniMax H3/continuous"

STABILIZE_TOOLTIP = (
    "Flatten the finished cut's slow drift away from its own opening. 0 is off.\n\n"
    "Chained takes slide steadily darker and less saturated without any one join "
    "showing it. That trend is slow by definition, so it separates cleanly: each "
    "channel's per-frame mean is smoothed over two seconds, and what is left is "
    "the drift. The correction is a gain aimed at segment 1's level -- gains keep "
    "black black -- and because the curve it comes from is smooth it is continuous "
    "across every join, so stabilising cannot introduce a step at a cut.\n\n"
    "Gains are bounded at +/-25%, because the one thing this cannot tell apart "
    "from drift is a shot that is genuinely darker after a real lighting change. "
    "Real changes are large and survive the bound; accumulated drift is small and "
    "does not.\n\n"
    "0.7-1.0 is the useful range. It costs one extra decoding pass and no GPU. "
    "This fixes colour only -- for identity drift use drift_arrest on H3 Chain "
    "Settings, which acts on the generation rather than on the finished file."
)

DRIFT_ARREST_TOOLTIP = (
    "Pull each handoff's colour and exposure back toward SEGMENT 1's before pinning "
    "it, as a fraction of the measured error. 0 is off.\n\n"
    "Chained generation has no absolute reference: each segment is told only to "
    "continue from the last, so wherever segment N ended up becomes the truth for "
    "N+1 and a small bias compounds. Left alone a six-shot chain loses about 8 L* of "
    "face brightness and 0.24 of ArcFace cosine, none of it visible at any one join.\n\n"
    "0.4-0.6 is the useful range. It is deliberately a fraction and not a reset: the "
    "previous segment really did end where it ended and the cut keeps those frames, "
    "so correcting the whole error would put a visible step at every join. Large "
    "per-channel corrections are also clamped, so a shot that is genuinely darker "
    "because the light changed is not fought.\n\n"
    "Needs handoff_mode 'latent' -- the correction is applied to the latent that gets "
    "pinned, and the pixel path has no latent to correct."
)

HANDOFF_MODE_TOOLTIP = (
    "How the previous segment's ending is carried into the next one.\n\n"
    "'latent' (default) slices the tail out of the previous segment's own sampled "
    "latent and pins that. No VAE in the handoff at all, so the anchor is exact by "
    "construction rather than 30 dB of a round trip -- and it skips a 39-frame VAE "
    "encode per segment, so it is also faster.\n\n"
    "'pixel' decodes, writes the tail as an mp4, reads it back and re-encodes it "
    "with MiniMaxH3AddGuide. Slower and lossy, but it is the only path that can "
    "rescale, so use it if you change resolution partway through a session.\n\n"
    "Both tails are always written to disk, so a session can be resumed in either "
    "mode. Sessions rendered before latent tails existed have only the mp4 and fall "
    "back to 'pixel' automatically, with a line in the log."
)


def _anchor_from_disk(sess, index, handoff_mode):
    """The anchor segment ``index`` hands to ``index+1``, read back off disk.

    Latent first when asked for, but never fatally: a session from before latent
    tails existed has only the mp4, and falling back to it is strictly better than
    refusing to resume.
    """
    tail = sess.tail_path(index)
    if handoff_mode == "latent":
        anchor = video_io.load_latent_tail(tail)
        if anchor is not None:
            return anchor
        log("no latent tail for %s -- resuming this handoff through the VAE instead",
            os.path.basename(tail))
    images, audio = video_io.load_clip(tail, prefer_wav=True)
    return {"images": images, "audio": audio}

H3Settings = io.Custom("H3_SETTINGS")
H3Chain = io.Custom("H3_CHAIN")


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------
class H3ChainSettingsNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainSettings",
            display_name="H3 Chain Settings",
            category=CATEGORY,
            description="Everything the chain and the repair node both need: models, "
                        "sampler, canvas and seed. Wire one of these into either.\n\n"
                        "session_name is read by H3 Render Segment (for the first segment "
                        "of a chain; later ones inherit it via chain_state) but NOT by H3 "
                        "Repair Segment, which keeps its own session_name -- a repair "
                        "targets a session by name explicitly and does not have to match "
                        "whatever this settings bundle's chain is currently rendering.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("vae", tooltip="The H3 *video* VAE."),
                io.Vae.Input("audio_vae", tooltip="The H3 *audio* VAE."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("width", default=864, min=32, max=16384, step=32),
                io.Int.Input("height", default=480, min=32, max=16384, step=32),
                # Deliberately not called "seed": the frontend bolts a
                # control_after_generate widget onto any widget with that name, and its
                # default is "randomize" -- which would silently re-roll the seed after
                # every run and re-render the entire chain from scratch each time.
                io.Int.Input("chain_seed", default=0, min=0, max=0xffffffffffffffff,
                             tooltip="Each segment derives its own seed from this one, so "
                                     "changing it re-renders the whole chain."),
                io.String.Input(
                    "session_name", default="my_chain", multiline=False,
                    tooltip="Folder under output/h3_continuous/. Lives here, not on each "
                            "H3 Render Segment, because a chain only ever has one session "
                            "no matter how many segment nodes are wired into it -- the "
                            "first segment (the one with chain_state unconnected) reads it "
                            "from here; every segment after that inherits the session "
                            "object it already resolved, off the chain_state wire."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                               tooltip="'match' scales reference images to the generation's "
                                       "pixel area; 'max' uses a 2048px short edge for the "
                                       "best identity fidelity and is several times slower, "
                                       "because reference tokens ride through every step."),
                io.Combo.Input("handoff_mode", options=["latent", "pixel"], default="latent",
                               advanced=True, tooltip=HANDOFF_MODE_TOOLTIP),
                io.Float.Input("drift_arrest", default=0.0, min=0.0, max=1.0, step=0.05,
                               round=False, advanced=True, tooltip=DRIFT_ARREST_TOOLTIP),
            ],
            outputs=[H3Settings.Output(display_name="settings")],
        )

    @classmethod
    def execute(cls, model, clip, vae, audio_vae, sampler, sigmas, width, height,
                chain_seed, session_name, ref_image_size, handoff_mode="latent",
                drift_arrest=0.0) -> io.NodeOutput:
        return io.NodeOutput({
            "model": model, "clip": clip, "vae": vae, "audio_vae": audio_vae,
            "sampler": sampler, "sigmas": sigmas,
            "width": width, "height": height, "seed": chain_seed,
            "session_name": session_name, "ref_image_size": ref_image_size,
            "handoff_mode": handoff_mode, "drift_arrest": float(drift_arrest),
        })


# ---------------------------------------------------------------------------
# render one segment
# ---------------------------------------------------------------------------
PROMPT_TOOLTIP = (
    "H3 prompt for this segment. Reference tags are numbered per type in socket "
    "order: image_0 is <Picture 1>, video_0 is <Video 1>, audio_0 is <Audio 1>.\n\n"
    "The clip carried in from the previous segment gets NO tag -- it never reaches "
    "the tokenizer -- so describe it in prose (\"the segment opens on an exact "
    "replay of the preceding shot and carries straight on from it\") and never "
    "label it. A <Video 1> written for it would be an unresolved reference.\n\n"
    "Nothing new can be scheduled inside the replayed opening: with a 1.63s "
    "handoff, an entrance written for \"the one-second mark\" lands in frames that "
    "are already pinned."
)

HANDOFF_TOOLTIP = (
    "How much of THIS segment's ending is replayed as the opening of the NEXT one.\n\n"
    "Snapped down to a valid guide length: 5, 22, 39 or 56 frames (0.21s, 0.92s, "
    "1.63s, 2.33s). 39 is the sweet spot -- guide fidelity peaks exactly at frames "
    "0, 17 and 34 and sags between, which is the latent grouping showing through.\n\n"
    "KEEP IT NEAR A FIFTH OF THE SEGMENT. 1.63s is 20% of an 8s segment and joins "
    "cleanly; the same 1.63s is 31% of a 5s segment, and there the seams measurably "
    "degrade -- too little free runway is left after the pinned region. Shorten the "
    "handoff for short segments, or lengthen the segments.\n\n"
    "A longer handoff also costs you that much new footage: at 8s segments with a "
    "1.63s handoff, each segment after the first contributes 6.37s to the finished cut.\n\n"
    "On the LAST segment of a take, set this to 0 -- there is no next segment to hand "
    "off to, and a nonzero value here just writes a tail clip nothing will ever read."
)


def _segment_schema():
    """The shot fields shared by H3 Render Segment and H3 Repair Segment."""
    return [
        io.String.Input("prompt", multiline=True, dynamic_prompts=False,
                        tooltip=PROMPT_TOOLTIP),
        io.Float.Input(
            "seconds", default=8.0, min=0.25, max=15.0, step=0.25, round=False,
            tooltip="Length of this segment, snapped up to H3's 17k+5 frame grid at "
                    "24 fps (8s -> 192 frames). The trained range is about 5-15s."),
        io.Float.Input("handoff_seconds", default=1.625, min=0.0, max=4.0, step=0.125,
                       round=False, advanced=True, tooltip=HANDOFF_TOOLTIP),
        io.Int.Input(
            "seed_override", default=0, min=0, max=0xffffffffffffffff, advanced=True,
            tooltip="0 = derive this segment's seed from the chain seed. Set anything "
                    "else to re-roll just this segment. (Not named 'seed' on purpose "
                    "-- that name gets an automatic randomise control, which would "
                    "re-render this segment on every run.)"),
        io.Autogrow.Input(
            "images", optional=True,
            tooltip="Reference images -> <Picture 1..9>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("image", optional=True,
                                     tooltip="Reference image for this segment."),
                prefix="image_", min=1, max=9)),
        io.Autogrow.Input(
            "videos", optional=True,
            tooltip="Reference videos (frames at 24 fps, 2-15s) -> <Video 1..3>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("video", optional=True,
                                     tooltip="Reference video frames."),
                prefix="video_", min=0, max=3)),
        io.Autogrow.Input(
            "video_audios", optional=True,
            tooltip="Soundtrack of the same-numbered reference video. Each one is "
                    "presented as its own <Audio j>, right before its <Video k>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("video_audio", optional=True,
                                     tooltip="Soundtrack of the same-numbered video."),
                prefix="video_audio_", min=0, max=3)),
        io.Autogrow.Input(
            "audios", optional=True,
            tooltip="Standalone reference audio -> <Audio 1..3>.",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("audio", optional=True,
                                     tooltip="Standalone reference audio."),
                prefix="audio_", min=0, max=3)),
    ]


def _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                          images=None, videos=None, video_audios=None, audios=None):
    length = generation_length(seconds_to_frames(seconds))
    handoff = guide_length(seconds_to_frames(handoff_seconds))
    if handoff >= length:
        raise ValueError(
            "handoff_seconds (%.2fs -> %d frames) must be shorter than the segment "
            "itself (%.2fs -> %d frames), or the next segment would be nothing but "
            "replay." % (handoff_seconds, handoff, seconds, length))
    return {
        "prompt": prompt, "seconds": seconds, "length": length, "handoff": handoff,
        "seed": int(seed_override),
        "images": [t for _, t in ordered_autogrow(images)],
        "videos": ordered_autogrow(videos),
        "video_audios": ordered_autogrow(video_audios),
        "audios": [a for _, a in ordered_autogrow(audios)],
    }


CHAIN_STATE_TOOLTIP = (
    "Wire in the PREVIOUS H3 Render Segment's chain_state to continue its take. Leave "
    "unconnected on the first segment of a session, or to start a new one.\n\n"
    "Also accepts H3 Load Session's chain output -- wiring that in here appends new "
    "segments onto a session that was already finished, without re-rendering it."
)


def _resolve_seed(settings, segment, index):
    if segment["seed"]:
        return int(segment["seed"])
    # 9973 is prime, so neighbouring segments never collide on a nearby chain seed.
    return (int(settings["seed"]) + (index + 1) * 9973) % (1 << 63)


def _summarize(chain):
    records = chain["segments"]
    total = sum(r["length"] for r in records) - sum(r["handoff"] for r in records[:-1])
    lines = ["%d segments -> %.2fs (%d frames) once the replayed handoffs come off"
             % (len(records), total / float(FPS), total)]
    for r in records:
        notes = "  (repaired)" if r.get("repaired") else ""
        if r.get("drift_correction"):
            notes += "  drift -%.3f" % r["drift_correction"]
        lines.append("  %2d  %5.2fs  handoff %4.2fs  seed %d%s"
                     % (r["index"] + 1, r["seconds"], r["handoff"] / float(FPS), r["seed"],
                        notes))
    lines.append("  in %s" % chain["dir"])
    return "\n".join(lines)


class H3RenderSegmentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RenderSegment",
            display_name="H3 Render Segment",
            category=CATEGORY,
            description="One shot in a continuous take: a prompt, its references, how long "
                        "it runs -- and it renders right here. Chain shots by wiring this "
                        "node's chain_state output into the next H3 Render Segment's "
                        "chain_state input; each one after the first opens on an exact "
                        "replay of the one before it, anchored with MiniMaxH3AddGuide, so "
                        "the cut is invisible.\n\n"
                        "This node's video output is real the moment THIS segment finishes "
                        "-- you do not wait for the rest of the chain to preview, save or "
                        "post-process it. The segment is also written to "
                        "output/h3_continuous/<session>/ and reused on the next queue, so a "
                        "crash, an interrupt or an edited prompt only costs the segments "
                        "that actually changed.",
            inputs=[
                H3Settings.Input("settings"),
                H3Chain.Input("chain_state", optional=True, tooltip=CHAIN_STATE_TOOLTIP),
                io.Boolean.Input("resume", default=True,
                                 tooltip="Reuse this segment if it is already rendered in the "
                                         "session with unchanged settings. A segment's "
                                         "fingerprint includes the one before it, so editing "
                                         "an earlier segment's prompt still invalidates this "
                                         "one even with resume on."),
                *_segment_schema(),
                io.Boolean.Input("unload_models_after", default=False, advanced=True,
                                 tooltip="Unload models after this segment renders, before the "
                                         "next one starts. OOM-only: it forces a full reload "
                                         "of the UNet and text encoder from disk, which "
                                         "dominates runtime on a small box. Turn it on for "
                                         "just the segment where you actually OOM."),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     H3Chain.Output(display_name="chain_state"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, settings, resume, prompt, seconds, handoff_seconds,
                seed_override, unload_models_after, chain_state=None, images=None,
                videos=None, video_audios=None, audios=None) -> io.NodeOutput:
        segment = _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                                        images, videos, video_audios, audios)
        handoff = segment["handoff"]

        if chain_state is None:
            sess = session_mod.Session(settings["session_name"])
            index, previous_key, anchor, records = 0, None, None, []
            reference = None
        else:
            sess = chain_state["sess_obj"]
            index = chain_state["index"] + 1
            previous_key = chain_state["key"]
            anchor = chain_state["anchor"]
            records = list(chain_state["segments"])
            # Segment 1's signature, carried down the whole chain. It is the only
            # absolute reference a chain has: the one shot rendered from the prompt
            # and the reference image alone, with no inherited anchor to be wrong about.
            reference = chain_state.get("reference")

        segment["resolved_seed"] = _resolve_seed(settings, segment, index)
        key = session_mod.segment_key(settings, segment, handoff, previous_key)
        manifest = sess.load() if resume else None

        if resume and sess.cached(manifest, index, key, needs_tail=bool(handoff)):
            log("%s -- reusing %s", describe(segment, index, anchor is not None),
                os.path.basename(sess.segment_path(index)))
            rendered_length = video_io.frame_count(sess.segment_path(index))
            new_anchor = (_anchor_from_disk(sess, index, settings["handoff_mode"])
                          if handoff else None)
            signature = session_mod.signature_from_record(manifest["segments"][index])
            # A reused segment still gets a thumbnail, so a resumed run visibly walks
            # through what it is keeping instead of appearing to stall.
            preview_frame = video_io.last_frame(sess.segment_path(index))
        else:
            log("%s", describe(segment, index, anchor is not None))
            images_out, audio_out, samples = render_segment(
                settings, segment, start_anchor=anchor)
            rendered_length = int(images_out.shape[0])
            video_io.save_clip(sess.segment_path(index), images_out, audio_out)

            # Both forms of the handoff go to disk whatever mode rendered it, so the
            # session can be resumed, repaired or extended in either mode later.
            pixel_tail = take_tail(images_out, audio_out, handoff)
            lat_tail = latent_tail(samples, handoff)
            signature = latent_signature(samples)
            if pixel_tail is not None:
                video_io.save_clip(sess.tail_path(index), pixel_tail[0], pixel_tail[1],
                                   exact_audio=True)
                # Saved uncorrected, deliberately: the file on disk is what this
                # segment actually ended on, which is what the repair node has to pin
                # against. Drift correction is applied when the anchor is consumed.
                video_io.save_latent_tail(sess.tail_path(index), lat_tail)

            if handoff:
                new_anchor = (lat_tail if settings["handoff_mode"] == "latent"
                              else {"images": pixel_tail[0], "audio": pixel_tail[1]})
            else:
                new_anchor = None
            preview_frame = images_out[-1].clone()
            del images_out, audio_out, samples, pixel_tail, lat_tail
            free_between_segments(unload_models_after)

        # Segment 1 seeds the reference and then it rides the chain unchanged: it is
        # the only shot rendered from the prompt and the reference image alone, with
        # no inherited anchor that could already be wrong.
        reference = reference or signature
        correction = 0.0
        if new_anchor is not None and settings.get("drift_arrest"):
            if new_anchor.get("video_latent") is None:
                log("drift_arrest is set but handoff_mode is 'pixel', which has no "
                    "latent to correct -- segment %d's handoff is going out uncorrected",
                    index + 1)
            corrected = arrest_drift(new_anchor, reference, signature,
                                     settings["drift_arrest"])
            correction = corrected.get("drift_correction", 0.0) if corrected else 0.0
            if correction:
                log("segment %d: drift correction %.4f per latent channel applied to "
                    "the handoff", index + 1, correction)
            new_anchor = corrected

        pbar = comfy.utils.ProgressBar(1)
        push_preview(pbar, preview_frame, 1, 1)
        del preview_frame

        record = {
            "index": index, "key": key, "length": rendered_length, "handoff": handoff,
            "seconds": round(rendered_length / float(FPS), 3),
            "seed": segment["resolved_seed"], "prompt": segment["prompt"],
            "file": os.path.basename(sess.segment_path(index)),
            # Kept so a resumed run can still measure how far it has drifted without
            # decoding every segment it is reusing.
            "signature": session_mod.signature_to_record(signature),
        }
        if correction:
            record["drift_correction"] = round(correction, 6)
        # Slicing to `index` rather than appending covers rewiring in a shorter earlier
        # segment upstream of a session that used to be longer.
        records = records[:index] + [record]
        # Written every segment, not once at the end: a hard kill (not just a clean
        # interrupt) still leaves the manifest exactly matching what is on disk, so the
        # next queue's resume never re-renders something that actually finished.
        sess.save(records, extra={"width": settings["width"],
                                  "height": settings["height"], "fps": FPS})

        chain = {"session": sess.name, "dir": sess.dir, "segments": records,
                 "sess_obj": sess, "anchor": new_anchor, "index": index, "key": key,
                 "reference": reference}
        summary = _summarize(chain)
        log("%s", summary.replace("\n", " | "))
        return io.NodeOutput(VideoFromFile(sess.segment_path(index)), chain, summary)


# ---------------------------------------------------------------------------
# join
# ---------------------------------------------------------------------------
class H3ChainToVideoNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainToVideo",
            display_name="H3 Chain to Video",
            category=CATEGORY,
            description="Joins a session's segments into one cut, dropping each segment's "
                        "replayed opening so the motion does not stutter once per join.",
            inputs=[
                H3Chain.Input("chain"),
                io.Float.Input("crf", default=14.0, min=0.0, max=51.0, step=1.0, advanced=True,
                               tooltip="Quality of the joined file. Lower is better and "
                                       "bigger; 0 is lossless."),
                io.Float.Input("stabilize", default=0.0, min=0.0, max=1.0, step=0.05,
                               round=False, advanced=True, tooltip=STABILIZE_TOOLTIP),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     io.String.Output(display_name="path")],
        )

    @classmethod
    def execute(cls, chain, crf, stabilize=0.0) -> io.NodeOutput:
        records = chain["segments"]
        if not records:
            raise ValueError("this chain has no rendered segments")
        parts = []
        for index, record in enumerate(records):
            path = os.path.join(chain["dir"], record["file"])
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "segment %d is missing from the session (%s). Re-queue the chain to "
                    "render it." % (index + 1, path))
            skip = 0 if index == 0 else int(records[index - 1]["handoff"])
            parts.append((path, skip))

        out_path = os.path.join(chain["dir"], "%s.mp4" % chain["session"])
        out_path, frames = video_io.join(parts, out_path, crf=crf,
                                         stabilize=stabilize)
        log("joined %d segments -> %d frames (%.2fs) -> %s",
            len(parts), frames, frames / float(FPS), out_path)
        return io.NodeOutput(VideoFromFile(out_path), out_path)


# ---------------------------------------------------------------------------
# repair
# ---------------------------------------------------------------------------
class H3RepairSegmentNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3RepairSegment",
            display_name="H3 Repair Segment",
            category=CATEGORY,
            description="Re-render one segment of a finished session, in place.\n\n"
                        "Normally regenerating segment N changes its ending, so every segment "
                        "after it has to be re-rendered too. This node pins BOTH ends: the "
                        "opening to the previous segment's handoff clip, and the ending to the "
                        "very clip the next segment already opens on. Nothing downstream moves.",
            inputs=[
                H3Settings.Input("settings"),
                io.String.Input("session_name", default="my_chain", multiline=False),
                io.Int.Input("segment_number", default=2, min=1, max=32,
                             tooltip="1-based, matching seg_NN.mp4 in the session folder."),
                *_segment_schema(),
                io.Boolean.Input("pin_ending", default=True,
                                 tooltip="Keep the ending the next segment continues from. Turn "
                                         "this off only for the last segment, or when you mean "
                                         "to re-render everything after this one."),
            ],
            outputs=[io.Video.Output(display_name="video"),
                     H3Chain.Output(display_name="chain"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def execute(cls, settings, session_name, segment_number, prompt, seconds,
                handoff_seconds, seed_override, pin_ending, images=None, videos=None,
                video_audios=None, audios=None) -> io.NodeOutput:
        segment = _segment_from_widgets(prompt, seconds, handoff_seconds, seed_override,
                                        images, videos, video_audios, audios)
        sess = session_mod.Session(session_name)
        manifest = sess.load()
        if not manifest:
            raise FileNotFoundError(
                "no manifest in %s -- run the chain workflow for this session first."
                % sess.dir)
        records = manifest["segments"]
        index = int(segment_number) - 1
        if index < 0 or index >= len(records):
            raise ValueError("session '%s' has %d segments; segment_number %d is out of range."
                             % (sess.name, len(records), segment_number))

        start_anchor = None
        if index > 0:
            previous_tail = sess.tail_path(index - 1)
            if not os.path.exists(previous_tail):
                raise FileNotFoundError(
                    "segment %d has no handoff clip (%s), so segment %d has nothing to open "
                    "on." % (index, previous_tail, segment_number))
            start_anchor = _anchor_from_disk(sess, index - 1, settings["handoff_mode"])

        end_anchor = None
        is_last = index == len(records) - 1
        if pin_ending and not is_last:
            own_tail = sess.tail_path(index)
            if not os.path.exists(own_tail):
                raise FileNotFoundError(
                    "segment %d has no handoff clip (%s) to pin its ending to."
                    % (segment_number, own_tail))
            end_anchor = picture_only(
                _anchor_from_disk(sess, index, settings["handoff_mode"]))
            end_frames = anchor_frames(end_anchor)
            if end_frames >= segment["length"]:
                raise ValueError(
                    "the pinned ending is %d frames but the replacement segment is only %d "
                    "long. Give the segment at least its original length."
                    % (end_frames, segment["length"]))

        resolved = dict(segment, resolved_seed=_resolve_seed(settings, segment, index))
        log("repairing segment %d of '%s'%s%s", segment_number, sess.name,
            " (opening pinned)" if start_anchor is not None else "",
            " (ending pinned)" if end_anchor is not None else "")

        pbar = comfy.utils.ProgressBar(1)
        images_out, audio_out, samples = render_segment(
            settings, resolved, start_anchor=start_anchor, end_anchor=end_anchor)
        push_preview(pbar, images_out[-1], 1, 1)

        target = sess.segment_path(index)
        if os.path.exists(target):
            shutil.copyfile(target,
                            os.path.join(sess.dir, "seg_%02d.replaced.mp4" % (index + 1)))
        video_io.save_clip(target, images_out, audio_out)
        length = int(images_out.shape[0])

        # The handoff clip is NOT rewritten while the ending is pinned: the file on disk
        # is what segment N+1 actually opens on, and the new ending was pinned to it.
        if not pin_ending and not is_last:
            handoff = records[index]["handoff"]
            tail = take_tail(images_out, audio_out, handoff)
            if tail is not None:
                video_io.save_clip(sess.tail_path(index), tail[0], tail[1], exact_audio=True)
                video_io.save_latent_tail(sess.tail_path(index),
                                          latent_tail(samples, handoff))
            log("pin_ending was off: segment %d's ending has moved, so segments %d..%d no "
                "longer join and have to be re-rendered.",
                segment_number, segment_number + 1, len(records))

        del images_out, audio_out, samples
        free_between_segments(False)

        records[index].update({
            "length": length,
            "seconds": round(length / float(FPS), 3),
            "seed": resolved["resolved_seed"],
            "prompt": resolved["prompt"],
            "repaired": True,
            "ending_pinned": end_anchor is not None,
        })
        sess.save(records, extra={k: v for k, v in manifest.items()
                                  if k not in ("version", "session", "segments")})

        chain = {"session": sess.name, "dir": sess.dir, "segments": records}
        summary = _summarize(chain)
        log("%s", summary.replace("\n", " | "))
        return io.NodeOutput(VideoFromFile(target), chain, summary)


# ---------------------------------------------------------------------------
# load an existing session
# ---------------------------------------------------------------------------
class H3LoadSessionNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3LoadSession",
            display_name="H3 Load Session",
            category=CATEGORY,
            description="Pick up a session already on disk -- to re-join it, feed the repair "
                        "workflow, or extend it -- without re-rendering anything.\n\n"
                        "Its chain output wires into H3 Chain to Video / H3 Repair Segment "
                        "as before, and now also into another H3 Render Segment's "
                        "chain_state, to render more shots onto the end of a session that "
                        "was already finished.",
            inputs=[
                io.String.Input("session_name", default="my_chain", multiline=False),
                io.Combo.Input("handoff_mode", options=["latent", "pixel"],
                               default="latent", advanced=True,
                               tooltip="Which form of the last segment's handoff to load, "
                                       "for extending the session with more H3 Render "
                                       "Segment nodes. Match the chain that will consume "
                                       "it. Falls back to 'pixel' if the session has no "
                                       "latent tail."),
            ],
            outputs=[H3Chain.Output(display_name="chain"),
                     io.String.Output(display_name="summary")],
        )

    @classmethod
    def fingerprint_inputs(cls, session_name, handoff_mode="latent"):
        # The folder changes underneath the graph, so never trust a cached result.
        sess = session_mod.Session(session_name)
        try:
            return "%s:%s" % (os.path.getmtime(sess.manifest_path), handoff_mode)
        except OSError:
            return "missing"

    @classmethod
    def execute(cls, session_name, handoff_mode="latent") -> io.NodeOutput:
        sess = session_mod.Session(session_name)
        manifest = sess.load()
        if not manifest:
            raise FileNotFoundError("no manifest in %s" % sess.dir)
        records = manifest["segments"]
        last = records[-1]
        anchor = None
        if last.get("handoff") and os.path.exists(sess.tail_path(last["index"])):
            anchor = _anchor_from_disk(sess, last["index"], handoff_mode)
        chain = {"session": sess.name, "dir": sess.dir, "segments": records,
                 "sess_obj": sess, "anchor": anchor, "index": last["index"],
                 "key": last["key"],
                 # Segment 1's signature, not the last one's. Extending a session has
                 # to keep measuring drift against the same shot the original chain
                 # did, or the new segments would treat an already-drifted ending as
                 # the reference and lock the drift in instead of correcting it.
                 "reference": session_mod.signature_from_record(records[0])}
        return io.NodeOutput(chain, _summarize(chain))


class H3ContinuousExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            H3ChainSettingsNode,
            H3RenderSegmentNode,
            H3ChainToVideoNode,
            H3RepairSegmentNode,
            H3LoadSessionNode,
        ]
