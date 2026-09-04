"""Generate the shipped .json workflows.

The graphs are built from a live ComfyUI's /object_info rather than typed out, so
widget order, socket names and combo values cannot drift away from the nodes. Run
it against a ComfyUI that has this pack installed:

    python tools/build_workflows.py --server http://127.0.0.1:8188
"""

import argparse
import json
import os
import sys
import uuid
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(HERE)

BLUE, GREEN, PURPLE, YELLOW, RED = "#3f789e", "#3f5159", "#535", "#653", "#533"

# ResolutionSelector (comfy_extras/nodes_resolution.py, ComfyUI core) offers these
# exact labels -- (label, ratio_w, ratio_h) -- used to pick the closest preset for a
# given width/height so the shipped default still lands close to the old fixed value.
ASPECT_RATIOS = [
    ("1:1 (Square)", 1, 1),
    ("2:3 (Portrait Photo)", 2, 3),
    ("3:2 (Photo)", 3, 2),
    ("3:4 (Portrait Standard)", 3, 4),
    ("4:3 (Standard)", 4, 3),
    ("9:16 (Portrait Widescreen)", 9, 16),
    ("16:9 (Widescreen)", 16, 9),
    ("21:9 (Ultrawide)", 21, 9),
]


def closest_aspect_ratio(width, height):
    target = width / float(height)
    label, _, _ = min(ASPECT_RATIOS, key=lambda t: abs((t[1] / float(t[2])) - target))
    return label

# The five files this pack runs on, in the shape ComfyUI templates use. A loader node
# that carries `properties.models` makes ComfyUI offer to download what is missing when
# the workflow is opened, instead of showing an empty dropdown with no explanation.
# Keyed on basename, so a loader still matches when the file sits in a subfolder.
HF = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/"

MODELS = {
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors": "diffusion_models",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors": "text_encoders",
    "minimax_h3_video_vae_fp16.safetensors": "vae",
    "minimax_h3_audio_vae_fp32.safetensors": "vae",
    "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors": "loras",
}


def model_entry(value):
    """The models[] entry for a loader widget value, or None if it names no known file."""
    if not isinstance(value, str):
        return None
    base = value.replace("\\", "/").rsplit("/", 1)[-1]
    directory = MODELS.get(base)
    if directory is None:
        return None
    return {"name": base, "url": HF + directory + "/" + base, "directory": directory}


# Types that render as a connection, not a widget.
SOCKET_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK", "AUDIO",
    "VIDEO", "SAMPLER", "SIGMAS", "GUIDER", "NOISE", "CLIP_VISION", "CONTROL_NET",
    "STYLE_MODEL", "GLIGEN", "UPSCALE_MODEL", "WEBCAM", "COMFY_AUTOGROW_V3",
    "H3_SETTINGS", "H3_CHAIN", "*",
}


class Builder:
    def __init__(self, info):
        self.info = info
        self.nodes = []
        self.links = []
        self.groups = []
        self.next_node = 1
        self.next_link = 1

    # -- schema helpers ------------------------------------------------------
    def spec(self, node_type):
        if node_type not in self.info:
            raise KeyError("%s is not installed on this ComfyUI" % node_type)
        return self.info[node_type]

    def widget_names(self, node_type):
        spec = self.spec(node_type)
        order = spec.get("input_order", {})
        names = []
        for section in ("required", "optional"):
            for name in order.get(section, []):
                declared = spec["input"][section][name][0]
                is_combo = isinstance(declared, list) or declared in (
                    "COMBO", "COMFY_DYNAMICCOMBO_V3")
                if is_combo or declared not in SOCKET_TYPES:
                    names.append(name)
        return names

    def socket_names(self, node_type):
        spec = self.spec(node_type)
        order = spec.get("input_order", {})
        out = []
        for section in ("required", "optional"):
            for name in order.get(section, []):
                declared = spec["input"][section][name][0]
                if not isinstance(declared, list) and declared in SOCKET_TYPES \
                        and declared != "COMFY_AUTOGROW_V3":
                    out.append((name, declared))
        return out

    def combo_options(self, node_type, widget):
        spec = self.spec(node_type)
        for section in ("required", "optional"):
            entry = spec["input"].get(section, {}).get(widget)
            if entry:
                if isinstance(entry[0], list):
                    return entry[0]
                if len(entry) > 1 and isinstance(entry[1], dict):
                    return entry[1].get("options", [])
        return []

    def pick(self, node_type, widget, *needles):
        """First combo option containing every needle -- so the shipped workflow
        points at the files this machine actually has."""
        for option in self.combo_options(node_type, widget):
            text = str(option).lower()
            if all(n.lower() in text for n in needles):
                return option
        return needles[0]

    # -- graph building ------------------------------------------------------
    def add(self, node_type, pos, widgets=None, size=None, title=None,
            autogrow=None, color=None, mode=0):
        spec = self.spec(node_type)
        node_id = self.next_node
        self.next_node += 1

        inputs = [{"name": name, "type": t, "link": None}
                  for name, t in self.socket_names(node_type)]
        for group, prefix, socket_type, count in (autogrow or []):
            for i in range(count):
                inputs.append({"label": "%s%d" % (prefix, i),
                               "name": "%s.%s%d" % (group, prefix, i),
                               "shape": 7, "type": socket_type, "link": None})

        names = self.widget_names(node_type)
        values = list(widgets or [])
        if len(values) != len(names):
            raise ValueError("%s takes widgets %s, got %d values"
                             % (node_type, names, len(values)))

        node = {
            "id": node_id, "type": node_type, "pos": list(pos),
            "size": list(size or [320, 120]), "flags": {}, "order": node_id,
            "mode": mode, "inputs": inputs,
            "outputs": [{"name": n, "type": t, "links": []} for n, t in
                        zip(spec.get("output_name") or spec["output"], spec["output"])],
            "properties": {"Node name for S&R": node_type},
            "widgets_values": values,
            "widgets_values_named": dict(zip(names, values)),
        }
        found = [m for m in (model_entry(v) for v in values) if m]
        if found:
            node["properties"]["models"] = found
        if title:
            node["title"] = title
        if color:
            node["color"] = color
            node["bgcolor"] = color
        self.nodes.append(node)
        return node

    def note(self, pos, size, text, title="Read me", color=YELLOW):
        node = {"id": self.next_node, "type": "MarkdownNote", "pos": list(pos),
                "size": list(size), "flags": {}, "order": self.next_node, "mode": 0,
                "inputs": [], "outputs": [], "title": title,
                "properties": {}, "widgets_values": [text],
                "color": color, "bgcolor": color}
        self.next_node += 1
        self.nodes.append(node)
        return node

    def convert_widget_to_input(self, node, name, socket_type):
        """Mimic the frontend's "Convert widget to input": add a socket for a widget
        that is normally baked into ``widgets_values``. The value already sitting in
        ``widgets_values`` is left alone -- ComfyUI keeps it as the display fallback
        and uses the link once one is attached, exactly as a manually-converted
        widget does."""
        node["inputs"].append({"name": name, "type": socket_type,
                               "widget": {"name": name}, "link": None})

    def link(self, src, src_slot, dst, dst_name):
        slot = next(i for i, s in enumerate(dst["inputs"]) if s["name"] == dst_name)
        link_id = self.next_link
        self.next_link += 1
        dst["inputs"][slot]["link"] = link_id
        src["outputs"][src_slot]["links"].append(link_id)
        self.links.append([link_id, src["id"], src_slot, dst["id"], slot,
                           src["outputs"][src_slot]["type"]])
        return link_id

    def group(self, title, bounding, color=BLUE):
        self.groups.append({"id": len(self.groups) + 1, "title": title,
                            "bounding": list(bounding), "color": color, "flags": {}})

    # A workflow carries its own saved viewport in extra.ds, and ComfyUI restores it
    # verbatim on open -- it does not fit the graph for you. A fixed scale/offset
    # therefore opens a tall graph somewhere off the top of the window, which looks
    # exactly like the file failing to load. Frame it from the real bounds instead.
    VIEWPORT = (1600, 900)   # the smallest window we promise to open cleanly in
    # ComfyUI floats its toolbar and sidebar over the full-window canvas, so graph
    # content parked at 0,0 opens underneath them. Clear both.
    MARGIN_X, MARGIN_Y = 110, 120

    def _fit(self):
        xs, ys = [], []
        for n in self.nodes:
            x, y = n["pos"]
            w, h = n.get("size", [200, 100])
            xs += [x, x + w]
            ys += [y, y + h]
        for g in self.groups:
            x, y, w, h = g["bounding"]
            xs += [x, x + w]
            ys += [y, y + h]
        if not xs:
            return {"scale": 0.55, "offset": [80, 80]}
        vw, vh = self.VIEWPORT
        scale = min((vw - self.MARGIN_X - 40) / float(max(xs) - min(xs)),
                    (vh - self.MARGIN_Y - 40) / float(max(ys) - min(ys)))
        # Never zoom out past readable. Below about 0.5 litegraph drops to its
        # low-quality pass and stops drawing widget text, so a "fits on screen"
        # scale of 0.23 opens on a wall of blank grey boxes -- which reads as a
        # failed load just as badly as an empty canvas does. Opening part-way down
        # a legible graph is the normal ComfyUI experience; opening on nothing is not.
        scale = max(0.5, min(0.8, round(scale, 3)))
        # litegraph draws at (pos + offset) * scale, so this puts the graph's
        # top-left corner MARGIN pixels inside the top-left of the canvas.
        return {"scale": scale,
                "offset": [round(self.MARGIN_X / scale - min(xs), 1),
                           round(self.MARGIN_Y / scale - min(ys), 1)]}

    def dump(self, path):
        for node in self.nodes:
            for out in node["outputs"]:
                if not out["links"]:
                    out["links"] = None
        # The id must be a real string, not null. ComfyUI itself writes null here and
        # generates one on load, but ComfyUI-Manager monkey-patches app.handleFile --
        # the code path behind File -> Open and drag-and-drop -- and its hook reads
        # `.datetime` off the FIRST value in the JSON object to sniff for a component
        # pack. With "id": null first, that is null.datetime: a TypeError thrown inside
        # an async FileReader.onload, so it surfaces as an unhandled rejection, the
        # original handleFile is never called, and the workflow silently does not open.
        # A uuid5 keyed on the filename is stable across rebuilds and sidesteps it.
        wid = str(uuid.uuid5(uuid.NAMESPACE_URL,
                             "comfyui-h3-continuous/" + os.path.basename(path)))
        graph = {
            "id": wid, "revision": 0,
            "last_node_id": self.next_node - 1, "last_link_id": self.next_link - 1,
            "nodes": self.nodes, "links": self.links, "groups": self.groups,
            "config": {}, "extra": {"ds": self._fit()},
            "version": 0.4,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)
        print("wrote %s (%d nodes, %d links)" % (path, len(self.nodes), len(self.links)))
        return graph


# ---------------------------------------------------------------------------
# shared pieces
# ---------------------------------------------------------------------------
def models_column(b, x=40, y=40, steps=20, width=864, height=480, chain_seed=12345,
                  session_name="my_chain"):
    """Loaders + sampler + scheduler -> one H3 Chain Settings node."""
    unet = b.add("UNETLoader", (x, y + 80), size=[400, 100],
                 widgets=[b.pick("UNETLoader", "unet_name", "minimax_h3_ref2va"), "default"],
                 title="H3 ref2va model")
    lora = b.add("LoraLoaderModelOnly", (x, y + 230), size=[400, 100], mode=4,
                 widgets=[b.pick("LoraLoaderModelOnly", "lora_name", "minimax_h3", "turbo"), 1.0],
                 title="Turbo LoRA (bypassed - Ctrl+B to enable)", color=PURPLE)
    clip = b.add("CLIPLoader", (x, y + 380), size=[400, 110],
                 widgets=[b.pick("CLIPLoader", "clip_name", "minimax_h3"), "minimax", "default"],
                 title="Qwen3-VL text encoder")
    vae = b.add("VAELoader", (x, y + 540), size=[400, 70],
                widgets=[b.pick("VAELoader", "vae_name", "minimax_h3_video")],
                title="Video VAE")
    avae = b.add("VAELoader", (x, y + 650), size=[400, 70],
                 widgets=[b.pick("VAELoader", "vae_name", "minimax_h3_audio")],
                 title="Audio VAE")
    sampler = b.add("KSamplerSelect", (x, y + 760), size=[400, 70],
                    widgets=["res_multistep"])
    sched = b.add("BasicScheduler", (x, y + 870), size=[400, 130],
                  widgets=["simple", steps, 1.0],
                  title="Steps (20 full quality / 4 with the turbo LoRA)")

    b.link(unet, 0, lora, "model")
    b.link(unet, 0, sched, "model")

    # Trailing widgets in schema order: ref_image_size, handoff_mode, drift_arrest.
    # Spelled out rather than left to ComfyUI's defaults so the shipped workflows
    # show what they are set to instead of showing nothing. drift_arrest ships at 0
    # on purpose: it measurably steers the latent but overshoots the picture and has
    # never been shown to improve a frame anyone watches. See docs/how-it-works.md.
    settings = b.add("H3ChainSettings", (x + 480, y + 80), size=[400, 440],
                     widgets=[width, height, chain_seed, session_name, "match",
                              "latent", 0.0],
                     title="H3 Chain Settings", color=GREEN)
    b.link(lora, 0, settings, "model")
    b.link(clip, 0, settings, "clip")
    b.link(vae, 0, settings, "vae")
    b.link(avae, 0, settings, "audio_vae")
    b.link(sampler, 0, settings, "sampler")
    b.link(sched, 0, settings, "sigmas")

    # width/height are plain widgets on H3ChainSettings; convert them to inputs (the
    # same click a user gets from "Convert widget to input") so a ResolutionSelector
    # can drive them, same as the other MiniMax H3 example workflows on this machine.
    b.convert_widget_to_input(settings, "width", "INT")
    b.convert_widget_to_input(settings, "height", "INT")
    resolution = b.add(
        "ResolutionSelector", (x + 480, y + 500), size=[400, 150],
        widgets=[closest_aspect_ratio(width, height), round(width * height / 1e6, 3), 32],
        title="Canvas resolution -> Settings width/height", color=GREEN)
    b.link(resolution, 0, settings, "width")
    b.link(resolution, 1, settings, "height")

    b.group("1 - Models, sampler, canvas", (x - 20, y, 980, 1060), BLUE)
    return settings, vae, avae


def segment_autogrow(image_count):
    """One spare socket past the images actually wired, plus one of each other kind.

    The frontend grows these itself as you connect things; declaring the spare just
    means the node looks right the moment the workflow opens.
    """
    return [("images", "image_", "IMAGE", image_count + 1),
            ("videos", "video_", "IMAGE", 1),
            ("video_audios", "video_audio_", "AUDIO", 1),
            ("audios", "audio_", "AUDIO", 1)]


def output_column(b, last_segment, x, y, prefix, mode=0):
    """``last_segment`` is the final H3RenderSegment / H3RepairSegment node -- both
    share the (video, chain, summary) output order. ``mode=4`` ships the whole
    column bypassed, to match a mode group that starts inactive."""
    join = b.add("H3ChainToVideo", (x, y), size=[380, 140], widgets=[14.0, 1.0],
                 title="Join the segments", color=GREEN, mode=mode)
    b.link(last_segment, 1, join, "chain")
    save = b.add("SaveVideo", (x, y + 160), size=[560, 560],
                 widgets=[prefix, "auto", "auto"], mode=mode)
    b.link(join, 0, save, "video")
    summary = b.add("PreviewAny", (x, y + 760), size=[560, 260],
                    title="What was rendered", mode=mode)
    b.link(last_segment, 2, summary, "source")
    return join, save


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
TEMPLATE_FIRST = """subject_definitions:
<Subject 1> is the person in <Picture 2>: DESCRIBE FACE, HAIR, BUILD, CLOTHING IN FULL.
Only their face, hair, body and clothing are taken from <Picture 2>; the background of
that image is not part of the target video.
<Subject 2> is the place in <Picture 1>: DESCRIBE THE LOCATION. Only the layout, shapes,
materials and colours are used.

summary:
[reference generation] DESCRIBE THIS SHOT IN ONE SENTENCE.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - LIST WHAT MUST NOT CHANGE.
<Subject 2> (appears in [Shot 1]): partially_preserved - LIST WHAT CARRIES OVER.

detailed_description:
The target video is photorealistic live-action, DESCRIBE THE STYLE, LENS, LIGHT.
[Shot 1] DESCRIBE THE ACTION BEAT BY BEAT. Repeat your camera and lighting rules
word for word in every segment - that verbatim repetition is what keeps the look
identical across the joins. Spoken lines go in <d>[English] like this.</d>

overall_soundscape:
DESCRIBE THE DIEGETIC SOUND.

non_diegetic_music:
N/A"""

TEMPLATE_NEXT = """subject_definitions:
<Subject 1> is the person in <Picture 2>: REPEAT THE DESCRIPTION FROM SEGMENT 1 VERBATIM.
<Subject 2> is the place in <Picture 1>: DESCRIBE THIS SEGMENT'S LOCATION.

summary:
[reference generation + keyframe completion] The target video opens on an exact replay
of the closing 1.6 seconds of the preceding segment of the same unbroken take and
carries straight on from it as DESCRIBE THIS SHOT.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - LIST WHAT MUST NOT CHANGE.
<Subject 2> (appears in [Shot 1]): partially_preserved - LIST WHAT CARRIES OVER.

detailed_description:
The target video is photorealistic live-action, REPEAT THE STYLE, LENS AND LIGHT RULES
FROM SEGMENT 1 WORD FOR WORD.
[Shot 1] The segment opens on an exact replay of the closing moments of the preceding
take and carries that motion straight through, so the join is invisible. The subject
holds the exact body pose, head angle, gaze direction and facial expression carried in
by the replayed opening, continuing them without any reset before moving into the
actions below. DESCRIBE THE NEW ACTION - and schedule nothing before the 1.7 second
mark, because the first 1.6 seconds are already pinned.

overall_soundscape:
The ambience carries straight on from the preceding segment without interruption.
DESCRIBE THIS SEGMENT'S SOUND.

non_diegetic_music:
N/A"""


REPO_RAW = "https://raw.githubusercontent.com/loopforge0/ComfyUI-H3-Continuous/refs/heads/main/"

MODEL_LINKS = """**diffusion_models**

- [minimax_h3_ref2va_pruned_int8_convrot.safetensors]({hf}diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors) (20 GB)

**text_encoders**

- [qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors]({hf}text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) (15 GB)

**vae**

- [minimax_h3_video_vae_fp16.safetensors]({hf}vae/minimax_h3_video_vae_fp16.safetensors)
- [minimax_h3_audio_vae_fp32.safetensors]({hf}vae/minimax_h3_audio_vae_fp32.safetensors)

**loras**

- 4 steps: [minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors]({hf}loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors)

It must be the **ref2va** model, not `fl2va`. This pack chains *reference*-conditioned
renders and the fl2va model takes no reference images.

The LoRA is optional but you almost certainly want it: it cuts sampling from 20 steps to
4, which on a 12 GB card is about 9 minutes a shot instead of 31. Select the **Turbo
LoRA** node, press **Ctrl+B** to un-bypass it, and set `steps` to 4.

## Model Storage Location

```
\U0001f4c2 ComfyUI/
├── \U0001f4c2 models/
│   ├── \U0001f4c2 diffusion_models/
│   │   └── minimax_h3_ref2va_pruned_int8_convrot.safetensors
│   ├── \U0001f4c2 text_encoders/
│   │   └── qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors
│   ├── \U0001f4c2 vae/
│   │   ├── minimax_h3_video_vae_fp16.safetensors
│   │   └── minimax_h3_audio_vae_fp32.safetensors
│   └── \U0001f4c2 loras/
│       └── minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
└── \U0001f4c2 input/
{input_tree}
```

Every loader in this workflow carries its download link, so if a file is missing ComfyUI
offers to fetch it when the workflow opens. Accept, or place the file by hand.

## Report Issue

Note: update ComfyUI first ([guide](https://docs.comfy.org/installation/update_comfyui)).
This pack needs **ComfyUI 0.34.0 or newer** - older builds have no MiniMax H3 nodes at
all, so nothing here will load. Check yours under *Settings → About*.

- This pack, its nodes or these workflows: [ComfyUI-H3-Continuous/issues](https://github.com/loopforge0/ComfyUI-H3-Continuous/issues)
- ComfyUI itself, or the H3 nodes it ships: [ComfyUI/issues](https://github.com/comfyanonymous/ComfyUI/issues)
- UI / frontend: [ComfyUI_frontend/issues](https://github.com/Comfy-Org/ComfyUI_frontend/issues)"""


def models_md(inputs=None):
    """The Model Links / Storage / Report Issue block for a workflow's Start-here note.

    Laid out the way Comfy-Org's own workflow templates present this: links grouped by
    the folder each file goes in, then a storage tree, then where to report a problem.

    ``inputs`` is an optional list of (filename, what it is) pairs for a workflow that
    ships with its reference images, rendered as the Input Assets section.
    """
    assets = ""
    tree = "    └── your reference images"
    if inputs:
        assets = "**Input Assets**\n\n%s\n\n" % "\n".join(
            "- [%s](%sexample_inputs/%s) - %s (`input/%s`)" % (n, REPO_RAW, n, what, n)
            for n, what in inputs)
        rows = ["    ├── %s" % n for n, _ in inputs[:-1]]
        rows.append("    └── %s" % inputs[-1][0])
        tree = "\n".join(rows)
    return ("## Model Links\n\n" + assets
            + MODEL_LINKS.format(hf=HF, input_tree=tree))


# ---------------------------------------------------------------------------
# workflow 1: the chain
# ---------------------------------------------------------------------------
CHAIN_INTRO = """# Continuous H3 video - chain workflow

Render any number of H3 segments as **one unbroken take**. Each segment opens on an
exact replay of the last 1.6 s of the segment before it, pinned with
`MiniMaxH3AddGuide`, so the joins are invisible rather than merely similar.

## Do this
1. **Models** (blue) - point the five loaders at your H3 files (listed below).
2. **Shots** (purple) - one **H3 Render Segment** per shot, wired in a row: each one's
   `chain_state` output feeds the next one's `chain_state` input. Write a prompt per
   segment and wire its reference images. `image_0` is `<Picture 1>`, `image_1` is
   `<Picture 2>`, and so on. The session name lives on **H3 Chain Settings**, one
   place for the whole chain.
3. **Join** (green) - the last segment's `chain_state` feeds H3 Chain to Video. Run.

## Each shot is its own output, live
An H3 Render Segment's `video` output is real the moment THAT segment finishes -- wire
a Preview Video node to segment 3 and it plays as soon as segment 3 is done, whether or
not segments 4+ have even started. You do not wait for the whole take to check one shot.

## Two things that will bite you
**Never label the carried-in clip.** The previous segment's tail is anchored, not
tokenized, so it has no `<Video>` tag. Describe it in prose. A `<Video 1>` written for
it is an unresolved reference and the segment falls apart.

**Nothing new can happen in the first 1.6 s** of any segment after the first - those
frames are a fixed replay. An entrance written for "the one-second mark" lands inside
frames that are already pinned.

## It resumes
Every finished segment is written to `output/h3_continuous/<session>/` -- and the
manifest is rewritten after every single segment, not just at the end, so even a hard
kill leaves it accurate. Re-running reuses everything that did not change, so editing
segment 3's prompt re-renders 3 onward and leaves 1 and 2 alone. Interrupting mid-chain
costs you only the segment in flight.

Broke one segment out of an otherwise good chain? Use the **repair** workflow - it
re-renders a single segment pinned at *both* ends, so nothing downstream has to move."""


SHOTS_NOTE = """## Your shots

One **H3 Render Segment** per shot, wired top to bottom: each node's `chain_state`
output goes into the next one's `chain_state` input. The first segment's `chain_state`
input stays unconnected -- that is the one that reads `session_name` off **H3 Chain
Settings**; every segment after it inherits the session through the wire instead.

- `seconds` - length of this shot (8 s = 192 frames).
- `handoff_seconds` *(advanced)* - how much of **this** shot's ending the **next** one
  opens on. 1.625 s is the sweet spot. Set to 0 on the last shot of the take.
- `seed_override` *(advanced)* - leave at 0 to derive it from the chain seed; set it to
  re-roll just this shot.

Reference images are numbered by socket: `image_0` is `<Picture 1>`, `image_1` is
`<Picture 2>`. One loader can feed several segments - that is how a character stays the
same person all the way through.

Need more shots? Drag in another H3 Render Segment and wire it into the chain. Need
fewer? Delete one and reconnect the `chain_state` wire around it."""


def build_chain(info, segments=4, prompts=None, session="my_chain",
                images=None, title_note=CHAIN_INTRO, seconds=8.0, handoff=1.625,
                width=864, height=480, chain_seed=12345, steps=20,
                example_inputs=None):
    """``images`` is one list of filenames per segment, in <Picture 1..N> order."""
    images = images or [["example.png"] for _ in range(segments)]
    b = Builder(info)
    b.note((40, -2000), (980, 960), models_md(example_inputs), "Models", BLUE)
    b.note((40, -1000), (980, 940), title_note, "Start here", YELLOW)
    settings, _, _ = models_column(b, steps=steps, width=width, height=height,
                                   chain_seed=chain_seed, session_name=session)

    x_img, x_seg = 1100, 1560
    b.note((x_seg, -520), (420, 420), SHOTS_NOTE, "Shots", PURPLE)

    loaders = {}   # filename -> node, so a shared character is loaded once
    row = 40
    prev_seg = None
    for i in range(segments):
        prompt = (prompts[i] if prompts else
                  (TEMPLATE_FIRST if i == 0 else TEMPLATE_NEXT))
        # seconds may be one value for the whole chain, or one per segment
        secs = seconds[i] if isinstance(seconds, (list, tuple)) else seconds
        this_handoff = 0.0 if i == segments - 1 else handoff
        seg = b.add("H3RenderSegment", (x_seg, row), size=[420, 660],
                    widgets=[True, prompt, secs, this_handoff, 0, False],
                    autogrow=segment_autogrow(len(images[i])),
                    title="Segment %d" % (i + 1), color=PURPLE)
        b.link(settings, 0, seg, "settings")
        if prev_seg is not None:
            b.link(prev_seg, 1, seg, "chain_state")  # chain_state -> chain_state
        stacked = 0
        for slot, filename in enumerate(images[i]):
            loader = loaders.get(filename)
            if loader is None:
                loader = b.add("LoadImage", (x_img, row + stacked * 330), size=[320, 300],
                               widgets=[filename], title="<Picture %d>" % (slot + 1))
                loaders[filename] = loader
                stacked += 1
            b.link(loader, 0, seg, "images.image_%d" % slot)
        prev_seg = seg
        row += 760

    b.group("2 - Your shots, in order -- each one renders here", (x_img - 20, 0, 940, row - 40),
            PURPLE)
    output_column(b, prev_seg, 2620, 120, "video/h3_continuous")
    b.group("3 - Join and save", (2600, 60, 600, 1080), GREEN)
    return b


# ---------------------------------------------------------------------------
# workflow 2: repair
# ---------------------------------------------------------------------------
REPAIR_INTRO = """# Continuous H3 video - repair workflow

One segment came out wrong. Normally re-rendering segment N changes its ending, so
every segment after it stops joining and has to be re-rendered too.

**H3 Repair Segment pins both ends.** The opening is pinned to the previous segment's
handoff clip, and the ending is pinned to *the very clip the next segment already opens
on*. The replacement lands between two fixed points, so nothing downstream moves.

## Do this
1. Point **Models** at the same files the chain used, and use the same `width`,
   `height` and `seed` in **H3 Chain Settings**.
2. Type the `session_name` you rendered with. **H3 Load Session** will print what is
   in it - segment numbers, lengths, seeds.
3. Set `segment_number` to the one you are replacing (1-based, same as `seg_NN.mp4`).
4. Rebuild that shot directly on the **H3 Repair Segment** node: the same reference
   images, the same length, the fixed prompt.
5. Run. The old file is kept as `seg_NN.replaced.mp4` before it is overwritten, the
   node's own `video` output plays the replacement immediately, and the whole cut is
   re-joined.

## Notes
- Leave `pin_ending` **on**. Turn it off only for the very last segment, or when you
  intend to re-render everything after this one.
- The pinned ending is anchored as picture only. Its audio comes from the render you
  are throwing away, and re-anchoring that would drag back exactly what you are
  removing.
- Keep the replacement the same length as the original. Repairs are for a bad take,
  not for a re-cut."""


def build_repair(info, session="my_chain", shared_image=None, image=None):
    b = Builder(info)
    b.note((40, -2040), (980, 980), models_md(), "Models", BLUE)
    b.note((40, -1040), (980, 980), REPAIR_INTRO, "Start here", RED)
    settings, _, _ = models_column(b)

    load = b.add("H3LoadSession", (560, 40), size=[380, 120],
                 widgets=[session, "latent"],
                 title="Which session?", color=GREEN)
    listing = b.add("PreviewAny", (560, 190), size=[380, 280],
                    title="What is in it")
    b.link(load, 1, listing, "source")
    b.group("2a - Which session? (read-only check)", (540, 0, 440, 510), GREEN)

    loaders = []
    row = 520
    if shared_image:
        loaders.append(b.add("LoadImage", (560, row), size=[380, 420],
                             widgets=[shared_image], title="Character"))
        row += 460
    if image:
        loaders.append(b.add("LoadImage", (560, row), size=[380, 420],
                             widgets=[image], title="Segment reference"))

    repair = b.add("H3RepairSegment", (1080, 40), size=[420, 660],
                   widgets=[session, 2, TEMPLATE_NEXT, 8.0, 1.625, 0, True],
                   autogrow=segment_autogrow(2),
                   title="H3 Repair Segment", color=RED)
    b.link(settings, 0, repair, "settings")
    for i, loader in enumerate(reversed(loaders)):
        b.link(loader, 0, repair, "images.image_%d" % i)

    output_column(b, repair, 1560, 40, "video/h3_continuous_repaired")
    b.group("2b - The segment you are replacing, join and save", (1060, 0, 940, 1080), RED)
    return b


# ---------------------------------------------------------------------------
# workflow 3: both modes, one canvas
# ---------------------------------------------------------------------------
COMBINED_INTRO = """# Continuous H3 video - one file, two modes

This canvas holds BOTH workflows, sharing one **H3 Chain Settings**. Only one mode
should be active when you Run -- the other's groups must be bypassed. It is the same
idiom as the turbo LoRA toggle in Models: shipped bypassed, Ctrl+B to enable.

## Which mode is active
- **A - RENDER** (purple/green, top) -- active by default. Chains H3 Render Segment
  nodes into a new (or resumed) take.
- **B - REPAIR** (red/green, bottom) -- shipped bypassed. Fixes one segment of a
  session that already finished, pinned at both ends so nothing downstream moves.

## To switch modes
Click a group's title bar to select every node inside it, then press **Ctrl+B** to
toggle bypass on the whole selection. Bypass A's two groups and un-bypass B's two
(or the reverse) before you Run -- running with both active renders, and pays GPU
time for, both at once.

Each mode's own group still carries its full walkthrough note -- read that one once
you have picked a mode."""


def build_combined(info, segments=4, prompts=None, session="my_chain", images=None,
                   seconds=8.0, handoff=1.625, width=864, height=480, chain_seed=12345,
                   steps=20, repair_session=None, repair_shared_image=None,
                   repair_image=None):
    """One file: the render chain (active) and the repair node (bypassed), sharing
    one H3 Chain Settings. Switching modes is a Ctrl+B on each mode's groups."""
    images = images or [["example.png"] for _ in range(segments)]
    repair_session = repair_session or session
    b = Builder(info)
    b.note((40, -600), (1420, 520), COMBINED_INTRO, "Start here -- read this first", YELLOW)
    settings, _, _ = models_column(b, steps=steps, width=width, height=height,
                                   chain_seed=chain_seed, session_name=session)

    # -- A: render chain, active by default -----------------------------------
    x_img, x_seg = 1100, 1560
    b.note((x_seg, -600), (420, 460), SHOTS_NOTE, "A - Shots", PURPLE)

    loaders = {}
    row = 40
    prev_seg = None
    for i in range(segments):
        prompt = (prompts[i] if prompts else
                  (TEMPLATE_FIRST if i == 0 else TEMPLATE_NEXT))
        secs = seconds[i] if isinstance(seconds, (list, tuple)) else seconds
        this_handoff = 0.0 if i == segments - 1 else handoff
        seg = b.add("H3RenderSegment", (x_seg, row), size=[420, 660],
                    widgets=[True, prompt, secs, this_handoff, 0, False],
                    autogrow=segment_autogrow(len(images[i])),
                    title="Segment %d" % (i + 1), color=PURPLE)
        b.link(settings, 0, seg, "settings")
        if prev_seg is not None:
            b.link(prev_seg, 1, seg, "chain_state")
        stacked = 0
        for slot, filename in enumerate(images[i]):
            loader = loaders.get(filename)
            if loader is None:
                loader = b.add("LoadImage", (x_img, row + stacked * 330), size=[320, 300],
                               widgets=[filename], title="<Picture %d>" % (slot + 1))
                loaders[filename] = loader
                stacked += 1
            b.link(loader, 0, seg, "images.image_%d" % slot)
        prev_seg = seg
        row += 760

    a_bottom = row
    output_column(b, prev_seg, 2620, 120, "video/h3_continuous")
    b.group("A - RENDER (active) - your shots, in order", (x_img - 20, -20, 940, a_bottom),
            PURPLE)
    b.group("A - RENDER (active) - join and save", (2600, 40, 600, 1080), GREEN)

    # -- B: repair, shipped bypassed -------------------------------------------
    y = max(1100, a_bottom) + 400
    b.note((40, y - 460), (980, 420), REPAIR_INTRO, "B - Repair", RED)

    load = b.add("H3LoadSession", (560, y), size=[380, 120],
                 widgets=[repair_session, "latent"],
                 title="Which session?", color=GREEN, mode=4)
    listing = b.add("PreviewAny", (560, y + 150), size=[380, 280],
                    title="What is in it", mode=4)
    b.link(load, 1, listing, "source")
    b.group("B - REPAIR (bypassed) - which session? (read-only check)",
            (540, y - 40, 440, 510), GREEN)

    repair_loaders = []
    row_r = y + 480
    if repair_shared_image:
        repair_loaders.append(b.add("LoadImage", (560, row_r), size=[380, 420],
                                    widgets=[repair_shared_image], title="Character",
                                    mode=4))
        row_r += 460
    if repair_image:
        repair_loaders.append(b.add("LoadImage", (560, row_r), size=[380, 420],
                                    widgets=[repair_image], title="Segment reference",
                                    mode=4))

    repair = b.add("H3RepairSegment", (1080, y), size=[420, 660],
                   widgets=[repair_session, 2, TEMPLATE_NEXT, 8.0, 1.625, 0, True],
                   autogrow=segment_autogrow(2), title="H3 Repair Segment", color=RED,
                   mode=4)
    b.link(settings, 0, repair, "settings")
    for i, loader in enumerate(reversed(repair_loaders)):
        b.link(loader, 0, repair, "images.image_%d" % i)

    output_column(b, repair, 1560, y, "video/h3_continuous_repaired", mode=4)
    b.group("B - REPAIR (bypassed) - the segment you are replacing, join and save",
            (1060, y - 40, 940, 1080), RED)

    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--out", default=os.path.join(HERE, "workflows"))
    ap.add_argument("--rebuild-example", action="store_true",
                    help="regenerate the stage example, discarding its hand-made layout")
    a = ap.parse_args()

    info = json.load(urllib.request.urlopen(a.server + "/object_info", timeout=60))
    for required in ("H3RenderSegment", "H3ChainToVideo", "MiniMaxH3ReferenceToVideo",
                     "ResolutionSelector"):
        if required not in info:
            raise SystemExit("%s is missing on %s -- install the pack and restart ComfyUI"
                             % (required, a.server))

    # The pack ships one workflow: the stage example, with the repair half appended
    # below it and bypassed (tools/append_repair.py). build_chain and build_repair stay
    # as the generators behind it -- append_repair sources its nodes from build_repair,
    # so both still have to track the live schema.

    # The stage example: three segments, two references, one continuous song.
    # Its shipped layout was arranged by hand and adopted with tools/adopt_workflow.py,
    # so regenerating it would throw that away. Only build it if it is missing, or if
    # --rebuild-example says to start over from the generated layout.
    sys.path.insert(0, os.path.join(HERE, "tools"))
    import kate_stage_prompts as stage
    example = os.path.join(a.out, "h3_continuous.json")
    if a.rebuild_example or not os.path.exists(example):
        build_chain(info, segments=3, prompts=stage.PROMPTS, session="stage_song",
                    images=stage.IMAGES, title_note=EXAMPLE_INTRO,
                    width=480, height=864,
                    example_inputs=[("kate.png", "`<Picture 1>`, the singer"),
                                    ("stage.png", "`<Picture 2>`, the stage")]).dump(example)
    else:
        print("kept %s (hand-arranged -- pass --rebuild-example to regenerate)" % example)


EXAMPLE_INTRO = """# Stage song - the worked example

Three shots, one unbroken take: a singer alone in a followspot performs a short
a cappella song and dances between the lines. Three 8-second renders joined with the
default 1.625 s handoff make about 20.75 seconds of finished video.

## Run it
1. Put `kate.png` and `stage.png` in `ComfyUI/input/` (they are in this repo, under
   `example_inputs/`), then Run. Nothing else needs setting.
2. On a 12 GB card with the turbo LoRA on and `steps` at 4, budget about 27 minutes.

Every segment carries the same two references: `image_0` is the singer (`<Picture 1>`)
and `image_1` is the stage (`<Picture 2>`). Feeding both into all three shots is what
holds her face and the room together across the joins.

## Read it
Put segment 2's prompt next to segment 1's and the whole method is visible:

- The **camera, lighting and subject blocks are repeated word for word** in all three.
  That verbatim repetition, not the anchor, is what keeps the look identical.
- Segments 2 and 3 **open by finishing a move already in progress** - the turn, then
  the pivot - because their first 1.625 s is a fixed replay of the segment before. The
  first sung word lands after the 1.7 s mark in both.
- Each segment **sings, then dances, never both at once**, and ends on a beat with her
  mouth closed. A dance move underneath a sung line freezes the mouth mid-word, and a
  half-sung word cannot be resumed out of a replay.

## Why there is no music
`non_diegetic_music` is `N/A` in all three, and the stage's drum kit, keyboards and
guitar are explicitly never played. Three independently generated score beds would not
match across the joins - the key would move and the tempo would drift, making the two
seams the most audible thing in the video. Score the finished cut in post instead,
where it is one continuous track."""


if __name__ == "__main__":
    main()
