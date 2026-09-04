"""Render the same chain under different settings and measure what changed.

A chain is expensive -- on a 12 GB card an 8 s segment at 20 steps is half an hour
-- so "does this change help" is not a question you can answer by eye on one run.
This builds the same six-shot take under N different settings, renders each through
a running ComfyUI, and hands the results to ``face_drift.py``.

Every arm renders the identical prompts, references, seed, resolution and sampler.
Only the named knob moves. Arms land in separate sessions so nothing is shared and
nothing is reused across them by accident.

    python bench_chain.py --arms pixel latent
    python bench_chain.py --arms latent --steps 20 --tag hq
    python bench_chain.py --report            # just re-measure what is on disk

Requires a ComfyUI already running with this pack installed (default
http://127.0.0.1:8188).
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kate_drift6_prompts import HANDOFF, PROMPTS, SECONDS  # noqa: E402


def comfy_root():
    """ComfyUI's root directory.

    Installed normally this pack sits in ComfyUI/custom_nodes/<pack>, so the root is two
    levels up. Set COMFYUI_DIR when the checkout lives somewhere else.
    """
    return os.environ.get("COMFYUI_DIR") or os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))


SERVER = os.environ.get("H3_COMFY", "http://127.0.0.1:8188")

UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
TURBO_LORA = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REFERENCE = "kate_closeup.png"
# A frame lifted out of segment 1's own render. A photo reference is out of
# distribution -- it was shot on a real camera, and the model has to translate it
# every segment. Its own segment 1 is not: same lens, same light, same rendition.
# Handing that back as <Picture 2> gives every later segment an absolute appearance
# target that costs it no translation.
SELF_REFERENCE = "kate_selfref.png"
SELF_REF_CLAUSE = (
    "\n\n<Picture 2> is a frame taken from earlier in this same continuous take. It "
    "shows exactly how <Subject 1> looks in this footage -- her face, hair, colouring, "
    "sweater, the lighting on her and the framing of the shot -- and the target video "
    "must match it exactly in all of those respects.")

# The knobs each arm moves. Anything not named here is identical across arms:
# same prompts, same reference image, same seed, same resolution, same sampler.
ARMS = {
    # What the pack shipped: decode the tail, write an mp4, VAE-encode it back.
    "pixel": {"handoff_mode": "pixel"},
    # Same chain, but the tail is sliced straight out of the sampled latent.
    "latent": {"handoff_mode": "latent"},
    # Latent handoff plus the closed loop back to segment 1's own colour.
    "arrest": {"handoff_mode": "latent", "drift_arrest": 0.5},
    "arrest_strong": {"handoff_mode": "latent", "drift_arrest": 0.8},
    # Every segment after the first also gets a still from segment 1's own output.
    "selfref": {"handoff_mode": "latent", "self_reference": True},
}


def build_prompt(session, *, handoff_mode="latent", drift_arrest=0.0, steps=4, turbo=True,
                 width=480, height=864, chain_seed=8814402, segments=6,
                 seconds=None, handoff=HANDOFF, prompts=None, reference=REFERENCE,
                 ref_image_size="match", join=True, self_reference=False):
    """The API-format graph: loaders, one H3 Render Segment per shot, one join."""
    seconds = seconds or SECONDS
    prompts = prompts or PROMPTS
    if segments > len(prompts):
        raise ValueError("only %d prompts available" % len(prompts))

    g = {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "avae": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "sampler": {"class_type": "KSamplerSelect",
                    "inputs": {"sampler_name": "res_multistep"}},
        "ref": {"class_type": "LoadImage", "inputs": {"image": reference}},
    }
    if self_reference:
        g["selfref"] = {"class_type": "LoadImage",
                        "inputs": {"image": SELF_REFERENCE}}
    model_src = ["unet", 0]
    if turbo:
        g["lora"] = {"class_type": "LoraLoaderModelOnly",
                     "inputs": {"model": ["unet", 0], "lora_name": TURBO_LORA,
                                "strength_model": 1.0}}
        model_src = ["lora", 0]

    g["sigmas"] = {"class_type": "BasicScheduler",
                   "inputs": {"model": model_src, "scheduler": "simple",
                              "steps": steps, "denoise": 1.0}}
    g["settings"] = {"class_type": "H3ChainSettings",
                     "inputs": {"model": model_src, "clip": ["clip", 0],
                                "vae": ["vae", 0], "audio_vae": ["avae", 0],
                                "sampler": ["sampler", 0], "sigmas": ["sigmas", 0],
                                "width": width, "height": height,
                                "chain_seed": chain_seed, "session_name": session,
                                "ref_image_size": ref_image_size,
                                "handoff_mode": handoff_mode,
                                "drift_arrest": drift_arrest}}

    previous = None
    for i in range(segments):
        node = "seg%d" % (i + 1)
        inputs = {
            "settings": ["settings", 0],
            "resume": True,
            "prompt": prompts[i],
            "seconds": seconds[i],
            # The last shot hands off to nothing, so it writes no tail.
            "handoff_seconds": handoff if i < segments - 1 else 0.0,
            "seed_override": 0,
            "unload_models_after": False,
            "images.image_0": ["ref", 0],
        }
        if self_reference and i > 0:
            # Segment 1 cannot reference itself, and does not need to.
            inputs["images.image_1"] = ["selfref", 0]
            inputs["prompt"] = prompts[i] + SELF_REF_CLAUSE
        if previous:
            inputs["chain_state"] = [previous, 1]
        g[node] = {"class_type": "H3RenderSegment", "inputs": inputs}
        previous = node

    # ComfyUI will not run a graph that reaches no output node. PreviewAny on a
    # string is the cheapest possible one: it makes the join (or the last segment)
    # reachable without asking for a second encode of footage the pack already wrote.
    if join:
        g["join"] = {"class_type": "H3ChainToVideo",
                     "inputs": {"chain": [previous, 1], "crf": 14.0}}
        g["out"] = {"class_type": "PreviewAny", "inputs": {"source": ["join", 1]}}
    else:
        g["out"] = {"class_type": "PreviewAny", "inputs": {"source": [previous, 2]}}
    return g


def post(path, payload):
    req = urllib.request.Request(
        SERVER + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        # ComfyUI puts the useful part -- which input on which node it rejected --
        # in the 400 body, which urllib otherwise throws away.
        raise RuntimeError("%s %s\n%s" % (exc.code, exc.reason,
                                          exc.read().decode("utf-8", "replace")[:4000]))


def get(path):
    with urllib.request.urlopen(SERVER + path, timeout=60) as r:
        return json.loads(r.read().decode())


def wait(prompt_id, poll=15, quiet_after=None):
    """Block until the prompt leaves the queue, then surface any node error."""
    started = time.time()
    while True:
        history = get("/history/%s" % prompt_id)
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error" or not status.get("completed", True):
                for message in status.get("messages", []):
                    if message[0] in ("execution_error", "execution_interrupted"):
                        raise RuntimeError("%s: %s" % (
                            message[1].get("node_type"),
                            message[1].get("exception_message", message[1])))
                raise RuntimeError("prompt %s did not complete: %s" % (prompt_id, status))
            return time.time() - started
        queue = get("/queue")
        running = queue.get("queue_running", [])
        pending = queue.get("queue_pending", [])
        if not any(item[1] == prompt_id for item in running + pending):
            # Not in history and not queued: it was cancelled or the server restarted.
            time.sleep(poll)
            history = get("/history/%s" % prompt_id)
            if prompt_id not in history:
                raise RuntimeError("prompt %s vanished from the queue" % prompt_id)
            continue
        if quiet_after is None or time.time() - started < quiet_after:
            print("    ... %5.1f min elapsed, %d running / %d queued"
                  % ((time.time() - started) / 60.0, len(running), len(pending)),
                  flush=True)
        time.sleep(poll)


def resolve_dir(comfy_root, session):
    return os.path.join(comfy_root, "output", "h3_continuous", session)


def run_arm(name, session, overrides, common):
    print("\n=== %s -> session %s" % (name, session), flush=True)
    print("    %s" % ", ".join("%s=%s" % kv for kv in sorted(overrides.items())),
          flush=True)
    graph = build_prompt(session, **dict(common, **overrides))
    response = post("/prompt", {"prompt": graph})
    if "error" in response:
        raise RuntimeError(json.dumps(response, indent=2)[:2000])
    prompt_id = response["prompt_id"]
    print("    queued %s" % prompt_id, flush=True)
    elapsed = wait(prompt_id)
    print("    done in %.1f min" % (elapsed / 60.0), flush=True)
    return elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["pixel", "latent"],
                    help="which arms to render (%s)" % ", ".join(sorted(ARMS)))
    ap.add_argument("--tag", default="drift", help="session name prefix")
    ap.add_argument("--segments", type=int, default=6)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=864)
    ap.add_argument("--seed", type=int, default=8814402)
    ap.add_argument("--full-steps", action="store_true",
                    help="drop the turbo LoRA and sample at --steps properly")
    ap.add_argument("--comfy-root", default=comfy_root(),
                    help="ComfyUI root (default: derived, or $COMFYUI_DIR)")
    ap.add_argument("--report", action="store_true",
                    help="skip rendering, just measure the sessions on disk")
    args = ap.parse_args()

    common = {
        "segments": args.segments,
        "steps": args.steps,
        "turbo": not args.full_steps,
        "width": args.width,
        "height": args.height,
        "chain_seed": args.seed,
    }

    sessions = {}
    for name in args.arms:
        if name not in ARMS:
            raise SystemExit("unknown arm %r; known: %s" % (name, ", ".join(sorted(ARMS))))
        sessions[name] = "%s_%s" % (args.tag, name)

    if not args.report:
        for name in args.arms:
            try:
                run_arm(name, sessions[name], ARMS[name], common)
            except Exception as exc:
                print("    ARM FAILED: %s" % exc, flush=True)

    print("\n" + "=" * 72)
    dirs = [resolve_dir(args.comfy_root, s) for s in sessions.values()]
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        print("nothing rendered yet")
        return
    import face_drift
    summary = {}
    for d in dirs:
        try:
            summary[os.path.basename(d)] = face_drift.report(
                os.path.basename(d), face_drift.analyse(d))
        except Exception as exc:
            print("  %s: %s" % (os.path.basename(d), exc))
    if len(summary) > 1:
        print("\nside by side (segment 1 -> last)")
        print("  %-26s %9s %8s %8s %10s" % ("arm", "identity", "dE", "detail", "per-hop"))
        for name, s in summary.items():
            print("  %-26s %9.4f %8.2f %8.2f %10.4f"
                  % (name, s["identity"], s["dE"], s["detail"],
                     (1.0 - s["identity"]) / s["hops"]))


if __name__ == "__main__":
    main()
